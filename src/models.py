"""
Models for CD-ISG-TS and baselines.

Core pieces
-----------
RevIN            : reversible per-window instance normalization. Removes the
                   per-window level & scale (encodes facility size/identity) and
                   restores it at the output -> prevents scale confound (caution #1)
                   and per-facility calibration leakage (caution #6).

WaveletOp        : a FIXED, ORTHONORMAL multi-level DWT written as a single matrix
                   W (built once from pywt, mode='periodization').  Because W is
                   orthonormal, IDWT = W^T and reconstruction is exact.  Implemented
                   as a torch buffer so the whole transform is differentiable and the
                   learnable gate acts directly in wavelet-coefficient space.

SubBandGate      : the CD-ISG-TS module.  Per staff-category (channel) and per
                   sub-band it holds a learnable gate g in (0,1).  Gates start OPEN
                   (g~1) so at initialization  IDWT(g . DWT(x)) == x  and the model
                   is exactly ERM; training then learns to suppress domain-variant
                   sub-bands.  Also exposes gated band energies grouped for the
                   cross-domain invariance regularizer.

Backbones        : DLinear (primary, CPU-cheap) and a small PatchTST-style
                   Transformer (secondary, for backbone-transfer tests).

Full models      : ERM, CDISGTS (ours), FixedBand (fixed low-pass heuristic),
                   WaveMaskModel (input wavelet augmentation), plus IRM & FFT-mask
                   variants handled in train.py / via flags here.
"""
import numpy as np, torch, torch.nn as nn, pywt


# ----------------------------------------------------------------------------- #
#  Reversible Instance Normalization
# ----------------------------------------------------------------------------- #
class RevIN(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps; self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):                      # x: [B, C, L]
        self.mean = x.mean(-1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(-1, keepdim=True, unbiased=False) + self.eps).detach()
        xn = (x - self.mean) / self.std
        if self.affine:
            xn = xn * self.gamma[None, :, None] + self.beta[None, :, None]
        return xn

    def denorm(self, y):                       # y: [B, C, H]
        if self.affine:
            y = (y - self.beta[None, :, None]) / (self.gamma[None, :, None] + self.eps ** 0.5)
        return y * self.std + self.mean


# ----------------------------------------------------------------------------- #
#  Fixed orthonormal multi-level DWT as a matrix operator
# ----------------------------------------------------------------------------- #
def build_wavelet_matrix(L, wavelet="db4", level=3):
    """Return (W [n,L] float32, band_id [n] int) with band 0=coarsest approx.

    W is orthonormal so IDWT == W.T ; band_id labels each coefficient's sub-band."""
    w = pywt.Wavelet(wavelet)
    # transform each unit vector to read off the analysis matrix
    cols, band_of = [], None
    lengths = None
    for i in range(L):
        e = np.zeros(L); e[i] = 1.0
        coeffs = pywt.wavedec(e, w, mode="periodization", level=level)
        if lengths is None:
            lengths = [len(c) for c in coeffs]      # [cA_L, cD_L, ..., cD_1]
        cols.append(np.concatenate(coeffs))
    W = np.stack(cols, axis=1)                       # [n_coeff, L]
    band_id = np.concatenate([np.full(l, b, int) for b, l in enumerate(lengths)])
    return W.astype(np.float32), band_id.astype(np.int64), lengths


class SubBandGate(nn.Module):
    """CD-ISG-TS core: differentiable DWT -> learnable per-(category,band) gate -> IDWT.

    marginal=True collapses the gate across categories (for the RQ2 ablation)."""
    def __init__(self, L, n_channels, wavelet="db4", level=3, marginal=False,
                 residual=True, gate_init=2.5, adaptive=False):
        super().__init__()
        W, band_id, lengths = build_wavelet_matrix(L, wavelet, level)
        self.register_buffer("W", torch.from_numpy(W))          # [n, L]
        self.register_buffer("band_id", torch.from_numpy(band_id))   # [n]
        self.n_bands = len(lengths)
        self.lengths = lengths
        self.C = n_channels
        self.marginal = marginal
        self.residual = residual
        self.adaptive = adaptive
        g0 = gate_init    # sigmoid(2.5)=0.924 -> near-open but responsive
        if marginal:
            self.gate_logit = nn.Parameter(torch.full((1, self.n_bands), g0))
        else:
            self.gate_logit = nn.Parameter(torch.full((n_channels, self.n_bands), g0))
        if adaptive:
            # Input-conditioned (amortized) gating. The static gate above must find
            # ONE compromise gating for every entity; this band-interaction map lets
            # the gate specialize to the window actually presented, using only the
            # input's own relative band energies. That is legal under domain
            # generalization: it conditions on x alone, never on target labels or
            # target-domain statistics, so it needs no adaptation step at deployment.
            # A[c] is a readable n_bands x n_bands matrix per category ("when this
            # entity carries more energy in D1, suppress D2"), so interpretability is
            # preserved. Zero-init makes the module reduce EXACTLY to the static gate
            # at initialization, which in turn reduces exactly to the identity, so the
            # safety guarantee of the original design is retained.
            self.band_map = nn.Parameter(torch.zeros(n_channels, self.n_bands,
                                                     self.n_bands))
        # map band -> coefficient positions
        self.register_buffer("band_onehot",
                             torch.nn.functional.one_hot(torch.from_numpy(band_id),
                                                         self.n_bands).float())  # [n, n_bands]
        self.register_buffer("band_len", torch.tensor(lengths, dtype=torch.float32))

    def _normalize(self, g):
        """Mean-one normalization: hold the coefficient-count-weighted mean gate at 1
        (sum_j n_j g_j = sum_j n_j). This rules out uniform shutdown of every band,
        but it does NOT preserve signal energy (sum_j g_j^2 E_j); retained energy is
        measured separately in experiments/gate.py."""
        scale = self.band_len.sum() / ((g * self.band_len).sum(-1, keepdim=True) + 1e-8)
        return g * scale

    def gates(self):
        """Static (entity-independent) gates -- the interpretable summary reported in
        the paper. With adaptive=True this is the baseline about which the
        input-conditioned offset moves."""
        g = torch.sigmoid(self.gate_logit)                       # [C or 1, n_bands]
        if self.marginal:
            g = g.expand(self.C, -1)
        return self._normalize(g)

    def _gates_for(self, coeff):
        """Per-sample gates [B, C, n_bands]. Static case broadcasts the shared gate."""
        if not self.adaptive:
            return self.gates()[None].expand(coeff.shape[0], -1, -1)
        # relative band energy of the INPUT window (detached: this is a descriptor of
        # the window, not a path we want the invariance penalty to game)
        e_in = torch.einsum("bcn,nk->bck", coeff.pow(2), self.band_onehot)
        rho = (e_in / (e_in.sum(-1, keepdim=True) + 1e-8)).detach()      # [B,C,n_bands]
        delta = torch.einsum("bck,ckj->bcj", rho, self.band_map)         # [B,C,n_bands]
        base = self.gate_logit
        if self.marginal:
            base = base.expand(self.C, -1)
        g = torch.sigmoid(base[None] + delta)                            # [B,C,n_bands]
        return self._normalize(g)

    def forward(self, x, return_energy=False):
        # x: [B, C, L]
        B, C, L = x.shape
        coeff = torch.einsum("nl,bcl->bcn", self.W, x)           # [B, C, n]
        g = self._gates_for(coeff)                               # [B, C, n_bands]
        gpos = torch.einsum("bcj,nj->bcn", g, self.band_onehot)  # [B, C, n]
        gcoeff = coeff * gpos                                    # [B, C, n]
        xr = torch.einsum("nl,bcn->bcl", self.W, gcoeff)         # IDWT = W^T (=> einsum nl,bcn->bcl)
        out = x + (xr - x) if self.residual else xr              # residual reconstruction
        if not return_energy:
            return out
        # GATED relative band-energy per sample/channel/band, WITH gradient. Its
        # cross-facility variance is the invariance regularizer (RO2-RO4): minimizing it
        # makes the RETAINED feature spectrum consistent across facilities, and gradients
        # flow into the stem (whole feature extractor), not just the 12 gate params.
        # Relative energies sum to 1 over bands, so the objective aligns spectral SHAPE
        # and (with the energy-preserving gate) cannot be trivially collapsed.
        e = gcoeff.pow(2)                                        # [B, C, n]  (gated)
        band_e = torch.einsum("bcn,nk->bck", e, self.band_onehot)                       # [B,C,n_bands]
        rel = band_e / (band_e.sum(-1, keepdim=True) + 1e-8)
        return out, rel


# ----------------------------------------------------------------------------- #
#  Backbones
# ----------------------------------------------------------------------------- #
class MovingAvg(nn.Module):
    def __init__(self, k=25):
        super().__init__(); self.k = k
        self.avg = nn.AvgPool1d(k, stride=1, padding=0)
    def forward(self, x):                       # x: [B,C,L]
        pad = (self.k - 1) // 2
        xp = torch.cat([x[..., :1].repeat(1, 1, pad),
                        x, x[..., -1:].repeat(1, 1, self.k - 1 - pad)], -1)
        return self.avg(xp)


class DLinear(nn.Module):
    """Decomposition linear forecaster (Zeng et al., 2023). Per-channel, shared weights."""
    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.lin_trend = nn.Linear(L, H)
        self.lin_season = nn.Linear(L, H)
    def forward(self, x):                       # [B,C,L] -> [B,C,H]
        trend = self.decomp(x)
        season = x - trend
        return self.lin_season(season) + self.lin_trend(trend)


class PatchTransformer(nn.Module):
    """Compact patch-based Transformer forecaster (secondary backbone)."""
    def __init__(self, L, H, patch=8, stride=8, d=48, heads=3, layers=2, ff=96, drop=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.npatch = (L - patch) // stride + 1
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, self.npatch, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True,
                                         activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(self.npatch * d, H)
    def forward(self, x):                       # [B,C,L] -> [B,C,H]
        B, C, L = x.shape
        xp = x.unfold(-1, self.patch, self.stride)          # [B,C,npatch,patch]
        xp = self.embed(xp) + self.pos[None]                # [B,C,npatch,d]
        xp = xp.reshape(B * C, self.npatch, -1)
        z = self.tr(xp).reshape(B * C, -1)
        return self.head(z).reshape(B, C, -1)


class Stem(nn.Module):
    """Lightweight NONLINEAR per-category temporal encoder producing the intermediate
    representation Z on which the sub-band gate acts (RO1).  The nonlinearity is what
    makes gating non-absorbable by a linear head: unlike gating the raw input (which a
    linear forecaster can undo, and which destroys predictive high-freq signal), gating
    this learned feature can suppress facility-specific nuisance while preserving the
    predictive content the head still needs.  groups=C keeps categories separate."""
    def __init__(self, C, L, width=4, kernel=5):
        super().__init__()
        p = kernel // 2
        self.conv1 = nn.Conv1d(C, C * width, kernel, padding=p, groups=C)
        self.conv2 = nn.Conv1d(C * width, C, kernel, padding=p, groups=C)
        self.act = nn.GELU()
    def forward(self, x):                                   # [B,C,L] -> [B,C,L]
        return x + self.conv2(self.act(self.conv1(x)))      # residual feature


class LinearForecaster(nn.Module):
    """DLinear-style channel-shared linear head (compute-efficient primary head)."""
    def __init__(self, L, H, kernel=25):
        super().__init__()
        self.f = DLinear(L, H, kernel)
    def forward(self, z):
        return self.f(z)


def make_forecaster(name, L, H):
    if name in ("dlinear", "linear", "mlp"):
        return LinearForecaster(L, H)
    if name == "transformer":
        return PatchTransformer(L, H)
    if name == "transformer_big":            # high-capacity: can overfit source facilities
        return PatchTransformer(L, H, patch=4, stride=4, d=128, heads=8, layers=4,
                                ff=256, drop=0.0)
    raise ValueError(name)


# ----------------------------------------------------------------------------- #
#  Full models  (RevIN -> Stem -> [gate] -> Forecaster)
# ----------------------------------------------------------------------------- #
class ERM(nn.Module):
    """Mandatory baseline: same stem + forecaster as CD-ISG-TS but NO gate."""
    def __init__(self, L, H, C, backbone="mlp", use_stem=True):
        super().__init__()
        self.revin = RevIN(C)
        self.stem = Stem(C, L) if use_stem else nn.Identity()
        self.net = make_forecaster(backbone, L, H)
    def encode(self, x):
        """Pooled intermediate feature for CORAL/DANN penalties: [B, C*L]."""
        z = self.stem(self.revin(x))
        return z.reshape(z.shape[0], -1)
    def forward(self, x, **kw):
        z = self.stem(self.revin(x))
        return self.revin.denorm(self.net(z))


class DLinearPlain(nn.Module):
    """Classic DLinear (Zeng et al. 2023) simplicity reference: RevIN + linear, no stem."""
    def __init__(self, L, H, C, backbone="dlinear"):
        super().__init__()
        self.revin = RevIN(C)
        self.net = make_forecaster("dlinear", L, H)
    def forward(self, x, **kw):
        return self.revin.denorm(self.net(self.revin(x)))


class FixedBand(nn.Module):
    """Fixed low-pass heuristic baseline: on the FEATURE, keep coarse bands, drop the
    finest detail bands (the 'domain-specific == high-frequency' assumption, e.g. Tao)."""
    def __init__(self, L, H, C, backbone="mlp", wavelet="db4", level=3, drop_fine=1,
                 use_stem=True):
        super().__init__()
        self.revin = RevIN(C)
        self.stem = Stem(C, L) if use_stem else nn.Identity()
        W, band_id, lengths = build_wavelet_matrix(L, wavelet, level)
        self.register_buffer("W", torch.from_numpy(W))
        keep = np.ones(len(lengths), np.float32)
        for k in range(drop_fine):
            keep[len(lengths) - 1 - k] = 0.0
        self.register_buffer("gpos", torch.from_numpy(keep[band_id]))
        self.net = make_forecaster(backbone, L, H)
    def forward(self, x, **kw):
        z = self.stem(self.revin(x))
        coeff = torch.einsum("nl,bcl->bcn", self.W, z)
        z2 = torch.einsum("nl,bcn->bcl", self.W, coeff * self.gpos[None, None])
        return self.revin.denorm(self.net(z2))


class CDISGTS(nn.Module):
    """Proposed CD-ISG-TS: RevIN -> nonlinear stem -> learnable category-conditioned
    sub-band gate on the intermediate feature -> forecaster."""
    def __init__(self, L, H, C, backbone="mlp", wavelet="db4", level=3,
                 marginal=False, use_fft=False, gate_init=2.5, use_stem=True,
                 aug_strength=None, adaptive=False, periodic=False, period=7):
        super().__init__()
        self.periodic = periodic
        self.period = period
        self.H = H
        if periodic:
            # Second frequency-domain mechanism (see class docstring): an
            # input-driven periodicity branch, blended by alpha. Named "blend_" so
            # the optimizer groups it with the gate (large step, no weight decay).
            self.blend_scale = nn.Parameter(torch.tensor(0.0))
            self.blend_bias = nn.Parameter(torch.tensor(-3.0))
            idx = [(L - period) + (h % period) for h in range(H)]
            self.register_buffer("sn_idx", torch.tensor(idx, dtype=torch.long))
        self.revin = RevIN(C)
        self.stem = Stem(C, L) if use_stem else nn.Identity()
        if use_fft:
            self.gate = FFTGate(L, C, marginal=marginal, gate_init=gate_init)
        else:
            self.gate = SubBandGate(L, C, wavelet, level, marginal=marginal,
                                    gate_init=gate_init, adaptive=adaptive)
        self.net = make_forecaster(backbone, L, H)
        # per-(category,band) augmentation strength: larger for domain-VARIANT bands
        # (from the category-conditioned cross-domain variance / ICC). Randomly
        # perturbing these bands during training performs invariance-guided domain
        # randomization -> forces the forecaster to rely on domain-STABLE sub-bands.
        if aug_strength is not None:
            self.register_buffer("aug_strength",
                                 torch.as_tensor(aug_strength, dtype=torch.float32))
        else:
            self.aug_strength = None

    def _augment(self, z):
        """Stochastic sub-band scaling of the feature, weighted by domain-variance."""
        W = self.gate.W                                        # [n,L]
        coeff = torch.einsum("nl,bcl->bcn", W, z)              # [B,C,n]
        s = self.aug_strength                                  # [C,n_bands]
        s_pos = torch.einsum("cb,nb->cn", s, self.gate.band_onehot)   # [C,n]
        noise = 1.0 + s_pos[None] * torch.randn_like(coeff)    # multiplicative jitter
        return torch.einsum("nl,bcn->bcl", W, coeff * noise)

    def _periodicity(self, xn):
        """Label-free evidence that the seasonal branch is worth trusting for THIS
        window, obtained by backtesting inside the window itself.

        Raw periodicity strength is not the right signal. A panel can carry a
        strong weekly rhythm that the trained model ALREADY reproduces, in which
        case blending in a seasonal-naive prediction only injects noise; that is
        exactly what harms the irreducible nursing panel. What we need is evidence
        that the seasonal rule beats what the model would otherwise do, and that
        evidence is available at deployment without any label: hold out the last
        H steps OF THE INPUT WINDOW, predict them from the earlier part with the
        seasonal rule, and measure the error actually incurred.

        We compare that against a persistence reference on the same held-out span,
        so the score is a relative skill rather than a scale-dependent error, and
        return it as a bounded signal. Everything is computed from x alone, so the
        domain-generalization protocol is respected.
        """
        H, p = self.H, self.period
        if xn.shape[-1] <= H + p:
            return torch.zeros(xn.shape[:2], device=xn.device)
        inner = xn[..., :-H]                       # the part we may condition on
        target = xn[..., -H:]                      # held-out tail of the input
        idx = torch.tensor([(inner.shape[-1] - p) + (h % p) for h in range(H)],
                           device=xn.device, dtype=torch.long)
        sn_hat = inner.index_select(-1, idx)
        pe_hat = inner[..., -1:].expand_as(target)
        e_sn = (sn_hat - target).abs().mean(-1)
        e_pe = (pe_hat - target).abs().mean(-1)
        skill = (e_pe - e_sn) / (e_pe + e_sn + 1e-8)      # in (-1, 1)
        return skill.clamp(-1, 1).detach()

    def _blend(self, out, xn):
        """out and xn are both in the normalized space."""
        if not self.periodic:
            return out
        sn = xn.index_select(-1, self.sn_idx)                    # [B,C,H]
        s = self._periodicity(xn)                                # [B,C]
        alpha = torch.sigmoid(self.blend_scale * s
                              + self.blend_bias).unsqueeze(-1)   # [B,C,1]
        return (1 - alpha) * out + alpha * sn

    def forward(self, x, return_energy=False, **kw):
        xn = self.revin(x)
        z = self.stem(xn)
        if self.training and self.aug_strength is not None:
            z = self._augment(z)
        if return_energy:
            zg, rel = self.gate(z, return_energy=True)
            return self.revin.denorm(self._blend(self.net(zg), xn)), rel
        zg = self.gate(z)
        return self.revin.denorm(self._blend(self.net(zg), xn))


class FFTGate(nn.Module):
    """Global-FFT learnable-mask variant (ablation vs. localized wavelet gate)."""
    def __init__(self, L, n_channels, n_bands=4, marginal=False, gate_init=2.5):
        super().__init__()
        self.L = L; self.C = n_channels; self.n_bands = n_bands; self.marginal = marginal
        nfreq = L // 2 + 1
        edges = np.linspace(0, nfreq, n_bands + 1).astype(int)
        band_id = np.zeros(nfreq, int)
        for b in range(n_bands):
            band_id[edges[b]:edges[b + 1]] = b
        self.register_buffer("band_onehot",
                             torch.nn.functional.one_hot(torch.from_numpy(band_id),
                                                         n_bands).float())
        blen = np.array([(band_id == b).sum() for b in range(n_bands)], np.float32)
        self.register_buffer("band_len", torch.from_numpy(blen))
        g0 = gate_init
        self.gate_logit = nn.Parameter(torch.full((1 if marginal else n_channels, n_bands), g0))

    def gates(self):
        g = torch.sigmoid(self.gate_logit)
        if self.marginal:
            g = g.expand(self.C, -1)
        scale = self.band_len.sum() / ((g * self.band_len).sum(-1, keepdim=True) + 1e-8)
        return g * scale

    def forward(self, x, return_energy=False):
        Xf = torch.fft.rfft(x, dim=-1)                          # [B,C,nfreq]
        g = self.gates()                                        # [C,n_bands]
        gpos = torch.einsum("cb,nb->cn", g, self.band_onehot)   # [C,nfreq]
        Xg = Xf * gpos[None].to(Xf.dtype)
        out = torch.fft.irfft(Xg, n=self.L, dim=-1)
        if not return_energy:
            return out
        e = Xg.abs().pow(2)                                      # gated spectrum (w/ grad)
        band_e = torch.einsum("bcn,nk->bck", e, self.band_onehot)
        rel = band_e / (band_e.sum(-1, keepdim=True) + 1e-8)
        return out, rel


# ----------------------------------------------------------------------------- #
#  Cross-domain invariance regularizer (category-conditioned)
# ----------------------------------------------------------------------------- #
def cross_domain_variance(rel, fac_ids):
    """Category-conditioned cross-domain invariance estimator (RO2).
    rel: [B,C,n_bands] RAW relative band energies; fac_ids: [B].
    Returns V_hat [C,n_bands] = variance ACROSS facilities of each band's
    per-facility mean relative energy. High V_hat == domain-variant sub-band."""
    uniq = torch.unique(fac_ids)
    if uniq.numel() < 2:
        return rel.mean(0) * 0.0
    fac_means = torch.stack([rel[fac_ids == f].mean(0) for f in uniq], 0)  # [F,C,nb]
    return fac_means.var(0, unbiased=False)                                # [C,nb]


def invariance_penalty(rel, fac_ids, gates=None):
    """Category-conditioned cross-domain invariance regularizer (RO3-RO4):
        L_inv = mean_{c,j} Var_facility( E_window[ rel_gated_{c,j} ] )
    i.e. the cross-facility variance of the retained (gated) feature spectrum.
    Gradients flow into the whole feature extractor (stem + gate), pushing it to
    produce a facility-invariant retained representation and thereby reducing the
    inter-facility feature discrepancy that bounds target-facility risk (RO4)."""
    return cross_domain_variance(rel, fac_ids).mean()


# ----------------------------------------------------------------------------- #
#  Fixed-decomposition SOTA baselines (compact, CPU-feasible, RevIN for fairness)
# ----------------------------------------------------------------------------- #
class AutoformerLite(nn.Module):
    """Compact Autoformer-style baseline (Wu et al., 2021): a FIXED moving-average
    series decomposition into trend + seasonal, a small self-attention encoder on the
    seasonal part, and a linear trend path. The decomposition rule is fixed (not
    conditioned on cross-domain statistics) -- the gap CD-ISG-TS targets (RQ1)."""
    def __init__(self, L, H, C, kernel=25, d=48, heads=3, layers=2, ff=96, drop=0.1):
        super().__init__()
        self.revin = RevIN(C)
        self.decomp = MovingAvg(kernel)
        self.season_tr = PatchTransformer(L, H, d=d, heads=heads, layers=layers,
                                          ff=ff, drop=drop)
        self.trend_lin = nn.Linear(L, H)
    def forward(self, x, **kw):
        z = self.revin(x)
        trend = self.decomp(z); season = z - trend
        y = self.season_tr(season) + self.trend_lin(trend)
        return self.revin.denorm(y)


class FEDformerLite(nn.Module):
    """Compact FEDformer-style baseline (Zhou et al., 2022): frequency-enhanced block
    that keeps a FIXED (randomly selected at init) subset of Fourier modes and applies a
    learned complex linear map on them, plus a trend path. Mode selection is fixed/random
    rather than learned from multi-domain statistics -- again the gap CD-ISG-TS targets."""
    def __init__(self, L, H, C, kernel=25, modes=16):
        super().__init__()
        self.revin = RevIN(C)
        self.decomp = MovingAvg(kernel)
        self.L, self.H = L, H
        nfreq = L // 2 + 1
        modes = min(modes, nfreq)
        rng = np.random.default_rng(0)
        sel = np.sort(rng.choice(nfreq, size=modes, replace=False))
        self.register_buffer("sel", torch.from_numpy(sel).long())
        # learned complex weights per selected mode mapping input spectrum -> horizon spectrum
        self.wr = nn.Parameter(torch.randn(modes) * 0.02)
        self.wi = nn.Parameter(torch.randn(modes) * 0.02)
        self.trend_lin = nn.Linear(L, H)
        self.season_lin = nn.Linear(L, H)
    def forward(self, x, **kw):
        z = self.revin(x)
        trend = self.decomp(z); season = z - trend
        Xf = torch.fft.rfft(season, dim=-1)                 # [B,C,nfreq]
        Xsel = Xf[..., self.sel]                            # keep fixed modes
        w = torch.complex(self.wr, self.wi)
        Xsel = Xsel * w[None, None]
        Xnew = torch.zeros_like(Xf)
        Xnew[..., self.sel] = Xsel
        season_f = torch.fft.irfft(Xnew, n=self.L, dim=-1)  # filtered seasonal (len L)
        y = self.season_lin(season_f) + self.trend_lin(trend)
        return self.revin.denorm(y)


# ----------------------------------------------------------------------------- #
#  PACE: Periodicity-Aware Cross-Entity forecasting
# ----------------------------------------------------------------------------- #
class PACE(nn.Module):
    """Periodicity-Aware Cross-Entity forecaster.

    MOTIVATION (diagnosed, not guessed). On a reducible split the source group may
    carry almost none of the periodic structure that the target group carries. A
    model trained on the source has then never seen that rhythm and cannot learn to
    emit it, which is why every invariance penalty in the literature fails to
    convert the headroom the diagnostic identifies. But the deployment INPUT window
    does contain the target's rhythm. The transferable object is therefore not a
    pattern to memorize from the source; it is the conditional rule "continue
    whatever periodicity this window already shows".

    PACE adds that rule as an explicit, input-driven branch and lets the network
    decide per window how far to trust it:

        y_hat = (1 - alpha(x)) * f_theta(x)  +  alpha(x) * SeasonalNaive(x)

    alpha(x) is a function of the window's OWN periodicity strength, measured by
    normalized autocorrelation at the seasonal lag, so it adapts to entities never
    seen in training. This is legal under domain generalization: it conditions on
    the input only, never on target labels or target-domain statistics.

    Properties mirroring the sub-band gate:
      * two blending parameters only;
      * initialized so alpha ~ 0, hence exactly the backbone at initialization,
        so it cannot start worse;
      * on aperiodic entities the measured strength is ~0 and the branch switches
        itself off, which is why it does not damage irreducible panels.
    """

    def __init__(self, L, H, C, backbone="mlp", period=7, use_stem=True,
                 gate=None):
        super().__init__()
        self.L, self.H, self.C, self.period = L, H, C, period
        self.revin = RevIN(C)
        self.stem = Stem(C, L) if use_stem else nn.Identity()
        self.gate = gate                      # optional CD-ISG-TS sub-band gate
        self.net = make_forecaster(backbone, L, H)
        # Blend head: alpha = sigmoid(scale * strength + bias), initialized nearly
        # closed. Named "blend_" so the optimizer places these with the gate
        # parameters (large step, no weight decay); under the backbone's weight
        # decay a near-zero-initialized blend would simply be held shut.
        self.blend_scale = nn.Parameter(torch.tensor(0.0))
        self.blend_bias = nn.Parameter(torch.tensor(-3.0))
        idx = [(L - period) + (h % period) for h in range(H)]
        self.register_buffer("sn_idx", torch.tensor(idx, dtype=torch.long))

    def _periodicity(self, xn):
        """Normalized autocorrelation at the seasonal lag, per sample and channel.
        Computed on the normalized window and detached: it describes the window,
        it is not a path we want the loss to game."""
        a = xn[..., self.period:]
        b = xn[..., :-self.period]
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        num = (a * b).sum(-1)
        den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
        return (num / den).clamp(-1, 1).detach()          # [B, C]

    def forward(self, x):
        xn = self.revin(x)                                 # [B,C,L] normalized
        z = self.stem(xn)
        if self.gate is not None:
            z = self.gate(z)
        base = self.net(z)                                 # [B,C,H] normalized space
        sn = xn.index_select(-1, self.sn_idx)              # seasonal-naive, same space
        s = self._periodicity(xn)                          # [B,C]
        alpha = torch.sigmoid(self.blend_scale * s
                              + self.blend_bias).unsqueeze(-1)      # [B,C,1]
        out = (1 - alpha) * base + alpha * sn
        return self.revin.denorm(out)


# ----------------------------------------------------------------------------- #
#  Calibration gate (experimental; not yet used by any manuscript result)
# ----------------------------------------------------------------------------- #
class OutputShrinkGate(nn.Module):
    """Input-conditioned rescaling of the FORECAST's sub-bands.

    Why this and not the existing gate. Under an absolute-error loss the optimal
    point forecast is shrunk toward the conditional mean, and the right amount of
    shrinkage depends on how predictable the entity is. A model fitted on one
    group learns a single shrinkage; deployed on a group of different
    predictability it is mis-calibrated -- it tracks too eagerly on noisy
    entities and too timidly on clean ones. That mis-calibration is a property
    of the emitted forecast, so no re-weighting of an internal feature can fix
    it, which is consistent with the static and input-conditioned FEATURE gates
    both failing.

    The descriptor is the presented window's own relative sub-band energy
    profile, so nothing is read from labels or from target-domain statistics and
    no adaptation step is required at deployment.

    Zero-initialized, so gain == 1 exactly and the module is the identity at
    initialization, preserving the safety guarantee of the original design.
    """

    def __init__(self, H, C, n_in_bands, wavelet="db4", level=1, shrink_only=False):
        super().__init__()
        # shrink_only forbids amplification. Under absolute-error loss the optimal
        # point forecast is a shrunk one, so a gain above 1 is almost never right;
        # allowing it lets the gate extrapolate the wrong way on target entities
        # whose descriptor falls outside the source range.
        self.shrink_only = shrink_only
        Wh, band_id, lengths = build_wavelet_matrix(H, wavelet, level)
        self.register_buffer("Wh", torch.from_numpy(Wh))
        self.n_out = len(lengths)
        self.register_buffer(
            "out_onehot",
            torch.nn.functional.one_hot(torch.from_numpy(band_id), self.n_out).float())
        # named "shrink_" so the optimizer groups them with the gate: large step,
        # no weight decay. Zero-init params would otherwise be pinned at zero by
        # the backbone's weight decay, which is what stalled the earlier gate.
        self.shrink_A = nn.Parameter(torch.zeros(C, self.n_out, n_in_bands))
        self.shrink_b = nn.Parameter(torch.zeros(C, self.n_out))

    def forward(self, y, rho_in):
        """y: [B,C,H] in normalized space. rho_in: [B,C,n_in_bands], detached."""
        coeff = torch.einsum("nh,bch->bcn", self.Wh, y)
        delta = torch.einsum("bck,cjk->bcj", rho_in, self.shrink_A) \
            + self.shrink_b[None]
        t = torch.tanh(delta)
        if self.shrink_only:
            t = t.clamp(max=0.0)                             # in (0,1]; ==1 at init
        gain = 1.0 + t                                       # in (0,2); ==1 at init
        gpos = torch.einsum("bcj,nj->bcn", gain, self.out_onehot)
        return torch.einsum("nh,bcn->bch", self.Wh, coeff * gpos)


class FADECalib(CDISGTS):
    """CDISGTS plus the output calibration gate of OutputShrinkGate."""

    def __init__(self, *a, shrink_only=False, **kw):
        super().__init__(*a, **kw)
        C = self.gate.C
        self.shrink = OutputShrinkGate(self.H, C, self.gate.n_bands,
                                       shrink_only=shrink_only)

    def _rho_in(self, xn):
        """Relative sub-band energy of the presented window; a descriptor of x only."""
        coeff = torch.einsum("nl,bcl->bcn", self.gate.W, xn)
        e = torch.einsum("bcn,nk->bck", coeff.pow(2), self.gate.band_onehot)
        return (e / (e.sum(-1, keepdim=True) + 1e-8)).detach()

    def forward(self, x, return_energy=False, **kw):
        xn = self.revin(x)
        z = self.stem(xn)
        rho = self._rho_in(xn)
        if return_energy:
            zg, rel = self.gate(z, return_energy=True)
            out = self._blend(self.net(zg), xn)
            return self.revin.denorm(self.shrink(out, rho)), rel
        zg = self.gate(z)
        out = self._blend(self.net(zg), xn)
        return self.revin.denorm(self.shrink(out, rho))
