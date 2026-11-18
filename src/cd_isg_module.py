"""
CD-ISG: Cross-Domain Invariant Sub-band Gating  —  the proposed ADD-ON MODULE only.

This single file contains the complete plug-in module and nothing else, so it can be
read and explained on its own. It has three parts:

  PART 1  build_wavelet_matrix : the FIXED (non-trainable) orthonormal DWT operator
  PART 2  SubBandGate          : the gating module itself  (the only trainable part)
  PART 3  invariance_penalty   : the cross-domain signal that trains the gate

The module is inserted inside an existing backbone:

    z = stem(revin(x))        # any feature extractor
    z = gate(z)               # <-- THIS MODULE
    y = forecast_head(z)      # any forecaster
"""
import numpy as np
import torch
import torch.nn as nn
import pywt


# =========================================================================== #
#  PART 1 — the fixed, orthonormal, multi-level DWT written as ONE matrix
# =========================================================================== #
def build_wavelet_matrix(L, wavelet="db4", level=3):
    """Build the analysis matrix W of a multi-level DWT.

    We transform each unit basis vector once and stack the results, which yields the
    linear operator of the whole multi-level transform as a single [L, L] matrix.

    Because the wavelet is orthonormal (and mode='periodization' keeps exactly L
    coefficients), W satisfies  W @ W.T = I.  Two consequences we rely on:
      * the inverse transform is just the transpose  ->  IDWT(c) = W.T @ c
      * with all gates equal to 1 the module is EXACTLY the identity.

    Returns
      W        : [L, L] float32   analysis matrix (fixed, never trained)
      band_id  : [L]  int64       which sub-band each coefficient belongs to
                                  (0 = coarsest approximation ... last = finest detail)
      lengths  : list[int]        number of coefficients per sub-band, e.g. [7, 7, 14, 28]
    """
    w = pywt.Wavelet(wavelet)
    cols, lengths = [], None
    for i in range(L):
        e = np.zeros(L); e[i] = 1.0
        coeffs = pywt.wavedec(e, w, mode="periodization", level=level)
        if lengths is None:
            lengths = [len(c) for c in coeffs]          # [cA_n, cD_n, ..., cD_1]
        cols.append(np.concatenate(coeffs))
    W = np.stack(cols, axis=1)                          # [n_coeff, L]
    band_id = np.concatenate([np.full(l, b, int) for b, l in enumerate(lengths)])
    return W.astype(np.float32), band_id.astype(np.int64), lengths


# =========================================================================== #
#  PART 2 — the gating module (the proposed add-on)
# =========================================================================== #
class SubBandGate(nn.Module):
    """DWT  ->  learnable per-(staff-category, sub-band) gate  ->  inverse DWT.

    Trainable parameters: ONE gate logit per (category, band).
    For 3 categories (RN, LPN, CNA) and 4 sub-bands this is 3 x 4 = 12 parameters.

    marginal=True shares the gate across categories (used for the ablation that tests
    whether category-conditioning actually matters).
    """

    def __init__(self, L, n_channels, wavelet="db4", level=3, marginal=False,
                 residual=True, gate_init=2.5):
        super().__init__()
        W, band_id, lengths = build_wavelet_matrix(L, wavelet, level)

        # register_buffer => saved with the model but NOT trained (no gradients)
        self.register_buffer("W", torch.from_numpy(W))                   # [n, L]
        self.register_buffer("band_id", torch.from_numpy(band_id))       # [n]

        self.n_bands = len(lengths)
        self.C = n_channels
        self.marginal = marginal
        self.residual = residual

        # ---- the ONLY trainable tensor in this module --------------------------
        # sigmoid(2.5) = 0.92 -> the gate starts nearly open (close to the identity)
        # but still in a responsive region of the sigmoid so it can move during training.
        g0 = gate_init
        if marginal:
            self.gate_logit = nn.Parameter(torch.full((1, self.n_bands), g0))
        else:
            self.gate_logit = nn.Parameter(torch.full((n_channels, self.n_bands), g0))

        # one-hot map: coefficient position -> its sub-band (lets us broadcast a
        # per-band gate onto the per-coefficient vector with a single einsum)
        self.register_buffer(
            "band_onehot",
            torch.nn.functional.one_hot(torch.from_numpy(band_id), self.n_bands).float())
        # number of coefficients in each band (used by the energy-preserving rescale)
        self.register_buffer("band_len", torch.tensor(lengths, dtype=torch.float32))

    # ----------------------------------------------------------------------- #
    def gates(self):
        """Return the ENERGY-PRESERVING gate g of shape [C, n_bands].

        Raw gate = sigmoid(logit) in (0, 1).  We then rescale each category's gates so
        that the coefficient-count-weighted mean gate is exactly 1:

              sum_j  n_j * g[c, j]   =   sum_j  n_j          (n_j = band size)

        Why this matters (this is the key design decision):
          * the gate can only REDISTRIBUTE emphasis - amplify some bands (g > 1) only by
            suppressing others (g < 1);
          * it therefore can NEVER collapse everything to zero, which is the classic
            failure mode of invariance penalties (the trivial "make all features
            constant, then they are perfectly invariant" solution).
        """
        g = torch.sigmoid(self.gate_logit)                    # [C or 1, n_bands]
        if self.marginal:
            g = g.expand(self.C, -1)
        scale = self.band_len.sum() / ((g * self.band_len).sum(-1, keepdim=True) + 1e-8)
        return g * scale

    # ----------------------------------------------------------------------- #
    def forward(self, x, return_energy=False):
        """x : [B, C, L]  intermediate feature  ->  [B, C, L] gated feature.

        B = batch, C = staff categories (RN/LPN/CNA), L = look-back length (56 days).
        """
        # 1) FORWARD DWT: project the feature onto the wavelet basis
        coeff = torch.einsum("nl,bcl->bcn", self.W, x)        # [B, C, n]

        # 2) GATE: broadcast the per-band gate onto every coefficient of that band
        g = self.gates()                                      # [C, n_bands]
        gpos = torch.einsum("cb,nb->cn", g, self.band_onehot)  # [C, n]
        gcoeff = coeff * gpos[None]                           # [B, C, n]

        # 3) INVERSE DWT: because W is orthonormal, the inverse is simply W transpose
        xr = torch.einsum("nl,bcn->bcl", self.W, gcoeff)      # [B, C, L]

        # residual form (keeps gradients well-conditioned; equals xr)
        out = x + (xr - x) if self.residual else xr

        if not return_energy:
            return out

        # 4) Per-band RELATIVE energy of the GATED feature, kept differentiable.
        #    Its cross-facility variance is the invariance regularizer (PART 3).
        #    Relative (sums to 1 across bands) => it compares the SHAPE of the
        #    spectrum, so it is scale-free and cannot be gamed by shrinking Z.
        e = gcoeff.pow(2)                                     # [B, C, n]
        band_e = torch.einsum("bcn,nk->bck", e, self.band_onehot)   # [B, C, n_bands]
        rel = band_e / (band_e.sum(-1, keepdim=True) + 1e-8)
        return out, rel


# =========================================================================== #
#  PART 3 — the cross-domain invariance signal that trains the gate
# =========================================================================== #
def cross_domain_variance(rel, fac_ids):
    """Category-conditioned cross-domain invariance ESTIMATOR.

    rel     : [B, C, n_bands]  relative band energies of the gated feature
    fac_ids : [B]              which facility (domain) each sample came from

    Step 1: average the band energies within each facility  -> one profile per facility
    Step 2: take the variance of those profiles ACROSS facilities

    Returns V_hat [C, n_bands].
      high V_hat  => that sub-band looks very different from facility to facility
                     => it is DOMAIN-SPECIFIC and should be suppressed
      low  V_hat  => that sub-band behaves the same everywhere
                     => it is INVARIANT and should be preserved
    """
    uniq = torch.unique(fac_ids)
    if uniq.numel() < 2:                     # need >= 2 domains to measure variance
        return rel.mean(0) * 0.0
    fac_means = torch.stack([rel[fac_ids == f].mean(0) for f in uniq], 0)  # [F, C, nb]
    return fac_means.var(0, unbiased=False)                                # [C, nb]


def invariance_penalty(rel, fac_ids, gates=None):
    """The regularizer added to the forecasting loss:

        L_total = MAE(forecast, target)  +  lambda * Omega
        Omega   = mean over (category, band) of cross-facility variance

    Minimizing Omega makes the RETAINED spectrum consistent across facilities.
    Gradients flow back into both the gate and the feature extractor, so the whole
    representation is pushed to be facility-invariant.
    """
    return cross_domain_variance(rel, fac_ids).mean()


# =========================================================================== #
#  HOW IT PLUGS INTO THE BACKBONE  (and a self-check)
# =========================================================================== #
if __name__ == "__main__":
    B, C, L = 8, 3, 56                     # batch, staff categories, look-back days
    gate = SubBandGate(L, C, wavelet="db4", level=3)

    n_trainable = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    print("trainable parameters in the module :", n_trainable)      # 12
    print("sub-band sizes                     :", gate.band_len.tolist())

    z = torch.randn(B, C, L)               # an intermediate feature from the backbone

    # (a) with the gate forced fully open, the module is EXACTLY the identity
    with torch.no_grad():
        gate.gate_logit.fill_(20.0)        # sigmoid(20) ~ 1
    print("max |gate(z) - z| when open        :", float((gate(z) - z).abs().max()))

    # (b) normal use inside a backbone
    gate.gate_logit.data.fill_(2.5)
    z_gated, rel = gate(z, return_energy=True)
    fac_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])   # 2 facilities in this batch
    print("gate matrix g (3 categories x 4 bands):\n", gate.gates().detach().numpy().round(3))
    print("energy-preserving check (should equal sum of band sizes):",
          float((gate.gates() * gate.band_len).sum(-1)[0]), "vs", float(gate.band_len.sum()))
    print("invariance penalty Omega           :", float(invariance_penalty(rel, fac_ids)))
