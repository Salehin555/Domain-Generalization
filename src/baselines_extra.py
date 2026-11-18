"""
Faithful (CPU-scaled) reimplementations of the transformer baselines and standard
domain-generalization baselines, sharing RevIN + channel-independent processing for a
fair comparison.

  AutoformerFull  : series decomposition + the Auto-Correlation mechanism with
                    time-delay aggregation (Wu et al., 2021), channel-independent.
  FEDformerFull   : series decomposition + a Frequency-Enhanced Block that keeps a
                    fixed random set of Fourier modes and applies a learned complex
                    linear map (Zhou et al., 2022), channel-independent.
  CORAL / DANN / GroupDRO : deep-CORAL covariance alignment, domain-adversarial training
                    with a gradient-reversal layer, and group-DRO worst-facility
                    reweighting, all on the shared RevIN+stem ERM backbone.
"""
import numpy as np, torch, torch.nn as nn
from src.models import RevIN, MovingAvg, DLinear


# --------------------------------------------------------------------------- #
#  Autoformer: Auto-Correlation + series decomposition
# --------------------------------------------------------------------------- #
class SeriesDecomp(nn.Module):
    def __init__(self, kernel=25):
        super().__init__(); self.ma = MovingAvg(kernel)
    def forward(self, x):                      # x: [B, L, d] -> season, trend
        xt = x.transpose(1, 2)                 # [B,d,L]
        trend = self.ma(xt).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """FFT-based autocorrelation with vectorized time-delay aggregation."""
    def __init__(self, factor=1):
        super().__init__(); self.factor = factor
    def forward(self, q, k, v):                # each [B, L, Hh, D]
        B, L, Hh, D = q.shape
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1), dim=-1)   # [B,Hh,D,Lf]
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=-1)   # [B,Hh,D,L]
        vv = v.permute(0, 2, 3, 1)                             # [B,Hh,D,L]
        top_k = max(1, int(self.factor * np.log(L)))
        mean_corr = corr.mean(1).mean(1)                      # [B,L]
        w, delay = torch.topk(mean_corr, top_k, dim=-1)       # [B,top_k]
        w = torch.softmax(w, dim=-1)
        vv2 = torch.cat([vv, vv], dim=-1)                     # [B,Hh,D,2L]
        idx0 = torch.arange(L, device=q.device).view(1, 1, 1, L).expand(B, Hh, D, L)
        out = torch.zeros_like(vv)
        for i in range(top_k):
            index = idx0 + delay[:, i].view(B, 1, 1, 1)
            out = out + torch.gather(vv2, -1, index) * w[:, i].view(B, 1, 1, 1)
        return out.permute(0, 3, 1, 2)                        # [B,L,Hh,D]


class AutoCorrLayer(nn.Module):
    def __init__(self, d, heads, kernel, ff):
        super().__init__()
        self.h = heads; self.dk = d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d); self.ac = AutoCorrelation()
        self.d1 = SeriesDecomp(kernel); self.d2 = SeriesDecomp(kernel)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def _split(self, x):
        B, L, d = x.shape
        return x.view(B, L, self.h, self.dk)
    def forward(self, x):
        B, L, d = x.shape
        a = self.ac(self._split(self.q(x)), self._split(self.k(x)), self._split(self.v(x)))
        a = self.o(a.reshape(B, L, d))
        s, _ = self.d1(x + a)
        s2, _ = self.d2(s + self.ff(s))
        return s2


class AutoformerFull(nn.Module):
    def __init__(self, L, H, C, d=32, heads=4, layers=2, ff=64, kernel=25):
        super().__init__()
        self.revin = RevIN(C); self.L, self.H = L, H
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.layers = nn.ModuleList([AutoCorrLayer(d, heads, kernel, ff) for _ in range(layers)])
        self.proj = nn.Linear(L * d, H)
        self.trend = nn.Linear(L, H)
        self.decomp = SeriesDecomp(kernel)
    def forward(self, x, **kw):
        z = self.revin(x)                       # [B,C,L]
        B, C, L = z.shape
        u = z.reshape(B * C, L, 1)
        season0, trend0 = self.decomp(u)        # [B*C,L,1]
        h = self.embed(season0) + self.pos
        for lyr in self.layers:
            h = lyr(h)
        seas = self.proj(h.reshape(B * C, -1))              # [B*C,H]
        tr = self.trend(trend0.squeeze(-1))                 # [B*C,H]
        y = (seas + tr).reshape(B, C, self.H)
        return self.revin.denorm(y)


# --------------------------------------------------------------------------- #
#  FEDformer: Frequency-Enhanced Block + series decomposition
# --------------------------------------------------------------------------- #
class FEB(nn.Module):
    def __init__(self, d, L, modes=16, seed=0):
        super().__init__()
        nfreq = L // 2 + 1
        modes = min(modes, nfreq)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(nfreq, size=modes, replace=False))
        self.register_buffer("idx", torch.from_numpy(idx).long())
        self.L = L; self.modes = modes
        scale = 1 / (d * d)
        self.wr = nn.Parameter(torch.randn(modes, d, d) * scale)
        self.wi = nn.Parameter(torch.randn(modes, d, d) * scale)
    def forward(self, x):                       # [B, L, d]
        B, L, d = x.shape
        xf = torch.fft.rfft(x, dim=1)           # [B, Lf, d]
        sel = xf[:, self.idx, :]                # [B, modes, d]
        w = torch.complex(self.wr, self.wi)     # [modes,d,d]
        out_sel = torch.einsum("bmd,mde->bme", sel, w)      # [B,modes,d]
        out_ft = torch.zeros(B, L // 2 + 1, d, dtype=torch.cfloat, device=x.device)
        out_ft[:, self.idx, :] = out_sel
        return torch.fft.irfft(out_ft, n=L, dim=1)          # [B,L,d]


class FEDLayer(nn.Module):
    def __init__(self, d, L, modes, kernel, ff, seed):
        super().__init__()
        self.feb = FEB(d, L, modes, seed)
        self.d1 = SeriesDecomp(kernel); self.d2 = SeriesDecomp(kernel)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
    def forward(self, x):
        s, _ = self.d1(x + self.feb(x))
        s2, _ = self.d2(s + self.ff(s))
        return s2


class FEDformerFull(nn.Module):
    def __init__(self, L, H, C, d=32, layers=2, ff=64, kernel=25, modes=16):
        super().__init__()
        self.revin = RevIN(C); self.L, self.H = L, H
        self.embed = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, L, d) * 0.02)
        self.layers = nn.ModuleList([FEDLayer(d, L, modes, kernel, ff, seed=i)
                                     for i in range(layers)])
        self.proj = nn.Linear(L * d, H)
        self.trend = nn.Linear(L, H)
        self.decomp = SeriesDecomp(kernel)
    def forward(self, x, **kw):
        z = self.revin(x); B, C, L = z.shape
        u = z.reshape(B * C, L, 1)
        season0, trend0 = self.decomp(u)
        h = self.embed(season0) + self.pos
        for lyr in self.layers:
            h = lyr(h)
        y = (self.proj(h.reshape(B * C, -1)) + self.trend(trend0.squeeze(-1))).reshape(B, C, self.H)
        return self.revin.denorm(y)


# --------------------------------------------------------------------------- #
#  Domain-generalization baselines (on the shared RevIN + stem backbone)
# --------------------------------------------------------------------------- #
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None

def grad_reverse(x, lamb=1.0):
    return GradReverse.apply(x, lamb)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, n_dom, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_dom))
    def forward(self, x):
        return self.net(x)

def coral_penalty(feats, fac_ids):
    """Deep-CORAL: mean squared Frobenius distance between per-facility feature
    covariances in the batch. feats: [B, d]; fac_ids: [B]."""
    uniq = torch.unique(fac_ids)
    if uniq.numel() < 2:
        return feats.sum() * 0.0
    covs = []
    for f in uniq:
        z = feats[fac_ids == f]
        if z.shape[0] < 2:
            continue
        zc = z - z.mean(0, keepdim=True)
        covs.append((zc.t() @ zc) / (z.shape[0] - 1))
    if len(covs) < 2:
        return feats.sum() * 0.0
    covs = torch.stack(covs, 0)                # [F,d,d]
    m = covs.mean(0, keepdim=True)
    d = covs.shape[-1]
    return ((covs - m) ** 2).sum() / (len(covs) * d * d)
