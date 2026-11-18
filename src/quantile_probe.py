"""Probabilistic-forecasting probe: does the invariance mechanism give better-CALIBRATED
prediction intervals on UNSEEN facilities (equal or better pinball loss + coverage closer
to nominal) than ERM? Staffing decisions need intervals, and DG for calibration is a
cleaner, decision-relevant claim than point MAE.

ERM-Q  = RevIN + stem + quantile head (pinball loss)
CD-Q   = RevIN + stem + energy-preserving sub-band gate + quantile head + invariance reg
"""
import numpy as np, torch, torch.nn as nn, os
from src.data import Panel
from src.models import RevIN, Stem, SubBandGate, DLinear, cross_domain_variance

torch.set_num_threads(max(1, os.cpu_count() - 2))
QS = [0.1, 0.5, 0.9]

class QHead(nn.Module):
    def __init__(self, L, H, Q):
        super().__init__(); self.H, self.Q = H, Q
        self.f = DLinear(L, H * Q)
    def forward(self, z):                       # [B,C,L] -> [B,C,H,Q]
        B, C, _ = z.shape
        return self.f(z).reshape(B, C, self.H, self.Q)

class QModel(nn.Module):
    def __init__(self, L, H, C, gate=False, level=3):
        super().__init__()
        self.revin = RevIN(C); self.stem = Stem(C, L)
        self.gate = SubBandGate(L, C, "db4", level) if gate else None
        self.head = QHead(L, H, len(QS))
    def forward(self, x, energy=False):
        z = self.stem(self.revin(x)); rel = None
        if self.gate is not None:
            if energy: z, rel = self.gate(z, return_energy=True)
            else: z = self.gate(z)
        y = self.head(z)                        # [B,C,H,Q]
        # denorm each quantile with RevIN stats
        m = self.revin.mean[..., None]; s = self.revin.std[..., None]
        if self.revin.affine:
            g = self.revin.gamma[None, :, None, None]; b = self.revin.beta[None, :, None, None]
            y = (y - b) / (g + self.revin.eps ** 0.5)
        y = y * s + m
        return (y, rel) if energy else y

def pinball(pred, y):                            # pred [B,C,H,Q], y [B,C,H]
    q = torch.tensor(QS).view(1, 1, 1, -1)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()

def train(P, tr, va, tm, vm, gate, lam=0.0, seed=0, epochs=26):
    np.random.seed(seed); torch.manual_seed(seed)
    M = QModel(P.L, P.H, 3, gate=gate)
    gp = [p for n, p in M.named_parameters() if "gate_logit" in n]
    rp = [p for n, p in M.named_parameters() if "gate_logit" not in n]
    groups = [dict(params=rp, lr=1e-3, weight_decay=1e-4)]
    if gp: groups.append(dict(params=gp, lr=0.01, weight_decay=0.0))
    opt = torch.optim.Adam(groups)
    best, best_state, pat = 1e9, None, 0
    for ep in range(epochs):
        ramp = 0.0 if ep < 0.4*epochs else min(1.0, (ep-0.4*epochs)/(0.3*epochs))
        M.train()
        for Xb, Yb, fb in P.grouped_batches(tr, 32, 8, seed=seed*97+ep, time_mask=tm):
            xb = torch.from_numpy(Xb); yb = torch.from_numpy(Yb); fbi = torch.from_numpy(fb)
            opt.zero_grad()
            if gate and lam > 0:
                pred, rel = M(xb, energy=True)
                loss = pinball(pred, yb) + lam*ramp*cross_domain_variance(rel, fbi).mean()
            else:
                loss = pinball(M(xb), yb)
            loss.backward(); opt.step()
        # val pinball
        M.eval(); vp = []
        with torch.no_grad():
            Xv, Yv, _ = P.eval_arrays(va, time_mask=vm)
            for i in range(0, len(Xv), 2048):
                vp.append(pinball(M(torch.from_numpy(Xv[i:i+2048])), torch.from_numpy(Yv[i:i+2048])).item())
        v = np.mean(vp)
        if v < best - 1e-6: best, best_state, pat = v, {k: t.clone() for k, t in M.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 6: break
    if best_state: M.load_state_dict(best_state)
    return M

def evaluate(M, P, te, vm):
    M.eval(); rows = []
    with torch.no_grad():
        for f in te:
            idx = P._pool(f, vm)
            if len(idx) == 0: continue
            pred = M(torch.from_numpy(P.Xw[idx])).numpy()       # [n,C,H,Q]
            y = P.Yw[idx]                                        # [n,C,H]
            pt = pred.sum(1); yt = y.sum(1)                      # total across categories -> [n,H,Q]
            lo, med, hi = pt[..., 0], pt[..., 1], pt[..., 2]
            cov = ((yt >= lo) & (yt <= hi)).mean()              # nominal 0.8
            # pinball on total
            q = np.array(QS)[None, None]
            e = yt[..., None] - pt
            pb = np.maximum(q*e, (q-1)*e).mean()
            rows.append(dict(cov=cov, pinball=pb, mae=np.abs(med-yt).mean()))
    return rows

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    P = Panel("cache/panel.npz", n_facilities=1500, L=56, H=14, stride=12, seed=0)
    tm, vm = P.is_2024, P.is_2025
    for proto in ["LFO-T", "LSO-T"]:
        print("==", proto)
        aggE, aggO = [], []
        for seed in [0, 1, 2]:
            tr, va, te = (P.split_states(seed) if proto == "LSO-T" else P.split_facilities(seed))
            E = evaluate(train(P, tr, va, tm, vm, gate=False, seed=seed), P, te, vm)
            O = evaluate(train(P, tr, va, tm, vm, gate=True, lam=1.0, seed=seed), P, te, vm)
            covE = np.mean([r["cov"] for r in E]); covO = np.mean([r["cov"] for r in O])
            pbE = np.mean([r["pinball"] for r in E]); pbO = np.mean([r["pinball"] for r in O])
            print(f"  s{seed}  ERM-Q: cov={covE:.3f} pinball={pbE:.4f} | CD-Q: cov={covO:.3f} pinball={pbO:.4f}"
                  f" | calib_err ERM={abs(covE-0.8):.3f} ours={abs(covO-0.8):.3f}")
            aggE += [(r["cov"], r["pinball"]) for r in E]; aggO += [(r["cov"], r["pinball"]) for r in O]
        aE = np.array(aggE); aO = np.array(aggO)
        print(f"  POOLED calib_err: ERM={abs(aE[:,0].mean()-0.8):.4f}  ours={abs(aO[:,0].mean()-0.8):.4f}"
              f" | pinball ERM={aE[:,1].mean():.4f} ours={aO[:,1].mean():.4f}")
