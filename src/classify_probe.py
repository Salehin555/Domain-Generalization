"""
Prototype #1: understaffing-risk classification (decision-relevant, not noise-saturated).

Task: from a 56-day lookback of [RN, LPN, CNA] hours-per-resident-day, predict for each
of the next 14 days whether TOTAL nurse HPRD < 3.48 (the CMS 2024 federal minimum) --
i.e. an "understaffed day". This is a per-horizon-day binary classification.

Rationale: understaffing is largely DAY-OF-WEEK driven (weekend dips), which is exactly
the shared weekly sub-band CD-ISG-TS models. We keep the absolute level (global
standardization, no RevIN) because the threshold is on the absolute scale.

Metric: AUROC and AUPRC (pooled over facility x window x horizon-day, and per-facility).
Compares DLinear / ERM / CD-ISG-TS under leave-facilities-out + 2024 -> 2025.
"""
import numpy as np, torch, torch.nn as nn, os
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data import Panel
from src.models import Stem, SubBandGate, DLinear, cross_domain_variance

torch.set_num_threads(max(1, os.cpu_count() - 2))
THR = 3.48

class Clf(nn.Module):
    def __init__(self, L, H, C, kind="erm", level=3):
        super().__init__()
        self.kind = kind
        self.register_buffer("mu", torch.zeros(C, 1))
        self.register_buffer("sd", torch.ones(C, 1))
        if kind == "dlinear":
            self.head = nn.Linear(C * L, H)
        else:
            self.stem = Stem(C, L)
            self.gate = SubBandGate(L, C, "db4", level) if kind == "cdisgts" else None
            self.head = nn.Sequential(nn.Linear(C * L, 128), nn.GELU(), nn.Linear(128, H))
    def set_norm(self, mu, sd):
        self.mu.copy_(mu); self.sd.copy_(sd)
    def forward(self, x, energy=False):
        z = (x - self.mu) / self.sd
        rel = None
        if self.kind == "dlinear":
            return self.head(z.reshape(z.shape[0], -1)), None
        z = self.stem(z)
        if self.gate is not None:
            if energy: z, rel = self.gate(z, return_energy=True)
            else: z = self.gate(z)
        return self.head(z.reshape(z.shape[0], -1)), rel

def train(P, tr, va, tm, vm, kind, lam=0.0, seed=0, epochs=24):
    np.random.seed(seed); torch.manual_seed(seed)
    L, H, C = P.L, P.H, 3
    M = Clf(L, H, C, kind)
    # global standardization from training windows (preserves cross-facility level)
    idx = np.concatenate([P._pool(f, tm) for f in tr])[:20000]
    Xt = P.Xw[idx]
    M.set_norm(torch.tensor(Xt.mean((0, 2), keepdims=True)[0]),
               torch.tensor(Xt.std((0, 2), keepdims=True)[0] + 1e-6))
    gp = [p for n, p in M.named_parameters() if "gate_logit" in n]
    rp = [p for n, p in M.named_parameters() if "gate_logit" not in n]
    groups = [dict(params=rp, lr=1e-3, weight_decay=1e-4)]
    if gp: groups.append(dict(params=gp, lr=0.01, weight_decay=0.0))
    opt = torch.optim.Adam(groups)
    bce = nn.BCEWithLogitsLoss()
    best, best_state, pat = -1, None, 0
    for ep in range(epochs):
        ramp = 0.0 if ep < 0.4 * epochs else min(1.0, (ep - 0.4*epochs) / (0.3*epochs))
        M.train()
        for Xb, Yb, fb in P.grouped_batches(tr, 32, 8, seed=seed*97+ep, time_mask=tm):
            xb = torch.from_numpy(Xb); fbi = torch.from_numpy(fb)
            lab = torch.from_numpy((Yb.sum(1) < THR).astype(np.float32))   # [B,H]
            opt.zero_grad()
            if kind == "cdisgts" and lam > 0:
                logit, rel = M(xb, energy=True)
                loss = bce(logit, lab) + lam*ramp*cross_domain_variance(rel, fbi).mean()
            else:
                logit, _ = M(xb)
                loss = bce(logit, lab)
            loss.backward(); opt.step()
        # val AUROC
        auc = auroc(M, P, va, vm)
        if auc > best + 1e-4: best, best_state, pat = auc, {k: t.clone() for k, t in M.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 6: break
    if best_state: M.load_state_dict(best_state)
    return M

@torch.no_grad()
def _scores(M, P, fac, mask):
    ys, ps, facs = [], [], []
    for f in fac:
        idx = P._pool(f, mask)
        if len(idx) == 0: continue
        lg, _ = M(torch.from_numpy(P.Xw[idx]))
        p = torch.sigmoid(lg).numpy().ravel()
        y = (P.Yw[idx].sum(1) < THR).astype(np.float32).ravel()
        ys.append(y); ps.append(p); facs.append(np.full(len(y), f))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(facs)

def auroc(M, P, fac, mask):
    y, p, _ = _scores(M, P, fac, mask)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5

def evaluate(M, P, fac, mask):
    y, p, fc = _scores(M, P, fac, mask)
    pooled_auc = roc_auc_score(y, p); pooled_ap = average_precision_score(y, p)
    per = []
    for f in np.unique(fc):
        m = fc == f
        if len(np.unique(y[m])) > 1:
            per.append(roc_auc_score(y[m], p[m]))
    return pooled_auc, pooled_ap, np.array(per)

if __name__ == "__main__":
    from scipy.stats import wilcoxon
    P = Panel("cache/panel.npz", n_facilities=2000, L=56, H=14, stride=12, seed=0)
    tm, vm = P.is_2024, P.is_2025
    print(f"understaffing classification (TOTAL HPRD < {THR}); base rate on test set shown below")
    for seed in [0, 1, 2]:
        tr, va, te = P.split_facilities(seed)
        res = {}
        for kind, lam in [("dlinear", 0), ("erm", 0), ("cdisgts", 2.0)]:
            M = train(P, tr, va, tm, vm, kind, lam=lam, seed=seed)
            res[kind] = evaluate(M, P, te, vm)
        # per-facility paired AUROC ours vs erm (align on facilities present in both)
        (a_dl, ap_dl, _), (a_er, ap_er, per_er) = res["dlinear"], res["erm"]
        a_ours, ap_ours, per_ours = res["cdisgts"]
        n = min(len(per_er), len(per_ours))
        try: pval = wilcoxon(per_ours[:n], per_er[:n]).pvalue
        except ValueError: pval = 1.0
        print(f"seed{seed}  AUROC  DLinear={a_dl:.4f}  ERM={a_er:.4f}  CD-ISG-TS={a_ours:.4f}"
              f"  | ours-erm={a_ours-a_er:+.4f}  per-fac win%={100*(per_ours[:n]>per_er[:n]).mean():.1f} p={pval:.1e}"
              f"  | AUPRC ERM={ap_er:.4f} ours={ap_ours:.4f}")
