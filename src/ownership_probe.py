"""
Prototype #2: domain splits by a real behavioral factor (proposal caution #4 hedge).
Merge CMS Provider Information (ownership type, urban/rural) with our panel and test
whether the cross-domain gap is genuinely larger -- and whether CD-ISG-TS helps -- when
we train on one ownership/geography group and deploy on an unseen group.

Splits (leave-group-out; validation drawn from source group only):
  OWN : train For-profit  -> deploy Non-profit
  GEO : train Urban       -> deploy Rural
Optionally combined with the 2024 -> 2025 temporal shift.
"""
import numpy as np, pandas as pd, os
from scipy.stats import wilcoxon
from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache")

def build_meta(P):
    d = pd.read_csv(os.path.join(CACHE, "NH_ProviderInfo.csv"), encoding="latin-1",
                    dtype=str, low_memory=False)
    d["CCN"] = d["CMS Certification Number (CCN)"].str.zfill(6)
    prov = pd.Series(P.prov).astype(str).str.zfill(6)
    m = d.set_index("CCN").reindex(prov)
    def grp(s):
        s = str(s).lower()
        if "non profit" in s or "nonprofit" in s: return "NonProfit"
        if "profit" in s: return "ForProfit"
        if "government" in s: return "Government"
        return "Other"
    own = m["Ownership Type"].map(grp).to_numpy()
    urban = m["Urban"].to_numpy()
    return own, urban

def masks(P, temporal):
    return (P.is_2024, P.is_2025) if temporal else (None, None)

def run(P, cfg, tr, va, te, tm, vm, seed):
    m, _ = train_model(P, cfg, tr, va, seed=seed, train_mask=tm, val_mask=vm)
    src = np.mean([x["mae_total"] for x in evaluate(m, P, va, time_mask=vm)])
    r = np.array([x["mae_total"] for x in evaluate(m, P, te, time_mask=vm)])
    return src, r

def main():
    P = Panel(os.path.join(CACHE, "panel.npz"), n_facilities=13402, L=56, H=14, stride=16, seed=0)
    own, urban = build_meta(P)
    idx = np.arange(P.nF)
    splits = {
        "OWN (ForProfit->NonProfit)": (idx[own == "ForProfit"], idx[own == "NonProfit"]),
        "GEO (Urban->Rural)":         (idx[urban == "Y"],       idx[urban == "N"]),
    }
    for temporal in [False, True]:
        tm, vm = masks(P, temporal)
        tag = "+2024->2025" if temporal else "same-period"
        print(f"\n========== {tag} ==========", flush=True)
        for name, (src_all, tgt) in splits.items():
            for seed in [0, 1, 2]:
                rng = np.random.default_rng(seed); s = src_all.copy(); rng.shuffle(s)
                nval = int(0.15 * len(s)); va = s[:nval]; tr = s[nval:]
                e_src, erm = run(P, dict(method="erm", epochs=22, patience=5, backbone="mlp"),
                                 tr, va, tgt, tm, vm, seed)
                _, dl = run(P, dict(method="dlinear_plain", epochs=22, patience=5),
                            tr, va, tgt, tm, vm, seed)
                _, ours = run(P, dict(method="cdisgts", epochs=22, patience=5, backbone="mlp",
                                      lam_inv=1.0, gate_lr=0.01), tr, va, tgt, tm, vm, seed)
                n = min(len(erm), len(ours))
                p = wilcoxon(ours[:n], erm[:n]).pvalue if not np.allclose(ours[:n], erm[:n]) else 1.0
                gap = 100 * (erm.mean() - e_src) / e_src
                print(f"  {name:28s} s{seed}  ERM_src={e_src:.4f} ERM_tgt={erm.mean():.4f} (gap {gap:+.1f}%) "
                      f"| DLinear={dl.mean():.4f} ERM={erm.mean():.4f} CD-ISG-TS={ours.mean():.4f} "
                      f"| ours-erm={ours.mean()-erm.mean():+.5f} win%={100*(ours[:n]<erm[:n]).mean():.1f} p={p:.1e}",
                      flush=True)

if __name__ == "__main__":
    main()
