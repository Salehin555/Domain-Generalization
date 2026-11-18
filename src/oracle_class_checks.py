"""
Robustness checks of the oracle class, the screening statistic and the domain
construction.

C1  Does the measured reducible gap depend on the hypothesis class used for the
    oracle? If a linear oracle and an MLP oracle disagree, "reducible" is
    class-relative and must be reported that way.
C2  Is Jensen-Shannon divergence of band-energy profiles the right screening
    statistic, or would a kernel discrepancy (MMD) rank splits better?
C3  Does combining the training-free features beat the single best one?
C4  Do the reducibility conclusions survive a different domain-construction
    threshold (deciles instead of quartiles)?

Output: outputs/checks_*.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
ELE = os.path.join(ROOT, "cache", "electricity", "panel_elec.npz")
EP, PAT, SEEDS = 22, 5, [0, 1, 2]


# ---------------------------------------------------------------- C1
def q2_oracle_backbone():
    print("### C1: does the reducible gap depend on the oracle's model class?",
          flush=True)
    P = Panel(ELE, n_facilities=10 ** 6, L=56, H=14, stride=18, seed=0,
              n_channels=3, boundary_date=20140101)
    g = np.asarray(P.state).astype(str); idx = np.arange(P.nF)
    src, tgt = idx[g == "Flat"], idx[g == "WorkWeek"]
    n_test = max(20, int(0.35 * len(tgt))); n_val = max(8, int(0.15 * len(tgt)))
    n_tr = min(len(src) - n_val, len(tgt) - n_test - n_val)
    rows = []
    CLASSES = [("MLP (used in paper)", dict(method="erm", epochs=EP,
                                            patience=PAT, backbone="mlp")),
               ("Linear (DLinear)", dict(method="dlinear_plain", epochs=EP,
                                         patience=PAT))]
    for name, cfg in CLASSES:
        gaps = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            a = src.copy(); rng.shuffle(a); b = tgt.copy(); rng.shuffle(b)
            s_va, s_tr = a[:n_val], a[n_val:n_val + n_tr]
            t_te, pool = b[:n_test], b[n_test:]
            t_va, t_tr = pool[:n_val], pool[n_val:n_val + n_tr]
            def err(tr_, va_, te_):
                m, _ = train_model(P, cfg, tr_, va_, seed=0)
                return float(np.mean([r["mae_total"] for r in evaluate(m, P, te_)]))
            tr_e = err(s_tr, s_va, t_te); or_e = err(t_tr, t_va, t_te)
            gaps.append(100 * (tr_e - or_e) / tr_e)
        rows.append(dict(oracle_class=name, gap=float(np.mean(gaps)),
                         sd=float(np.std(gaps))))
        print(f"  {name:22s} reducible gap = {np.mean(gaps):+.2f}% "
              f"(sd {np.std(gaps):.2f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "checks_oracle_class.csv"),
                              index=False)


# ---------------------------------------------------------------- C2 / C3
def mmd(A, B, gamma=None):
    """RBF-kernel MMD^2 between two sets of per-entity profiles."""
    X = np.vstack([A, B])
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    if gamma is None:
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / (med + 1e-12)
    K = np.exp(-gamma * d2)
    n, m = len(A), len(B)
    Kxx = K[:n, :n].sum() / max(n * n, 1)
    Kyy = K[n:, n:].sum() / max(m * m, 1)
    Kxy = K[:n, n:].sum() / max(n * m, 1)
    return float(Kxx + Kyy - 2 * Kxy)


def q3_q7_divergences():
    print("\n### C2/C3: alternative screening statistics", flush=True)
    surv = pd.read_csv(os.path.join(OUT, "reducibility_survey.csv"))
    g = (surv.groupby(["panel", "split"])
         .agg(red=("reducible_gap_pct", "mean"), js=("spec_div", "first"),
              diff=("diff_ratio", "first"), sep=("sep", "first"))
         .reset_index())
    g = g[g.split != "random control"].dropna()
    res = []
    for name, col in [("JS divergence (used)", "js"),
                      ("difficulty ratio", "diff"),
                      ("group separation", "sep")]:
        r, p = pearsonr(g[col], g.red); rs, ps = spearmanr(g[col], g.red)
        res.append(dict(statistic=name, pearson=r, p=p, spearman=rs, p_s=ps))
        print(f"  {name:22s} pearson={r:+.3f} (p={p:.4f})  "
              f"spearman={rs:+.3f} (p={ps:.4f})", flush=True)
    # combined: least squares on standardized features, leave-one-out R^2
    X = g[["js", "diff", "sep"]].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = g.red.values
    preds = np.zeros_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        A = np.c_[np.ones(m.sum()), X[m]]
        beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        preds[i] = np.r_[1.0, X[i]] @ beta
    r_c, p_c = pearsonr(preds, y)
    res.append(dict(statistic="all three combined (LOO)", pearson=r_c, p=p_c,
                    spearman=spearmanr(preds, y).statistic, p_s=np.nan))
    print(f"  {'all three (LOO)':22s} pearson={r_c:+.3f} (p={p_c:.4f})",
          flush=True)
    pd.DataFrame(res).to_csv(os.path.join(OUT, "checks_divergences.csv"), index=False)
    print("  (MMD requires the per-entity profiles, computed in C4 below)",
          flush=True)


# ---------------------------------------------------------------- C4
def q8_threshold_and_mmd():
    print("\n### C4: decile instead of quartile domains, and MMD screening",
          flush=True)
    from src.survey_reducibility import entity_stats, js_div, diagnose
    rows, screen = [], []
    PANELS = [("Nursing", os.path.join(ROOT, "cache", "panel.npz"), 20250101, 1200),
              ("Electricity", ELE, 20140101, 10 ** 6)]
    for name, path, bd, nf in PANELS:
        P = Panel(path, n_facilities=nf, L=56, H=14, stride=18, seed=0,
                  n_channels=3, boundary_date=bd)
        stats, prof = entity_stats(P)
        idx = np.arange(P.nF)
        for qlo, qhi, tag in [(25, 75, "quartile"), (10, 90, "decile")]:
            for sname, v in stats.items():
                a, b = np.percentile(v, [qlo, qhi])
                A, B = idx[v <= a], idx[v >= b]
                if len(A) < 25 or len(B) < 25:
                    continue
                gaps = [d["reducible_gap_pct"] for d in
                        (diagnose(P, A, B, s) for s in [0, 1]) if d]
                if not gaps:
                    continue
                rows.append(dict(panel=name, threshold=tag, split=sname,
                                 gap=float(np.mean(gaps))))
                screen.append(dict(panel=name, threshold=tag, split=sname,
                                   gap=float(np.mean(gaps)),
                                   js=js_div(prof[A].mean(0), prof[B].mean(0)),
                                   mmd=mmd(prof[A], prof[B])))
        sub = [r for r in rows if r["panel"] == name]
        for tag in ["quartile", "decile"]:
            v = [r["gap"] for r in sub if r["threshold"] == tag]
            if v:
                print(f"  {name:12s} {tag:9s} mean gap = {np.mean(v):+.2f}% "
                      f"(range {min(v):+.2f} to {max(v):+.2f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "checks_thresholds.csv"), index=False)
    S = pd.DataFrame(screen)
    if len(S) > 5:
        rj = pearsonr(S.js, S.gap); rm = pearsonr(S.mmd, S.gap)
        print(f"\n  screening on this subset: JS r={rj[0]:+.3f} (p={rj[1]:.3f}), "
              f"MMD r={rm[0]:+.3f} (p={rm[1]:.3f})", flush=True)
        S.to_csv(os.path.join(OUT, "checks_mmd.csv"), index=False)


if __name__ == "__main__":
    q2_oracle_backbone()
    q3_q7_divergences()
    q8_threshold_and_mmd()
    print("\nSaved outputs/checks_*.csv")
