"""Analysis of the budget, coverage and decision-date experiments (saved rows only; no training).

1. Coverage versus history length (coverage_rows.csv). Seeds averaged within
   split. For a split and a history placement h, E is the estimate on h and D_cov,
   D_unc the realized headroom on the deployment half that h covers and on the one
   it does not. Reported per panel: mean |E - D| and bias in each condition,
   Pearson r over (split, placement) points, sign agreement, and the paired
   difference  delta = |E - D_unc| - |E - D_cov|  (positive: coverage helps), with
   a bootstrap over splits (both placements of a split resampled together).

2. Several decision dates and explicit costs (origin_rows_*.csv). Seeds averaged
   within split. Unit: split x origin. A retraining costs c percentage points of
   the source model's error, i.e. it pays only if realized headroom D exceeds c.
   Net benefit per unit, relative to never retraining (0):
       always      D - c
       rule        (D - c) * 1[E >= c]
       rule_bc     (D - c) * 1[E - b >= c], b = mean(E - D) over all splits of the
                   panel's EARLIER origins (available at the decision date); first
                   origins have no correction and are excluded from rule_bc means
       oracle      max(D - c, 0)       (hindsight; not available in practice)
   Costs c in {0, 1, 2, 3, 5}. Bootstrap over splits (all origins of a split
   resampled together) for rule - always and rule - never.

3. Joint training and fine-tuning (transfer_rows_*.csv, *_entities_*.csv).
   Error reduction of each arm relative to the deployed model and relative to the
   target-trained reference (mean over seeds of per-seed ratios); the
   negative-transfer gap of Wang et al. (2019) for the joint arm is its error minus
   the reference's error; entity-bootstrap intervals (seed-averaged per-entity
   errors, 2,000 resamples) for the main contrasts.

Outputs: results/summary_coverage.csv, summary_origins.csv,
         summary_costs.csv, summary_transfer.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import scipy.stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
B = 2000
COSTS = (0.0, 1.0, 2.0, 3.0, 5.0)


def boot_splits(units, fn, rng, key="split"):
    ids = units[key].unique()
    out = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        u = pd.concat([units[units[key] == i] for i in pick])
        out.append(fn(u))
    return np.percentile(out, [2.5, 97.5])


def coverage(rng):
    p = os.path.join(RES, "coverage_rows.csv")
    if not os.path.exists(p):
        return
    R = pd.read_csv(p)
    R = R.groupby(["panel", "split", "history"]).mean(numeric_only=True).reset_index()
    cov_of = {"H1": "D1", "H2": "D2"}
    unc_of = {"H1": "D2", "H2": "D1"}
    R["E"] = R.headroom_est_mae_cat
    R["Dc"] = [r[f"headroom_{cov_of[r.history]}_mae_cat"] for _, r in R.iterrows()]
    R["Du"] = [r[f"headroom_{unc_of[r.history]}_mae_cat"] for _, r in R.iterrows()]
    out = []
    for pn, g in R.groupby("panel"):
        ac, au = (g.E - g.Dc).abs(), (g.E - g.Du).abs()
        delta = lambda u: float(((u.E - u.Du).abs() - (u.E - u.Dc).abs()).mean())
        lo, hi = boot_splits(g, delta, rng)
        out.append(dict(
            panel=pn, n_splits=g.split.nunique(), n_points=len(g),
            mae_cov=ac.mean(), mae_unc=au.mean(), delta=delta(g), delta_lo=lo, delta_hi=hi,
            share_better=float((au > ac).mean()),
            bias_cov=(g.E - g.Dc).mean(), bias_unc=(g.E - g.Du).mean(),
            r_cov=ss.pearsonr(g.E, g.Dc)[0], r_unc=ss.pearsonr(g.E, g.Du)[0],
            sign_cov=float((np.sign(g.E) == np.sign(g.Dc)).mean()),
            sign_unc=float((np.sign(g.E) == np.sign(g.Du)).mean()),
            win_h1=float(g[g.history == "H1"].train_windows_per_entity.mean()),
            win_h2=float(g[g.history == "H2"].train_windows_per_entity.mean())))
    pd.DataFrame(out).to_csv(os.path.join(RES, "summary_coverage.csv"), index=False)
    print(pd.DataFrame(out).round(3).to_string())


def origins(rng):
    fs = sorted(glob.glob(os.path.join(RES, "origin_rows_*.csv")))
    if not fs:
        return
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    R = (R.groupby(["panel", "origin", "origin_idx", "split"]).mean(numeric_only=True)
         .reset_index())
    R["E"], R["D"] = R.headroom_obs_mae_cat, R.headroom_dep_mae_cat
    # bias correction from earlier origins of the same panel only
    R["b"] = np.nan
    for pn, g in R.groupby("panel"):
        for j in sorted(g.origin_idx.unique()):
            prev = g[g.origin_idx < j]
            if len(prev):
                R.loc[(R.panel == pn) & (R.origin_idx == j), "b"] = float((prev.E - prev.D).mean())
    so, sc = [], []
    for pn, g in R.groupby("panel"):
        for o, h in g.groupby("origin"):
            so.append(dict(panel=pn, origin=o, history_days=int(h.history_days.iloc[0]),
                           n_splits=len(h), r=ss.pearsonr(h.E, h.D)[0] if len(h) > 2 else np.nan,
                           rho=ss.spearmanr(h.E, h.D)[0] if len(h) > 2 else np.nan,
                           bias=float((h.E - h.D).mean()), mae=float((h.E - h.D).abs().mean()),
                           mean_D=float(h.D.mean())))
        so.append(dict(panel=pn, origin="pooled", history_days=np.nan, n_splits=len(g),
                       r=ss.pearsonr(g.E, g.D)[0], rho=ss.spearmanr(g.E, g.D)[0],
                       bias=float((g.E - g.D).mean()), mae=float((g.E - g.D).abs().mean()),
                       mean_D=float(g.D.mean())))
    groups = list(R.groupby("panel")) + [("All", R)]
    for pn, g in groups:
        for c in COSTS:
            nb = lambda u, c=c: dict(
                always=float((u.D - c).mean()),
                rule=float(((u.D - c) * (u.E >= c)).mean()),
                oracle=float(np.maximum(u.D - c, 0).mean()))
            v = nb(g)
            gb = g[g.b.notna()]
            rbc = float(((gb.D - c) * ((gb.E - gb.b) >= c)).mean()) if len(gb) else np.nan
            rraw_later = float(((gb.D - c) * (gb.E >= c)).mean()) if len(gb) else np.nan
            key = ["panel", "split"] if pn == "All" else ["split"]
            u = g.assign(uid=g[key].astype(str).agg("|".join, axis=1))
            d_al = boot_splits(u, lambda x: nb(x)["rule"] - nb(x)["always"], rng, key="uid")
            d_nv = boot_splits(u, lambda x: nb(x)["rule"], rng, key="uid")
            act = g.E >= c; ben = g.D >= c
            sc.append(dict(panel=pn, cost=c, n_units=len(g), never=0.0, **v,
                           rule_minus_always=v["rule"] - v["always"],
                           rma_lo=d_al[0], rma_hi=d_al[1], rule_lo=d_nv[0], rule_hi=d_nv[1],
                           rule_later=rraw_later, rule_bc_later=rbc, n_later=len(gb),
                           retrain_rate=float(act.mean()),
                           unnecessary=int((act & ~ben).sum()), missed=int((~act & ben).sum()),
                           beneficial=int((act & ben).sum())))
    pd.DataFrame(so).to_csv(os.path.join(RES, "summary_origins.csv"), index=False)
    pd.DataFrame(sc).to_csv(os.path.join(RES, "summary_costs.csv"), index=False)
    print(pd.DataFrame(so).round(3).to_string())
    print(pd.DataFrame(sc).round(3).to_string())


def transfer(rng):
    fs = sorted(glob.glob(os.path.join(RES, "transfer_rows_*.csv")))
    if not fs:
        return
    R = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    E = pd.concat([pd.read_csv(f) for f in
                   sorted(glob.glob(os.path.join(RES, "transfer_entities_*.csv")))],
                  ignore_index=True)
    out = []
    for (pn, sp), g in R.groupby(["panel", "split"]):
        w = g.pivot(index="seed", columns="arm", values="mae_cat")
        ent = (E[(E.panel == pn) & (E.split == sp)]
               .groupby(["arm", "fac"]).mae_cat.mean().unstack(0))
        for arm in w.columns:
            red_dep = (100 * (w["dep"] - w[arm]) / w["dep"])
            red_ref = (100 * (w["ref"] - w[arm]) / w["ref"])
            # entity bootstrap of the reduction relative to the deployed model
            a, d = ent[arm].to_numpy(), ent["dep"].to_numpy()
            n = len(a); bs = []
            for _ in range(B):
                i = rng.integers(0, n, n)
                bs.append(100 * (d[i].mean() - a[i].mean()) / d[i].mean())
            out.append(dict(panel=pn, split=sp, arm=arm, n_seeds=len(w),
                            mae=float(w[arm].mean()),
                            red_vs_dep=float(red_dep.mean()), red_vs_dep_sd=float(red_dep.std(ddof=1)) if len(w) > 1 else np.nan,
                            red_vs_dep_lo=np.percentile(bs, 2.5), red_vs_dep_hi=np.percentile(bs, 97.5),
                            red_vs_ref=float(red_ref.mean()),
                            ntg_vs_ref=float((w[arm] - w["ref"]).mean())))
    pd.DataFrame(out).to_csv(os.path.join(RES, "summary_transfer.csv"), index=False)
    print(pd.DataFrame(out).round(3).to_string())


def budget(rng):
    """4. Budget curve (budget_curve_rows.csv): comparisons at 1x, 2x, 4x the
    deployed optimization budget, every arm at the same budget. Mean over seeds of
    per-seed ratios; the change in the deployed model's error with budget shows how
    far the standard budget is from convergence."""
    p = os.path.join(RES, "budget_curve_rows.csv")
    if not os.path.exists(p):
        return
    Bd = pd.read_csv(p)
    w = Bd.pivot_table(index=["panel", "split", "seed", "mult"], columns="arm",
                       values="mae_cat")
    d1 = w.xs(1, level="mult")["dep"]
    o = pd.DataFrame(index=w.index)
    o["h_sm"] = 100 * (w.src - w.ref) / w.src
    o["h_dep"] = 100 * (w.dep - w.ref) / w.dep
    o["joint_vs_dep"] = 100 * (w.dep - w.joint_dep) / w.dep
    o["joint_vs_ref"] = 100 * (w.ref - w.joint_dep) / w.ref
    o["dep_gain"] = [100 * (d1.loc[i[:3]] - w.loc[i, "dep"]) / d1.loc[i[:3]] for i in w.index]
    s = o.groupby(level=["panel", "split", "mult"]).agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    s.to_csv(os.path.join(RES, "summary_budget.csv"), index=False)
    print(s.round(3).to_string())


def main():
    rng = np.random.default_rng(0)
    coverage(rng)
    origins(rng)
    transfer(rng)
    budget(rng)


if __name__ == "__main__":
    main()
