"""Analysis: retraining decisions against the deployed model.

Input: results/origin_deployed_*.csv (origins_deployed.py).
Unit: split x decision date, the two allocation seeds averaged (as in an_budget_dates.py).

For each unit:
  E   estimate: reduction in observed-era error of the equal-budget target-trained
      reference relative to the deployed model (percent of the deployed model's error)
  D   realized: the same reduction in the deployment period
  E_s, D_s  the same for the size-matched pair (r5 definition, for comparison)
  L_E, L_D  gain of training the deployed model for twice its budget
  B_E, B_D  gain of the source-validated seasonal blend of the deployed model

Policies at retraining cost c (same units as D):
  never, always, rule (retrain if E >= c), oracle (retrain if D > c, hindsight)
Multi-action (costs c for retraining, c/4 for training longer, 0 for the blend):
  best fixed action, rule (largest estimated net benefit), oracle.
Outputs: results/decision_*.csv and decision_stats.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COSTS = (0.0, 1.0, 2.0, 3.0, 5.0)
B = 2000
MT = sys.argv[1] if len(sys.argv) > 1 else "mae_cat"


def red(a, b):
    return 100.0 * (a - b) / a


def load():
    fs = [f for f in glob.glob(os.path.join(RES, "origin_deployed_*.csv")) if "entities" not in f]
    df = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["panel", "origin", "split", "seed"])
    out = pd.DataFrame(dict(panel=df.panel, origin=df.origin, split=df.split, seed=df.seed))
    for era, k in (("obs", "E"), ("dep", "D")):
        out[k] = red(df[f"dep_{era}_{MT}"], df[f"refE_{era}_{MT}"])
        out[k + "_s"] = red(df[f"src_{era}_{MT}"], df[f"ref_{era}_{MT}"])
        out["L_" + k] = red(df[f"dep_{era}_{MT}"], df[f"dep2_{era}_{MT}"])
        out["B_" + k] = red(df[f"dep_{era}_{MT}"], df[f"blend_{era}_{MT}"])
        out["SM_" + k] = red(df[f"dep_{era}_{MT}"], df[f"src_{era}_{MT}"])  # deployed vs size-matched
    units = out.groupby(["panel", "origin", "split"], as_index=False).mean(numeric_only=True)
    return units.drop(columns="seed")


def policies(u, c):
    E, D = u.E.values, u.D.values
    rule = E >= c
    orac = D > c
    nb = lambda r: np.where(r, D - c, 0.0)
    return dict(never=0.0, always=float(np.mean(D - c)), rule=float(nb(rule).mean()),
                oracle=float(nb(orac).mean()), retrain_rate=float(rule.mean()),
                unnecessary=int((rule & ~orac).sum()), missed=int((~rule & orac).sum()),
                n=len(u))


def multi(u, c):
    """Net benefit of each action; estimates choose, realized values score."""
    est = np.column_stack([np.zeros(len(u)), u.E - c, u.L_E - c / 4, u.B_E])
    real = np.column_stack([np.zeros(len(u)), u.D - c, u.L_D - c / 4, u.B_D])
    names = ["keep", "retrain", "longer", "blend"]
    fixed = {n: float(real[:, i].mean()) for i, n in enumerate(names)}
    pick = est.argmax(1)
    rule = float(real[np.arange(len(u)), pick].mean())
    oracle = float(real.max(1).mean())
    return fixed, rule, oracle, np.bincount(pick, minlength=4) / len(u)


def boot(u, fn, panel_level=False, seed=0):
    """Percentile interval: resample splits within panel (all dates of a split together),
    or whole panels when panel_level is True."""
    rng = np.random.default_rng(seed)
    vals = []
    groups = {p: g for p, g in u.groupby("panel")}
    for _ in range(B):
        if panel_level:
            ps = rng.choice(list(groups), len(groups), replace=True)
            s = pd.concat([groups[p] for p in ps])
        else:
            parts = []
            for p, g in groups.items():
                sp = g.split.unique()
                pick = rng.choice(sp, len(sp), replace=True)
                parts.append(pd.concat([g[g.split == k] for k in pick]))
            s = pd.concat(parts)
        vals.append(fn(s))
    return np.percentile(vals, [2.5, 97.5])


def main():
    u = load()
    u.to_csv(os.path.join(RES, f"decision_units_{MT}.csv"), index=False)
    stats = {}
    # association of estimate with realized value, deployed and size-matched
    rows = []
    for p, g in list(u.groupby("panel")) + [("All panels", u)]:
        rows.append(dict(panel=p, n_units=len(g), n_splits=g.groupby(["panel", "split"]).ngroups,
                         r_dep=np.corrcoef(g.E, g.D)[0, 1], bias_dep=float((g.E - g.D).mean()),
                         mae_dep=float((g.E - g.D).abs().mean()), mean_D=float(g.D.mean()),
                         r_sm=np.corrcoef(g.E_s, g.D_s)[0, 1], bias_sm=float((g.E_s - g.D_s).mean()),
                         mean_D_sm=float(g.D_s.mean()), mean_SM_D=float(g.SM_D.mean()),
                         mean_L_D=float(g.L_D.mean()), mean_B_D=float(g.B_D.mean()),
                         share_D_pos=float((g.D > 0).mean())))
    assoc = pd.DataFrame(rows)
    assoc.to_csv(os.path.join(RES, f"decision_assoc_{MT}.csv"), index=False)
    print(assoc.round(3).to_string())

    # retrain-or-keep policies with costs, pooled and per panel, with intervals
    rows = []
    for c in COSTS:
        for p, g in list(u.groupby("panel")) + [("All panels", u)]:
            r = dict(panel=p, cost=c, **policies(g, c))
            if p == "All panels":
                lo, hi = boot(g, lambda s: policies(s, c)["rule"] - policies(s, c)["always"])
                plo, phi = boot(g, lambda s: policies(s, c)["rule"] - policies(s, c)["always"],
                                panel_level=True)
                eq = np.mean([policies(gg, c)["rule"] - policies(gg, c)["always"]
                              for _, gg in g.groupby("panel")])
                r.update(rma_lo=lo, rma_hi=hi, rma_panel_lo=plo, rma_panel_hi=phi, rma_equal_panel=eq)
            else:
                pass
            r["rule_minus_always"] = r["rule"] - r["always"]
            r["regret_rule"] = r["oracle"] - r["rule"]
            r["regret_best_fixed"] = r["oracle"] - max(r["always"], 0.0)
            r["voi"] = r["rule"] - max(r["always"], 0.0)
            rows.append(r)
    pol = pd.DataFrame(rows)
    pol.to_csv(os.path.join(RES, f"decision_costs_{MT}.csv"), index=False)
    print(pol[["panel", "cost", "always", "rule", "oracle", "rule_minus_always", "voi", "unnecessary",
               "missed", "retrain_rate"]].round(2).to_string())

    # leave one panel out (pooled rule minus always)
    lopo = []
    for c in COSTS:
        for p in u.panel.unique():
            g = u[u.panel != p]
            s = policies(g, c)
            lopo.append(dict(left_out=p, cost=c, rule_minus_always=s["rule"] - s["always"],
                             voi=s["rule"] - max(s["always"], 0)))
    pd.DataFrame(lopo).to_csv(os.path.join(RES, f"decision_lopo_{MT}.csv"), index=False)

    # asymmetric costs: an unnecessary retraining costs extra k points beyond c
    asym = []
    for c in (1.0, 3.0):
        for k in (0.0, 2.0, 5.0):
            for p, g in list(u.groupby("panel")) + [("All panels", u)]:
                rule = g.E >= c
                nb = np.where(rule, g.D - c - k * (g.D < c), 0.0).mean()
                alw = (g.D - c - k * (g.D < c)).mean()
                orac = np.where(g.D > c, g.D - c, 0.0).mean()
                asym.append(dict(panel=p, cost=c, extra_false=k, rule=nb, always=alw, oracle=orac))
    pd.DataFrame(asym).to_csv(os.path.join(RES, f"decision_asym_{MT}.csv"), index=False)

    # calibration at decision thresholds
    cal = []
    for t in (1.0, 3.0, 5.0):
        for p, g in list(u.groupby("panel")) + [("All panels", u)]:
            sel = g.E >= t
            cal.append(dict(panel=p, threshold=t, n_selected=int(sel.sum()),
                            precision=float((g.D[sel] >= t).mean()) if sel.any() else np.nan,
                            recall=float((sel & (g.D >= t)).sum() / max(1, (g.D >= t).sum())),
                            mean_E_sel=float(g.E[sel].mean()) if sel.any() else np.nan,
                            mean_D_sel=float(g.D[sel].mean()) if sel.any() else np.nan))
    pd.DataFrame(cal).to_csv(os.path.join(RES, f"decision_calibration_{MT}.csv"), index=False)

    # multi-action decisions
    ma = []
    for c in COSTS:
        for p, g in list(u.groupby("panel")) + [("All panels", u)]:
            fixed, rule, oracle, share = multi(g, c)
            ma.append(dict(panel=p, cost=c, **{f"fixed_{k}": v for k, v in fixed.items()},
                           rule=rule, oracle=oracle, best_fixed=max(fixed.values()),
                           share_keep=share[0], share_retrain=share[1], share_longer=share[2],
                           share_blend=share[3]))
    ma = pd.DataFrame(ma)
    ma.to_csv(os.path.join(RES, f"decision_multi_{MT}.csv"), index=False)
    print(ma.round(2).to_string())
    json.dump(dict(n_units=len(u)), open(os.path.join(RES, f"decision_stats_{MT}.json"), "w"))


if __name__ == "__main__":
    main()
