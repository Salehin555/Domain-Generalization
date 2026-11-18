"""Coverage-gated decision rules and the sequential procedure (benchmark panels).

Inputs: origin_deployed_*.csv, origin_seq_*.csv, coverage_measures.csv.
Unit: split x decision date, seeds averaged.

Coverage gate (exploratory: motivated by the solar failure and evaluated on the same
data): the estimate may authorize retraining only if every deployment day has the same
date one year earlier in the observed history (lag365 coverage = 100%); otherwise the
rule falls back to the cheapest source-validated alternative (keep, or blend).

Sequential procedure: the model in service is first improved without target data
(trained to convergence and blended with seasonal naive); the residual benefit of
retraining (converged reference, blended) is estimated on observed history (E_res)
and realized in the deployment period (D_res). Net benefits are expressed relative to
the improved model and, for the total, relative to the deployed model at 1x.
Outputs: results/gated_*.csv, seq_*.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COSTS = (0.0, 1.0, 2.0, 3.0, 5.0)
MT = sys.argv[1] if len(sys.argv) > 1 else "mae_cat"
B = 2000


def red(a, b):
    return 100.0 * (a - b) / a


def coverage():
    c = pd.read_csv(os.path.join(RES, "coverage_measures.csv"))
    c = c[c.setting.str.startswith("origin")]
    c["origin"] = c.setting.str.split().str[1].astype(int)
    return c.set_index(["panel", "origin"]).lag365


def units():
    fs = [f for f in glob.glob(os.path.join(RES, "origin_deployed_*.csv")) if "entities" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["panel", "origin", "split", "seed"])
    s = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "origin_seq_*.csv"))])
    m = d.merge(s, on=["panel", "origin", "split", "seed"], how="inner", suffixes=("", "_s"))
    o = pd.DataFrame(dict(panel=m.panel, origin=m.origin, split=m.split))
    for era, k in (("obs", "E"), ("dep", "D")):
        o[k] = red(m[f"dep_{era}_{MT}"], m[f"refE_{era}_{MT}"])
        o["L_" + k] = red(m[f"dep_{era}_{MT}"], m[f"dep2_{era}_{MT}"])
        o["B_" + k] = red(m[f"dep_{era}_{MT}"], m[f"blend_{era}_{MT}"])
        o["I_" + k] = red(m[f"dep_{era}_{MT}"], m[f"depCb_{era}_{MT}"])        # improvement, no target data
        o["T_" + k] = red(m[f"dep_{era}_{MT}"], m[f"refCb_{era}_{MT}"])        # improved retrained, vs deployed
        o["R_" + k] = red(m[f"depCb_{era}_{MT}"], m[f"refCb_{era}_{MT}"])      # residual benefit
    u = o.groupby(["panel", "origin", "split"], as_index=False).mean(numeric_only=True)
    cov = coverage()
    u["covered"] = [cov.loc[(p, int(og))] >= 0.999 for p, og in zip(u.panel, u.origin)]
    return u


def two_action(g, c, est, real, gate):
    ok = g.covered.values if gate else np.ones(len(g), bool)
    rule = ok & (g[est].values >= c)
    nb = lambda r: float(np.where(r, g[real] - c, 0.0).mean())
    return dict(always=float((g[real] - c).mean()), rule=nb(rule), oracle=nb(g[real].values > c),
                unnecessary=int((rule & (g[real].values < c)).sum()),
                missed=int((~rule & (g[real].values > c)).sum()), n=len(g))


def four_action(g, c, gate):
    est = np.column_stack([np.zeros(len(g)), g.E - c, g.L_E - c / 4, g.B_E])
    real = np.column_stack([np.zeros(len(g)), g.D - c, g.L_D - c / 4, g.B_D])
    pick = est.argmax(1)
    if gate:
        pick = np.where(g.covered.values, pick, 3)            # blend when coverage fails
    return float(real[np.arange(len(g)), pick].mean())


def totals(g, c):
    """Overall net reduction relative to the deployed model at the standard budget, per unit, then averaged.

    The improvement I and the residual reduction R have different denominators (the deployed and the improved
    model), so they compound: G = I + (1 - I) R. The retraining cost c is stated in percent of the error of the
    model in service when the decision is made (the improved model), i.e. c (1 - I) in deployed-model units.
      total_rule        improve, then retrain if the residual estimate >= c
      total_gate_retrain  as above, but the coverage gate only blocks the retraining step
      total_gate_full   the gate is applied before the first action: if coverage is incomplete, neither
                        longer training nor retraining is used and the deployed model is only blended with
                        seasonal naive (weight on source validation); otherwise as total_rule
    """
    I, R = g.I_D.values / 100.0, g.R_D.values / 100.0
    rule = g.R_E.values >= c
    cov = g.covered.values

    def tot(retrain):
        return 100.0 * (I + (1 - I) * R * retrain) - c * (1 - I) * retrain

    full = np.where(cov, tot(rule), g.B_D.values)
    return dict(total_improve=float((100.0 * I).mean()), total_rule=float(tot(rule).mean()),
                total_gate_retrain=float(tot(rule & cov).mean()), total_gate_full=float(full.mean()),
                total_always=float(tot(np.ones(len(g), bool)).mean()))


def boot_pooled(u, fn, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    groups = {p: g for p, g in u.groupby("panel")}
    for _ in range(B):
        parts = []
        for g in groups.values():
            sp = g.split.unique()
            parts.append(pd.concat([g[g.split == k] for k in rng.choice(sp, len(sp))]))
        out.append(fn(pd.concat(parts)))
    return np.percentile(out, [2.5, 97.5])


def main():
    u = units()
    u.to_csv(os.path.join(RES, f"seq_units_{MT}.csv"), index=False)
    groups = list(u.groupby("panel")) + [("All panels", u)]

    # four-action rule, with and without the coverage gate, against always blending
    rows = []
    for c in COSTS:
        for p, g in groups:
            r = dict(panel=p, cost=c, always_blend=float(g.B_D.mean()),
                     always_retrain=float((g.D - c).mean()), always_longer=float((g.L_D - c / 4).mean()),
                     rule=four_action(g, c, False), rule_gated=four_action(g, c, True),
                     oracle=float(np.column_stack([np.zeros(len(g)), g.D - c, g.L_D - c / 4, g.B_D]).max(1).mean()))
            if p == "All panels":
                r["gated_minus_blend_lo"], r["gated_minus_blend_hi"] = boot_pooled(
                    g, lambda s: four_action(s, c, True) - s.B_D.mean())
                r["rule_minus_blend_lo"], r["rule_minus_blend_hi"] = boot_pooled(
                    g, lambda s: four_action(s, c, False) - s.B_D.mean())
            rows.append(r)
    fa = pd.DataFrame(rows)
    fa.to_csv(os.path.join(RES, f"gated_four_action_{MT}.csv"), index=False)
    print(fa.round(2).to_string())

    # two-action rule against the deployed model, gated
    rows = []
    for c in COSTS:
        for p, g in groups:
            a = two_action(g, c, "E", "D", False)
            b = two_action(g, c, "E", "D", True)
            rows.append(dict(panel=p, cost=c, always=a["always"], rule=a["rule"], rule_gated=b["rule"],
                             oracle=a["oracle"], unn=a["unnecessary"], unn_gated=b["unnecessary"]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"gated_two_action_{MT}.csv"), index=False)
    print(pd.DataFrame(rows).round(2).to_string())

    # sequential procedure
    assoc = []
    for p, g in groups:
        assoc.append(dict(panel=p, n=len(g), improve_D=float(g.I_D.mean()), retrained_improved_D=float(g.T_D.mean()),
                          resid_E=float(g.R_E.mean()), resid_D=float(g.R_D.mean()),
                          r_resid=float(np.corrcoef(g.R_E, g.R_D)[0, 1]), bias_resid=float((g.R_E - g.R_D).mean()),
                          share_resid_pos=float((g.R_D > 0).mean())))
    pd.DataFrame(assoc).to_csv(os.path.join(RES, f"seq_assoc_{MT}.csv"), index=False)
    print(pd.DataFrame(assoc).round(2).to_string())
    rows = []
    for c in COSTS:
        for p, g in groups:
            a = two_action(g, c, "R_E", "R_D", False)
            b = two_action(g, c, "R_E", "R_D", True)
            r = dict(panel=p, cost=c, always=a["always"], rule=a["rule"], rule_gated=b["rule"],
                     oracle=a["oracle"], unnecessary=a["unnecessary"], missed=a["missed"],
                     improve_only=float(g.I_D.mean()), **totals(g, c))
            if p == "All panels":
                r["rule_minus_best_lo"], r["rule_minus_best_hi"] = boot_pooled(
                    g, lambda s: two_action(s, c, "R_E", "R_D", False)["rule"]
                    - max(0.0, two_action(s, c, "R_E", "R_D", False)["always"]))
            rows.append(r)
    sq = pd.DataFrame(rows)
    sq.to_csv(os.path.join(RES, f"seq_costs_{MT}.csv"), index=False)
    print(sq.round(2).to_string())


if __name__ == "__main__":
    main()
