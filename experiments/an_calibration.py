"""Can the positive bias of the observed-history estimate be
corrected before deployment?

Question: the observed-history estimate overstates deployment headroom
on every panel; is there a practical bias correction?

Uses only temporal_A.csv (standard variant, primary metric). Each row is one
split x seed: estimate = headroom_obs, realized = headroom_dep. Corrections are
always fitted on data that would be available when the corrected panel is
estimated, i.e. never on that panel's own realized values:

    raw                 the estimate as reported
    shift (cross-panel) estimate minus the mean bias of the OTHER panels whose
                        observed history spans a full seasonal cycle
    line (cross-panel)  realized ~ a + b * estimate fitted on those other panels
    shift (earlier splits, same panel), line (earlier splits, same panel)
                        leave-one-split-out within the panel, both seeds of the
                        held-out split excluded; this assumes realized headroom
                        from earlier deployments on the same panel exists, so it
                        is an optimistic reference, not a pre-deployment rule

Solar, whose history does not span a seasonal cycle, is corrected with the
cross-panel rules too, to show whether calibration can rescue a violated
precondition. It never serves as a calibration source.

Output: calibration_table.csv, calibration_stats.json
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FULL = ["Nursing", "Electricity", "Traffic"]          # full seasonal coverage
THR = 3.0


def summarize(est, real):
    est, real = np.asarray(est, float), np.asarray(real, float)
    return dict(mae=float(np.mean(np.abs(est - real))), bias=float(np.mean(est - real)),
                pearson=float(np.corrcoef(est, real)[0, 1]),
                agree=float(np.mean((est >= THR) == (real >= THR))))


def main():
    d = pd.read_csv(os.path.join(RES, "temporal_A.csv"))
    d = d[d.variant == "standard"].copy()
    d["est"], d["real"] = d.headroom_obs_mae_cat, d.headroom_dep_mae_cat
    out = []
    for p in FULL + ["Solar"]:
        g = d[d.panel == p]
        others = d[d.panel.isin([q for q in FULL if q != p])]
        shift = float((others.est - others.real).mean())
        b, a = np.polyfit(others.est, others.real, 1)
        variants = {"raw": g.est.to_numpy(),
                    "shift (cross-panel)": (g.est - shift).to_numpy(),
                    "line (cross-panel)": (a + b * g.est).to_numpy()}
        if p in FULL:
            sh, ln = np.empty(len(g)), np.empty(len(g))
            for i, (_, r) in enumerate(g.iterrows()):
                o = g[g.split != r.split]
                sh[i] = r.est - float((o.est - o.real).mean())
                bb, aa = np.polyfit(o.est, o.real, 1)
                ln[i] = aa + bb * r.est
            variants["shift (earlier splits, same panel)"] = sh
            variants["line (earlier splits, same panel)"] = ln
        for name, est in variants.items():
            out.append(dict(panel=p, correction=name, n=len(g), **summarize(est, g.real)))
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(RES, "calibration_table.csv"), index=False)
    stats = {}
    for name, g in T[T.panel.isin(FULL)].groupby("correction"):
        stats[name] = dict(mean_mae=float(g.mae.mean()), mean_abs_bias=float(g.bias.abs().mean()),
                           mean_agree=float(g.agree.mean()), per_panel=g.set_index("panel")[
                               ["mae", "bias", "agree"]].to_dict("index"))
    stats["solar"] = T[T.panel == "Solar"].set_index("correction")[
        ["mae", "bias", "pearson", "agree"]].to_dict("index")
    json.dump(stats, open(os.path.join(RES, "calibration_stats.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
