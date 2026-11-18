"""Summaries of the prospective temporal experiments.

Table 4: per panel (and, for nursing, per sensitivity variant) the agreement
between the observed-era headroom estimate and realised deployment headroom:
Pearson r, Spearman rho, mean bias (estimate - realised, percentage points) and
agreement at the prespecified 3% analysis threshold. Units are split x seed
rows, as before; the number of distinct splits is reported alongside, because
the two seeds of a split share entities and are not independent.

Also: seasonal-truncation summary (Supplement F) and the target-GROUP history
curve (days after which the estimate stays within 0.5 pp of its final value).

Outputs: temporal_table.csv, seasonal_table.csv, history_table.csv,
         temporal_stats.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MET = ["mae_cat", "mae_total"]
THR = 3.0
SPAN = {"Nursing": 12, "Electricity": 24, "Traffic": 16, "Solar": 8}


def summarise(g, mt):
    e, d = g[f"headroom_obs_{mt}"].to_numpy(), g[f"headroom_dep_{mt}"].to_numpy()
    ok = np.isfinite(e) & np.isfinite(d)
    e, d = e[ok], d[ok]
    return dict(n=int(len(e)), n_splits=int(g.split.nunique()),
                pearson=float(ss.pearsonr(e, d)[0]) if len(e) > 2 else np.nan,
                spearman=float(ss.spearmanr(e, d)[0]) if len(e) > 2 else np.nan,
                bias=float(np.mean(e - d)),
                agree=float(np.mean((e >= THR) == (d >= THR))),
                realised_mean=float(d.mean()))


def main():
    out = {}
    A = pd.read_csv(os.path.join(RES, "temporal_A.csv"))
    rows = []
    for (p, v), g in A.groupby(["panel", "variant"]):
        for mt in MET:
            rows.append(dict(panel=p, variant=v, metric=mt, span_months=SPAN[p],
                             **summarise(g, mt)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "temporal_table.csv"), index=False)
    out["temporal"] = T.to_dict("records")
    aud = os.path.join(RES, "temporal_A_window_audit.csv")
    if os.path.exists(aud):
        out["window_audit"] = pd.read_csv(aud).to_dict("records")

    sp = os.path.join(RES, "seasonal_truncation.csv")
    if os.path.exists(sp):
        S = pd.read_csv(sp)
        srows = [dict(panel=p, months=m, metric=mt, **summarise(g, mt))
                 for (p, m), g in S.groupby(["panel", "months"]) for mt in MET]
        ST = pd.DataFrame(srows)
        ST.to_csv(os.path.join(RES, "seasonal_table.csv"), index=False)
        out["seasonal"] = ST.to_dict("records")

    bp = os.path.join(RES, "temporal_B.csv")
    if os.path.exists(bp):
        B = pd.read_csv(bp)
        hrows = []
        for p, g in B.groupby("panel"):
            for mt in MET:
                c = g.groupby("days")[f"headroom_{mt}"].mean().sort_index()
                final = c.iloc[-1]
                within = c[(c - final).abs() <= 0.5]
                # first ladder point from which the curve STAYS within 0.5 pp
                settle = None
                for d in c.index:
                    if ((c.loc[d:] - final).abs() <= 0.5).all():
                        settle = int(d); break
                hrows.append(dict(panel=p, metric=mt, final=float(final),
                                  settle_day=settle,
                                  settle_months=round((settle or np.nan) / 30.44, 1),
                                  ladder=list(map(int, c.index)),
                                  curve=list(map(float, c.values)),
                                  n_splits=int(g.split.nunique())))
        HT = pd.DataFrame(hrows)
        HT.to_csv(os.path.join(RES, "history_table.csv"), index=False)
        out["history"] = HT.to_dict("records")
    with open(os.path.join(RES, "temporal_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
