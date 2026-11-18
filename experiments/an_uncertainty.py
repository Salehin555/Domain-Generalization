"""Analysis: uncertainty of the survey panel means at separate levels.

For each panel's mean headroom over its eight behavioral splits (primary metric):
  entity       entity-bootstrap 95% interval (models fixed; survey_stats.json)
  allocation   t interval over the three allocation seeds (panel mean per seed)
  grouping     t interval over the four groupings (two directions averaged), i.e.
               uncertainty from which partitions of the panel were studied
  temporal     range of the panel mean of realized deployment headroom over the
               decision dates of origins (size-matched, primary metric)
Across panels: mean and SD of the four panel means.
Output: results/uncertainty_levels.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def tint(x):
    x = np.asarray(x, float)
    m, s, n = x.mean(), x.std(ddof=1), len(x)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return m, m - h, m + h


def main():
    sv = pd.read_csv(os.path.join(RES, "survey_splits.csv"))
    sv = sv[~sv.split.str.contains("random")]
    sv["grouping"] = sv.split.str.split(":").str[0]
    st = json.load(open(os.path.join(RES, "survey_stats.json")))
    org = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in
                     ("origin_rows_Electricity_Traffic_Nursing.csv", "origin_rows_Solar.csv")])
    rows = []
    for p, g in sv.groupby("panel"):
        mean = g.groupby("split").headroom_mae_cat.mean().mean()
        e_lo, e_hi = st[f"{p}_mae_cat"]["mean_ci"]
        seed_means = g.groupby("seed").headroom_mae_cat.mean()
        _, a_lo, a_hi = tint(seed_means)
        grp = g.groupby("grouping").headroom_mae_cat.mean()
        _, g_lo, g_hi = tint(grp)
        o = org[org.panel == p].groupby("origin").headroom_dep_mae_cat.mean()
        rows.append(dict(panel=p, mean=mean, entity_lo=e_lo, entity_hi=e_hi, alloc_lo=a_lo,
                         alloc_hi=a_hi, grouping_lo=g_lo, grouping_hi=g_hi,
                         temporal_min=o.min(), temporal_max=o.max(), n_dates=len(o)))
    df = pd.DataFrame(rows)
    pm = df["mean"].values
    df = pd.concat([df, pd.DataFrame([dict(panel="Across panels", mean=pm.mean(),
                                           panel_sd=pm.std(ddof=1))])])
    df.to_csv(os.path.join(RES, "uncertainty_levels.csv"), index=False)
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
