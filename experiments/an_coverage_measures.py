"""Analysis: alternative calendar-coverage measures.

Settings: the four full-history panels and the truncated histories
(coverage_table.csv), and every decision date of origins (summary_origins.csv).
For each setting the observed history and the deployment period are rebuilt from
dates, and three coverage measures are computed from dates alone:

  month     share of deployment days whose calendar month occurs at least once in
            the history (the measure of the article)
  month14   the same, requiring at least 14 observed days of that month
  lag365    share of deployment days whose date one year earlier lies in the
            history (the same season was observed a year before)
plus history length in years. Each is related to the agreement between estimate
and realized headroom (Pearson r over split x seed rows, and absolute bias).
Output: results/coverage_measures.csv, coverage_assoc.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SPAN = {"Electricity": ("2012-01-01", "2014-12-31", "2014-01-01"),
        "Traffic": ("2015-01-01", "2016-12-31", "2016-05-04"),
        "Solar": ("2006-01-01", "2006-12-31", "2006-09-02"),
        "Nursing": ("2024-01-01", "2025-12-31", "2025-01-01")}
ORIG = {"Electricity": [20130101, 20130701, 20140101, 20140701],
        "Traffic": [20160101, 20160504, 20160901],
        "Nursing": [20250101, 20250701], "Solar": [20060601, 20060902]}


def measures(h0, h1, d0, d1):
    hist = pd.date_range(h0, h1 - pd.Timedelta(days=1))
    dep = pd.date_range(d0, d1)
    cnt = pd.Series(1, index=hist).groupby(hist.month).sum()
    month = np.isin(dep.month, cnt.index).mean()
    month14 = np.isin(dep.month, cnt[cnt >= 14].index).mean()
    lag = (dep - pd.Timedelta(days=365))
    lag365 = ((lag >= hist[0]) & (lag <= hist[-1])).mean()
    return dict(month=month, month14=month14, lag365=lag365, years=len(hist) / 365.0)


def main():
    rows = []
    ct = pd.read_csv(os.path.join(RES, "coverage_table.csv"))
    for _, r in ct.iterrows():
        s, e, b = (pd.Timestamp(x) for x in SPAN[r.panel])
        h0 = s if r.setting == "full history" else b - pd.Timedelta(days=int(round(r.months * 30.44)))
        m = measures(max(h0, s), b, b, e)
        rows.append(dict(panel=r.panel, setting=f"{r.setting} {int(r.months)} mo", r=r.pearson,
                         abs_bias=abs(r.bias), source="single date", **m))
    so = pd.read_csv(os.path.join(RES, "summary_origins.csv"))
    so = so[so.origin != "pooled"]
    for _, r in so.iterrows():
        s, e, _ = (pd.Timestamp(x) for x in SPAN[r.panel])
        o = int(r.origin); lst = ORIG[r.panel]; j = lst.index(o)
        o_ts = pd.Timestamp(str(o))
        end = pd.Timestamp(str(lst[j + 1])) - pd.Timedelta(days=1) if j + 1 < len(lst) else e
        m = measures(s, o_ts, o_ts, end)
        rows.append(dict(panel=r.panel, setting=f"origin {o}", r=r.r, abs_bias=abs(r.bias),
                         source="several dates", **m))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "coverage_measures.csv"), index=False)
    print(df.round(2).to_string())
    out = []
    for k in ("month", "month14", "lag365", "years"):
        rr = stats.spearmanr(df[k], df.r)
        rb = stats.spearmanr(df[k], df.abs_bias)
        full = df[df[k] >= 0.999]
        part = df[df[k] < 0.999]
        out.append(dict(measure=k, spearman_r=rr.statistic, p_r=rr.pvalue,
                        spearman_absbias=rb.statistic, n_full=len(full), n_partial=len(part),
                        min_r_full=full.r.min() if len(full) else np.nan,
                        median_r_full=full.r.median() if len(full) else np.nan,
                        median_r_partial=part.r.median() if len(part) else np.nan,
                        max_absbias_full=full.abs_bias.max() if len(full) else np.nan))
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(RES, "coverage_assoc.csv"), index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
