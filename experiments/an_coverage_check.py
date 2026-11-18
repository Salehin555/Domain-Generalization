"""A formal seasonal-coverage check for the observed-history
estimate.

Aim: formalize a simple seasonality/coverage diagnostic that says
when observed-history estimates can be trusted.

Coverage is computed from dates alone, with no labels and no model:

    coverage = share of deployment-era days whose calendar month occurs at least
               once in the observed history available to the estimate

The observed history is every day before the boundary for the main analysis
(temporal_A), and the last round(k * 30.44) days before the boundary for the
truncation analysis, exactly as seasonal.py truncates. The deployment era is
every day from the boundary to the end of the panel.

The table joins coverage with the agreement statistics already computed by
an_temporal.py, so the relation between coverage and reliability can be read
off directly.

Output: coverage_table.csv, coverage_stats.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PANELS, RES  # noqa: E402


def months_of(dates):
    return (np.asarray(dates) // 100) % 100


def coverage(dates, bidx, obs_days=None):
    d = np.asarray(dates)
    obs = d[:bidx] if obs_days is None else d[max(0, bidx - obs_days):bidx]
    dep = d[bidx:]
    have = set(months_of(obs).tolist())
    return float(np.mean([m in have for m in months_of(dep)])) if len(dep) else np.nan


def main():
    t = json.load(open(os.path.join(RES, "temporal_stats.json")))
    T = pd.DataFrame(t["temporal"])
    S = pd.DataFrame(t["seasonal"])
    rows = []
    for p, spec in PANELS.items():
        dates = np.load(spec["path"], allow_pickle=True)["dates"]
        bidx = int(np.searchsorted(dates, spec["boundary"]))
        g = T[(T.panel == p) & (T.variant == "standard") & (T.metric == "mae_cat")]
        if len(g):
            c = g.iloc[0]
            rows.append(dict(panel=p, setting="full history", months=int(c.span_months),
                             coverage=coverage(dates, bidx), pearson=float(c.pearson),
                             agree=float(c.agree), bias=float(c.bias)))
        for mo, gg in S[(S.panel == p) & (S.metric == "mae_cat")].groupby("months"):
            c = gg.iloc[0]
            rows.append(dict(panel=p, setting="truncated", months=int(mo),
                             coverage=coverage(dates, bidx, int(round(mo * 30.44))),
                             pearson=float(c.pearson), agree=float(c.agree),
                             bias=float(c.bias)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "coverage_table.csv"), index=False)
    full = D[D.coverage >= 0.999]
    part = D[D.coverage < 0.999]
    stats = dict(
        n_full=int(len(full)), n_partial=int(len(part)),
        full_min_r=float(full.pearson.min()), full_median_r=float(full.pearson.median()),
        partial_max_r=float(part.pearson.max()), partial_median_r=float(part.pearson.median()),
        full_max_bias=float(full.bias.max()), partial_median_bias=float(part.bias.median()),
        spearman_coverage_r=float(pd.Series(D.coverage).corr(D.pearson, method="spearman")),
    )
    json.dump(stats, open(os.path.join(RES, "coverage_stats.json"), "w"), indent=1)
    pd.set_option("display.width", 200)
    print(D.round(3).to_string())
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
