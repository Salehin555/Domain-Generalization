"""Facility-level statistics for the full-scale nursing benchmark.

Procedure (seeds averaged to
one value per facility; Wilcoxon signed-rank; matched-pairs rank-biserial;
Holm across baselines within a protocol; bootstrap CI of the mean paired
difference), for BOTH endpoints and with the reference model ERM-MLP, so
that every method is compared with the pooled baseline rather than with FADE.

Outputs: fullscale_means.csv, fullscale_wilcoxon.csv, fullscale_stats.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MET = ["mae_cat", "mae_total"]
REF = "ERM-MLP"
NB = 5000


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(run, 1.0)
    return adj


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    df = pd.concat([pd.read_csv(f) for f in
                    glob.glob(os.path.join(RES, "fullscale_*_*.csv")) if "means" not in f and "wilcoxon" not in f], ignore_index=True)
    rng = np.random.default_rng(0)
    means = (df.groupby(["protocol", "method", "seed"])[MET].mean()
               .groupby(["protocol", "method"]).agg(["mean", "std", "count"]))
    means.columns = ["_".join(c) for c in means.columns]
    means = means.reset_index()
    means.to_csv(os.path.join(RES, "fullscale_means.csv"), index=False)
    rows = []
    for mt in MET:
        fac = df.groupby(["protocol", "method", "fac"], as_index=False)[mt].mean()
        for proto, sub in fac.groupby("protocol"):
            w = sub.pivot(index="fac", columns="method", values=mt)
            if REF not in w:
                continue
            recs = []
            for b in [m for m in w.columns if m != REF]:
                pr = w[[REF, b]].dropna()
                d = (pr[REF] - pr[b]).to_numpy()          # > 0: method b beats ERM
                try:
                    p = wilcoxon(d).pvalue
                except ValueError:
                    p = 1.0
                bi = rng.integers(0, len(d), size=(NB, len(d)))
                bm = d[bi].mean(1)
                recs.append(dict(metric=mt, protocol=proto, method=b,
                                 n_facilities=len(d),
                                 pct_vs_erm=float(100 * d.mean() / pr[REF].mean()),
                                 win_pct=float((d > 0).mean() * 100),
                                 mean_diff=float(d.mean()),
                                 ci_lo=float(np.percentile(bm, 2.5)),
                                 ci_hi=float(np.percentile(bm, 97.5)),
                                 effect_rb=rank_biserial(d), p=float(p)))
            for r, a in zip(recs, holm([r["p"] for r in recs])):
                r["p_holm"] = float(a)
            rows += recs
    W = pd.DataFrame(rows)
    W.to_csv(os.path.join(RES, "fullscale_wilcoxon.csv"), index=False)
    stats = dict(
        n_facilities_test={p: int(df[df.protocol == p].fac.nunique())
                           for p in df.protocol.unique()},
        seeds={m: int(df[df.method == m].seed.nunique()) for m in df.method.unique()},
        max_abs_pct_vs_erm={f"{mt}|{p}": float(W[(W.metric == mt) & (W.protocol == p)]
                                               .pct_vs_erm.abs().max())
                            for mt in MET for p in W.protocol.unique()},
    )
    with open(os.path.join(RES, "fullscale_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(W[W.metric == "mae_cat"].round(4).to_string())


if __name__ == "__main__":
    main()
