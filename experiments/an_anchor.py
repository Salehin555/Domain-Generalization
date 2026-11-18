"""Anchor tables from the common manifest (anchor.py).

For each anchor split, manifest and method: error on the target test entities
(mean over allocation seeds, SD over seeds) for both endpoints, and the error
reduction relative to the source-trained ERM-MLP on the SAME entities and
windows, with a paired entity-level bootstrap CI (entities resampled within each
seed, reductions averaged over seeds). Target-retraining headroom is the same
calculation with the target-trained reference in place of a method, so every
number in the anchor table shares one denominator population.

Outputs: anchor_table.csv, anchor_stats.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MET = ["mae_cat", "mae_total"]
REF = "Target-trained reference (ERM-MLP)"
BASE = "ERM-MLP"
NB = 2000


def load(kind):
    fs = sorted(glob.glob(os.path.join(RES, f"anchor_{kind}_*.csv")))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def main():
    R, E = load("rows"), load("entities")
    rng = np.random.default_rng(0)
    rows = []
    for (panel, split, man), g in R.groupby(["panel", "split", "manifest"]):
        eg = E[(E.panel == panel) & (E.split == split)]
        seeds = sorted(g.seed.unique())
        # entity-level resampling indices per seed, shared by every method
        idx = {}
        for s in seeds:
            facs = eg[(eg.seed == s) & (eg.method == BASE) &
                      (eg.manifest == man)].fac.to_numpy()
            idx[s] = (facs, rng.integers(0, len(facs), size=(NB, len(facs))))
        methods = [m for m in g.method.unique() if m != "source error (s_te)"]
        for meth in methods:
            man_m = "A" if meth == REF else man
            r = dict(panel=panel, split=split, manifest=man, method=meth,
                     n_train=int(g[g.method == meth].n_train.iloc[0]),
                     n_test=int(g[g.method == meth].n_test.iloc[0]),
                     n_seeds=len(seeds))
            for mt in MET:
                vals = g[g.method == meth].set_index("seed")[mt]
                base = g[g.method == BASE].set_index("seed")[mt]
                red = 100 * (base - vals) / base
                r[mt] = float(vals.mean()); r[f"{mt}_sd"] = float(vals.std(ddof=1))
                r[f"reduction_{mt}"] = float(red.mean())
                r[f"reduction_{mt}_sd"] = float(red.std(ddof=1))
                boots = np.zeros(NB)
                for s in seeds:
                    facs, ix = idx[s]
                    em = eg[(eg.seed == s) & (eg.method == meth) &
                            (eg.manifest == man_m)].set_index("fac")[mt].reindex(facs).to_numpy()
                    eb = eg[(eg.seed == s) & (eg.method == BASE) &
                            (eg.manifest == man)].set_index("fac")[mt].reindex(facs).to_numpy()
                    bm, bb = em[ix].mean(1), eb[ix].mean(1)
                    boots += 100 * (bb - bm) / bb
                boots /= len(seeds)
                r[f"reduction_{mt}_ci"] = [float(np.percentile(boots, 2.5)),
                                           float(np.percentile(boots, 97.5))]
                if meth == BASE:
                    so = g[g.method == "source error (s_te)"].set_index("seed")[mt]
                    app = 100 * (vals - so) / so
                    r[f"source_{mt}"] = float(so.mean())
                    r[f"apparent_gap_{mt}"] = float(app.mean())
                    r[f"apparent_gap_{mt}_sd"] = float(app.std(ddof=1))
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "anchor_table.csv"), index=False)
    stats = {}
    for (panel, split, man), g in T.groupby(["panel", "split", "manifest"]):
        key = f"{panel}|{split}|{man}"
        stats[key] = {m: {k: v for k, v in row.items() if k not in
                          ("panel", "split", "manifest", "method")}
                      for m, row in g.set_index("method").iterrows()}
        # headroom per seed, minimum over seeds
        rr = R[(R.panel == panel) & (R.split == split)]
        h = {}
        for mt in MET:
            tr = rr[(rr.method == BASE) & (rr.manifest == "A")].set_index("seed")[mt]
            rf = rr[rr.method == REF].set_index("seed")[mt]
            hs = 100 * (tr - rf) / tr
            h[mt] = dict(mean=float(hs.mean()), sd=float(hs.std(ddof=1)),
                         min=float(hs.min()), max=float(hs.max()),
                         n_positive=int((hs > 0).sum()), n=int(len(hs)))
        stats[key]["headroom_per_seed"] = h
    with open(os.path.join(RES, "anchor_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1, default=float)
    show = T[T.manifest == "A"][["panel", "split", "method", "mae_cat",
                                 "reduction_mae_cat", "reduction_mae_cat_ci",
                                 "reduction_mae_total"]]
    pd.set_option("display.width", 220)
    print(show.to_string())


if __name__ == "__main__":
    main()
