"""Analysis: tuned domain-generalization baselines (dg_tuned.py).

Per anchor, method and domain definition: error reduction relative to the ERM run of
the same seed, (a) for the configuration chosen on source validation and (b) for the
best configuration on the target test set (hindsight). Mean and SD over five seeds.
Output: results/dg_summary.csv
"""
import glob
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "dg_tuned_rows_*.csv"))])
    out = []
    for (p, sp, seed), g in d.groupby(["panel", "split", "seed"]):
        erm = g[g.method == "ERM"]
        for (m, dom), gm in g[g.method != "ERM"].groupby(["method", "domains"]):
            pick = gm.loc[gm.val_mae.idxmin()]
            for mt in ("mae_cat", "mae_total"):
                e0 = erm[mt].iloc[0]
                out.append(dict(panel=p, split=sp, seed=seed, method=m, domains=dom, metric=mt,
                                selected=100 * (e0 - pick[mt]) / e0,
                                hindsight=100 * (e0 - gm[mt].min()) / e0, weight=pick.weight))
    o = pd.DataFrame(out)
    s = (o.groupby(["metric", "panel", "split", "method", "domains"])
         .agg(selected=("selected", "mean"), selected_sd=("selected", "std"),
              hindsight=("hindsight", "mean")).reset_index())
    s.to_csv(os.path.join(RES, "dg_summary.csv"), index=False)
    print(s[s.metric == "mae_cat"].round(2).to_string())
    print("max selected", s[s.metric == "mae_cat"].selected.max(), "max hindsight",
          s[s.metric == "mae_cat"].hindsight.max())


if __name__ == "__main__":
    main()
