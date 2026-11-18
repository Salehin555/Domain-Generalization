"""Domain-generalization baselines on every benchmark anchor.

Motivation: the domain-generalization baselines were evaluated on
electricity only. This runs IRM, CORAL, DANN and GroupDRO on the traffic and
solar anchors with the settings used for electricity (fixed hyperparameters,
source-validation early stopping, manifest A), scored exactly as anchor.py
scores every other method, so an_anchor.py picks the rows up unchanged.

Output: anchor_rows_dgext.csv, anchor_entities_dgext.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, RES, allocate, fit, per_entity, predict  # noqa: E402
from anchor import METHODS, anchors  # noqa: E402

DG = {"IRM", "CORAL", "DANN", "GroupDRO"}


def main():
    want = sys.argv[1:] or ["Traffic", "Solar"]
    rows, ents = [], []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if panel not in ("Traffic", "Solar"):
            continue
        for seed in seeds:
            t0 = time.time()
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            for label, cfg in METHODS:
                if label not in DG:
                    continue
                m = fit(P, cfg, al["s_tr"], al["s_va"], seed)
                _, Yt, Pt, fid = predict(m, P, al["t_te"])
                e = per_entity(Pt, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, manifest="A", method=label,
                         n_train=len(al["s_tr"]), n_test=len(al["t_te"]))
                for mt in METRICS:
                    r[mt] = e[mt].mean()
                rows.append(r)
                ents.append(e.reset_index().assign(panel=panel, split=split, seed=seed,
                                                   manifest="A", method=label))
            pd.DataFrame(rows).to_csv(os.path.join(RES, "anchor_rows_dgext.csv"), index=False)
            pd.concat(ents).to_csv(os.path.join(RES, "anchor_entities_dgext.csv"), index=False)
            print(f"  {panel} {split} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    print(pd.DataFrame(rows).groupby(["panel", "method"]).mae_cat.mean().round(4).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
