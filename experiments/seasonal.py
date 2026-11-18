"""Observed-history truncation test (Supplement F), both
endpoints. Design: the observed-era estimate is allowed only the last k months before the
boundary; realised deployment headroom is unchanged.

Output: seasonal_truncation.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ERM, METRICS, RES, fit, headroom, load_panel, per_entity, predict  # noqa
from temporal import match_source, split_target                  # noqa: E402
from src.label_free_calibration import make_splits                  # noqa: E402
from src.survey_reducibility import entity_stats                    # noqa: E402

LADDER = {"Electricity": [4, 6, 8, 12, 18, 24], "Traffic": [4, 6, 8, 12, 16]}
BOUND = {"Electricity": 20140101, "Traffic": 20160504}


def main():
    rows = []
    for name, months in LADDER.items():
        P = load_panel(name)
        bidx = int(np.searchsorted(P.dates, BOUND[name]))
        stats, _ = entity_stats(P)
        splits = make_splits(stats, 10, np.random.default_rng(0))
        for m in months:
            trunc = P.is_2024 & (P.wstart >= bidx - int(round(m * 30.44)))
            for k, (A, B) in enumerate(splits):
                for sd in [0, 1]:
                    rng = np.random.default_rng(sd)
                    t_tr, t_va, t_te = split_target(B, rng)
                    if len(t_tr) < 12:
                        continue
                    s_tr, s_va = match_source(A, B, len(t_tr), rng)
                    if len(s_tr) < 12:
                        continue
                    ms = fit(P, ERM, s_tr, s_va, sd, trunc, trunc)
                    mr = fit(P, ERM, t_tr, t_va, sd, trunc, trunc)
                    r = dict(panel=name, months=m, split=k, seed=sd)
                    for era, msk in (("obs", trunc), ("dep", P.is_2025)):
                        _, Y, Ps, fid = predict(ms, P, t_te, msk)
                        _, _, Pr, _ = predict(mr, P, t_te, msk)
                        es, er = per_entity(Ps, Y, fid), per_entity(Pr, Y, fid)
                        for mt in METRICS:
                            r[f"headroom_{era}_{mt}"] = headroom(es[mt].mean(),
                                                                 er[mt].mean())
                    rows.append(r)
            d = pd.DataFrame([x for x in rows if x["panel"] == name and x["months"] == m])
            print(f"  {name} {m:2d} mo n={len(d)} r={np.corrcoef(d.headroom_obs_mae_cat, d.headroom_dep_mae_cat)[0, 1]:+.2f}",
                  flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "seasonal_truncation.csv"),
                                      index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
