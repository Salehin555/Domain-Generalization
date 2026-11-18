"""Optimization-budget control for the joint-training arms.

The joint arm on the deployed pool (joint_finetune.py, joint_dep) trains on
(deployed + target training) entities, so it receives 1.6-1.7 times the deployed
model's gradient steps per epoch on the benchmark anchors. This control gives the
deployed model (dep_long) and the target-only reference (ref_long) exactly that
longer budget, by tiling their entity lists to the joint_dep length, on the same
manifests and seeds. If the joint arm's advantage came from extra steps, dep_long
and ref_long would gain as much.

Output: results/budget_rows.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, RES, allocate, fit, per_entity, predict  # noqa: E402
from anchor import anchors  # noqa: E402
from joint_finetune import ERM, KEEP, tiled  # noqa: E402


def main():
    rows = []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(["Electricity", "Traffic", "Solar"]):
        if (panel, split) not in KEEP:
            continue
        for seed in seeds:
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            used = set(al["s_va"])
            s_all = np.array([e for e in S if e not in used])
            nlong = len(s_all) + len(al["t_tr"])
            m = dict(dep_long=fit(P, ERM, tiled(s_all, nlong), al["s_va"], seed),
                     ref_long=fit(P, ERM, tiled(al["t_tr"], nlong), al["t_va"], seed))
            for arm, model in m.items():
                _, Yt, Pr, fid = predict(model, P, al["t_te"])
                e = per_entity(Pr, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, arm=arm, n_budget=nlong,
                         n_dep=len(s_all))
                for mt in METRICS:
                    r[mt] = float(e[mt].mean())
                rows.append(r)
            print(f"  {panel} {split} seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "budget_rows.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
