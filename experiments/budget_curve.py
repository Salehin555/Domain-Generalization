"""Sensitivity of headroom and of the joint arm to the optimization budget.

The budget control (budget_control.py) showed that training longer lowers the
error of the deployed model and of the target-only reference, so the anchor models
are not trained to convergence at the standard budget. This script measures, on the
benchmark anchor manifests and seeds, how the comparisons move with the budget.

Budget multiplier m in {1, 2, 4}: every entity list is tiled to m x (deployed size)
entries per epoch (24 epochs, patience 5), so each arm receives the same number of
gradient steps at a given m:
    src        size-matched source training entities
    ref        target training entities (target-only reference)
    dep        every source entity outside s_va
    joint_dep  every source entity outside s_va plus the target training entities
               (tiled to the same length; at m = 1 this is shorter than the
               joint_dep list itself, so m starts where all arms are comparable
               only from m = 2; m = 1 is kept for reference)
Output: results/budget_curve_rows.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, RES, allocate, fit, per_entity, predict  # noqa: E402
from anchor import anchors  # noqa: E402
from joint_finetune import ERM, KEEP, tiled  # noqa: E402

MULTS = (1, 2, 4)


def sub(ents, n):
    """Exactly n entries: tile when shorter; a fixed-order truncation is never used,
    so an arm is never deprived of its own entities (longer lists are kept whole)."""
    return tiled(ents, n)


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
            nb = len(s_all)
            pool = np.concatenate([s_all, al["t_tr"]])
            for mlt in MULTS:
                n = mlt * nb
                arms = dict(src=(sub(al["s_tr"], n), al["s_va"]),
                            ref=(sub(al["t_tr"], n), al["t_va"]),
                            dep=(sub(s_all, n), al["s_va"]),
                            joint_dep=(sub(pool, n), al["t_va"]))
                for arm, (tr, va) in arms.items():
                    model = fit(P, ERM, tr, va, seed)
                    _, Yt, Pr, fid = predict(model, P, al["t_te"])
                    e = per_entity(Pr, Yt, fid)
                    r = dict(panel=panel, split=split, seed=seed, mult=mlt, arm=arm,
                             n_list=len(tr), n_dep=nb)
                    for mt in METRICS:
                        r[mt] = float(e[mt].mean())
                    rows.append(r)
            print(f"  {panel} {split} seed {seed} done", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "budget_curve_rows.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
