"""How many target entities are needed?

Aim: (i) a learning curve in the NUMBER of target entities, not
only in history length; (ii) a simple decision aid for keeping the pooled model
versus retraining on K target entities, where the size-matched and deployed
comparisons differ.

For each anchor split and allocation seed (the anchor.py manifest) the
target-trained reference is refitted on the first K of its training entities
(t_tr is already shuffled by the allocation), with the same early-stopping set
t_va, for a ladder of K. It is compared with two source-trained models, both
evaluated on the same target test entities:

    matched    a source model trained on the first K source training entities,
               so both sides have K entities (the headroom definition at size K)
    deployed   the source model a practitioner already has: every source entity
               outside the early-stopping set, independent of K

headroom_deployed(K) answers the operational question directly: does training on
K target entities beat the model already in service?

OPTIMIZATION BUDGET. Training draws batches of 32 entities x 8 windows for a fixed
number of epochs, so a model with K <= 32 entities receives one gradient step per
epoch while the deployed model, with every source entity, receives several. A
learning curve run that way confounds the number of entities with the number of
optimizer steps. With EXP_STEPMATCH=1 (the reported analysis) every model receives
the SAME budget as the deployed model: its entity list is tiled up to the deployed
size before batching, so batch composition, epochs, patience and validation
frequency are identical and only the number of distinct entities varies. The
default run keeps the article's standard epoch budget and is kept as a
sensitivity analysis.

Output: tentity_rows_<panels>[_stepmatched].csv
Usage:  [EXP_STEPMATCH=1] python tentities.py [panel ...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, RES, allocate, fit, per_entity, predict  # noqa: E402
from anchor import anchors  # noqa: E402

ERM = dict(method="erm", epochs=24, patience=5, backbone="mlp")
LADDER = {"Electricity": [5, 10, 20, 30], "Traffic": [5, 10, 20, 40, 70],
          "Solar": [5, 10, 15], "Nursing": [25, 100, 400]}
# nursing fits train on about 9,000 deployed facilities, so the nursing curve is run
# with one allocation seed (EXP_SEEDS=0) to keep it within the CPU budget
SEEDS = [int(s) for s in os.environ["EXP_SEEDS"].split(",")] if os.environ.get("EXP_SEEDS") else None
KEEP = {("Electricity", "Flat -> WorkWeek"), ("Electricity", "WorkWeek -> Flat"),
        ("Traffic", "Flat -> WorkWeek"), ("Solar", "Flat -> WorkWeek"),
        ("Nursing", "Ownership (For-profit -> Non-profit)")}


STEPMATCH = os.environ.get("EXP_STEPMATCH", "0") == "1"


def tiled(ents, n_budget):
    """Repeat an entity list up to n_budget entries (no-op when already that size)."""
    ents = np.asarray(ents)
    if not STEPMATCH or len(ents) >= n_budget:
        return ents
    reps = int(np.ceil(n_budget / len(ents)))
    return np.tile(ents, reps)[:n_budget]


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar", "Nursing"]
    tag = "_".join(want) + ("_stepmatched" if STEPMATCH else "")
    out = os.path.join(RES, f"tentity_rows_{tag}.csv")
    rows = []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if (panel, split) not in KEEP:
            continue
        for seed in (SEEDS if SEEDS is not None else seeds):
            t0 = time.time()
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            n = len(al["t_tr"])
            ladder = sorted({k for k in LADDER[panel] if k < n} | {n})
            used = set(al["s_va"])
            s_all = np.array([e for e in S if e not in used])
            dep = fit(P, ERM, s_all, al["s_va"], seed)
            _, Yt, Pd, fid = predict(dep, P, al["t_te"])
            ed = per_entity(Pd, Yt, fid)
            nb = len(s_all)
            for K in ladder:
                ref = fit(P, ERM, tiled(al["t_tr"][:K], nb), al["t_va"], seed)
                src = fit(P, ERM, tiled(al["s_tr"][:K], nb), al["s_va"], seed)
                _, _, Pr, _ = predict(ref, P, al["t_te"])
                _, _, Ps, _ = predict(src, P, al["t_te"])
                er, es = per_entity(Pr, Yt, fid), per_entity(Ps, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, K=K, K_max=n,
                         n_deployed=len(s_all), n_test=len(al["t_te"]),
                         budget="deployed" if STEPMATCH else "standard epochs")
                for m in METRICS:
                    rf, sm, sd = float(er[m].mean()), float(es[m].mean()), float(ed[m].mean())
                    r[f"reference_{m}"] = rf
                    r[f"matched_{m}"] = sm
                    r[f"deployed_{m}"] = sd
                    r[f"headroom_matched_{m}"] = 100.0 * (sm - rf) / sm
                    r[f"headroom_deployed_{m}"] = 100.0 * (sd - rf) / sd
                rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  {panel} {split} seed {seed}: K={ladder} ({time.time() - t0:.0f}s)",
                  flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["panel", "split", "K"])[["headroom_matched_mae_cat",
                                               "headroom_deployed_mae_cat"]].mean()
          .round(2).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
