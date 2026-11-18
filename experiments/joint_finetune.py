"""Joint source+target training and fine-tuning.

Headroom compares a source-only model with a target-only reference. A practitioner
with target data could instead train on both groups, or fine-tune the model already
in service. This script adds those arms on the anchor manifests (anchor.py
allocation, same seeds) and scores every arm on the same target test entities.

Arms (all ERM-Stem, 24 epochs, patience 5):
    src        size-matched source model: s_tr, early stopping on s_va
    ref        target-trained reference: t_tr, early stopping on t_va
    dep        deployed model: every source entity outside s_va, early stopping s_va
    joint      s_tr + t_tr, early stopping on t_va
    joint_dep  every source entity outside s_va + t_tr, early stopping on t_va
    ft_src     src warm-started, then trained on t_tr (lr 1e-4), early stopping t_va
    ft_dep     dep warm-started, then trained on t_tr (lr 1e-4), early stopping t_va

Optimization budget. As in the equal-budget learning curve (tentities.py,
EXP_STEPMATCH=1), every entity list is tiled up to the deployed size before
batching, so src, ref, dep and joint receive the same number of gradient steps per
epoch; joint_dep, whose list is larger, receives slightly more. The fine-tuning arms
add one further budget of that size on top of their starting model, and the
starting model is itself a candidate for early stopping, so fine-tuning can return
it unchanged. The extra compute of fine-tuning is therefore reported, not hidden.

Early stopping of the arms that use target training data is on the target
validation set t_va, which the target-trained reference also uses; the source-only
arms stop on s_va. No arm sees t_te during training or selection.

Output: results/transfer_rows.csv (one row per panel, split, seed, arm) and
        results/transfer_entities.csv (per-entity errors on t_te)
Usage:  python joint_finetune.py [panel ...]; EXP_SEEDS=0,1 to restrict seeds
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
FT = dict(ERM, lr=1e-4)
KEEP = {("Electricity", "Flat -> WorkWeek"), ("Electricity", "WorkWeek -> Flat"),
        ("Traffic", "Flat -> WorkWeek"), ("Solar", "Flat -> WorkWeek"),
        ("Nursing", "Ownership (For-profit -> Non-profit)")}
SEEDS = ([int(s) for s in os.environ["EXP_SEEDS"].split(",")]
         if os.environ.get("EXP_SEEDS") else None)


def tiled(ents, n_budget):
    """Repeat an entity list up to n_budget entries (no-op when already that size)."""
    ents = np.asarray(ents)
    if len(ents) >= n_budget:
        return ents
    return np.tile(ents, int(np.ceil(n_budget / len(ents))))[:n_budget]


def state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar", "Nursing"]
    tag = "_".join(want)
    rows, ents = [], []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if (panel, split) not in KEEP:
            continue
        for seed in (SEEDS if SEEDS is not None else seeds):
            t0 = time.time()
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            used = set(al["s_va"])
            s_all = np.array([e for e in S if e not in used])
            nb = len(s_all)
            m = {}
            m["src"] = fit(P, ERM, tiled(al["s_tr"], nb), al["s_va"], seed)
            m["ref"] = fit(P, ERM, tiled(al["t_tr"], nb), al["t_va"], seed)
            m["dep"] = fit(P, ERM, s_all, al["s_va"], seed)
            m["joint"] = fit(P, ERM, tiled(np.concatenate([al["s_tr"], al["t_tr"]]), nb),
                             al["t_va"], seed)
            m["joint_dep"] = fit(P, ERM, np.concatenate([s_all, al["t_tr"]]), al["t_va"], seed)
            m["ft_src"] = fit(P, dict(FT, init_state=state(m["src"])),
                              tiled(al["t_tr"], nb), al["t_va"], seed)
            m["ft_dep"] = fit(P, dict(FT, init_state=state(m["dep"])),
                              tiled(al["t_tr"], nb), al["t_va"], seed)
            Yt = fid = None
            for arm, model in m.items():
                _, Yt, Pr, fid = predict(model, P, al["t_te"])
                e = per_entity(Pr, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, arm=arm,
                         n_s_tr=len(al["s_tr"]), n_t_tr=len(al["t_tr"]),
                         n_t_va=len(al["t_va"]), n_t_te=len(al["t_te"]), n_dep=nb)
                for mt in METRICS:
                    r[mt] = float(e[mt].mean())
                rows.append(r)
                ee = e.reset_index()
                ee["panel"], ee["split"], ee["seed"], ee["arm"] = panel, split, seed, arm
                ents.append(ee)
            D = pd.DataFrame([x for x in rows if x["panel"] == panel and
                              x["split"] == split and x["seed"] == seed]).set_index("arm")
            base = D.loc["dep", "mae_cat"]
            print(f"  {panel} {split} seed {seed} ({time.time() - t0:.0f}s): " +
                  "  ".join(f"{a} {100 * (base - D.loc[a, 'mae_cat']) / base:+.2f}"
                            for a in D.index if a != "dep"), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"transfer_rows_{tag}.csv"),
                                      index=False)
            pd.concat(ents).to_csv(os.path.join(RES, f"transfer_entities_{tag}.csv"),
                                   index=False)
    print("done")


if __name__ == "__main__":
    main()
