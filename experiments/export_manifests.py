"""Export the entity manifests used by the experiments (no training).

Every allocation in the study is deterministic given the split and the seed. This
script re-creates the allocations with the same calls the experiments use and
writes them out, one row per (analysis, panel, split, seed, entity), with the
entity's role and its identifier in the source data.

    anchors    anchor.anchors() + common.allocate()   (anchor splits,
               learning curve, model families, classical references, joint
               training; roles s_te, s_va, s_tr, t_te, t_va, t_tr)
    temporal   temporal.make_splits / split_target / match_source (observed-
               history estimate, crossed coverage design, several decision dates;
               roles s_va, s_tr, t_te, t_va, t_tr; the several-dates analysis
               derives its groupings from the history before its earliest date,
               which is exported separately as temporal_earliest)

Output: results/manifests_anchors.csv, results/manifests_temporal.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, allocate, load_panel  # noqa: E402
from anchor import anchors  # noqa: E402
from temporal import NFAC, N_SPLITS, SEEDS, match_source, split_target  # noqa: E402
from origins import ORIGINS, panel_at  # noqa: E402
from src.label_free_calibration import make_splits  # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402


def ids(P, idx):
    return np.asarray(P.prov)[np.asarray(idx)]


def main():
    rows = []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(
            ["Electricity", "Traffic", "Solar", "Nursing"]):
        for seed in seeds:
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            if al is None:
                continue
            for role, ents in al.items():
                for e, pid in zip(ents, ids(P, ents)):
                    rows.append(dict(panel=panel, split=split, seed=seed, role=role,
                                     entity_index=int(e), entity_id=str(pid)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "manifests_anchors.csv"), index=False)
    print("anchors", len(rows))
    rows = []
    for name in ("Nursing", "Electricity", "Traffic", "Solar"):
        for tag, P in (("temporal", load_panel(name, NFAC.get(name, 10 ** 6))),
                       ("temporal_earliest", panel_at(name, ORIGINS[name][0], stride=18))):
            stats, _ = entity_stats(P)
            splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
            for k, (A, B) in enumerate(splits):
                for sd in SEEDS:
                    rng = np.random.default_rng(sd)
                    t_tr, t_va, t_te = split_target(B, rng)
                    if len(t_tr) < 12:
                        continue
                    s_tr, s_va = match_source(A, B, len(t_tr), rng)
                    if len(s_tr) < 12:
                        continue
                    for role, ents in (("t_tr", t_tr), ("t_va", t_va), ("t_te", t_te),
                                       ("s_tr", s_tr), ("s_va", s_va)):
                        for e, pid in zip(ents, ids(P, ents)):
                            rows.append(dict(analysis=tag, panel=name, split=k, seed=sd,
                                             role=role, entity_index=int(e),
                                             entity_id=str(pid)))
            del P
    pd.DataFrame(rows).to_csv(os.path.join(RES, "manifests_temporal.csv"), index=False)
    print("temporal", len(rows))


if __name__ == "__main__":
    main()
