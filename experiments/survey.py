"""36-split headroom survey (main text Section 5.3).

Differences from src/survey_reducibility.py (the split protocol itself is unchanged):
  * both endpoints recorded; mae_cat is primary
  * source error measured on held-out source entities (s_te), not on the
    early-stopping set
  * model initialisation seed = allocation seed (fixed at 0 there)
  * seasonal-naive and source-validation blend evaluated on the same windows
  * per-entity errors and per-entity screen features saved for resampling

This is a RETROSPECTIVE CROSS-ENTITY experiment: all windows of each entity are
used (no temporal masks); groups are defined from observed-era inputs only.

Outputs (experiments/results/):
  survey_splits.csv, survey_entities.csv, survey_source_entities.csv,
  survey_features_<panel>.npz
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, load_panel              # noqa: E402
from diag import core_screens, entity_features, run_split   # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402

SEEDS = [0, 1, 2]
NFAC = {"Nursing": 1500}


def survey_splits(stats, n):
    idx = np.arange(n)
    splits = {}
    for sname, v in stats.items():
        q1, q3 = np.percentile(v, [25, 75])
        A, B = idx[v <= q1], idx[v >= q3]
        if len(A) >= 30 and len(B) >= 30:
            splits[f"{sname}: low->high"] = (A, B)
            splits[f"{sname}: high->low"] = (B, A)
    p = np.random.default_rng(0).permutation(idx)
    h = len(p) // 2
    splits["random control"] = (p[:h], p[h:])
    return splits


def main():
    panels = sys.argv[1:] or ["Nursing", "Electricity", "Traffic", "Solar"]
    rows, ents, srcs = [], [], []
    for name in panels:
        t0 = time.time()
        P = load_panel(name, NFAC.get(name, 10 ** 6))
        stats, _ = entity_stats(P)
        F = entity_features(P)
        splits = survey_splits(stats, P.nF)
        memb = {}
        print(f"\n### {name}: nF={P.nF}, {len(splits)} splits", flush=True)
        for sname, (A, B) in splits.items():
            memb[sname] = (A, B)
            sc = core_screens(F, A, B)
            for seed in SEEDS:
                out = run_split(P, A, B, seed)
                if out is None:
                    continue
                row, ent, src = out
                rows.append(dict(panel=name, split=sname, seed=seed, n_A=len(A),
                                 n_B=len(B), **sc, **row))
                ents.append(ent.assign(panel=name, split=sname, seed=seed))
                srcs.append(src.assign(panel=name, split=sname, seed=seed))
            d = [r for r in rows if r["panel"] == name and r["split"] == sname]
            print(f"  {sname:20s} headroom cat={np.mean([r['headroom_mae_cat'] for r in d]):+6.2f}%"
                  f"  tot={np.mean([r['headroom_mae_total'] for r in d]):+6.2f}%"
                  f"  apparent cat={np.mean([r['apparent_gap_mae_cat'] for r in d]):+6.1f}%",
                  flush=True)
        np.savez_compressed(
            os.path.join(RES, f"survey_features_{name}.npz"), **F,
            split_names=np.array(list(memb)),
            **{f"A_{i}": v[0] for i, v in enumerate(memb.values())},
            **{f"B_{i}": v[1] for i, v in enumerate(memb.values())})
        print(f"  [{name} done in {time.time() - t0:.0f}s]", flush=True)
        # write incrementally so a crash never loses finished panels
        tag = "" if len(sys.argv) == 1 else "_" + "_".join(panels)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"survey_splits{tag}.csv"), index=False)
        pd.concat(ents).to_csv(os.path.join(RES, f"survey_entities{tag}.csv"), index=False)
        pd.concat(srcs).to_csv(os.path.join(RES, f"survey_source_entities{tag}.csv"),
                               index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
