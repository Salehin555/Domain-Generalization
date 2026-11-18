"""Screen DISCOVERY and CONFIRMATION split collections.

These are separate from the 36-split survey. Each panel gets 26 candidate
random-direction splits (src.label_free_calibration.make_splits): groups are the
extreme q% tails, q drawn from {10,...,35}, of a random linear combination of
the four standardised observed-era descriptors (size, vol, weekly, trend); the
direction is flipped with probability 1/2. Splits with a group of fewer than 25
entities, or too few entities to size-match, are dropped.

  discovery    make_splits(..., rng seed 0)   used to choose the statistic and
                                              fit calibration coefficients
  confirmation make_splits(..., rng seed 1)   fresh splits, evaluated with the
                                              discovery choices frozen

Headroom per split = mean over allocation seeds {0, 1}; both endpoints recorded;
per-entity errors saved. Nursing uses a random 1,200-facility working set (the
full panel would make 104 splits x 2 seeds x 2 models prohibitive).

Outputs: screen_<set>_splits.csv, screen_<set>_entities.csv,
         screen_features_<panel>.npz
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, load_panel                      # noqa: E402
from diag import core_screens, entity_features, run_split   # noqa: E402
from src.label_free_calibration import make_splits        # noqa: E402
from src.screen_statistics_probe import per_entity_features, screens  # noqa: E402
from src.survey_reducibility import entity_stats           # noqa: E402

SEEDS = [0, 1]
NFAC = {"Nursing": 1200}
SETS = {"discovery": 0, "confirmation": 1}


def main():
    panels = sys.argv[1:] or ["Nursing", "Electricity", "Traffic", "Solar"]
    tag = "" if len(sys.argv) == 1 else "_" + "_".join(panels)
    out = {k: ([], []) for k in SETS}
    for name in panels:
        t0 = time.time()
        P = load_panel(name, NFAC.get(name, 10 ** 6))
        stats, _ = entity_stats(P)
        F8 = per_entity_features(P)
        F3 = entity_features(P)
        save = {}
        for sset, rs in SETS.items():
            splits = make_splits(stats, 26, np.random.default_rng(rs))
            print(f"\n### {name} {sset}: {len(splits)} splits", flush=True)
            for k, (A, B) in enumerate(splits):
                save[f"{sset}_A_{k}"], save[f"{sset}_B_{k}"] = A, B
                sc = {**screens(F8, A, B), **core_screens(F3, A, B)}
                runs = [run_split(P, A, B, s, floors=(15, 6)) for s in SEEDS]
                runs = [(s, r) for s, r in zip(SEEDS, runs) if r is not None]
                if not runs:
                    continue
                for s, (row, ent, _) in runs:
                    out[sset][0].append(dict(panel=name, set=sset, split=k, seed=s,
                                             n_A=len(A), n_B=len(B), **sc, **row))
                    out[sset][1].append(ent.assign(panel=name, set=sset, split=k,
                                                   seed=s))
                h = np.mean([r["headroom_mae_cat"] for _, (r, _, _) in runs])
                print(f"  split {k:2d} n=({len(A)},{len(B)}) acf={sc['acf_l2']:.4f} "
                      f"headroom cat={h:+6.2f}%", flush=True)
        np.savez_compressed(os.path.join(RES, f"screen_features_{name}.npz"),
                            **F3, **save)
        for sset in SETS:
            pd.DataFrame(out[sset][0]).to_csv(
                os.path.join(RES, f"screen_{sset}_splits{tag}.csv"), index=False)
            pd.concat(out[sset][1]).to_csv(
                os.path.join(RES, f"screen_{sset}_entities{tag}.csv"), index=False)
        print(f"  [{name} done in {time.time() - t0:.0f}s]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
