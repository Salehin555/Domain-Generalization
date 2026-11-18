"""Labeled target-data requirement of the learning curve (no training).

The learning curve (tentities.py) refits the target-trained reference on the
first K target training entities but keeps the full early-stopping set t_va, and
scores on the target test set t_te. "K entities" therefore understates the labeled
target data a practitioner needs: K training entities plus |t_va| early-stopping
entities (the test entities are needed only to evaluate the decision). The
allocation is deterministic given the seed, so the set sizes are recomputed here
exactly, with the same calls as tentities.py, and no model is fitted.

Output: results/tentity_sizes.csv (panel, split, seed, n_tr, n_va, n_te)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, allocate  # noqa: E402
from anchor import anchors  # noqa: E402
from tentities import KEEP  # noqa: E402


def main():
    rows = []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(
            ["Electricity", "Traffic", "Solar", "Nursing"]):
        if (panel, split) not in KEEP:
            continue
        for seed in seeds:
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            rows.append(dict(panel=panel, split=split, seed=seed, n_tr=len(al["t_tr"]),
                             n_va=len(al["t_va"]), n_te=len(al["t_te"])))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "tentity_sizes.csv"), index=False)
    print(D.to_string())


if __name__ == "__main__":
    main()
