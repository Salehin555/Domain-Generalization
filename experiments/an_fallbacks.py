"""Count classical fits that fell back to seasonal naive, from the forecast caches.

classical.py replaces a failed or non-finite fit by the seasonal-naive forecast
of that series. A fallback is therefore identified exactly, after the fact, as a
forecast equal to the seasonal-naive forecast of a series that is not constant
(constant series take the flat forecast by design and are counted separately).
Counting from the caches gives one consistent figure for every panel, including a
panel whose run was stopped before it wrote its in-run counts. The in-run counts
that do exist are compared as a check.

Output: classical_fallbacks.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, load_panel, seasonal_naive  # noqa: E402
from classical import CLASSICAL  # noqa: E402


def main():
    rows = []
    for panel in ("Electricity", "Traffic", "Solar"):
        cpath = os.path.join(RES, f"classical_cache_{panel}.pkl")
        if not os.path.exists(cpath):
            print(f"{panel}: no cache yet"); continue
        cache = pickle.load(open(cpath, "rb"))
        P = load_panel(panel)
        n_series = 0; const = 0
        fb = {m: 0 for m in CLASSICAL}
        for e, out in cache.items():
            X = P.eval_arrays([e])[0]                      # [nw, C, L]
            sn = seasonal_naive(X)
            flat = np.ptp(X, axis=2) < 1e-9                # [nw, C]
            n_series += flat.size; const += int(flat.sum())
            for m in CLASSICAL:
                if m.startswith("Persistence"):
                    continue
                same = np.all(np.abs(out[m] - sn) < 1e-6, axis=2)
                fb[m] += int((same & ~flat).sum())
        for m in CLASSICAL:
            rows.append(dict(panel=panel, method=m, series=n_series, fallbacks=fb[m],
                             constant_series=const))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(RES, "classical_fallbacks.csv"), index=False)
    print(D.to_string())
    run = os.path.join(RES, "classical_fallbacks_Solar.csv")
    if os.path.exists(run):
        print("\nin-run counts (Solar) for comparison:")
        print(pd.read_csv(run).to_string())


if __name__ == "__main__":
    main()
