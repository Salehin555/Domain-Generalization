"""Is the autocorrelation screen real, or the best of eight lucky draws?

screen_statistics_probe.py scored eight candidate screens on the same splits and
acf_l2 won on all four panels. Winning once is unremarkable when eight things are
tried; winning on every panel is a much stronger pattern, but "much stronger"
needs a number, not an adjective. Three checks:

  1. MULTIPLICITY. Permute the measured gaps within each panel, rescore ALL eight
     candidates, and record how often the best of them clears the same bar
     acf_l2 cleared (minimum leave-one-out r across the four panels). This prices
     in the selection, because the permuted run gets the same eight attempts.

  2. USEFULNESS, not just correlation. A correlation can be significant and still
     be worthless for a decision. Report mean absolute error against the
     predict-the-mean baseline, and accuracy of the 3 percent go/no-go call.

  3. WHAT IT IS MEASURING. If the account is right, the autocorrelation distance
     should carry information the energy divergence does not. Partial correlation
     of gap with acf_l2 controlling for js_mean tests exactly that.

Output: printed report; no files written.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

from src.label_free_calibration import loo_predict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
PANEL_ORDER = ["Nursing", "Electricity", "Solar", "Traffic"]


def loo_r(x, y):
    if len(y) < 6:
        return np.nan
    return ss.pearsonr(loo_predict(x, y), y)[0]


def main():
    df = pd.read_csv(os.path.join(OUT, "screen_statistics_probe.csv"))
    cands = [c for c in df.columns if c not in ("panel", "split", "gap")]

    # ---------------------------------------------------------------- check 1
    obs = {c: min(loo_r(df[df.panel == p][c].values, df[df.panel == p].gap.values)
                  for p in PANEL_ORDER) for c in cands}
    best_c = max(obs, key=obs.get)
    bar = obs[best_c]
    print("=" * 74)
    print("1. MULTIPLICITY: is the winner just the luckiest of eight?")
    print("=" * 74)
    print(f"   worst-panel LOO r, per candidate (higher = works everywhere):")
    for c in sorted(obs, key=obs.get, reverse=True):
        print(f"      {c:<14} {obs[c]:+.3f}")
    print(f"\n   winner: {best_c}, clearing a bar of min-r = {bar:+.3f}")

    rng = np.random.default_rng(0)
    NPERM = 2000
    hits = 0
    null_best = []
    for _ in range(NPERM):
        b = -np.inf
        for c in cands:
            worst = np.inf
            for p in PANEL_ORDER:
                g = df[df.panel == p]
                worst = min(worst, loo_r(g[c].values, rng.permutation(g.gap.values)))
            b = max(b, worst)
        null_best.append(b)
        hits += (b >= bar)
    null_best = np.array(null_best)
    pval = (hits + 1) / (NPERM + 1)
    print(f"   under {NPERM} permutations of the gaps, the BEST of the same eight")
    print(f"   candidates reached min-r = {null_best.mean():+.3f} +/- {null_best.std():.3f}")
    print(f"   (95th pct {np.percentile(null_best,95):+.3f}, max {null_best.max():+.3f})")
    print(f"   p = {pval:.4f}  <- probability of this pattern with eight tries and no signal")

    # ---------------------------------------------------------------- check 2
    print("\n" + "=" * 74)
    print("2. IS IT USEFUL? error and decision quality, acf_l2 vs the current screen")
    print("=" * 74)
    print(f"{'panel':<13}{'screen':<12}{'LOO r':>8}{'MAE pp':>9}"
          f"{'mean-MAE':>10}{'decide@3%':>11}")
    print("-" * 63)
    for p in PANEL_ORDER:
        g = df[df.panel == p]
        y = g.gap.values
        base = np.mean(np.abs(y - y.mean()))
        for c in ["js_mean", "acf_l2"]:
            pred = loo_predict(g[c].values, y)
            r = ss.pearsonr(pred, y)[0]
            mae = np.mean(np.abs(pred - y))
            acc = np.mean((pred >= 3.0) == (y >= 3.0))
            print(f"{p:<13}{c:<12}{r:>+8.3f}{mae:>9.2f}{base:>10.2f}{100*acc:>10.0f}%")
        print("-" * 63)

    # ---------------------------------------------------------------- check 3
    print("=" * 74)
    print("3. DOES IT ADD ANYTHING BEYOND THE ENERGY DIVERGENCE?")
    print("=" * 74)
    print("   partial correlation of gap with acf_l2, controlling for js_mean")
    for p in PANEL_ORDER:
        g = df[df.panel == p]
        y, a, j = g.gap.values, g.acf_l2.values, g.js_mean.values
        ry = y - np.polyval(np.polyfit(j, y, 1), j)
        ra = a - np.polyval(np.polyfit(j, a, 1), j)
        pr, pp = ss.pearsonr(ry, ra)
        raw = ss.pearsonr(y, a)
        print(f"      {p:<13} raw r={raw[0]:+.3f}   partial r={pr:+.3f} (p={pp:.3g})")


if __name__ == "__main__":
    main()
