"""Out-of-sample confirmation of the autocorrelation screen on FRESH splits.

acf_l2 was chosen after looking at 95 splits generated with seed 0. The
permutation test says the pattern is not selection noise (p=0.0005), but the
statistic was still selected on the same data it is scored on. The only clean
answer is new splits it has never seen.

This regenerates the whole experiment with a different split seed, measures the
reducible gap from scratch, and scores the two screens with no refitting of
anything. Nothing about acf_l2 is tuned here; it is applied as already defined.

If acf_l2 holds up on these and js_mean stays weak on solar and traffic, the
finding is real. If acf_l2 collapses, it was an artifact of the first draw and
must be reported as such.

Output: outputs/screen_confirm_holdout.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

from src.data import Panel
from src.survey_reducibility import entity_stats, js_div
from src.label_free_calibration import PANELS, make_splits, measure_gap, loo_predict
from src.screen_statistics_probe import per_entity_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SPLIT_SEED = 1           # <- the whole point: NOT the seed used for discovery
SEEDS = [0, 1]
PANEL_ORDER = ["Nursing", "Electricity", "Solar", "Traffic"]


def main():
    rows = []
    for name in PANEL_ORDER:
        spec = PANELS[name]
        P = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6), L=56, H=14,
                  stride=18, seed=0, n_channels=3, boundary_date=spec["boundary"])
        stats, _ = entity_stats(P)
        splits = make_splits(stats, 26, np.random.default_rng(SPLIT_SEED))
        F = per_entity_features(P)
        print(f"\n### {name}: {len(splits)} fresh splits (split seed {SPLIT_SEED})",
              flush=True)
        for k, (A, B) in enumerate(splits):
            js = js_div(F["prof4"][A].mean(0), F["prof4"][B].mean(0))
            acf = float(np.linalg.norm(F["acf"][A].mean(0) - F["acf"][B].mean(0)))
            gaps = [g for g in (measure_gap(P, A, B, s) for s in SEEDS)
                    if g is not None]
            if not gaps:
                continue
            gap = float(np.mean(gaps))
            rows.append(dict(panel=name, split=k, gap=gap, js_mean=js, acf_l2=acf))
            print(f"  split {k:2d}  js={js:.5f}  acf={acf:.4f}  gap={gap:+6.2f}%",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "screen_confirm_holdout.csv"), index=False)

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE CONFIRMATION (fresh splits, nothing retuned)")
    print("=" * 70)
    print(f"{'panel':<13}{'screen':<11}{'LOO r':>9}{'MAE pp':>9}{'mean-MAE':>10}"
          f"{'decide@3%':>11}")
    print("-" * 63)
    for p in PANEL_ORDER:
        g = df[df.panel == p]
        if len(g) < 6:
            continue
        y = g.gap.values
        base = np.mean(np.abs(y - y.mean()))
        for c in ["js_mean", "acf_l2"]:
            pred = loo_predict(g[c].values, y)
            r = ss.pearsonr(pred, y)[0]
            print(f"{p:<13}{c:<11}{r:>+9.3f}{np.mean(np.abs(pred-y)):>9.2f}"
                  f"{base:>10.2f}{100*np.mean((pred>=3.0)==(y>=3.0)):>10.0f}%")
        print("-" * 63)


if __name__ == "__main__":
    main()
