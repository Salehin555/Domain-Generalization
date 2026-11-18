"""Several decision dates per panel.

The observed-history analysis uses one decision date per panel. Here each panel
has two to four decision dates (origins) with non-overlapping deployment periods:
at origin o both models are trained and early-stopped only on windows that end
before o, the estimate is scored on the target test entities' windows before o,
and realized headroom on their windows lying entirely inside [o, next origin).

    Electricity  2013-01-01, 2013-07-01, 2014-01-01, 2014-07-01   (6-month periods)
    Traffic      2016-01-01, 2016-05-04, 2016-09-01               (4-month periods)
    Nursing      2025-01-01, 2025-07-01                           (6-month periods)
    Solar        2006-06-01, 2006-09-02                           (3 and 4 months)

No information from after an origin is used for that origin:
  * Scale (benchmark panels). The cached panels divide each entity by its mean
    total daily value over the original observed era, which ends after the early
    origins. For each origin the arrays are rescaled by the entity's mean total
    over the days before that origin (a per-entity constant, so the rescaling is
    exact) and written to a temporary panel file.
  * Groups. Split statistics are computed from windows before the EARLIEST origin,
    so the same candidate groupings are re-evaluated at every date without using
    any later data.
  * Nursing. Within-year interpolation is two-sided, so windows with an imputed
    target day or an input day imputed from after their forecast origin are
    excluded from training, selection and scoring (the audited clean-window mask).
    The cohort-selection caveat of the main analysis still applies.

Splits: 12 random-combination splits (as in temporal.py), seeds 0 and 1,
windows of stride 7. Output: results/origin_rows_<panels>.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ERM, METRICS, PANELS, RES, ROOT, fit, per_entity, predict  # noqa: E402
from temporal import (NFAC, N_SPLITS, SEEDS, clean_window_mask,  # noqa: E402
                         match_source, split_target)
from src.data import Panel  # noqa: E402
from src.label_free_calibration import make_splits  # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402

ORIGINS = {"Electricity": [20130101, 20130701, 20140101, 20140701],
           "Traffic": [20160101, 20160504, 20160901],
           "Nursing": [20250101, 20250701],
           "Solar": [20060601, 20060902]}
STRIDE = 7
TMP = os.path.join(ROOT, "cache", "origins")
os.makedirs(TMP, exist_ok=True)


def panel_at(name, origin, stride=STRIDE):
    """Panel whose observed era ends at `origin`, rescaled with pre-origin data only."""
    src = PANELS[name]["path"]
    if name != "Nursing":
        d = dict(np.load(src, allow_pickle=True))
        b = int(np.searchsorted(d["dates"], origin))
        X = d["X"].astype(np.float64)
        s = X[:, :b, :3].sum(2).mean(1)                       # pre-origin mean total
        d["X"] = (X / s[:, None, None]).astype(np.float32)
        src = os.path.join(TMP, f"{name}_{origin}.npz")
        np.savez(src, **d)
    return Panel(src, n_facilities=NFAC.get(name, 10 ** 6), L=56, H=14, stride=stride,
                 seed=0, n_channels=3, boundary_date=origin)


def headroom(s, r):
    return 100.0 * (s - r) / s


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar", "Nursing"]
    rows = []
    for name in want:
        t0 = time.time()
        orig = ORIGINS[name]
        P0 = panel_at(name, orig[0], stride=18)             # groups from the earliest history
        stats, _ = entity_stats(P0)
        splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
        del P0
        for j, o in enumerate(orig):
            P = panel_at(name, o)
            d = np.asarray(P.dates)
            end = orig[j + 1] if j + 1 < len(orig) else int(d[-1]) + 1
            last = d[np.minimum(P.wstart + P.span - 1, len(d) - 1)]
            ok = (clean_window_mask(P)[0] if name == "Nursing"
                  else np.ones(len(P.fid), bool))
            obs = P.is_2024 & ok
            dep = P.is_2025 & (last < end) & (P.wstart + P.span <= len(d)) & ok
            for sp, (A, B) in enumerate(splits):
                for sd in SEEDS:
                    rng = np.random.default_rng(sd)
                    t_tr, t_va, t_te = split_target(B, rng)
                    if len(t_tr) < 12:
                        continue
                    s_tr, s_va = match_source(A, B, len(t_tr), rng)
                    if len(s_tr) < 12:
                        continue
                    m_s = fit(P, ERM, s_tr, s_va, sd, obs, obs)
                    m_r = fit(P, ERM, t_tr, t_va, sd, obs, obs)
                    r = dict(panel=name, origin=o, origin_idx=j, deploy_end=end, split=sp,
                             seed=sd, history_days=int(np.searchsorted(d, o)),
                             n_tgt_train=len(t_tr), n_tgt_test=len(t_te))
                    for era, msk in (("obs", obs), ("dep", dep)):
                        _, Y, Ps, fid = predict(m_s, P, t_te, msk)
                        _, _, Pr, _ = predict(m_r, P, t_te, msk)
                        es, er = per_entity(Ps, Y, fid), per_entity(Pr, Y, fid)
                        for mt in METRICS:
                            r[f"headroom_{era}_{mt}"] = headroom(es[mt].mean(), er[mt].mean())
                    rows.append(r)
                    print(f"  {name} origin {o} split {sp:2d} seed {sd}: "
                          f"est={r['headroom_obs_mae_cat']:+6.2f} dep={r['headroom_dep_mae_cat']:+6.2f}",
                          flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"origin_rows_{'_'.join(want)}.csv"), index=False)
            del P
        print(f"[{name} done {time.time() - t0:.0f}s]", flush=True)
    print("done")


if __name__ == "__main__":
    main()
