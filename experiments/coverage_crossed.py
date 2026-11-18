"""Calendar coverage separated from history length.

The history-truncation analysis shortens coverage and the amount of training data
together. This design holds history length fixed and varies only which calendar
months the history covers, with a crossed layout at the panel's decision date:

    history placements   H1 = first half of the last observed year
                         H2 = second half of the last observed year
    deployment halves    D1 = first half of the deployment year (starting at the
                              decision date), D2 = second half

Both models of a split (source-trained and target-trained reference) are fitted and
early-stopped on windows lying entirely inside one placement, so H1 and H2 carry
the same number of days and entities. The estimate is scored on the target test
entities' windows inside that placement; realized headroom is scored on windows
lying entirely inside D1 and inside D2. A model fitted on H1 is therefore evaluated
on covered months (D1: the same calendar months one year later) and on uncovered
months (D2), and a model fitted on H2 the other way round. Averaged over the two
placements, covered and uncovered cells have the same history length, the same
training data, and the same mean gap between history and deployment (six months on
electricity and nursing, eight on traffic), so the contrast isolates coverage.

Panels: electricity, traffic, nursing (1,200-facility working set). Solar has one
year of data and cannot be crossed this way. Splits and target/source allocation
are those of the observed-history analysis (temporal.py, 12 random-combination
splits, seeds 0 and 1), computed on the same stride-18 panel; training here uses
stride 7 so that each half-year placement holds enough windows per entity.
Stride changes the windows available but not the number of gradient steps, which
is fixed by the number of entities per epoch.

Output: results/coverage_rows.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ERM, METRICS, RES, fit, load_panel, per_entity, predict  # noqa: E402
from temporal import NFAC, N_SPLITS, SEEDS, match_source, split_target  # noqa: E402
from src.label_free_calibration import make_splits  # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402

RANGES = {
    "Electricity": dict(H1=(20130101, 20130630), H2=(20130701, 20131231),
                        D1=(20140101, 20140630), D2=(20140701, 20141231)),
    "Traffic": dict(H1=(20150504, 20150831), H2=(20150901, 20151231),
                    D1=(20160504, 20160831), D2=(20160901, 20161231)),
    "Nursing": dict(H1=(20240101, 20240630), H2=(20240701, 20241231),
                    D1=(20250101, 20250630), D2=(20250701, 20251231)),
}
STRIDE = 7


def inside(P, lo, hi):
    """Windows whose input and horizon both lie inside [lo, hi] (YYYYMMDD)."""
    d = np.asarray(P.dates)
    first = d[P.wstart]
    last = d[np.minimum(P.wstart + P.span - 1, len(d) - 1)]
    return (first >= lo) & (last <= hi) & (P.wstart + P.span <= len(d))


def headroom(s, r):
    return 100.0 * (s - r) / s


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Nursing"]
    rows = []
    for name in want:
        t0 = time.time()
        P18 = load_panel(name, NFAC.get(name, 10 ** 6))            # splits as in temporal
        stats, _ = entity_stats(P18)
        splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
        del P18
        P = load_panel(name, NFAC.get(name, 10 ** 6), stride=STRIDE)
        R = RANGES[name]
        M = {k: inside(P, *v) for k, v in R.items()}
        for k in ("H1", "H2"):
            assert M[k].any() and (P.wstart[M[k]] + P.span <= np.searchsorted(
                P.dates, R["D1"][0])).all(), "history window crosses the decision date"
        for sp, (A, B) in enumerate(splits):
            for sd in SEEDS:
                rng = np.random.default_rng(sd)
                t_tr, t_va, t_te = split_target(B, rng)
                if len(t_tr) < 12:
                    continue
                s_tr, s_va = match_source(A, B, len(t_tr), rng)
                if len(s_tr) < 12:
                    continue
                for h in ("H1", "H2"):
                    m_s = fit(P, ERM, s_tr, s_va, sd, M[h], M[h])
                    m_r = fit(P, ERM, t_tr, t_va, sd, M[h], M[h])
                    r = dict(panel=name, split=sp, seed=sd, history=h,
                             n_src=len(s_tr), n_tgt_train=len(t_tr), n_tgt_test=len(t_te),
                             train_windows_per_entity=float(M[h].sum()) / P.nF)
                    for ev in (h, "D1", "D2"):
                        _, Y, Ps, fid = predict(m_s, P, t_te, M[ev])
                        _, _, Pr, _ = predict(m_r, P, t_te, M[ev])
                        es, er = per_entity(Ps, Y, fid), per_entity(Pr, Y, fid)
                        tag = "est" if ev == h else ev
                        for mt in METRICS:
                            r[f"headroom_{tag}_{mt}"] = headroom(es[mt].mean(), er[mt].mean())
                    rows.append(r)
                    print(f"  {name} split {sp:2d} seed {sd} {h}: est={r['headroom_est_mae_cat']:+6.2f} "
                          f"D1={r['headroom_D1_mae_cat']:+6.2f} D2={r['headroom_D2_mae_cat']:+6.2f}",
                          flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "coverage_rows.csv"), index=False)
        print(f"[{name} done {time.time() - t0:.0f}s]", flush=True)
    print("done")


if __name__ == "__main__":
    main()
