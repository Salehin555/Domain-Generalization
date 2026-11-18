"""
Generic adapter for the multivariate benchmark panels (traffic, solar) into the
.npz schema consumed by src.data.Panel.

Identical treatment to the electricity panel so the survey stays comparable:
  * sub-daily meters aggregated into three daily channels by time-of-day third,
    the analogue of the three staff categories in the nursing panel;
  * each channel divided by the entity's own mean total daily value, the analogue
    of dividing nursing hours by resident census, which removes entity size while
    preserving composition;
  * behavioral domain labels derived from the TRAINING period only, taken at the
    extreme quartiles of the weekend-to-weekday ratio.

Panels
------
traffic : 862 freeway sensors, hourly, 731 days. Occupancy rates. Commuter
          corridors and through-routes differ strongly in their work-week rhythm,
          so a weekly-profile split is behaviorally meaningful here.
solar   : 137 photovoltaic plants, 10-minute, 365 days. Generation is driven by
          the sun, which has no weekly cycle, so the weekly split is expected to
          be close to meaningless on this panel. That makes solar a useful extra
          control rather than a weakness: a protocol that reports headroom there
          would be suspect.
"""
import gzip
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache", "panels")

SPECS = {
    "traffic": dict(file="traffic.txt.gz", per_day=24, start="2015-01-01",
                    train_frac=0.67),
    "solar":   dict(file="solar.txt.gz",  per_day=144, start="2006-01-01",
                    train_frac=0.67),
}


def build(name):
    sp = SPECS[name]
    path = os.path.join(CACHE, sp["file"])
    print(f"\n=== {name} ===", flush=True)
    with gzip.open(path, "rt") as f:
        A = np.loadtxt(f, delimiter=",", dtype=np.float32)     # [T_sub, nEntities]
    T, nE = A.shape
    pd_ = sp["per_day"]
    nD = T // pd_
    A = A[:nD * pd_]
    print(f"  raw {A.shape} -> {nD} days, {nE} entities")

    # three daily channels by time-of-day third
    seg = pd_ // 3
    X = np.zeros((nE, nD, 3), np.float32)
    R = A.reshape(nD, pd_, nE)
    for c in range(3):
        X[:, :, c] = R[:, c * seg:(c + 1) * seg, :].sum(1).T

    dates = pd.date_range(sp["start"], periods=nD, freq="D")

    total = X.sum(axis=2)
    keep = (total.mean(axis=1) > 0) & ((total > 0).mean(axis=1) > 0.5)
    X, total = X[keep], total[keep]
    print(f"  kept {keep.sum()} / {nE} entities")

    # Observed (training) window. Everything derived from the series below is
    # restricted to it, so that no deployment-period value can enter either the
    # normalization denominator or the domain definition.
    cut = dates[int(sp["train_frac"] * nD)]
    tr = dates <= cut

    # Scale normalization uses the OBSERVED-PERIOD mean only. Using the mean of
    # the whole series would put deployment-period information into the
    # denominator of every training window (temporal leakage).
    scale = total[:, tr].mean(axis=1, keepdims=True)
    Xn = (X / scale[:, :, None]).astype(np.float32)

    # behavioral grouping from the training period only, extreme quartiles
    dow = dates.dayofweek.values
    wknd, wkdy = tr & (dow >= 5), tr & (dow < 5)
    ratio = total[:, wknd].mean(1) / (total[:, wkdy].mean(1) + 1e-9)
    q1, q3 = np.percentile(ratio, [25, 75])
    grp = np.full(len(ratio), "Middle", dtype=object)
    grp[ratio <= q1] = "WorkWeek"
    grp[ratio >= q3] = "Flat"
    print(f"  weekend/weekday ratio median={np.median(ratio):.3f} "
          f"Q1={q1:.3f} Q3={q3:.3f}")
    for g in ["WorkWeek", "Middle", "Flat"]:
        print(f"    {g:9s} n={(grp == g).sum()}")

    out = os.path.join(CACHE, f"panel_{name}.npz")
    np.savez_compressed(
        out, X=Xn,
        dates=np.array([int(d.strftime("%Y%m%d")) for d in dates], np.int64),
        provnum=np.arange(keep.sum()).astype(str),
        state=grp.astype(str),
        medcensus=scale.ravel().astype(np.float32),
        ratio=ratio.astype(np.float32))
    print(f"  wrote {out}  X={Xn.shape}")
    return int(str(dates[int(sp['train_frac'] * nD)].strftime("%Y%m%d")))


if __name__ == "__main__":
    for n in SPECS:
        b = build(n)
        print(f"  boundary_date for {n}: {b}")
