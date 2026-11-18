"""
Second-dataset adapter: electricity load panel -> .npz matching src.data.Panel.

WHY A SECOND DATASET
--------------------
The CMS nurse-staffing panel turned out to sit in the IRREDUCIBLE regime: the
oracle diagnostic showed a large apparent cross-group shift with essentially no
reducible component, which is why every sound method converged there. A single
dataset can only ever demonstrate one regime. To test whether the diagnostic
correctly PREDICTS the opposite regime, we need a panel of many named entities
whose domains plausibly differ in *spectral* structure, since that is what the
CD-ISG-TS gate acts on. Electricity clients qualify: industrial consumers follow a
hard five-day work cycle while residential consumers follow a seven-day cycle with
evening peaks. That is a genuine difference in the frequency content of the series,
not merely a difference in level.

DATA
----
The standard multivariate electricity benchmark (321 clients, hourly, 2012-01-01
to 2014-12-31, 26,304 hours). This is the same variant used by the Informer,
Autoformer, and DLinear literature that supplies this study's baselines, so the
panel is directly comparable to published forecasting work.

CHANNEL CONSTRUCTION (the analogue of RN / LPN / CNA)
----------------------------------------------------
Each client's hourly meter is aggregated into three daily channels by time-of-day
segment:
    ch0 = night      00:00-08:00
    ch1 = working    08:00-16:00
    ch2 = evening    16:00-24:00
This mirrors the nursing panel, where one facility carries three staff-category
series. It is also the axis along which industrial and residential clients differ
most, so the category-conditioned gate is genuinely exercised rather than nominal.

SCALE NORMALIZATION (the analogue of hours per resident-day)
-----------------------------------------------------------
Clients differ enormously in absolute consumption, exactly as facilities differ in
size. In the nursing panel we divided hours by census to obtain a scale-free
target. Here we divide every channel by the client's OWN mean total daily load:

    X[i, t, c] = load[i, t, c] / mean_t( total_load[i, t] )

This removes client size while PRESERVING the day-versus-evening composition, just
as HPRD removes facility size while preserving the staff-category mix. This is the
honest analogue. Normalizing each channel by its own mean would erase the very
composition signal that distinguishes the domains; skipping normalization would
manufacture a large apparent gap out of client size alone and guarantee a
flattering result.

DOMAIN GROUPING
---------------
This dataset ships no industrial/residential label, so the behavioral grouping is
DERIVED, and derived only from the TRAINING period so that no deployment-period
information leaks into the domain definition. We compute each client's
weekend-to-weekday total-load ratio over 2012-2013 and split at the median: a low
ratio indicates work-week-driven (industrial-like) demand, a high ratio indicates
flat or home-centered (residential-like) demand. The manuscript states plainly
that this grouping is derived, in contrast to the CMS ownership and urban/rural
labels, which are external metadata.

Output: cache/electricity/panel_elec.npz with keys
    X [nF, T, 3] float32, dates [T] int (YYYYMMDD), provnum [nF],
    state [nF] (derived behavioral group), medcensus [nF] (mean total daily load,
    the size covariate analogous to bed count), ratio [nF]
"""
import gzip
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache", "electricity")

RAW = os.path.join(CACHE, "electricity_lstnet.txt.gz")
OUTF = os.path.join(CACHE, "panel_elec.npz")

START = pd.Timestamp("2012-01-01 00:00")
TRAIN_END = pd.Timestamp("2013-12-31")
SEGMENTS = [(0, 8), (8, 16), (16, 24)]
SEG_NAMES = ["night", "working", "evening"]


def main():
    print("reading hourly meter matrix ...", flush=True)
    with gzip.open(RAW, "rt") as f:
        A = np.loadtxt(f, delimiter=",", dtype=np.float32)     # [T_hours, nClients]
    T_h, nC = A.shape
    stamps = pd.date_range(START, periods=T_h, freq="h")
    print(f"  hourly matrix {A.shape}  ({stamps[0].date()} .. {stamps[-1].date()})")

    # ---- aggregate hourly -> daily, per time-of-day segment ------------------
    hour = stamps.hour.values
    dayix = ((stamps - stamps[0]).days).values
    nD = dayix.max() + 1
    X = np.zeros((nC, nD, len(SEGMENTS)), np.float32)
    for c, (a, b) in enumerate(SEGMENTS):
        m = (hour >= a) & (hour < b)
        seg = np.zeros((nD, nC), np.float32)
        np.add.at(seg, dayix[m], A[m])
        X[:, :, c] = seg.T
    dates = pd.date_range(START.normalize(), periods=nD, freq="D")
    print(f"  daily panel {X.shape}  ({dates[0].date()} .. {dates[-1].date()})")

    # ---- keep clients with a complete, non-degenerate record -----------------
    total = X.sum(axis=2)                                      # [nClients, nDays]
    nonzero_frac = (total > 0).mean(axis=1)
    keep = (nonzero_frac > 0.99) & (total.mean(axis=1) > 0)
    X = X[keep]; total = total[keep]
    clients = np.arange(nC)[keep]
    print(f"  kept {keep.sum()} / {nC} clients with >99% nonzero coverage")

    # ---- scale normalization (analogue of HPRD) ------------------------------
    # observed era only (dates <= TRAIN_END), frozen and applied to the whole series,
    # so no deployment-era value enters the denominator
    tr_norm = np.asarray(dates <= TRAIN_END)
    scale = total[:, tr_norm].mean(axis=1, keepdims=True)      # [nClients, 1]
    Xn = (X / scale[:, :, None]).astype(np.float32)

    # ---- derived behavioral grouping, TRAINING PERIOD ONLY -------------------
    # The weekend/weekday ratio is tightly concentrated near 1.0 (interquartile
    # range roughly 0.97 to 1.02), so a median split would place behaviorally
    # near-identical clients on opposite sides and define domains that do not
    # actually differ. We therefore take the EXTREME QUARTILES as the two domains
    # and leave the middle half unassigned, which is the analogue of contrasting
    # for-profit against non-profit facilities rather than splitting a continuous
    # proxy at its median. This choice is made from the training period only and
    # is reported explicitly in the manuscript.
    tr = dates <= TRAIN_END
    dow = dates.dayofweek.values
    wknd = tr & (dow >= 5)
    wkdy = tr & (dow < 5)
    ratio = total[:, wknd].mean(axis=1) / total[:, wkdy].mean(axis=1)
    q1, q3 = np.percentile(ratio, [25, 75])
    group = np.full(len(ratio), "Middle", dtype=object)
    group[ratio <= q1] = "WorkWeek"      # work-week driven (industrial-like)
    group[ratio >= q3] = "Flat"          # flat / home-centered (residential-like)
    group = group.astype(str)
    print(f"  weekend/weekday ratio: median={np.median(ratio):.3f} "
          f"range=[{ratio.min():.3f}, {ratio.max():.3f}]  Q1={q1:.3f} Q3={q3:.3f}")
    for g in ["WorkWeek", "Middle", "Flat"]:
        print(f"    {g:9s} n={(group == g).sum()}")

    # validation that the two extreme groups really are spectrally distinct
    c0 = total - total.mean(axis=1, keepdims=True)
    F = np.abs(np.fft.rfft(c0, axis=1)) / total.shape[1]
    freq = np.fft.rfftfreq(total.shape[1], 1.0)
    rel7 = F[:, int(np.argmin(abs(freq - 1 / 7.0)))] / (c0.std(axis=1) + 1e-9)
    a, b = rel7[group == "WorkWeek"], rel7[group == "Flat"]
    print(f"  relative weekly (7 d) spectral amplitude: "
          f"WorkWeek {a.mean():.3f} vs Flat {b.mean():.3f}")

    dates_int = np.array([int(d.strftime("%Y%m%d")) for d in dates], np.int64)
    np.savez_compressed(
        OUTF,
        X=Xn,
        dates=dates_int,
        provnum=clients.astype(str),
        state=group.astype(str),
        medcensus=scale.ravel().astype(np.float32),
        ratio=ratio.astype(np.float32),
        seg_names=np.array(SEG_NAMES),
    )
    print(f"\nwrote {OUTF}")
    print(f"  X {Xn.shape}  daily total per client averages "
          f"{Xn.sum(axis=2).mean():.3f} (scale-free)")


if __name__ == "__main__":
    main()
