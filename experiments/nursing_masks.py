"""Audit inputs for the nursing panel.

Rebuilds, from the raw PBJ files and with exactly the row-level quality gates of
src/build_panel.py:

1. IMPUTATION MASKS for the analysis cohort (cache/panel.npz, 13,402 facilities)
     imputed[i, d]   facility-day d was missing and filled by interpolation
     right[i, d]     for an imputed day, the index of the observed day AFTER it
                     that the within-era interpolation used (-1 if the fill was
                     clamped at an era edge, i.e. used only earlier values)
   From these, window-level flags are derived in temporal.py:
     imputed target      any imputed day inside the 14-day forecast horizon
     post-origin input   an imputed input day whose fill used an observation
                         AFTER the window's forecast origin (last input day)

2. PROSPECTIVE COHORT. The analysis cohort requires >=97% daily coverage over
   BOTH years, i.e. it is selected using deployment-era data. The prospective
   cohort requires >=97% coverage over the observed era (2024) only, which is
   what a practitioner could apply on 1 January 2025. Its panel keeps deployment
   days that are missing as NaN; they are never interpolated across and windows
   touching them are excluded from training and scoring.

Outputs: results/nursing_imputation_masks.npz, results/nursing_cohort_audit.csv,
         cache/panel_prospective.npz (new file; cache/panel.npz untouched)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.build_panel import load                     # noqa: E402

RES = os.path.join(HERE, "results")
BOUNDARY_DATE = 20250101


def gated(df):
    df = df[df["MDScensus"] >= 5]
    df = df[(df["TOTAL"] >= 0) & (df["RN"] >= 0) & (df["LPN"] >= 0) & (df["CNA"] >= 0)]
    h = df["TOTAL"] / df["MDScensus"]
    return df[(h > 0.3) & (h < 12.0)]


def right_anchor(present_row, b):
    """For each day, the observed day after it inside the same era (or -1)."""
    T = len(present_row)
    out = np.full(T, -1, np.int32)
    for lo, hi in ((0, b), (b, T)):
        nxt = -1
        for d in range(hi - 1, lo - 1, -1):
            if present_row[d]:
                nxt = d
            elif nxt != -1:
                out[d] = nxt
    # clamp: a missing day with no EARLIER observation in its era is filled from
    # the right only; np.interp clamps and still uses that right value
    return out


def interp_era(a, b):
    idx = np.arange(a.shape[1])
    for lo, hi in ((0, b), (b, a.shape[1])):
        seg = a[:, lo:hi]
        for i in range(seg.shape[0]):
            m = ~np.isnan(seg[i])
            if m.all() or not m.any():
                continue
            seg[i] = np.interp(idx[:hi - lo], idx[:hi - lo][m], seg[i][m])
        a[:, lo:hi] = seg
    return a


def main():
    df = load()
    df["TOTAL"] = df["RN"] + df["LPN"] + df["CNA"]
    dates = np.sort(df["WorkDate"].unique()).astype(np.int32)
    n_days = len(dates)
    b = int(np.searchsorted(dates, BOUNDARY_DATE))
    d2i = {int(v): i for i, v in enumerate(dates)}
    df = gated(df)
    df["di"] = df["WorkDate"].map(d2i)

    cov_all = df.groupby("PROVNUM")["WorkDate"].nunique()
    cov_obs = df[df.di < b].groupby("PROVNUM")["WorkDate"].nunique()
    cov_dep = df[df.di >= b].groupby("PROVNUM")["WorkDate"].nunique()
    both = set(cov_all[cov_all >= int(0.97 * n_days)].index)
    pros = set(cov_obs[cov_obs >= int(0.97 * b)].index)

    # ---------------------------------------------- 1. analysis-cohort masks
    panel = np.load(os.path.join(ROOT, "cache", "panel.npz"), allow_pickle=True)
    prov = panel["provnum"].astype(str)
    assert set(prov) == both, "analysis cohort does not reproduce cache/panel.npz"
    p2i = {p: i for i, p in enumerate(prov)}
    sub = df[df.PROVNUM.isin(both)]
    present = np.zeros((len(prov), n_days), bool)
    present[sub.PROVNUM.map(p2i).to_numpy(), sub.di.to_numpy()] = True
    right = np.stack([right_anchor(present[i], b) for i in range(len(prov))])
    np.savez_compressed(os.path.join(RES, "nursing_imputation_masks.npz"),
                        imputed=~present, right=right, boundary=b, dates=dates)

    # ---------------------------------------------- 2. prospective cohort panel
    pf = np.sort(np.array(list(pros)))
    q2i = {p: i for i, p in enumerate(pf)}
    sp = df[df.PROVNUM.isin(pros)]
    X = np.full((len(pf), n_days, 4), np.nan, np.float32)
    C = np.full((len(pf), n_days), np.nan, np.float32)
    fi, di = sp.PROVNUM.map(q2i).to_numpy(), sp.di.to_numpy()
    cen = sp.MDScensus.to_numpy()
    C[fi, di] = cen
    for k, col in enumerate(["RN", "LPN", "CNA", "TOTAL"]):
        X[fi, di, k] = sp[col].to_numpy() / cen
    pres_p = ~np.isnan(X[:, :, 0])
    # observed era: interpolate as in the analysis cohort (>=97% present there).
    # deployment era: fill ONLY short internal gaps (<=3 days); longer gaps and
    # leading/trailing absences stay NaN and their windows are excluded.
    Xo = X[:, :b].copy(); Co = C[:, :b].copy()
    for k in range(4):
        Xo[:, :, k] = interp_era(Xo[:, :, k], b)
    Co = interp_era(Co, b)
    X[:, :b], C[:, :b] = Xo, Co
    for i in range(len(pf)):
        miss = np.flatnonzero(~pres_p[i, b:]) + b
        if miss.size == 0:
            continue
        for r in np.split(miss, np.flatnonzero(np.diff(miss) != 1) + 1):
            lo, hi = r[0], r[-1]
            if lo > b and hi < n_days - 1 and (hi - lo + 1) <= 3:
                for k in range(4):
                    X[i, lo:hi + 1, k] = np.interp(np.arange(lo, hi + 1),
                                                   [lo - 1, hi + 1],
                                                   [X[i, lo - 1, k], X[i, hi + 1, k]])
    meta = sp.groupby("PROVNUM").agg(STATE=("STATE", lambda s: s.mode().iloc[0]),
                                     FIPS=("COUNTY_FIPS", lambda s: s.mode().iloc[0]),
                                     medcen=("MDScensus", "median")).loc[pf]
    np.savez_compressed(os.path.join(ROOT, "cache", "panel_prospective.npz"),
                        X=X, census=C, dates=dates, provnum=pf.astype("U6"),
                        state=meta.STATE.to_numpy().astype("U2"),
                        county_fips=meta.FIPS.to_numpy().astype("U5"),
                        medcensus=meta.medcen.to_numpy().astype(np.float32),
                        observed=pres_p)

    dep_cov = cov_dep.reindex(pf).fillna(0).to_numpy() / (n_days - b)
    rows = dict(
        analysis_cohort_both_years=len(both),
        prospective_cohort_2024_only=len(pros),
        in_prospective_not_analysis=len(pros - both),
        in_analysis_not_prospective=len(both - pros),
        prospective_with_no_2025_data=int((dep_cov == 0).sum()),
        prospective_with_2025_coverage_below_97pct=int((dep_cov < 0.97).sum()),
        imputed_cells_analysis_cohort=int((~present).sum()),
        pct_imputed_cells=100 * float((~present).mean()),
    )
    pd.DataFrame([rows]).to_csv(os.path.join(RES, "nursing_cohort_audit.csv"),
                                index=False)
    print(pd.Series(rows).to_string())


if __name__ == "__main__":
    main()
