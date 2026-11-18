"""
Step 1 (Methodology): Prepare Data, Domains, and Protocol.

Builds a clean per-facility daily panel from the 8 quarterly CMS PBJ Daily Nurse
Staffing public-use files (CY2024Q1 .. CY2025Q4).

Design decisions grounded in the proposal + its 6 cautionary notes:
  * NORMALIZATION (note #1, "close to critical"): we forecast a *scale-normalized*
    quantity -- nurse Hours Per Resident-Day (HPRD) -- not raw hours, so the learned
    invariance cannot merely capture facility size.
  * Each facility (PROVNUM = CMS Certification Number) is one DOMAIN.
  * Staff categories follow the CMS Five-Star grouping:
        RN  = Hrs_RNDON + Hrs_RNadmin + Hrs_RN
        LPN = Hrs_LPNadmin + Hrs_LPN
        CNA = Hrs_CNA + Hrs_NAtrn + Hrs_MedAide
  * Quality filtering removes facilities with missing census or implausible hours,
    and keeps only facilities with (near-)complete daily coverage over all 8 quarters
    so every domain contributes a comparable, long, continuous series.

Output: cache/panel.npz
  X          : float32 [n_fac, n_days, 4]  channels = [RN, LPN, CNA, TOTAL] HPRD
  census     : float32 [n_fac, n_days]
  dates      : int32  [n_days]             YYYYMMDD
  provnum    : str    [n_fac]
  state      : str    [n_fac]
  county_fips: str    [n_fac]
"""
import numpy as np
import pandas as pd
import os, glob, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache")
os.makedirs(CACHE, exist_ok=True)

FILES = sorted(glob.glob(os.path.join(ROOT, "PBJ_dailynursestaffing_CY*.csv")))
assert len(FILES) == 8, f"expected 8 quarter files, found {len(FILES)}"

USECOLS = ["PROVNUM", "STATE", "COUNTY_FIPS", "WorkDate", "MDScensus",
           "Hrs_RNDON", "Hrs_RNadmin", "Hrs_RN",
           "Hrs_LPNadmin", "Hrs_LPN",
           "Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide"]
DTYPE = {"PROVNUM": str, "STATE": str, "COUNTY_FIPS": str, "WorkDate": np.int32,
         "MDScensus": np.float32,
         "Hrs_RNDON": np.float32, "Hrs_RNadmin": np.float32, "Hrs_RN": np.float32,
         "Hrs_LPNadmin": np.float32, "Hrs_LPN": np.float32,
         "Hrs_CNA": np.float32, "Hrs_NAtrn": np.float32, "Hrs_MedAide": np.float32}

def load():
    parts = []
    for f in FILES:
        print("  reading", os.path.basename(f), flush=True)
        df = pd.read_csv(f, usecols=USECOLS, dtype=DTYPE, encoding="latin-1")
        df["RN"]  = df["Hrs_RNDON"] + df["Hrs_RNadmin"] + df["Hrs_RN"]
        df["LPN"] = df["Hrs_LPNadmin"] + df["Hrs_LPN"]
        df["CNA"] = df["Hrs_CNA"] + df["Hrs_NAtrn"] + df["Hrs_MedAide"]
        df = df[["PROVNUM", "STATE", "COUNTY_FIPS", "WorkDate", "MDScensus",
                 "RN", "LPN", "CNA"]]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

def main():
    print("Loading 8 quarterly files ...", flush=True)
    df = load()
    print("  raw rows:", len(df), flush=True)

    # canonical calendar over the whole 2-year span
    dates = np.sort(df["WorkDate"].unique()).astype(np.int32)
    n_days = len(dates)
    date_to_idx = {int(d): i for i, d in enumerate(dates)}
    print("  calendar days:", n_days, "from", dates[0], "to", dates[-1], flush=True)

    df["TOTAL"] = df["RN"] + df["LPN"] + df["CNA"]

    # ---- quality gates at the row level -------------------------------------
    # positive census (needed for HPRD), plausible hours
    n0 = len(df)
    df = df[df["MDScensus"] >= 5]                       # avoid noisy ratios from tiny census
    df = df[(df["TOTAL"] >= 0) & (df["RN"] >= 0) & (df["LPN"] >= 0) & (df["CNA"] >= 0)]
    hprd_total = df["TOTAL"] / df["MDScensus"]
    df = df[(hprd_total > 0.3) & (hprd_total < 12.0)]   # implausible total HPRD removed
    print(f"  row-level quality filter kept {len(df)}/{n0}", flush=True)

    # ---- keep facilities with (near) complete daily coverage ----------------
    cov = df.groupby("PROVNUM")["WorkDate"].nunique()
    keep_fac = cov[cov >= int(0.97 * n_days)].index      # >=97% of calendar days present
    df = df[df["PROVNUM"].isin(keep_fac)]
    fac_list = np.sort(df["PROVNUM"].unique())
    print("  facilities with near-complete coverage:", len(fac_list), flush=True)

    fac_to_idx = {f: i for i, f in enumerate(fac_list)}
    n_fac = len(fac_list)

    # ---- assemble dense arrays ----------------------------------------------
    X = np.full((n_fac, n_days, 4), np.nan, np.float32)   # RN,LPN,CNA,TOTAL (HPRD)
    C = np.full((n_fac, n_days), np.nan, np.float32)      # census

    fi = df["PROVNUM"].map(fac_to_idx).to_numpy()
    di = df["WorkDate"].map(date_to_idx).to_numpy()
    cen = df["MDScensus"].to_numpy()
    C[fi, di] = cen
    X[fi, di, 0] = df["RN"].to_numpy()  / cen
    X[fi, di, 1] = df["LPN"].to_numpy() / cen
    X[fi, di, 2] = df["CNA"].to_numpy() / cen
    X[fi, di, 3] = df["TOTAL"].to_numpy() / cen

    # ---- gap-fill short missing runs by linear interp along time ------------
    # (needed for windowed forecasting; only small gaps remain after 97% filter)
    #
    # np.interp is TWO-SIDED, so a missing run that straddles the temporal
    # deployment boundary would be filled using a deployment-era observation and
    # carry future information back into the observed era. We therefore fill each
    # era independently: within an era np.interp clamps at the edges, so no value
    # crosses the boundary in either direction. Compared with a single two-sided
    # fill, this changes 26 of 13,402 facilities and 96 observed-era facility-days.
    BOUNDARY_DATE = 20250101
    b = int(np.searchsorted(dates, BOUNDARY_DATE))

    def interp_fill(a):
        n, T = a.shape
        idx = np.arange(T)
        for i in range(n):
            row = a[i]
            m = ~np.isnan(row)
            if m.all() or not m.any():
                continue
            a[i] = np.interp(idx, idx[m], row[m])
        return a

    def interp_fill_by_era(a):
        """Fill within the observed era and the deployment era separately."""
        a[:, :b] = interp_fill(a[:, :b])
        a[:, b:] = interp_fill(a[:, b:])
        return a

    for k in range(4):
        X[:, :, k] = interp_fill_by_era(X[:, :, k])
    C = interp_fill_by_era(C)

    # facility metadata (mode state / fips)
    meta = df.groupby("PROVNUM").agg(STATE=("STATE", lambda s: s.mode().iloc[0]),
                                     FIPS=("COUNTY_FIPS", lambda s: s.mode().iloc[0]),
                                     medcen=("MDScensus", "median"))
    meta = meta.loc[fac_list]

    np.savez_compressed(
        os.path.join(CACHE, os.environ.get("PANEL_OUT", "panel.npz")),
        X=X, census=C, dates=dates,
        provnum=fac_list.astype("U6"),
        state=meta["STATE"].to_numpy().astype("U2"),
        county_fips=meta["FIPS"].to_numpy().astype("U5"),
        medcensus=meta["medcen"].to_numpy().astype(np.float32),
    )
    print("Saved cache/panel.npz  X shape:", X.shape, flush=True)
    print("  states:", len(np.unique(meta['STATE'])),
          " median census range:", float(meta['medcen'].min()), "-", float(meta['medcen'].max()),
          flush=True)

if __name__ == "__main__":
    main()
