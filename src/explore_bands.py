"""
Step 3 pre-study / caution #4b: an inexpensive exploratory pass on the real PBJ
data to check *how much cross-facility variance actually exists per frequency
sub-band* -- i.e. whether there is a real domain-generalization effect to find,
and which sub-bands carry facility-specific vs. shared structure.

Method
------
For every facility we slice its 731-day per-resident-day series into non-overlapping
64-day windows, instance-normalize each window (remove per-window mean/std -> keep
*shape* only, exactly as the model will via RevIN), take a level-4 db4 DWT, and read
off the relative energy in each sub-band [A4, D4, D3, D2, D1].

For each sub-band we then decompose the variation of its relative energy into a
BETWEEN-facility part and a WITHIN-facility part and form the intraclass correlation

        ICC_j = Var_between / (Var_between + Var_within).

High ICC  -> the band level is strongly determined by *which facility* you are in
            = domain-specific  -> a DG model should suppress it.
Low  ICC  -> the band is shared across facilities (temporal, calendar-driven)
            = invariant       -> a DG model should preserve it.

We compute this per staff category (RN/LPN/CNA) to test whether category-conditioning
is warranted (RQ2): if the ICC profile differs by category, a category-conditioned
invariance estimator is justified.

Outputs (for later figure + tables):
  cache/explore.npz, outputs/band_icc.csv
"""
import numpy as np, pandas as pd, pywt, os

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
d = np.load(os.path.join(ROOT, "cache", "panel.npz"), allow_pickle=True)
X = d["X"]                      # [n_fac, 731, 4]  RN,LPN,CNA,TOTAL
state = d["state"]; medcen = d["medcensus"]
n_fac, T, _ = X.shape
CH = ["RN", "LPN", "CNA", "TOTAL"]
BANDS = ["A4", "D4", "D3", "D2", "D1"]
W = 64                          # window length
WAVELET, LEVEL = "db4", 4

def band_energies(win):
    """relative energy per sub-band of a 1-D window (instance-normalized)."""
    m, s = win.mean(), win.std() + 1e-6
    w = (win - m) / s
    coeffs = pywt.wavedec(w, WAVELET, level=LEVEL)   # [cA4, cD4, cD3, cD2, cD1]
    e = np.array([np.sum(c**2) for c in coeffs], np.float64)
    return e / (e.sum() + 1e-12)

n_win = T // W
print(f"{n_fac} facilities x {n_win} windows/facility, W={W}", flush=True)

# per-facility mean & var of band energy, per channel
fac_mean = np.zeros((n_fac, 4, len(BANDS)))
fac_var  = np.zeros((n_fac, 4, len(BANDS)))
all_win_energy = {c: [] for c in range(4)}   # for approx freq period annotation
for i in range(n_fac):
    for c in range(4):
        series = X[i, :, c]
        es = np.stack([band_energies(series[k*W:(k+1)*W]) for k in range(n_win)])  # [n_win, 5]
        fac_mean[i, c] = es.mean(0)
        fac_var[i, c]  = es.var(0)
    if (i + 1) % 2000 == 0:
        print("  ", i + 1, flush=True)

# ICC per channel per band
rows = []
icc = np.zeros((4, len(BANDS)))
for c in range(4):
    grand = fac_mean[:, c, :].mean(0)
    var_between = fac_mean[:, c, :].var(0)      # variance of facility means
    var_within  = fac_var[:, c, :].mean(0)      # mean within-facility variance
    icc[c] = var_between / (var_between + var_within + 1e-12)
    for j, b in enumerate(BANDS):
        rows.append(dict(channel=CH[c], band=b, mean_energy=grand[j],
                         var_between=var_between[j], var_within=var_within[j],
                         ICC=icc[c, j]))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "outputs", "band_icc.csv"), index=False)

# approximate day-period covered by each band (daily sampling)
periods = {"A4": ">32 d (trend)", "D4": "16-32 d", "D3": "8-16 d",
           "D2": "4-8 d (weekly)", "D1": "2-4 d (fast noise)"}
print("\nBand   period            TOTAL: mean-energy   ICC(between/within)")
for j, b in enumerate(BANDS):
    print(f"  {b:3s} {periods[b]:18s}  {df[(df.channel=='TOTAL')&(df.band==b)].mean_energy.values[0]:.3f}"
          f"           {icc[3, j]:.3f}")

print("\nICC by category (domain-specificity of each band):")
print(df.pivot_table(index="band", columns="channel", values="ICC").reindex(BANDS).round(3))

np.savez(os.path.join(ROOT, "cache", "explore.npz"),
         fac_mean=fac_mean, fac_var=fac_var, icc=icc,
         bands=np.array(BANDS), ch=np.array(CH),
         state=state, medcen=medcen)
print("\nSaved outputs/band_icc.csv and cache/explore.npz")
