"""Rigorous paired statistical testing of CD-ISG-TS vs each baseline on the
full-scale results, with a worked Wilcoxon signed-rank example for the write-up.

Pairing unit: (facility, seed) -- the same facility+seed is forecast by both methods,
so their per-facility MAEs are naturally paired. We use the Wilcoxon signed-rank test
(non-parametric, no normality assumption) and Holm-Bonferroni correction across the
multiple baselines. Effect size = matched-pairs rank-biserial correlation.
"""
import numpy as np, pandas as pd, os
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
df = pd.read_csv(os.path.join(OUT, "fullscale_per_facility.csv"))

OURS = "CD-ISG-TS"
BASELINES = ["ERM-DLinear", "Autoformer", "FEDformer", "Wave-Mask", "FixedBand",
             "GroupDRO", "IRM", "CORAL", "DANN", "ERM-MLP"]

def paired(proto, base):
    sub = df[df.protocol == proto]
    piv = sub.pivot_table(index=["fac", "seed"], columns="method", values="mae_total")
    a = piv[[OURS, base]].dropna()
    o = a[OURS].to_numpy(); b = a[base].to_numpy()
    d = b - o                                   # positive => ours better (lower MAE)
    n = len(d)
    nz = d[d != 0]
    try:
        W, p = wilcoxon(o, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        W, p = np.nan, 1.0
    win = (o < b).mean()
    # matched-pairs rank-biserial effect size from signed ranks
    r = np.abs(nz).argsort().argsort() + 1.0    # ranks of |d|
    Rpos = r[nz > 0].sum(); Rneg = r[nz < 0].sum()
    rb = (Rpos - Rneg) / (Rpos + Rneg)          # in [-1, 1]
    return dict(baseline=base, n=n, win_pct=100*win, med_impr=np.median(d),
                mean_impr=d.mean(), W=W, p=p, effect_rb=rb)

rows = {}
for proto in ["LFO", "LFO-T", "LSO-T"]:
    res = [paired(proto, base) for base in BASELINES]
    # Holm-Bonferroni across baselines
    res = sorted(res, key=lambda r: r["p"])
    m = len(res)
    for i, r in enumerate(res):
        r["p_holm"] = min(1.0, r["p"] * (m - i))
        r["sig"] = "yes" if r["p_holm"] < 0.05 else "no"
    rows[proto] = pd.DataFrame(res)

out = pd.concat([v.assign(protocol=k) for k, v in rows.items()])[
    ["protocol", "baseline", "n", "win_pct", "med_impr", "mean_impr", "W", "p", "p_holm", "effect_rb", "sig"]]
out.to_csv(os.path.join(OUT, "wilcoxon_fullscale.csv"), index=False)

pd.set_option("display.width", 200)
print("=== Wilcoxon signed-rank, CD-ISG-TS vs baselines (LFO-T) ===")
show = rows["LFO-T"].sort_values("mean_impr", ascending=False)
for _, r in show.iterrows():
    print(f"  {r.baseline:12s} n={r.n:5d} win={r.win_pct:5.1f}%  mean_impr={r.mean_impr:+.5f} "
          f"median={r.med_impr:+.5f}  p_holm={r.p_holm:.1e}  rank-biserial={r.effect_rb:+.3f}  sig={r.sig}")

# ---------------------------------------------------------------------------
# Full-data test details for CD-ISG-TS vs ERM-DLinear (the mandatory baseline)
sub = df[df.protocol == "LFO-T"]
pv = sub.pivot_table(index=["fac", "seed"], columns="method", values="mae_total")[
    [OURS, "ERM-DLinear"]].dropna()
o = pv[OURS].to_numpy(); b = pv["ERM-DLinear"].to_numpy(); d = b - o
nz = d[d != 0]; n = len(nz)
rk = np.abs(nz).argsort().argsort() + 1.0
Rpos = rk[nz > 0].sum(); Rneg = rk[nz < 0].sum()
Wstat = min(Rpos, Rneg)
muW = n*(n+1)/4.0; sdW = np.sqrt(n*(n+1)*(2*n+1)/24.0)
z = (Wstat - muW)/sdW
print("\n=== FULL-DATA Wilcoxon (CD-ISG-TS vs ERM-DLinear, LFO-T) ===")
print(f"  n(non-zero pairs)={n}  R+={Rpos:.0f}  R-={Rneg:.0f}  W=min={Wstat:.0f}")
print(f"  E[W]={muW:.0f}  SD[W]={sdW:.0f}  z=(W-E)/SD={z:.1f}  win%={(o<b).mean()*100:.1f}")

# ---------------------------------------------------------------------------
# Worked mini-example: representative random sample of 10 test facilities
rng = np.random.default_rng(7)
s0 = df[(df.protocol == "LFO-T") & (df.seed == 0)]
p0 = s0.pivot_table(index="fac", columns="method", values="mae_total")[[OURS, "ERM-DLinear"]].dropna()
best = None
for trial in range(200):                       # pick a sample whose win-rate ~ population 0.70
    samp = p0.sample(10, random_state=trial)
    wr = (samp[OURS] < samp["ERM-DLinear"]).mean()
    if abs(wr - 0.70) < 0.06:
        best = samp; break
ex = (best if best is not None else p0.sample(10, random_state=0)).copy()
ex.columns = ["ours", "erm"]
ex["d"] = ex["erm"] - ex["ours"]
ex["absd"] = ex["d"].abs()
ex["rank"] = ex["absd"].rank()
ex["signed_rank"] = np.sign(ex["d"]) * ex["rank"]
Wp = ex.loc[ex.d > 0, "rank"].sum(); Wn = ex.loc[ex.d < 0, "rank"].sum()
ne = len(ex); muE = ne*(ne+1)/4.0; sdE = np.sqrt(ne*(ne+1)*(2*ne+1)/24.0)
zE = (min(Wp, Wn) - muE)/sdE
print("\n=== Worked example (10 representative test facilities, CD-ISG-TS vs ERM-DLinear) ===")
print(ex.round(4).to_string())
print(f"\n  R+ = {Wp:.1f}   R- = {Wn:.1f}   W = min(R+,R-) = {min(Wp,Wn):.1f}   n = {ne}")
print(f"  E[W]=n(n+1)/4={muE:.1f}   SD[W]=sqrt(n(n+1)(2n+1)/24)={sdE:.2f}   z={zE:.2f}")
ex.round(4).to_csv(os.path.join(OUT, "wilcoxon_worked_example.csv"))
print("\nSaved outputs/wilcoxon_fullscale.csv and wilcoxon_worked_example.csv")
