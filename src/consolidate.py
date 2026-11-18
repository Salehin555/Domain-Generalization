"""Consolidate raw outputs into clean, paper-ready CSV tables."""
import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

_fs = os.path.join(OUT, "fullscale_main.csv")
FULL = os.path.exists(_fs)
main = pd.read_csv(_fs if FULL else os.path.join(OUT, "main_results.csv"))
_sig = os.path.join(OUT, "fullscale_significance.csv" if FULL else "significance.csv")
sig = pd.read_csv(_sig)
# fullscale significance is stored with ref=CD-ISG-TS implicitly (columns baseline/ours_minus_baseline)
if FULL and "ref" not in sig.columns:
    sig = sig.rename(columns={"baseline": "method", "ours_minus_baseline": "delta_mean",
                              "ours_win_pct": "ref_better_frac"})
    sig["ref"] = "CD-ISG-TS"; sig["delta_mean"] = -sig["delta_mean"]  # -> other - ours

ORDER = ["ERM-DLinear", "ERM-MLP", "Autoformer", "FEDformer", "IRM", "CORAL",
         "DANN", "GroupDRO", "Wave-Mask", "FixedBand", "CD-ISG-TS"]
PROTO = ["LFO", "LFO-T", "LSO-T"]

# ---- Table 1: main OOD MAE (mean +- std) per method x protocol, + worst-decile
rows = []
for m in ORDER:
    r = {"Method": m}
    for p in PROTO:
        s = main[(main.method == m) & (main.protocol == p)]
        if len(s):
            s = s.iloc[0]
            r[f"{p}_MAE"] = f"{s['mae_total_mean']:.4f} +/- {s['mae_total_std']:.4f}"
            if "mae_total_worst10_mean" in s.index:
                r[f"{p}_worst10"] = f"{s['mae_total_worst10_mean']:.4f}"
    rows.append(r)
t1 = pd.DataFrame(rows)
t1.to_csv(os.path.join(OUT, "table_main_benchmark.csv"), index=False)

# ---- Table 2: per-category MAE for CD-ISG-TS vs ERM-DLinear (LFO-T)
rows = []
for m in ["ERM-DLinear", "ERM-MLP", "CD-ISG-TS"]:
    s = main[(main.method == m) & (main.protocol == "LFO-T")].iloc[0]
    rows.append(dict(Method=m,
                     RN=f"{s['mae_RN_mean']:.4f}", LPN=f"{s['mae_LPN_mean']:.4f}",
                     CNA=f"{s['mae_CNA_mean']:.4f}", Total=f"{s['mae_total_mean']:.4f}",
                     RMSE=f"{s['rmse_total_mean']:.4f}", sMAPE=f"{s['smape_total_mean']:.4f}"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_per_category.csv"), index=False)

# ---- Table 3: significance of CD-ISG-TS vs each baseline (paired Wilcoxon)
rows = []
s = sig[sig.ref == "CD-ISG-TS"]
for p in PROTO:
    for m in ORDER:
        if m == "CD-ISG-TS":
            continue
        sub = s[(s.protocol == p) & (s.method == m)]
        if not len(sub):
            continue
        sub = sub.iloc[0]
        # delta_mean = other - ours ; positive => ours better ; ref_better_frac = ours wins
        rows.append(dict(protocol=p, baseline=m,
                         ours_minus_baseline=f"{-sub['delta_mean']:+.5f}",
                         ours_win_pct=f"{100*sub['ref_better_frac']:.1f}%",
                         p_value=f"{sub['pvalue']:.2e}",
                         significant=("yes" if sub['pvalue'] < 0.05 else "no")))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table_significance_vs_ours.csv"), index=False)

print("=== Table 1: main benchmark ===")
print(t1.to_string(index=False))
print("\n=== Table 3 (LFO-T): CD-ISG-TS vs baselines ===")
d = pd.DataFrame(rows)
print(d[d.protocol == "LFO-T"].to_string(index=False))
print("\nWrote table_main_benchmark.csv, table_per_category.csv, table_significance_vs_ours.csv")
