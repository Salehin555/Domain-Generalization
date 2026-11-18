"""Independent check of the reported numbers.

Recomputes key quantities directly from the raw result files (not from the
analysis scripts) and compares them with the macros in
Manuscript/generated/numbers.tex. Also lists every number typed by hand in the
running text of main.tex, so it can be reviewed.

Run:  python experiments/verify_numbers.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MS = os.path.join(os.path.dirname(HERE), "Manuscript")
M = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}",
                    open(os.path.join(MS, "generated", "numbers.tex"), encoding="utf-8").read()))
ok = []


def chk(label, got, macro, tol):
    want = float(M[macro].replace("{,}", "").replace(",", ""))
    good = abs(got - want) <= tol
    ok.append(good)
    print(f"[{'OK ' if good else 'FAIL'}] {label:55s} macro {macro}={M[macro]:>8}  recomputed={got:+.4f}")


# survey panel means (behavioral splits; per-split mean over seeds)
S = pd.read_csv(os.path.join(RES, "survey_splits.csv"))
b = S[S.split != "random control"].groupby(["panel", "split"]).headroom_mae_cat.mean()
for p, a in (("Nursing", "Nur"), ("Traffic", "Tra"), ("Solar", "Sol"), ("Electricity", "Ele")):
    chk(f"survey mean headroom, {p}", b[p].mean(), f"Sv{a}Mean", 0.006)
bt = S[S.split != "random control"].groupby(["panel", "split"]).headroom_mae_total.mean()
chk("survey mean headroom, Electricity (total)", bt["Electricity"].mean(), "SvEleMeanT", 0.006)

# within-panel Spearman, apparent gap vs headroom
g = S[S.split != "random control"].groupby(["panel", "split"])[["apparent_gap_mae_cat", "headroom_mae_cat"]].mean().reset_index()
r = np.mean([ss.spearmanr(x.apparent_gap_mae_cat, x.headroom_mae_cat)[0] for _, x in g.groupby("panel")])
chk("mean within-panel Spearman, apparent gap", r, "SvAppWithin", 0.006)

# anchors: headroom = mean over seeds of per-seed relative reduction
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "anchor_rows_*.csv"))])
REF = "Target-trained reference (ERM-MLP)"
for split, macro in (("Flat -> WorkWeek", "AnEleFWH"), ("WorkWeek -> Flat", "AnEleWFH")):
    e = R[(R.panel == "Electricity") & (R.split == split)]
    tr = e[(e.method == "ERM-MLP") & (e.manifest == "A")].set_index("seed").mae_cat
    rf = e[e.method == REF].set_index("seed").mae_cat
    chk(f"electricity {split} headroom", (100 * (tr - rf) / tr).mean(), macro, 0.006)
    bl = e[(e.method == "Blend, a chosen on source validation") & (e.manifest == "A")].set_index("seed").mae_cat
    chk(f"electricity {split} blend reduction", (100 * (tr - bl) / tr).mean(),
        "AcEleFWBlend" if split.startswith("Flat") else "AcEleWFBlend", 0.06)
n = R[(R.panel == "Nursing") & R.split.str.startswith("Ownership")]
tr = n[n.method == "ERM-MLP"].set_index("seed").mae_cat
rf = n[n.method == REF].set_index("seed").mae_cat
chk("nursing ownership headroom", (100 * (tr - rf) / tr).mean(), "AnNurOwnH", 0.006)

# temporal
A = pd.read_csv(os.path.join(RES, "temporal_A.csv"))
for p, a in (("Nursing", "Nur"), ("Electricity", "Ele"), ("Traffic", "Tra"), ("Solar", "Sol")):
    x = A[(A.panel == p) & (A.variant == "standard")]
    chk(f"temporal Pearson r, {p}", ss.pearsonr(x.headroom_obs_mae_cat, x.headroom_dep_mae_cat)[0],
        f"Tp{a}R", 0.006)

# screen: confirmation Spearman of the autocorrelation distance (split means)
C = pd.read_csv(os.path.join(RES, "screen_confirmation_splits.csv"))
cm = C.groupby(["panel", "split"]).agg(acf=("acf_l2", "first"), h=("headroom_mae_cat", "mean")).reset_index()
for p, a in (("Nursing", "Nur"), ("Electricity", "Ele"), ("Solar", "Sol"), ("Traffic", "Tra")):
    x = cm[cm.panel == p]
    chk(f"confirmation Spearman (acf), {p}", ss.spearmanr(x.acf, x.h)[0], f"ScConfRho{a}", 0.006)

# gate: mean per-run Pearson between suppression and aligned ICC
G = pd.read_csv(os.path.join(RES, "gate_runs.csv"))
I = pd.read_csv(os.path.join(RES, "gate_icc_aligned.csv"))
icc = np.array([[I[(I.channel == c) & (I.band == bnd)].ICC.iloc[0] for bnd in ("A3", "D3", "D2", "D1")]
                for c in ("RN", "LPN", "CNA")]).ravel()
rs = [np.corrcoef(icc, 1 - np.array([row[f"g_{c}_{bnd}"] for c in ("RN", "LPN", "CNA")
                                     for bnd in ("A3", "D3", "D2", "D1")]))[0, 1]
      for _, row in G.iterrows()]
chk("gate-ICC Pearson, mean over runs", float(np.mean(rs)), "GaRmean", 0.006)

# cohort
co = pd.read_csv(os.path.join(RES, "nursing_cohort_audit.csv")).iloc[0]
chk("prospective cohort size", co.prospective_cohort_2024_only, "CoProsp", 0)

# full benchmark (if available)
fs = glob.glob(os.path.join(RES, "fullscale_light_*.csv"))
if fs and "FsLfotGate" in M:
    F = pd.concat([pd.read_csv(f) for f in fs])
    x = F[F.protocol == "LFO-T"].groupby(["method", "fac"]).mae_cat.mean().unstack(0)
    d = x["ERM-MLP"] - x["FADE gate only (static)"]
    chk("full benchmark LFO-T, gate vs ERM-Stem (%)", 100 * d.mean() / x["ERM-MLP"].mean(), "FsLfotGate", 0.006)

print(f"\n{sum(ok)}/{len(ok)} checks passed")

# numbers typed by hand in main.tex running text (outside tables/macros/labels);
# skipped when the manuscript source is not present (e.g. in the code repository)
if not os.path.exists(os.path.join(MS, "main.tex")):
    print("\nmain.tex not found; hand-typed-number scan skipped")
    sys.exit(0 if all(ok) else 1)
tex = open(os.path.join(MS, "main.tex"), encoding="utf-8").read()
body = tex[tex.index("\\begin{document}"):]
body = re.sub(r"(?<!\\)%.*", "", body)
body = re.sub(r"\\(label|ref|eqref|cite[tp]?|citep|includegraphics|input)\{[^}]*\}", "", body)
hand = sorted(set(re.findall(r"(?<![A-Za-z\\{])[-+]?\d+(?:[.,]\d+)?(?:\{,\}\d+)?\s?\\?%", body)))
print("\nhand-typed percentages in main.tex:", hand)
sys.exit(0 if all(ok) else 1)
