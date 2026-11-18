"""Figure 3: error reduction relative to the source-trained ERM-Stem on
the four benchmark anchor splits, all rows on one common entity manifest
(results/anchor_table.csv, manifest A, primary endpoint), with that split's
target-retraining headroom drawn as a reference line (different quantity, not a
bound). Error bars: paired entity-bootstrap 95% interval.
Writes Manuscript/component_ablation.pdf (600 dpi).
"""
import json
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.figstyle import FAINT, INK, OI, save, set_style, soft_grid   # noqa: E402

set_style()
CASES = [("Electricity", "Flat -> WorkWeek", "Electricity, Flat$\\to$WorkWeek"),
         ("Electricity", "WorkWeek -> Flat", "Electricity, WorkWeek$\\to$Flat"),
         ("Traffic", "Flat -> WorkWeek", "Traffic, Flat$\\to$WorkWeek"),
         ("Solar", "Flat -> WorkWeek", "Solar, Flat$\\to$WorkWeek")]
CLASSICAL = ("Persistence (last value)", "Holt-Winters, additive weekly season",
             "STL + ETS, damped trend", "Theta, weekly deseasonalized",
             "SARIMA(1,0,0)(0,1,1)7, fixed order")
ROWS = [("best of four DG baselines (hindsight)", "__dg", OI["blue"]),
        ("best classical local model (hindsight)", "__cl", OI["sky"] if "sky" in OI else OI["blue"]),
        ("seasonal naive", "Seasonal naive", OI["green"]),
        ("blend, source-val. $\\alpha$", "Blend, a chosen on source validation", OI["green"]),
        ("combination incl. Holt-Winters, source-val. weights",
         "Combination: source model, seasonal naive, persistence, Holt-Winters", OI["green"]),
        ("FADE gate only", "FADE gate only (static)", OI["purple"]),
        ("FADE periodicity only", "FADE periodicity only", OI["orange"]),
        ("FADE gate + periodicity", "FADE (gate + periodicity)", OI["orange"])]
REF = "Target-trained reference (ERM-MLP)"


def main():
    T = pd.read_csv(os.path.join(HERE, "results", "anchor_table.csv"))
    T = T[T.manifest == "A"]
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6), sharey=False)
    for ax, (p, sp, title) in zip(axes, CASES):
        g = T[(T.panel == p) & (T.split == sp)].set_index("method")
        vals, lo, hi, cols, labs = [], [], [], [], []
        for lab, meth, col in ROWS:
            if meth == "__dg":
                dg = g.loc[[m for m in ("IRM", "CORAL", "DANN", "GroupDRO") if m in g.index]]
                meth = dg.reduction_mae_cat.idxmax() if len(dg) else None
            elif meth == "__cl":
                cl = g.loc[[m for m in CLASSICAL if m in g.index]]
                meth = cl.reduction_mae_cat.idxmax() if len(cl) else None
            if meth is None or meth not in g.index:        # not run on this case
                vals.append(np.nan); lo.append(0); hi.append(0); cols.append(col)
                labs.append(lab); continue
            r = g.loc[meth]
            ci = [float(v) for v in str(r.reduction_mae_cat_ci).strip("[]").split(",")]
            vals.append(r.reduction_mae_cat); lo.append(r.reduction_mae_cat - ci[0])
            hi.append(ci[1] - r.reduction_mae_cat); cols.append(col); labs.append(lab)
        y = np.arange(len(vals))[::-1]
        v0 = np.nan_to_num(vals)
        ax.barh(y, v0, xerr=[lo, hi], color=cols, alpha=0.85, height=0.62,
                error_kw=dict(lw=0.8, capsize=2, ecolor=INK))
        for yy, v in zip(y, vals):
            if not np.isfinite(v):
                ax.text(0.3, yy, "not run", fontsize=7, color=FAINT, va="center")
        h = g.loc[REF, "reduction_mae_cat"]
        ax.axvline(h, color=INK, ls="--", lw=1.1)
        ax.set_ylim(-1.25, len(vals) - 0.45)
        ax.text(h, -0.95, f" headroom $H_T$ {h:+.1f}% ", fontsize=7.6, color=INK,
                va="center", ha="left" if h <= np.nanmean(v0 if False else [0]) + 1e9 and
                h < ax.get_xlim()[1] * 0.55 else "right",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
        ax.axvline(0, color=FAINT, lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels(labs if ax is axes[0] else [""] * len(labs),
                                             fontsize=8)
        ax.set_title(title, loc="left", fontsize=9.2, fontweight="semibold")
        ax.set_xlabel("error reduction vs ERM-Stem (%)")
        soft_grid(ax)
    out = os.environ.get("FIG_OUT") or os.path.join(ROOT, "Manuscript", "component_ablation.pdf")
    save(fig, out)
    shutil.copy(out, os.path.join(HERE, "figures", "component_ablation.pdf"))


if __name__ == "__main__":
    main()
