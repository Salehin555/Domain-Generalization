"""Figure 4: error reduction relative to the source-trained ERM-Stem on the
four benchmark anchors (common manifest, primary metric, mean over five seeds,
paired entity-bootstrap 95% intervals), with methods that can be deployed without
target data separated from hindsight references.

Inputs: results/anchor_table.csv (case study), results/dg_tuned_rows_*.csv
(tuned domain-generalization baselines; their reduction is computed per seed
against the ERM run of the same script). Output: FIG_OUT (PDF, 600 dpi).
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.figstyle import FAINT, INK, OI, save, set_style, soft_grid  # noqa: E402

set_style()
plt.rcParams.update({"font.size": 9.5})
CASES = [("Electricity", "Flat -> WorkWeek", "Electricity, Flat$\\to$WorkWeek"),
         ("Electricity", "WorkWeek -> Flat", "Electricity, WorkWeek$\\to$Flat"),
         ("Traffic", "Flat -> WorkWeek", "Traffic, Flat$\\to$WorkWeek"),
         ("Solar", "Flat -> WorkWeek", "Solar, Flat$\\to$WorkWeek")]
CLASSICAL = ("Persistence (last value)", "Holt-Winters, additive weekly season",
             "STL + ETS, damped trend", "Theta, weekly deseasonalized",
             "SARIMA(1,0,0)(0,1,1)7, fixed order")
DEPLOY = [("Seasonal naive", "Seasonal naive"),
          ("Blend with seasonal naive", "Blend, a chosen on source validation"),
          ("Combination incl. Holt-Winters", "Combination: source model, seasonal naive, persistence, Holt-Winters"),
          ("FADE (gate + periodicity)", "FADE (gate + periodicity)"),
          ("Best tuned DG baseline", "__dgtuned")]
HIND = [("Best DG baseline (hindsight)", "__dg"),
        ("Best local model (hindsight)", "__cl")]
REF = "Target-trained reference (ERM-MLP)"


def tuned_dg():
    fs = glob.glob(os.path.join(HERE, "results", "dg_tuned_rows_*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs])
    out = {}
    for (p, sp), g in d.groupby(["panel", "split"]):
        best = []
        for seed, gs in g.groupby("seed"):
            erm = gs[gs.method == "ERM"].mae_cat.iloc[0]
            dg = gs[gs.method != "ERM"]
            # per method: configuration chosen on source validation
            red = []
            for m, gm in dg.groupby("method"):
                pick = gm.loc[gm.val_mae.idxmin()]
                red.append(100 * (erm - pick.mae_cat) / erm)
            best.append(red)
        best = np.array(best)                     # seeds x methods
        means = best.mean(0)
        k = int(np.argmax(means))
        v = best[:, k]
        out[(p, sp)] = (v.mean(), v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1))
    return out


def main():
    T = pd.read_csv(os.path.join(HERE, "results", "anchor_table.csv"))
    T = T[T.manifest == "A"]
    dgt = tuned_dg()
    rows = DEPLOY + HIND
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    axes = axes.ravel()
    for i, (ax, (p, sp, title)) in enumerate(zip(axes, CASES)):
        g = T[(T.panel == p) & (T.split == sp)].set_index("method")
        vals, err, cols, hatch = [], [[], []], [], []
        for lab, meth in rows:
            hind = (lab, meth) in HIND
            if meth == "__dgtuned":
                if dgt is None or (p, sp) not in dgt:
                    v, lo, hi = np.nan, 0, 0
                else:
                    v, lo, hi = dgt[(p, sp)]
            else:
                if meth == "__dg":
                    dg = g.loc[[m for m in ("IRM", "CORAL", "DANN", "GroupDRO") if m in g.index]]
                    meth = dg.reduction_mae_cat.idxmax()
                elif meth == "__cl":
                    cl = g.loc[[m for m in CLASSICAL if m in g.index]]
                    meth = cl.reduction_mae_cat.idxmax()
                r = g.loc[meth]
                ci = [float(x) for x in str(r.reduction_mae_cat_ci).strip("[]").split(",")]
                v, lo, hi = r.reduction_mae_cat, ci[0], ci[1]
            vals.append(v); err[0].append(max(0, v - lo) if np.isfinite(v) else 0)
            err[1].append(max(0, hi - v) if np.isfinite(v) else 0)
            cols.append("#BBBBBB" if hind else OI["green"])
            hatch.append("////" if hind else "")
        y = np.arange(len(vals))[::-1]
        bars = ax.barh(y, np.nan_to_num(vals), xerr=err, color=cols, height=0.64,
                       edgecolor="#555555", linewidth=0.5,
                       error_kw=dict(lw=0.8, capsize=2, ecolor=INK))
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        h = g.loc[REF, "reduction_mae_cat"]
        ax.axvline(h, color=OI["orange"], ls="--", lw=1.4)
        ax.axvline(0, color=FAINT, lw=0.9)
        ax.axhline(len(HIND) - 0.5, color=FAINT, lw=0.6, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if i % 2 == 0 else [""] * len(rows), fontsize=8.8)
        ax.set_title(f"{title}; headroom {h:+.1f}%", loc="left", fontsize=8.8, fontweight="semibold")
        if i >= 2:
            ax.set_xlabel("Error reduction relative to ERM-Stem (%)", fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
        soft_grid(ax)
    fig.legend(handles=[Patch(facecolor=OI["green"], edgecolor="#555555", label="Deployable: chosen on source validation"),
                        Patch(facecolor="#BBBBBB", edgecolor="#555555", hatch="////",
                              label="Hindsight reference: chosen on the target test set"),
                        plt.Line2D([], [], color=OI["orange"], ls="--", lw=1.4,
                                   label="Target-retraining headroom (separate quantity)")],
               loc="lower center", ncol=2, fontsize=8.4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    out = os.environ.get("FIG_OUT", os.path.join(HERE, "figures", "fig4_components.pdf"))
    save(fig, out)


if __name__ == "__main__":
    main()
