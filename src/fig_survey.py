"""
Reducibility survey figure (three panels), four datasets.

A: reducibility varies systematically ACROSS PANELS and forms a gradient rather
   than a binary split; every random control sits at zero, which is what validates
   the procedure.
B: the apparent gap, the quantity a practitioner would normally inspect, does not
   rank splits by reducibility.
C: a training-free spectral divergence does, which is what makes the decision cheap.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from src.figstyle import set_style, OI, INK, FAINT, save, soft_grid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "figures")
set_style()

PC = {"Nursing": OI["blue"], "Traffic": OI["green"],
      "Solar": OI["purple"], "Electricity": OI["orange"]}


def main():
    d = pd.read_csv(os.path.join(OUT, "reducibility_survey.csv"))
    g = (d.groupby(["panel", "split"])
           .agg(red=("reducible_gap_pct", "mean"), app=("apparent_gap_pct", "mean"),
                spec=("spec_div", "first")).reset_index())
    beh = g[g.split != "random control"]
    ctl = g[g.split == "random control"]
    order = beh.groupby("panel").red.mean().sort_values().index.tolist()

    fig = plt.figure(figsize=(12.4, 4.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34,
                          top=0.74, bottom=0.17, left=0.075, right=0.985)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    # ---------------- A: per-panel gradient ----------------
    rng = np.random.default_rng(0)
    for i, p in enumerate(order):
        v = beh[beh.panel == p].red.values
        axA.scatter(rng.normal(i, 0.075, len(v)), v, s=52, color=PC[p],
                    edgecolor="white", linewidth=0.7, zorder=3)
        axA.hlines(v.mean(), i - 0.28, i + 0.28, color=PC[p], lw=2.4, zorder=4)
        c = ctl[ctl.panel == p].red.values
        axA.scatter([i], c, marker="_", s=260, color=INK, lw=1.8, zorder=5)
    axA.axhline(0, color=INK, lw=1.0)
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(order, fontsize=8.6)
    axA.set_ylabel("measured headroom (%)")
    axA.set_title("Measured headroom differs across panels", loc="left",
                  fontweight="semibold", fontsize=10)
    axA.scatter([], [], marker="_", s=160, color=INK, lw=1.8,
                label="random control")
    axA.legend(frameon=False, fontsize=7.8, loc="upper left")
    soft_grid(axA)

    # ---------------- B / C: predictors ----------------
    # Correlations are scored on the 32 BEHAVIORAL splits only. The random
    # controls are constructed to sit at ~zero headroom and ~zero divergence, so
    # including them adds a cluster at the origin that inflates any association.
    for ax, x, name, extra in [
            (axB, "app", "Apparent gap does not predict measured headroom",
             "   [what you would normally look at]"),
            (axC, "spec", "A training-free statistic does predict it", "")]:
        r, p = pearsonr(beh[x], beh.red)
        rs, ps = spearmanr(beh[x], beh.red)
        for p_, sub in beh.groupby("panel"):
            ax.scatter(sub[x], sub.red, s=44, color=PC[p_], label=p_,
                       edgecolor="white", linewidth=0.6, zorder=3)
        if x == "spec":
            b1, b0 = np.polyfit(beh[x], beh.red, 1)
            xs = np.linspace(beh[x].min(), beh[x].max(), 50)
            ax.plot(xs, b0 + b1 * xs, ls="--", lw=1.2, color=INK, zorder=2)
        ax.axhline(0, color=FAINT, lw=0.9)
        ax.set_xlabel(("apparent gap (%)" if x == "app"
                       else "spectral divergence between groups") + extra)
        ax.set_ylabel("measured headroom (%)")
        ax.set_title(f"{name}\nSpearman = {rs:+.2f} (p = {ps:.3f})",
                     loc="left", fontweight="semibold", fontsize=9.4)
        soft_grid(ax)
    axC.legend(frameon=False, fontsize=7.4, loc="upper left", ncol=2)

    fig.suptitle("How much headroom is measured? 36 splits across four panels",
                 x=0.006, y=0.965, ha="left", fontsize=11.4,
                 fontweight="semibold", color=INK)
    save(fig, os.path.join(FIG, "reducibility_survey.pdf"))


if __name__ == "__main__":
    main()
