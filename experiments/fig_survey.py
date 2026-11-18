"""Figure 2: the 36-split survey, primary endpoint.

A  headroom of the 32 behavioral splits by panel, panel mean, random control
B  apparent gap vs headroom
C  autocorrelation distance (principal screen) vs headroom
Association statistics printed on B and C are the dependence-aware ones:
mean within-panel Spearman with the stratified pair-preserving permutation
p-value (survey_exact_within.json). Source: results/survey_split_table.csv.
Writes Manuscript/reducibility_survey.pdf (600 dpi) and a copy in figures/.
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
PC = {"Nursing": OI["blue"], "Traffic": OI["green"], "Solar": OI["purple"],
      "Electricity": OI["orange"]}


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    t = pd.read_csv(os.path.join(HERE, "results", "survey_split_table.csv"))
    ex = json.load(open(os.path.join(HERE, "results", "survey_exact_within.json")))
    beh, ctl = t[t.behavioral], t[~t.behavioral]
    order = beh.groupby("panel").headroom_mae_cat.mean().sort_values().index.tolist()
    fig = plt.figure(figsize=(12.4, 4.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34,
                          top=0.80, bottom=0.17, left=0.075, right=0.985)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    rng = np.random.default_rng(0)
    for i, p in enumerate(order):
        v = beh[beh.panel == p].headroom_mae_cat.values
        axA.scatter(rng.normal(i, 0.075, len(v)), v, s=52, color=PC[p],
                    edgecolor="white", linewidth=0.7, zorder=3)
        axA.hlines(v.mean(), i - 0.28, i + 0.28, color=PC[p], lw=2.4, zorder=4)
        axA.scatter([i], ctl[ctl.panel == p].headroom_mae_cat.values, marker="_",
                    s=260, color=INK, lw=1.8, zorder=5)
    axA.axhline(0, color=INK, lw=1.0)
    axA.set_xticks(range(len(order))); axA.set_xticklabels(order, fontsize=8.6)
    axA.set_ylabel("target-retraining headroom (%)")
    axA.set_title("Headroom by panel (bar = panel mean)", loc="left",
                  fontweight="semibold", fontsize=10)
    axA.scatter([], [], marker="_", s=160, color=INK, lw=1.8, label="random control")
    axA.legend(frameon=False, fontsize=7.8, loc="upper left")
    soft_grid(axA)
    for ax, x, key, lab in [
            (axB, "apparent_gap_mae_cat", "apparent_gap|mae_cat",
             "increase in target error, apparent gap (%)"),
            (axC, "acf_l2", "acf_l2|mae_cat",
             "autocorrelation distance between groups")]:
        for p_, sub in beh.groupby("panel"):
            ax.scatter(sub[x], sub.headroom_mae_cat, s=44, color=PC[p_], label=p_,
                       edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(0, color=FAINT, lw=0.9)
        ax.set_xlabel(lab); ax.set_ylabel("target-retraining headroom (%)")
        e = ex[key]
        ax.set_title(f"mean within-panel Spearman = {e['mean_within_panel_spearman']:+.2f}\n"
                     f"(stratified permutation, {fmt_p(e['p_exact'])})",
                     loc="left", fontweight="semibold", fontsize=9.2)
        soft_grid(ax)
    axB.set_xscale("symlog", linthresh=10)
    axC.legend(frameon=False, fontsize=7.4, loc="upper left", ncol=2)
    out = os.path.join(ROOT, "Manuscript", "reducibility_survey.pdf")
    save(fig, out)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    shutil.copy(out, os.path.join(HERE, "figures", "reducibility_survey.pdf"))


if __name__ == "__main__":
    main()
