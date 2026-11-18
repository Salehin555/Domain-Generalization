"""Figure 3: survey of 36 splits (saved results: survey_split_table.csv).

(a) Headroom of the 32 behavioral splits by panel, with panel means and the random control.
(b) Apparent gap against headroom (symmetric-log axis).
(c) Autocorrelation distance between the two groups against headroom.
Within-panel Spearman correlations and stratified permutation p-values as reported in Section 5.3/6.1.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from style import INK, MUTED, PANEL_COLOR, PANEL_MARKER, clean, panel_label, setup

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "Manuscript", "figures", "fig3_survey.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ORDER = ["Nursing", "Traffic", "Solar", "Electricity"]

setup()
d = pd.read_csv(os.path.join(RES, "survey_split_table.csv"))
beh = d[d.behavioral]
ctl = d[~d.behavioral]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), gridspec_kw=dict(width_ratios=[1.2, 1.1, 1.1], wspace=0.3))
rng = np.random.default_rng(3)

# (a) strip plot by panel
ax = axes[0]
for k, p in enumerate(ORDER):
    g = beh[beh.panel == p]
    x = k + rng.uniform(-0.16, 0.16, len(g))
    ax.scatter(x, g.headroom_mae_cat, s=16, marker=PANEL_MARKER[p], color=PANEL_COLOR[p],
               edgecolor="white", linewidth=0.5, zorder=3)
    m = g.headroom_mae_cat.mean()
    ax.plot([k - 0.3, k + 0.3], [m, m], color=INK, lw=1.2, zorder=4)
    c = ctl[ctl.panel == p].headroom_mae_cat.iloc[0]
    ax.plot([k - 0.3, k + 0.3], [c, c], color=MUTED, lw=1.0, ls=(0, (2, 1.5)), zorder=4)
ax.axhline(0, color="#8A8A8A", lw=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(["Nursing", "Traffic", "Solar", "Electr."], fontsize=6.8)
ax.set_xlim(-0.55, 3.55)
ax.set_ylabel("Headroom $H_T$ (%)")
ax.set_title("Headroom by panel", pad=11)
ax.legend(handles=[Line2D([], [], color=INK, lw=1.2, label="panel mean"),
                   Line2D([], [], color=MUTED, lw=1.0, ls=(0, (2, 1.5)), label="random control")],
          loc="upper left", frameon=False, handlelength=1.6)
clean(ax)
panel_label(ax, "a", x=-0.2, y=1.13)


def scatter(ax, col, xlabel, title, stat):
    for p in ORDER:
        g = beh[beh.panel == p]
        ax.scatter(g[col], g.headroom_mae_cat, s=16, marker=PANEL_MARKER[p], color=PANEL_COLOR[p],
                   edgecolor="white", linewidth=0.5, zorder=3, label=p)
    ax.axhline(0, color="#8A8A8A", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=11)
    ax.text(0.0, 1.015, stat, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=MUTED)
    clean(ax, grid_axis="both")


ax = axes[1]
scatter(ax, "apparent_gap_mae_cat", "Apparent gap (%, symmetric-log scale)", "Apparent gap vs headroom",
        "within-panel Spearman $+0.08$, $p = 0.697$")
ax.set_xscale("symlog", linthresh=10)
ax.margins(x=0.06)
ax.set_xticks([-50, 0, 10, 100])
ax.set_xticklabels(["$-$50", "0", "10", "100"])
ax.set_ylabel("Headroom $H_T$ (%)")
panel_label(ax, "b", x=-0.16, y=1.13)

ax = axes[2]
scatter(ax, "acf_l2", "Autocorrelation distance $S_{\\mathrm{ACF}}$", "Input screen vs headroom",
        "within-panel Spearman $+0.39$, $p = 0.015$")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55), frameon=False, handletextpad=0.3, borderaxespad=0.2)
panel_label(ax, "c", x=-0.1, y=1.13)
for ax in axes[1:]:
    ax.set_ylim(axes[0].get_ylim())

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT, "splits", len(beh), "controls", len(ctl))
