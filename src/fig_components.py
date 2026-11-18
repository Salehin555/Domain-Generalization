"""
Component ablation across all four panels.

The point of this figure is to keep the two mechanisms from being credited for
each other's effect. For every panel it shows the backbone, each mechanism alone,
and the unified module, under one fixed configuration applied unchanged to all
four panels. The reading is deliberately unflattering where it should be: the
gate contributes interpretability rather than accuracy, and the periodicity
branch contributes accuracy only where the diagnostic found periodic headroom.

The numbers are read from outputs/unified_results.csv at draw time. They were
hard-coded here once, drifted away from that file, and put an unreproducible
FADE column into the paper; deriving them keeps figure, table and data in step.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.figstyle import set_style, OI, INK, MUTED, FAINT, save, soft_grid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
FIG = os.path.join(ROOT, "figures")
set_style()

# Panel-level MEAN measured headroom over the behavioral splits of each panel.
# They are DERIVED at draw time from the same file the manuscript quotes, so
# the figure cannot disagree with the text. This is a different quantity from the
# single anchor-split headroom quoted in the main text for electricity
# (+7.9% on Flat -> WorkWeek); the panel label below says so explicitly.
def panel_mean_headroom():
    d = pd.read_csv(os.path.join(OUT, "reducibility_survey.csv"))
    beh = d[d.split != "random control"]
    per_split = beh.groupby(["panel", "split"]).reducible_gap_pct.mean()
    return per_split.groupby("panel").mean().to_dict()


RED = panel_mean_headroom()
CASE = {"Electricity": "Electricity (reducible)", "Traffic": "Traffic (mild)",
        "Solar": "Solar (reducible)", "Nursing": "Nursing (irreducible)"}
# Keys on the right are the variant labels as written in unified_results.csv and
# must not change; the display names used in the figure are set in LAB below.
VAR = {"erm": "ERM-MLP (backbone)", "gate": "+ sub-band gate",
       "per": "+ periodicity branch", "uni": "Unified CD-ISG-TS"}


def load():
    """Panel-mean MAE per variant, straight from the experiment output."""
    df = pd.read_csv(os.path.join(OUT, "unified_results.csv"))
    m = df.groupby(["case", "variant"]).mae_total.mean()
    return {p: dict(red=RED[p], **{k: float(m[CASE[p], v]) for k, v in VAR.items()})
            for p in RED}


DATA = load()
ORDER = ["Electricity", "Traffic", "Solar", "Nursing"]
LAB = ["backbone", "+ gate\n(mech. 1)", "+ periodicity\n(mech. 2)", "FADE\n(both)"]
KEYS = ["erm", "gate", "per", "uni"]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.5))
    for ax, p in zip(axes, ORDER):
        d = DATA[p]
        base = d["erm"]
        vals = [100 * (base - d[k]) / base for k in KEYS]
        cols = [FAINT, OI["blue"], OI["orange"], OI["green"]]
        ax.bar(range(4), vals, color=cols, width=0.68)
        ax.axhline(0, color=INK, lw=1.0)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.25 if v >= 0 else -0.55), f"{v:+.1f}",
                    ha="center", fontsize=7.6,
                    color=INK, fontweight="semibold" if i == 3 else "normal")
        ax.set_xticks(range(4))
        ax.set_xticklabels(LAB, fontsize=7.0)
        ax.set_title(f"{p}\nmeasured headroom {d['red']:.1f}% (panel mean)",
                     loc="left", fontsize=9.0, fontweight="semibold")
        ax.set_ylim(min(min(vals) - 1.6, -1.6), max(max(vals) + 2.2, 3.0))
        soft_grid(ax)
    axes[0].set_ylabel("error reduction vs backbone (%)")
    fig.suptitle("Error reduction relative to the pooled ERM backbone, shown "
                 "alongside measured target-retraining headroom",
                 x=0.006, y=0.975, ha="left", fontsize=11.4,
                 fontweight="semibold", color=INK)
    fig.text(0.006, 0.905,
             "Bars are error reduction against the backbone. The panel headroom "
             "printed above each plot is a separate model-family-relative "
             "diagnostic, shown for reference and not as an upper bound.",
             ha="left", fontsize=8.0, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, os.path.join(FIG, "component_ablation.pdf"))


if __name__ == "__main__":
    main()
