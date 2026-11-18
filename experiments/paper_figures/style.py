"""Shared figure style for the manuscript (Figures 2-4)."""
import matplotlib.pyplot as plt

INK, MUTED, GRID, AXIS = "#222222", "#5F5F5F", "#E6E6E6", "#8A8A8A"
SRC, TGT = "#0072B2", "#D55E00"                      # validated pair (Okabe-Ito)
# panel identity: validated Okabe-Ito set; marker shape is the secondary encoding
PANEL_COLOR = {"Nursing": "#0072B2", "Traffic": "#009E73", "Solar": "#CC79A7", "Electricity": "#E69F00"}
PANEL_MARKER = {"Nursing": "o", "Traffic": "s", "Solar": "^", "Electricity": "D"}
NEUTRAL = "#9A9A9A"


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "savefig.dpi": 600,
        "axes.titleweight": "normal", "axes.titlelocation": "left",
    })


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, s, x=-0.12, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")
