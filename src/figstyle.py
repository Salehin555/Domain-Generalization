"""
Shared publication figure style: one coherent visual system for every figure.
Colorblind-safe (Okabe-Ito), thin marks, recessive axes, 600-dpi vector PDF,
editable text. Titles are descriptive (never 'Figure N').
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe categorical palette (fixed, peer-reviewed)
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", purple="#CC79A7",
          vermilion="#D55E00", sky="#56B4E9", yellow="#F0E442", black="#222222",
          grey="#8C8C8C")

# fixed method -> color/marker mapping (color follows the entity, never its rank)
METHOD_STYLE = {
    "ERM-DLinear": dict(c="#9AA0A6", m="o"),
    "ERM-MLP":     dict(c="#5F6368", m="o"),
    "Autoformer":  dict(c="#E69F00", m="D"),
    "FEDformer":   dict(c="#D55E00", m="D"),
    "Wave-Mask":   dict(c="#CC79A7", m="s"),
    "IRM":         dict(c="#56B4E9", m="^"),
    "FixedBand":   dict(c="#009E73", m="v"),
    "CORAL":       dict(c="#44AA99", m="P"),
    "DANN":        dict(c="#999933", m="X"),
    "GroupDRO":    dict(c="#882255", m="<"),
    "CD-ISG-TS":   dict(c="#0072B2", m="*"),
}
CAT_COLOR = {"RN": "#0072B2", "LPN": "#E69F00", "CNA": "#009E73", "TOTAL": "#5F6368"}
INK = "#1A1A1A"; MUTED = "#6B6B6B"; FAINT = "#B8B8B8"
SEQ = "Blues"          # single-hue sequential
DIVERGE = "RdBu_r"     # diverging for deltas


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,       # editable/embedded TrueType
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "medium",
        "axes.labelsize": 8.5, "axes.labelcolor": INK,
        "axes.edgecolor": "#3A3A3A", "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": 7.6, "legend.frameon": False,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.1,
        "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
        "axes.grid": False,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def soft_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color="#ECECEC", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def title_block(ax, title, subtitle=None):
    """Descriptive title (bold) + optional grey subtitle, left-aligned."""
    ax.set_title(title, loc="left", fontsize=10, fontweight="semibold",
                 color=INK, pad=10 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.8, color=MUTED)


def save(fig, path, keep_titles=False):
    """Write the figure. Figure-level titles are stripped by default: in the
    article every figure already carries a numbered LaTeX caption, so an in-image
    title duplicates it. Axes-level titles are kept, since those label individual
    panels and carry information the caption does not."""
    if not keep_titles:
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_visible(False)
        for _t in list(fig.texts):
            _t.set_visible(False)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)                 # 600-dpi PDF
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight",
                pad_inches=0.03, dpi=150)                                    # preview PNG
    plt.close(fig)
    print("wrote", path)
