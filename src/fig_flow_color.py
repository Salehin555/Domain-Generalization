"""
Compact, colorful, unique-design methodological workflow diagram (Figure 1).

A wide, three-panel landscape flow (Data -> Model -> Evaluation) that fits within
roughly half a page. Each panel has a soft vertical gradient wash (eye-soothing,
low-saturation) behind flat, non-overlapping content boxes. Every piece of text is
pure black, set in Times New Roman. 600-dpi vector PDF.
"""
import numpy as np, os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures", "paper"); os.makedirs(FIG, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.style": "normal", "mathtext.default": "regular",
    "savefig.dpi": 600,
})

BLACK = "#000000"

# panel accent identities (echo Figure 2's palette: blue = backbone/data,
# plum-wine = proposed module, sage green = training/validation signal)
BLUE_TOP, BLUE_EDGE   = "#D9E7F3", "#5C82A6"
WINE_TOP, WINE_EDGE   = "#F2DEE8", "#8E5473"
SAGE_TOP, SAGE_EDGE   = "#DCEBDA", "#71916C"
WHITE = "#FFFFFF"
ARROW = "#3A3A3A"

fig, ax = plt.subplots(figsize=(13.0, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 42.5); ax.axis("off")

def T(x, y, s, size=8.5, weight="normal", ha="center", va="center", z=6):
    ax.text(x, y, s, fontsize=size, color=BLACK, fontweight=weight, ha=ha, va=va,
            style="normal", zorder=z)

def box(x, y, w, h, fill, edge, lw=1.2, r=1.0, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=z))

def arrow(x0, y0, x1, y1, color=ARROW, lw=1.6, ms=15, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))

def gradient_panel(x, y, w, h, top_color, r=2.2, z=1):
    """Soft vertical gradient wash (top_color -> white) clipped to a rounded panel."""
    cmap = LinearSegmentedColormap.from_list("g", [top_color, WHITE])
    grad = np.linspace(1, 0, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x, x + w, y, y + h], aspect="auto", cmap=cmap,
                   zorder=z, origin="upper", interpolation="bicubic")
    clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                          transform=ax.transData)
    im.set_clip_path(clip)
    return clip

def panel_border(x, y, w, h, edge, r=2.2, z=3, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc="none", ec=edge, lw=lw, zorder=z))

PANELS = [
    dict(x=1,  edge=BLUE_EDGE, top=BLUE_TOP, title="1.  Panels and Splits",
        steps=[("Four public panels", ("nursing, electricity,", "traffic, solar")),
               ("Scale-normalized target", ("removes entity size,", "keeps composition")),
               ("Uniform split protocol", ("36 leave-group-out splits", "+ random controls"))]),
    dict(x=35, edge=WINE_EDGE, top=WINE_TOP, title="2.  Reducibility Diagnostic",
        steps=[("Transfer versus oracle", ("train on source, train on target,", "one unseen test set")),
               ("Reducible gap  (proposed)", ("how much of the shift", "any method could remove")),
               ("Training-free predictor", ("spectral divergence estimates", "it before any training"))],
        emphasize=1),
    dict(x=69, edge=SAGE_EDGE, top=SAGE_TOP, title="3.  Decision and Forecasting",
        steps=[("Act on the answer", ("near zero: deploy a simple model;", "large: keep searching")),
               ("FADE module", ("gate reads transferable bands;", "periodicity branch converts them")),
               ("Benchmark and validation", ("10 methods, 3 protocols,", "paired significance tests"))]),
]
PW, PY, PH = 30, 2, 39.5

for pi, p in enumerate(PANELS):
    x = p["x"]
    gradient_panel(x, PY, PW, PH, p["top"])
    panel_border(x, PY, PW, PH, p["edge"])
    T(x + PW / 2, PY + PH - 3.2, p["title"], size=13.5, weight="bold")

    step_h, gap = 9.0, 1.9
    top_y = PY + PH - 7.2
    for si, (head, sub) in enumerate(p["steps"]):
        by = top_y - (si + 1) * step_h - si * gap
        emph = p.get("emphasize") == si
        fill = p["top"] if emph else WHITE
        lw = 2.0 if emph else 1.1
        box(x + 1.4, by, PW - 2.8, step_h, fill, p["edge"], lw=lw, r=0.9)
        T(x + PW / 2, by + step_h - 2.7, head, size=11.0, weight="bold")
        # sub may be a single-line string or a manually curated 2-line tuple
        if isinstance(sub, tuple):
            T(x + PW / 2, by + step_h - 5.6, sub[0], size=9.4)
            T(x + PW / 2, by + step_h - 7.8, sub[1], size=9.4)
        else:
            T(x + PW / 2, by + step_h - 5.8, sub, size=9.4)
        if si < len(p["steps"]) - 1:
            arrow(x + PW / 2, by - 0.05, x + PW / 2, by - gap + 0.05, lw=1.3, ms=10)

# ---- inter-panel arrows -----------------------------------------------------
mid_y = PY + PH / 2 + 2.5
arrow(1 + PW + 0.4, mid_y, 35 - 0.4, mid_y, color=BLUE_EDGE, lw=2.2, ms=20)
arrow(35 + PW + 0.4, mid_y, 69 - 0.4, mid_y, color=WINE_EDGE, lw=2.2, ms=20)

out = os.path.join(FIG, "flow_diagram_color.pdf")
fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close(fig)
print("wrote", out)
