"""
Black-and-white (grayscale) publication figures for the manuscript:
  (1) methodological process-flow diagram (sketch/flowchart style)
  (2) detailed CD-ISG-TS architecture diagram
No color, no italic fonts, 600-dpi vector PDF (+ PNG preview).
"""
import numpy as np, os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures", "paper"); os.makedirs(FIG, exist_ok=True)
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.style": "normal", "mathtext.default": "regular", "savefig.dpi": 600,
})
BLACK = "#000000"; INK = "#111111"; DKGREY = "#444444"; MIDGREY = "#777777"
LTGREY = "#EDEDED"; MDGREY = "#DCDCDC"; WHITE = "#FFFFFF"; BANDBG = "#F6F6F6"

def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), dpi=600, bbox_inches="tight",
                pad_inches=0.06, facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight",
                pad_inches=0.06, facecolor="white")
    plt.close(fig); print("wrote", name)

def T(ax, x, y, s, size=8.2, color=INK, weight="normal", ha="center", va="center", z=6, rot=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style="normal", zorder=z, rotation=rot)

def box(ax, x, y, w, h, fill=WHITE, edge=BLACK, lw=1.2, r=0.6, z=3, style="round"):
    bs = f"round,pad=0,rounding_size={r}" if style == "round" else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=bs, fc=fill, ec=edge, lw=lw, zorder=z))

def arrow(ax, x0, y0, x1, y1, lw=1.3, ms=12, z=5, ls="-", color=BLACK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

def cyl(ax, x, y, w, h, fill=MDGREY, edge=BLACK, lw=1.2, z=3):     # data store (cylinder)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=fill, ec=edge, lw=lw, zorder=z))
    ax.add_patch(matplotlib.patches.Ellipse((x + w/2, y + h), w, h*0.28, fc=fill, ec=edge, lw=lw, zorder=z+1))

def diamond(ax, cx, cy, w, h, fill=WHITE, edge=BLACK, lw=1.2, z=3):
    ax.add_patch(Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                         closed=True, fc=fill, ec=edge, lw=lw, zorder=z))


# ======================================================================= #
#  FIGURE 1 — methodological process flow (three parts)
# ======================================================================= #
def flow():
    fig, ax = plt.subplots(figsize=(9.6, 12.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 130); ax.axis("off")

    def band(y0, y1, title):
        ax.add_patch(Rectangle((1, y0), 98, y1 - y0, fc=BANDBG, ec=DKGREY, lw=1.0,
                               ls=(0, (5, 3)), zorder=1))
        T(ax, 3.2, y1 - 2.4, title, size=10.5, weight="bold", ha="left")

    # ---- Part 1: data ----
    band(92.5, 128.5, "Part 1:  Data preparation")
    cyl(ax, 6, 118, 20, 5.2, fill=MDGREY)
    T(ax, 16, 120.4, "CMS PBJ daily\nnurse staffing\n(8 quarters)", size=7.6)
    cyl(ax, 33, 118, 20, 5.2, fill=MDGREY)
    T(ax, 43, 120.4, "CMS Provider\nInformation\n(ownership, region)", size=7.6)
    box(ax, 62, 117.6, 33, 6.4)
    T(ax, 78.5, 122.0, "Merge by CMS Certification Number", size=8.0, weight="bold")
    T(ax, 78.5, 119.2, "13,387 of 13,402 facilities matched", size=7.4, color=DKGREY)
    arrow(ax, 26.3, 120.6, 61.7, 120.6); arrow(ax, 53.3, 120.6, 61.7, 120.6)

    box(ax, 12, 108.5, 30, 6.2)
    T(ax, 27, 112.9, "Quality filtering", size=8.4, weight="bold")
    T(ax, 27, 109.9, "census >= 5, plausible hours,\n>= 97% daily coverage", size=7.2, color=DKGREY)
    box(ax, 55, 108.5, 33, 6.2)
    T(ax, 71.5, 112.9, "Scale normalization", size=8.4, weight="bold")
    T(ax, 71.5, 109.9, "hours per resident-day (HPRD),\nper staff category  RN, LPN, CNA", size=7.2, color=DKGREY)
    arrow(ax, 78.5, 117.5, 71.5, 114.8); arrow(ax, 55.0, 111.6, 42.2, 111.6)

    cyl(ax, 32, 96, 36, 6.0, fill=LTGREY)
    T(ax, 50, 98.9, "Per-facility daily panel", size=8.4, weight="bold")
    T(ax, 50, 96.4, "13,402 facilities  x  731 days  x  3 categories", size=7.4, color=DKGREY)
    arrow(ax, 27, 108.4, 45, 102.4); arrow(ax, 71.5, 108.4, 55, 102.4)
    arrow(ax, 50, 95.9, 50, 92.7)

    # ---- Part 2: method ----
    band(56.5, 91.0, "Part 2:  CD-ISG-TS model")
    y = 82.0
    chain = [("Lookback window\n56 days x 3", WHITE),
             ("RevIN\nremove level, scale", WHITE),
             ("Nonlinear stem\nfeature Z", WHITE)]
    xs = [6, 30, 54]; w = 20
    for (t, f), x in zip(chain, xs):
        box(ax, x, y, w, 6.2, fill=f); T(ax, x + w/2, y + 3.1, t, size=7.8)
    arrow(ax, 26.2, y+3.1, 29.8, y+3.1); arrow(ax, 50.2, y+3.1, 53.8, y+3.1)
    # gate block
    box(ax, 78, y-1.0, 18, 8.0, fill=LTGREY, edge=BLACK, lw=1.5)
    T(ax, 87, y+5.6, "Sub-band gate", size=8.0, weight="bold")
    T(ax, 87, y+3.0, "multi-level DWT", size=6.9)
    T(ax, 87, y+1.4, "energy-preserving gate", size=6.9)
    T(ax, 87, y-0.1, "inverse DWT", size=6.9)
    arrow(ax, 74.2, y+3.1, 78.0, y+3.1)

    box(ax, 40, 70.0, 24, 6.4)
    T(ax, 52, 73.8, "Forecast head", size=8.2, weight="bold")
    T(ax, 52, 70.9, "linear (DLinear) or Transformer", size=7.0, color=DKGREY)
    arrow(ax, 87, y-1.0, 60, 76.4)
    box(ax, 71, 70.0, 24, 6.4)
    T(ax, 83, 73.8, "14-day forecast", size=8.2, weight="bold")
    T(ax, 83, 70.9, "per staff category", size=7.0, color=DKGREY)
    arrow(ax, 64.2, 73.2, 70.8, 73.2)
    # invariance regularizer
    box(ax, 8, 60.0, 44, 6.6, fill=LTGREY, style="round")
    T(ax, 30, 64.4, "Cross-domain invariance regularizer", size=8.0, weight="bold")
    T(ax, 30, 61.4, "group batch by facility  ->  variance of retained\nsub-band energy  ->  add to the loss", size=6.9, color=DKGREY)
    arrow(ax, 87, y-1.0, 30, 66.6, ls=(0, (4, 3)))
    T(ax, 55, 63.4, "trains the gate", size=6.6, color=DKGREY, ha="left")
    arrow(ax, 52, 70.0, 52, 67.0, ls=(0, (4, 3)))
    arrow(ax, 52, 60.0, 52, 57.2)

    # ---- Part 3: evaluation ----
    band(2.5, 55.0, "Part 3:  Evaluation and diagnosis")
    box(ax, 8, 44.6, 40, 7.0)
    T(ax, 28, 49.7, "Domain-generalization protocols", size=8.2, weight="bold")
    T(ax, 28, 46.4, "leave-facilities-out, leave-states-out,\ntrain 2024 -> deploy 2025", size=7.0, color=DKGREY)
    box(ax, 55, 44.6, 40, 7.0)
    T(ax, 75, 49.7, "Baselines", size=8.2, weight="bold")
    T(ax, 75, 46.4, "DLinear, Autoformer, FEDformer, IRM,\nCORAL, DANN, GroupDRO, Wave-Mask", size=7.0, color=DKGREY)
    arrow(ax, 40, 56.9, 28, 51.7); arrow(ax, 63, 56.9, 75, 51.7)

    box(ax, 8, 35.0, 40, 8.2)
    T(ax, 28, 41.0, "Accuracy and significance", size=8.2, weight="bold")
    T(ax, 28, 37.3, "per-facility MAE, RMSE, sMAPE;\nWilcoxon signed-rank, Holm correction", size=7.0, color=DKGREY)
    box(ax, 55, 35.0, 40, 8.2, fill=LTGREY, edge=BLACK, lw=1.5)
    T(ax, 75, 41.0, "Reducibility diagnostic (oracle)", size=8.2, weight="bold")
    T(ax, 75, 37.3, "transfer error vs. in-domain oracle error\non leave-group-out (ownership, region)", size=7.0, color=DKGREY)
    arrow(ax, 28, 44.6, 28, 43.2); arrow(ax, 75, 44.6, 75, 43.2)

    diamond(ax, 50, 27.0, 30, 8.0, fill=WHITE)
    T(ax, 50, 27.0, "Is the cross-facility\ngap reducible?", size=7.6, weight="bold")
    arrow(ax, 28, 35.0, 42, 29.2); arrow(ax, 75, 35.0, 58, 29.2)

    box(ax, 6, 15.0, 40, 7.0, fill=WHITE)
    T(ax, 26, 20.0, "Yes: apply invariance module", size=7.8, weight="bold")
    T(ax, 26, 16.7, "gate suppresses domain-variant bands", size=6.9, color=DKGREY)
    box(ax, 56, 15.0, 40, 7.0, fill=WHITE)
    T(ax, 76, 20.0, "No: a simple pooled model suffices", size=7.8, weight="bold")
    T(ax, 76, 16.7, "complex DG cannot help (this study)", size=6.9, color=DKGREY)
    arrow(ax, 44, 24.6, 30, 22.0); T(ax, 34, 24.7, "yes", size=6.8, ha="left")
    arrow(ax, 56, 24.6, 72, 22.0); T(ax, 62, 24.7, "no", size=6.8, ha="left")

    box(ax, 26, 5.5, 48, 6.6, fill=LTGREY)
    T(ax, 50, 9.8, "Interpretation and practical guidance", size=8.2, weight="bold")
    T(ax, 50, 6.9, "interpretable frequency structure + a decision rule for when to invest in domain generalization",
      size=7.0, color=DKGREY)
    arrow(ax, 26, 15.0, 45, 12.1); arrow(ax, 76, 15.0, 55, 12.1)
    _save(fig, "flow_diagram_bw")


# ======================================================================= #
#  FIGURE 2 — detailed architecture (grayscale)
# ======================================================================= #
def architecture():
    fig, ax = plt.subplots(figsize=(13.0, 7.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    tt = np.linspace(0, 1, 90)
    T(ax, 0.6, 98.4, "CD-ISG-TS architecture: a plug-in sub-band gating module inside a standard forecasting backbone",
      size=11.6, weight="bold", ha="left")
    T(ax, 0.6, 95.4, "With the gate open the module is exactly the identity, so the network reduces to the plain backbone.",
      size=8.0, color=DKGREY, ha="left")

    # top pipeline
    PY, PH = 83.0, 9.6; cy = PY + PH/2
    blocks = [("Staffing window\nRN, LPN, CNA", MDGREY, "3 x 56"),
              ("RevIN\nremove level, scale", WHITE, "3 x 56"),
              ("Temporal stem\nfeature Z", WHITE, "3 x 56"),
              ("Sub-band gate\n(proposed)", LTGREY, "3 x 56"),
              ("Forecast head", WHITE, "3 x 14"),
              ("RevIN inverse", MDGREY, "3 x 14")]
    xs = [1, 17.5, 32, 47.5, 66, 81.5]; ws = [13.5, 11, 12.5, 15, 12.5, 12.5]
    for (t, f, sh), x, w in zip(blocks, xs, ws):
        emph = t.startswith("Sub-band")
        box(ax, x, PY, w, PH, fill=f, edge=BLACK, lw=2.0 if emph else 1.2)
        T(ax, x+w/2, PY+PH-2.6, t.split("\n")[0], size=9.0, weight="bold")
        T(ax, x+w/2, PY+3.3, "\n".join(t.split("\n")[1:]), size=7.2, color=DKGREY)
        T(ax, x+w/2, PY-2.0, sh, size=7.0, color=MIDGREY)
    for i in range(len(xs)-1):
        arrow(ax, xs[i]+ws[i]+0.4, cy, xs[i+1]-0.6, cy)

    # zoom panel
    ZX, ZY, ZW, ZH = 1, 4.5, 94, 63
    box(ax, ZX, ZY, ZW, ZH, fill="#FCFCFC", edge=BLACK, lw=1.6, r=1.4, z=1)
    for (x0, x1) in [(47.5, ZX+0.6), (62.5, ZX+ZW-0.6)]:
        ax.plot([x0, x1], [PY-0.3, ZY+ZH], ls=(0, (4, 3)), lw=1.0, color=DKGREY, zorder=2)
    T(ax, ZX+1.6, ZY+ZH-3.2, "Inside the module", size=10.2, weight="bold", ha="left")
    T(ax, ZX+1.6, ZY+ZH-6.0, "The gate acts on the learned feature Z and re-weights localized frequency sub-bands per staff category.",
      size=8.0, color=DKGREY, ha="left")

    MY, MH = 42, 13; mcy = MY+MH/2
    box(ax, 4, MY, 11, MH, fill=WHITE)
    T(ax, 9.5, MY+MH-2.2, "Feature Z", size=8.4, weight="bold")
    for k, lab in enumerate(["RN", "LPN", "CNA"]):
        yy = MY+8-k*2.7; s = 0.5+0.42*np.sin(2*np.pi*tt*7+k)+0.13*np.sin(2*np.pi*tt*19+2*k)
        ax.plot(6+tt*7.4, yy+s*1.3-0.65, lw=0.9, color=["#111","#555","#999"][k], zorder=6)
        T(ax, 5.5, yy, lab, size=5.6, color=["#111","#555","#999"][k], ha="right")
    arrow(ax, 15.2, mcy, 18.2, mcy)
    box(ax, 18.6, MY, 12.4, MH, fill=MDGREY)
    T(ax, 24.8, MY+MH-2.3, "Multi-level DWT", size=8.4, weight="bold")
    T(ax, 24.8, MY+7, "fixed orthonormal W", size=7.0); T(ax, 24.8, MY+5, "db4, 3 levels", size=7.0)
    T(ax, 24.8, MY+2.3, "no trainable weights", size=6.4, color=DKGREY)
    T(ax, 24.8, MY-2.0, "inverse = W transpose", size=6.8, color=MIDGREY)
    arrow(ax, 31.2, mcy, 34.2, mcy)
    box(ax, 34.6, MY, 15.6, MH, fill=WHITE)
    T(ax, 42.4, MY+MH-2.2, "Sub-band coefficients", size=8.4, weight="bold")
    T(ax, 42.4, MY+8.2, "slow, shared         fast, specific", size=6.1, color=DKGREY)
    labs = ["A3","D3","D2","D1"]; per = ["> 16 d","8-16 d","4-8 d","2-4 d"]
    grays = ["#111111","#555555","#8A8A8A","#C8C8C8"]
    bx0, bw = 36.2, 3.3
    for k in range(4):
        ax.add_patch(Rectangle((bx0+k*bw, MY+4.6), bw, 2.9, fc=grays[k], ec="white", lw=0.9, zorder=6))
        T(ax, bx0+k*bw+bw/2, MY+6.0, labs[k], size=6.6, color="white" if k<2 else INK, weight="bold")
        T(ax, bx0+k*bw+bw/2, MY+2.9, per[k], size=5.5, color=DKGREY)
    T(ax, 42.4, MY-2.0, "sizes  7, 7, 14, 28  =  56", size=6.8, color=MIDGREY)
    ax.add_patch(Circle((53, mcy), 1.9, fc=WHITE, ec=BLACK, lw=1.6, zorder=6)); T(ax, 53, mcy, "x", size=12, weight="bold")
    arrow(ax, 50.6, mcy, 50.9, mcy); T(ax, 53, mcy+3.6, "per band", size=6.2, color=DKGREY); arrow(ax, 55.1, mcy, 58.0, mcy)
    box(ax, 58.4, MY, 12, MH, fill=MDGREY)
    T(ax, 64.4, MY+MH-2.3, "Inverse DWT", size=8.4, weight="bold")
    T(ax, 64.4, MY+6.6, "W transpose", size=7.4); T(ax, 64.4, MY+3.6, "exact reconstruction", size=6.4, color=DKGREY)
    arrow(ax, 70.6, mcy, 73.6, mcy)
    box(ax, 74, MY, 11.4, MH, fill=LTGREY, edge=BLACK, lw=1.4)
    T(ax, 79.7, MY+MH-2.2, "Gated feature", size=8.4, weight="bold")
    for k in range(3):
        yy = MY+8-k*2.7; s = 0.5+0.42*np.sin(2*np.pi*tt*7+k)+0.035*np.sin(2*np.pi*tt*19+2*k)
        ax.plot(75.5+tt*8.4, yy+s*1.3-0.65, lw=0.9, color=["#111","#555","#999"][k], zorder=6)
    arrow(ax, 85.6, mcy, 90.6, mcy); T(ax, 88.3, mcy+2.4, "to head", size=6.6, color=DKGREY)

    # gating network
    GX, GY, GW, GH = 31, 18, 40, 19
    box(ax, GX, GY, GW, GH, fill=WHITE, edge=BLACK, lw=1.5, r=1.0)
    T(ax, GX+1.4, GY+GH-2.4, "Gating network  (category-conditioned, energy-preserving)", size=8.2, weight="bold", ha="left")
    gy0 = 23.6; lx0 = 33.6
    T(ax, lx0+4.1, GY+GH-5.4, "learnable logits", size=6.9); T(ax, lx0+4.1, GY+GH-7.3, "theta  (3 x 4)", size=6.4, color=DKGREY)
    for r in range(3):
        for c in range(4):
            ax.add_patch(Rectangle((lx0+c*2.05, gy0+(2-r)*1.9), 1.9, 1.75, fc="#EDEDED", ec="#888", lw=0.6, zorder=6))
        T(ax, lx0-0.4, gy0+(2-r)*1.9+0.88, ["RN","LPN","CNA"][r], size=5.5, color=DKGREY, ha="right")
    for c in range(4): T(ax, lx0+c*2.05+0.95, gy0-1.3, labs[c], size=5.5, color=DKGREY)
    arrow(ax, lx0+8.5, gy0+2.8, lx0+10.6, gy0+2.8, ms=10); T(ax, lx0+9.55, gy0+6.6, "sigmoid", size=6.2, color=DKGREY)
    nx0 = lx0+11.2
    box(ax, nx0, GY+3.4, 12.2, 11, fill=LTGREY, edge=BLACK, lw=1.0, r=0.7)
    T(ax, nx0+6.1, GY+12.2, "energy-preserving", size=7.0, weight="bold")
    T(ax, nx0+6.1, GY+10.0, "rescale so that", size=6.3, color=DKGREY)
    T(ax, nx0+6.1, GY+7.8, "sum n(j) g(c,j)", size=6.8); T(ax, nx0+6.1, GY+6.0, "=  sum n(j)", size=6.8)
    T(ax, nx0+6.1, GY+4.3, "n(j) = band size", size=5.8, color=DKGREY)
    arrow(ax, nx0+12.6, gy0+2.8, nx0+14.7, gy0+2.8, ms=10)
    mx0 = nx0+15.3
    T(ax, mx0+4.1, GY+GH-5.4, "gate g", size=7.0, weight="bold"); T(ax, mx0+4.1, GY+GH-7.3, "(3 x 4)", size=6.4, color=DKGREY)
    demo = np.array([[1.29,1.23,1.02,0.86],[1.27,1.24,1.04,0.85],[1.30,1.16,0.99,0.89]])
    for r in range(3):
        for c in range(4):
            v = demo[r,c]; shade = str(min(0.92, max(0.15, 0.5 + (v-1)*1.4)))
            ax.add_patch(Rectangle((mx0+c*2.05, gy0+(2-r)*1.9), 1.9, 1.75, fc=shade, ec="white", lw=0.7, zorder=6))
            T(ax, mx0+c*2.05+0.95, gy0+(2-r)*1.9+0.88, f"{v:.2f}", size=5.1, color="white" if v>1.05 else INK)
    for c in range(4): T(ax, mx0+c*2.05+0.95, gy0-1.3, labs[c], size=5.5, color=DKGREY)
    T(ax, mx0+3.9, GY+1.9, "below 1 = suppress    above 1 = amplify", size=5.4, color=DKGREY)
    arrow(ax, mx0+4.1, GY+GH+0.3, 54, mcy-2.3, ms=12)

    # invariance estimator strip
    IX, IY, IW, IH = 4, 6.4, 87, 8.8
    box(ax, IX, IY, IW, IH, fill=LTGREY, edge=BLACK, lw=1.2, r=0.8)
    T(ax, IX+1.6, IY+IH-1.9, "Cross-domain invariance estimator  (trains the gate)", size=8.2, weight="bold", ha="left")
    steps = [(20,"band energy of\nthe gated feature"),(40,"group the batch\nby facility"),
             (61,"variance across facilities\nOmega (per category, band)"),(82,"objective\nL = MAE + lambda x Omega")]
    for k,(sx,txt) in enumerate(steps):
        T(ax, sx, IY+3.1, txt, size=6.8)
        if k < 3: arrow(ax, sx+9, IY+3.1, steps[k+1][0]-9, IY+3.1, ms=10)
    arrow(ax, 79.7, MY-0.3, 79.7, IY+IH+0.3, ls=(0,(3,2)))
    arrow(ax, 8, IY+IH+0.3, 33.4, GY+0.6, ls=(0,(4,2.5))); T(ax, 19, 17.6, "gradient", size=6.5, color=DKGREY)

    # legend (grayscale)
    for k,(lab,fc) in enumerate([("standard backbone",WHITE),("proposed module",LTGREY),
                                 ("data / tensors",MDGREY)]):
        x = 4+k*24
        ax.add_patch(FancyBboxPatch((x,0.8),2.6,1.7,boxstyle="round,pad=0,rounding_size=0.4",fc=fc,ec=BLACK,lw=1.1,zorder=6))
        T(ax, x+3.3, 1.65, lab, size=7.2, color=DKGREY, ha="left")
    _save(fig, "architecture_bw")


if __name__ == "__main__":
    flow(); architecture()
