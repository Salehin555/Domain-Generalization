"""
Publication architecture figure for CD-ISG-TS.

Top    : the end-to-end pipeline (the plug-in module sits inside a standard backbone).
Bottom : a zoomed view of the plug-in module -- multi-level DWT, the category-conditioned
         energy-preserving gating network, inverse DWT, and the cross-domain invariance
         estimator that trains the gate.

All text is upright (no italics); saved as a 600-dpi vector PDF.
"""
import numpy as np, os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.style": "normal", "mathtext.default": "regular",
    "savefig.dpi": 600,
})

INK        = "#15181C"
MUTED      = "#6B7280"
BB_FILL    = "#EDF3FA"; BB_EDGE  = "#2E6FA7"      # standard backbone
MOD_FILL   = "#FEF6E9"; MOD_EDGE = "#C8871B"      # proposed module
PANEL_FILL = "#FDFAF4"
INV_FILL   = "#EDF7F2"; INV_EDGE = "#0E7C5A"      # training signal
GREY_FILL  = "#F3F4F6"; GREY_EDGE = "#9CA3AF"
BAND_COLORS = ["#0B3C5D", "#1F6FA8", "#5FA8D3", "#B7DCF0"]
ARROW      = "#5B6169"
CATC       = ["#2E6FA7", "#C8871B", "#0E7C5A"]

fig, ax = plt.subplots(figsize=(13.2, 8.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def T(x, y, s, size=8.5, color=INK, weight="normal", ha="center", va="center", z=6):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            style="normal", zorder=z)

def boxr(x, y, w, h, fill, edge, lw=1.3, r=1.4, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=z))

def arrow(x0, y0, x1, y1, color=ARROW, lw=1.5, ls="-", ms=13, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

tt = np.linspace(0, 1, 90)

# =========================================================================== #
#  TITLE
# =========================================================================== #
T(1.0, 98.6, "CD-ISG-TS: a lightweight plug-in sub-band gating module inside a standard forecasting backbone",
  size=12.6, weight="bold", ha="left")
T(1.0, 95.6, "The module has no trainable weights except a 3 x 4 gate. With the gate open it is exactly the identity, so the network reduces to the plain backbone.",
  size=8.2, color=MUTED, ha="left")

# =========================================================================== #
#  TOP  —  end-to-end pipeline
# =========================================================================== #
PY, PH = 82.0, 10.4
cy = PY + PH / 2

blocks = [
    dict(x=1.0,  w=13.2, fill=GREY_FILL, edge=GREY_EDGE,
         title="Staffing window", sub="RN, LPN, CNA\nhours per resident-day", shape="3 x 56"),
    dict(x=17.6, w=10.8, fill=BB_FILL, edge=BB_EDGE,
         title="RevIN", sub="remove per-window\nlevel and scale", shape="3 x 56"),
    dict(x=31.8, w=12.4, fill=BB_FILL, edge=BB_EDGE,
         title="Temporal stem", sub="depthwise conv + GELU\n(residual)", shape="Z : 3 x 56"),
    dict(x=47.6, w=15.0, fill=MOD_FILL, edge=MOD_EDGE,
         title="CD-ISG module", sub="sub-band gating\n(proposed, +12 parameters)", shape="3 x 56"),
    dict(x=66.0, w=12.2, fill=BB_FILL, edge=BB_EDGE,
         title="Forecast head", sub="DLinear  or\npatch Transformer", shape="3 x 14"),
    dict(x=81.6, w=12.6, fill=GREY_FILL, edge=GREY_EDGE,
         title="RevIN inverse", sub="restore level\nand scale", shape="forecast 3 x 14"),
]
for i, b in enumerate(blocks):
    emph = b["title"] == "CD-ISG module"
    boxr(b["x"], PY, b["w"], PH, b["fill"], b["edge"], lw=2.2 if emph else 1.3)
    T(b["x"] + b["w"]/2, PY + PH - 2.6, b["title"], size=9.4, weight="bold",
      color=MOD_EDGE if emph else INK)
    T(b["x"] + b["w"]/2, PY + 3.6, b["sub"], size=7.4, color="#414750")
    T(b["x"] + b["w"]/2, PY - 2.0, b["shape"], size=7.1, color=MUTED)
    if i < len(blocks) - 1:
        arrow(b["x"] + b["w"] + 0.5, cy, blocks[i+1]["x"] - 0.7, cy)

# =========================================================================== #
#  ZOOM PANEL
# =========================================================================== #
ZX, ZY, ZW, ZH = 1.0, 4.2, 93.6, 62.0
boxr(ZX, ZY, ZW, ZH, PANEL_FILL, MOD_EDGE, lw=1.8, r=1.6, z=1)

# zoom callout: module box bottom corners -> panel top corners (region kept empty)
for (x0, x1) in [(47.6, ZX + 0.6), (62.6, ZX + ZW - 0.6)]:
    ax.plot([x0, x1], [PY - 0.3, ZY + ZH], ls=(0, (4, 3)), lw=1.0,
            color=MOD_EDGE, alpha=0.8, zorder=2)

T(ZX + 1.8, ZY + ZH - 3.4, "Inside the plug-in module", size=10.6, weight="bold",
  color=MOD_EDGE, ha="left")
T(ZX + 1.8, ZY + ZH - 6.4,
  "The gate acts on the learned feature Z, not the raw input: it re-weights localized frequency sub-bands separately for each staff category.",
  size=8.0, color=MUTED, ha="left")

MY, MH = 42.0, 13.0
mcy = MY + MH/2

# ---- 1. feature Z ----------------------------------------------------------
boxr(4.0, MY, 11.2, MH, "#FFFFFF", GREY_EDGE, lw=1.1)
T(9.6, MY + MH - 2.2, "Feature  Z", size=8.6, weight="bold")
for k, (lab, col) in enumerate(zip(["RN", "LPN", "CNA"], CATC)):
    yy = MY + 8.0 - k*2.7
    s = 0.5 + 0.42*np.sin(2*np.pi*tt*7 + k) + 0.13*np.sin(2*np.pi*tt*19 + 2*k)
    ax.plot(6.1 + tt*7.6, yy + s*1.4 - 0.7, lw=0.8, color=col, zorder=6)
    T(5.6, yy, lab, size=5.8, color=col, ha="right")
T(9.6, MY - 2.0, "3 x 56", size=7.0, color=MUTED)

arrow(15.6, mcy, 18.4, mcy)

# ---- 2. DWT ---------------------------------------------------------------
boxr(18.8, MY, 12.4, MH, BB_FILL, BB_EDGE, lw=1.2)
T(25.0, MY + MH - 2.4, "Multi-level DWT", size=8.6, weight="bold")
T(25.0, MY + 7.2, "fixed orthonormal matrix W", size=7.2, color="#414750")
T(25.0, MY + 5.0, "db4, 3 levels", size=7.2, color="#414750")
T(25.0, MY + 2.4, "no trainable weights", size=6.6, color=MUTED)
T(25.0, MY - 2.0, "inverse = W transpose", size=7.0, color=MUTED)

arrow(31.6, mcy, 34.4, mcy)

# ---- 3. sub-band coefficients (equal-width bands for label legibility) -----
boxr(34.8, MY, 15.6, MH, "#FFFFFF", GREY_EDGE, lw=1.1)
T(42.6, MY + MH - 2.2, "Sub-band coefficients", size=8.6, weight="bold")
T(42.6, MY + 8.3, "slow, shared            fast, specific", size=6.2, color=MUTED)
band_lab = ["A3", "D3", "D2", "D1"]
band_per = ["> 16 d", "8-16 d", "4-8 d", "2-4 d"]
bx0, bwid = 36.2, 3.3
for k in range(4):
    ax.add_patch(Rectangle((bx0 + k*bwid, MY + 4.6), bwid, 2.9, fc=BAND_COLORS[k],
                           ec="white", lw=0.9, zorder=6))
    T(bx0 + k*bwid + bwid/2, MY + 6.05, band_lab[k], size=6.8,
      color="white" if k < 2 else INK, weight="bold")
    T(bx0 + k*bwid + bwid/2, MY + 2.9, band_per[k], size=5.7, color=MUTED)
T(42.6, MY - 2.0, "sizes  7, 7, 14, 28  =  56", size=7.0, color=MUTED)

# ---- 4. element-wise product ----------------------------------------------
ax.add_patch(Circle((53.0, mcy), 1.9, fc="#FFFFFF", ec=MOD_EDGE, lw=1.8, zorder=6))
T(53.0, mcy, "x", size=12, weight="bold", color=MOD_EDGE)
arrow(50.6, mcy, 50.9, mcy)
T(53.0, mcy + 3.7, "per band", size=6.4, color=MUTED)
arrow(55.1, mcy, 57.9, mcy)

# ---- 5. inverse DWT --------------------------------------------------------
boxr(58.3, MY, 12.2, MH, BB_FILL, BB_EDGE, lw=1.2)
T(64.4, MY + MH - 2.4, "Inverse DWT", size=8.6, weight="bold")
T(64.4, MY + 7.0, "W transpose", size=7.6, color="#414750")
T(64.4, MY + 4.0, "perfect reconstruction", size=6.6, color=MUTED)
T(64.4, MY + 2.2, "when the gate is open", size=6.6, color=MUTED)

arrow(70.9, mcy, 73.7, mcy)

# ---- 6. gated feature ------------------------------------------------------
boxr(74.1, MY, 11.2, MH, MOD_FILL, MOD_EDGE, lw=1.5)
T(79.7, MY + MH - 2.2, "Gated feature", size=8.6, weight="bold", color=MOD_EDGE)
for k, col in enumerate(CATC):
    yy = MY + 8.0 - k*2.7
    s = 0.5 + 0.42*np.sin(2*np.pi*tt*7 + k) + 0.035*np.sin(2*np.pi*tt*19 + 2*k)
    ax.plot(75.6 + tt*8.2, yy + s*1.4 - 0.7, lw=0.8, color=col, zorder=6)
T(79.7, MY - 2.0, "3 x 56", size=7.0, color=MUTED)

arrow(85.7, mcy, 91.4, mcy)
T(88.6, mcy + 2.4, "to forecast head", size=6.8, color=MUTED)

# =========================================================================== #
#  GATING NETWORK
# =========================================================================== #
GX, GY, GW, GH = 31.0, 18.0, 40.0, 19.0
boxr(GX, GY, GW, GH, "#FFFFFF", MOD_EDGE, lw=1.5, r=1.2, z=3)
T(GX + 1.4, GY + GH - 2.4, "Gating network   (category-conditioned, energy-preserving)",
  size=8.5, weight="bold", color=MOD_EDGE, ha="left")

cats = ["RN", "LPN", "CNA"]
gy0 = 23.6
# (a) learnable logits
lx0 = 33.6
T(lx0 + 4.1, GY + GH - 5.4, "learnable logits", size=7.0)
T(lx0 + 4.1, GY + GH - 7.3, "theta  (3 x 4)", size=6.5, color=MUTED)
for r in range(3):
    for c in range(4):
        ax.add_patch(Rectangle((lx0 + c*2.05, gy0 + (2-r)*1.9), 1.9, 1.75,
                               fc="#E8EDF3", ec="#AFBAC6", lw=0.6, zorder=6))
    T(lx0 - 0.4, gy0 + (2-r)*1.9 + 0.88, cats[r], size=5.6, color=MUTED, ha="right")
for c in range(4):
    T(lx0 + c*2.05 + 0.95, gy0 - 1.3, band_lab[c], size=5.6, color=MUTED)

arrow(lx0 + 8.5, gy0 + 2.8, lx0 + 10.6, gy0 + 2.8, lw=1.2, ms=10)
T(lx0 + 9.55, gy0 + 6.6, "sigmoid", size=6.4, color=MUTED)

# (b) energy-preserving normalization
nx0 = lx0 + 11.2
boxr(nx0, GY + 3.4, 12.2, 11.0, INV_FILL, INV_EDGE, lw=1.1, r=0.9)
T(nx0 + 6.1, GY + 12.2, "energy-preserving", size=7.1, weight="bold", color=INV_EDGE)
T(nx0 + 6.1, GY + 10.0, "rescale so that", size=6.4, color="#3B4A44")
T(nx0 + 6.1, GY + 7.8, "sum of  n(j) g(c,j)", size=6.9)
T(nx0 + 6.1, GY + 6.0, "=  sum of  n(j)", size=6.9)
T(nx0 + 6.1, GY + 4.3, "n(j) = band size", size=5.9, color=MUTED)

arrow(nx0 + 12.6, gy0 + 2.8, nx0 + 14.7, gy0 + 2.8, lw=1.2, ms=10)

# (c) resulting gate
mx0 = nx0 + 15.3
T(mx0 + 4.1, GY + GH - 5.4, "gate  g", size=7.2, weight="bold", color=MOD_EDGE)
T(mx0 + 4.1, GY + GH - 7.3, "(3 x 4)", size=6.5, color=MUTED)
gate_demo = np.array([[1.29, 1.23, 1.02, 0.86],
                      [1.27, 1.24, 1.04, 0.85],
                      [1.30, 1.16, 0.99, 0.89]])
cmap = plt.get_cmap("RdBu_r"); vmax = 0.34
for r in range(3):
    for c in range(4):
        val = gate_demo[r, c]
        ax.add_patch(Rectangle((mx0 + c*2.05, gy0 + (2-r)*1.9), 1.9, 1.75,
                               fc=cmap(0.5 + (val - 1)/(2*vmax)), ec="white", lw=0.7, zorder=6))
        T(mx0 + c*2.05 + 0.95, gy0 + (2-r)*1.9 + 0.88, f"{val:.2f}", size=5.2,
          color=INK if abs(val - 1) < 0.16 else "white")
for c in range(4):
    T(mx0 + c*2.05 + 0.95, gy0 - 1.3, band_lab[c], size=5.6, color=MUTED)
T(mx0 + 3.6, GY + 1.9, "below 1 = suppress    above 1 = amplify", size=5.5, color=MUTED)

# gate -> element-wise product
arrow(mx0 + 4.1, GY + GH + 0.3, 54.0, mcy - 2.3, color=MOD_EDGE, lw=1.4, ms=12)

# =========================================================================== #
#  CROSS-DOMAIN INVARIANCE ESTIMATOR
# =========================================================================== #
IX, IY, IW, IH = 4.0, 6.4, 87.0, 8.8
boxr(IX, IY, IW, IH, INV_FILL, INV_EDGE, lw=1.3, r=1.0, z=3)
T(IX + 1.6, IY + IH - 1.9, "Cross-domain invariance estimator   (this is what trains the gate)",
  size=8.4, weight="bold", color=INV_EDGE, ha="left")

steps = [(20.0, "band energy of\nthe gated feature"),
         (40.0, "group the batch\nby facility"),
         (61.0, "variance across facilities\nOmega  (per category, per band)"),
         (82.0, "total objective\nL  =  MAE  +  lambda x Omega")]
for k, (sx, txt) in enumerate(steps):
    T(sx, IY + 3.1, txt, size=6.9, color="#22403A")
    if k < len(steps) - 1:
        arrow(sx + 9.0, IY + 3.1, steps[k+1][0] - 9.0, IY + 3.1,
              color=INV_EDGE, lw=1.1, ms=10)

# gated feature -> estimator ; estimator -> gate logits (gradient)
arrow(83.2, MY - 0.3, 83.2, IY + IH + 0.3, color=INV_EDGE, lw=1.1, ls=(0, (3, 2)))
arrow(8.0, IY + IH + 0.3, 33.4, GY + 0.6, color=INV_EDGE, lw=1.3, ls=(0, (4, 2.5)))
T(19.0, 17.6, "gradient", size=6.6, color=INV_EDGE)

# =========================================================================== #
#  LEGEND
# =========================================================================== #
lx, ly = 4.0, 0.8
for k, (lab, fc, ec) in enumerate([("standard backbone", BB_FILL, BB_EDGE),
                                   ("proposed module", MOD_FILL, MOD_EDGE),
                                   ("training signal", INV_FILL, INV_EDGE),
                                   ("data / tensors", GREY_FILL, GREY_EDGE)]):
    x = lx + k*20.0
    ax.add_patch(FancyBboxPatch((x, ly), 2.6, 1.7,
                                boxstyle="round,pad=0.0,rounding_size=0.5",
                                fc=fc, ec=ec, lw=1.1, zorder=6))
    T(x + 3.3, ly + 0.85, lab, size=7.2, color="#414750", ha="left")

out = os.path.join(FIG, "architecture_cd_isg_ts.pdf")
fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight",
            pad_inches=0.06, facecolor="white")
plt.close(fig)
print("wrote", out)
