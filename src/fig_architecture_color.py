"""
Elegant, muted-color version of the CD-ISG-TS architecture diagram.

Design intent: a bespoke, editorial palette (soft slate blue / muted plum-wine /
sage green / warm taupe) rather than a generic bright "AI-tool" gradient look.
Flat matte fills, thin strokes, no drop shadows, no italics, generous whitespace,
precise grid alignment. Every piece of text is pure black, set in Times New Roman,
so all color lives in the shapes/fills/lines, never in the writing. 600-dpi vector PDF.
"""
import numpy as np, os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

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

# ---- bespoke, muted, eye-soothing palette (shapes/fills/lines only) --------
BB_FILL, BB_EDGE   = "#EAF1F7", "#5C82A6"   # soft slate blue  -- standard backbone
MOD_FILL, MOD_EDGE = "#F5ECF1", "#8E5473"   # muted plum-wine  -- proposed module
SIG_FILL, SIG_EDGE = "#EEF4EC", "#71916C"   # sage green       -- training signal
DAT_FILL, DAT_EDGE = "#F5F2ED", "#A79C8A"   # warm taupe       -- data / tensors

# light enough, at every step, for pure-black text to stay legible on top
BAND_COLORS = ["#A9C6DC", "#C4D8E7", "#DBE8F0", "#EEF3F8"]      # light blue ramp, coarse -> fine
ARROW = "#7A756C"
CATC  = ["#5C82A6", "#8E5473", "#71916C"]                       # RN, LPN, CNA line colors (lines only)

def _diverging(t):
    """Custom soft diverging color: light blue (t=0) -> cream (t=0.5) -> light wine (t=1).
    Anchors are kept light so pure-black text always reads clearly on top."""
    lo = np.array([0xAE, 0xC6, 0xD4]) / 255.0
    mid = np.array([0xF5, 0xF0, 0xE6]) / 255.0
    hi = np.array([0xDB, 0xB4, 0xC6]) / 255.0
    t = np.clip(t, 0, 1)
    if t < 0.5:
        c = lo + (mid - lo) * (t / 0.5)
    else:
        c = mid + (hi - mid) * ((t - 0.5) / 0.5)
    return tuple(c)

fig, ax = plt.subplots(figsize=(13.0, 7.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
tt = np.linspace(0, 1, 90)

def T(x, y, s, size=8.5, weight="normal", ha="center", va="center", z=6):
    """All figure text is pure black; color differentiation lives in shapes/lines."""
    ax.text(x, y, s, fontsize=size, color=BLACK, fontweight=weight, ha=ha, va=va,
            style="normal", zorder=z)

def box(x, y, w, h, fill, edge, lw=1.3, r=1.4, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw, zorder=z))

def arrow(x0, y0, x1, y1, color=ARROW, lw=1.5, ls="-", ms=13, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, lw=lw, color=color,
                                 linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

# ---- top pipeline (centred) --------------------------------------------------
PY, PH = 88.5, 9.4; cy = PY + PH / 2
blocks = [("Input window\nthree channels", DAT_FILL, DAT_EDGE, "3 x 56"),
          ("RevIN\nremove level, scale", BB_FILL, BB_EDGE, "3 x 56"),
          ("Temporal stem\nfeature Z", BB_FILL, BB_EDGE, "3 x 56"),
          ("Sub-band gate\nmechanism 1", MOD_FILL, MOD_EDGE, "3 x 56"),
          ("Forecast head", BB_FILL, BB_EDGE, "3 x 14"),
          ("Blend\nby alpha", MOD_FILL, MOD_EDGE, "3 x 14")]
ws = [13.0, 12.0, 12.5, 14.0, 12.0, 11.5]
GAP = 2.6
TOTAL = sum(ws) + GAP * (len(ws) - 1)
X0 = (100 - TOTAL) / 2                      # centre the whole block
xs = []
_x = X0
for w in ws:
    xs.append(_x); _x += w + GAP

for (t_, f, e, sh), x, w in zip(blocks, xs, ws):
    emph = t_.startswith("Sub-band") or t_.startswith("Blend")
    box(x, PY, w, PH, f, e, lw=2.1 if emph else 1.3)
    T(x + w/2, PY + PH - 2.9, t_.split("\n")[0], size=10.2, weight="bold")
    T(x + w/2, PY + 3.2, "\n".join(t_.split("\n")[1:]), size=8.4)
    T(x + w/2, PY - 2.2, sh, size=8.0)
for i in range(len(xs) - 1):
    arrow(xs[i] + ws[i] + 0.15, cy, xs[i + 1] - 0.15, cy)

# ---- mechanism 2: the input-driven periodicity branch -----------------------
BY, BH = 76.0, 7.6; bcy = BY + BH / 2
PB_X, PB_W = xs[1] + 1.0, 30.0                  # periodicity block
SN_X, SN_W = PB_X + PB_W + 3.4, 14.0            # seasonal continuation block
box(PB_X, BY, PB_W, BH, MOD_FILL, MOD_EDGE, lw=2.1)
T(PB_X + PB_W/2, BY + BH - 2.6, "Periodicity branch   (mechanism 2)",
  size=9.8, weight="bold")
T(PB_X + PB_W/2, BY + 2.4, "extend the rhythm the input already shows", size=8.4)
box(SN_X, BY, SN_W, BH, "#FFFFFF", DAT_EDGE, lw=1.1)
T(SN_X + SN_W/2, BY + BH - 2.6, "seasonal", size=9.2, weight="bold")
T(SN_X + SN_W/2, BY + 2.4, "continuation", size=8.4)

# risers, offset from the shape labels so nothing is written over
DROP = xs[0] + 2.4                              # leaves the input box near its left edge
RISE = xs[5] + 2.4                              # re-enters the blend box near its left edge
ax.plot([DROP, DROP], [PY - 0.35, bcy], lw=1.4, color=ARROW, zorder=4)
arrow(DROP, bcy, PB_X - 0.15, bcy)
arrow(PB_X + PB_W + 0.15, bcy, SN_X - 0.15, bcy)
ax.plot([SN_X + SN_W, RISE], [bcy, bcy], lw=1.4, color=ARROW, zorder=4)
arrow(RISE, bcy, RISE, PY - 0.35)

# alpha, set by an in-window backtest
AL_X, AL_W = PB_X - 2.0, PB_W + SN_W + 7.4
box(AL_X, BY - 10.0, AL_W, 6.4, SIG_FILL, SIG_EDGE, lw=1.3)
T(AL_X + AL_W/2, BY - 5.2, "alpha from an in-window backtest, no labels used",
  size=8.6, weight="bold")
T(AL_X + AL_W/2, BY - 8.2,
  "hold out the last 14 days of the input and score the seasonal rule there",
  size=7.6)
arrow(AL_X + AL_W/2, BY - 3.6, AL_X + AL_W/2, BY - 0.4)

# ---- zoom panel ---------------------------------------------------------------
ZX, ZY, ZW, ZH = 1, 3.0, 94, 61.0
box(ZX, ZY, ZW, ZH, "#FCFBF9", MOD_EDGE, lw=1.6, r=1.4, z=1)
for (x0, x1) in [(xs[3], ZX + 0.6), (xs[3] + ws[3], ZX + ZW - 0.6)]:
    ax.plot([x0, x1], [BY - 10.2, ZY + ZH], ls=(0, (4, 3)), lw=0.9,
            color=MOD_EDGE, zorder=2, alpha=0.55)
T(ZX + 1.6, ZY + ZH - 3.4, "Inside mechanism 1: the sub-band gate", size=11.4, weight="bold", ha="left")
T(ZX + 1.6, ZY + ZH - 6.4,
  "The gate re-weights localized frequency sub-bands of the learned feature, one weight per channel and band.",
  size=8.8, ha="left")

MY, MH = 42, 13; mcy = MY + MH / 2
box(4, MY, 11, MH, "#FFFFFF", DAT_EDGE, lw=1.1)
T(9.5, MY + MH - 2.2, "Feature Z", size=8.4, weight="bold")
for k, lab in enumerate(["ch 1", "ch 2", "ch 3"]):
    yy = MY + 8 - k * 2.7
    s = 0.5 + 0.42 * np.sin(2 * np.pi * tt * 7 + k) + 0.13 * np.sin(2 * np.pi * tt * 19 + 2 * k)
    ax.plot(6 + tt * 7.4, yy + s * 1.3 - 0.65, lw=1.0, color=CATC[k], zorder=6)
    T(5.5, yy, lab, size=5.6, ha="right")
arrow(15.2, mcy, 18.2, mcy)

box(18.6, MY, 12.4, MH, BB_FILL, BB_EDGE, lw=1.2)
T(24.8, MY + MH - 2.3, "Multi-level DWT", size=8.4, weight="bold")
T(24.8, MY + 7, "fixed orthonormal W", size=7.0)
T(24.8, MY + 5, "db4, 3 levels", size=7.0)
T(24.8, MY + 2.3, "no trainable weights", size=6.4)
T(24.8, MY - 2.0, "inverse = W transpose", size=6.8)
arrow(31.2, mcy, 34.2, mcy)

box(34.6, MY, 15.6, MH, "#FFFFFF", DAT_EDGE, lw=1.1)
T(42.4, MY + MH - 2.2, "Sub-band coefficients", size=8.4, weight="bold")
T(42.4, MY + 8.2, "slow, shared         fast, specific", size=6.1)
labs = ["A3", "D3", "D2", "D1"]; per = ["> 16 d", "8-16 d", "4-8 d", "2-4 d"]
bx0, bw = 36.2, 3.3
for k in range(4):
    ax.add_patch(Rectangle((bx0 + k * bw, MY + 4.6), bw, 2.9, fc=BAND_COLORS[k],
                           ec="white", lw=0.9, zorder=6))
    T(bx0 + k * bw + bw / 2, MY + 6.0, labs[k], size=6.6, weight="bold")
    T(bx0 + k * bw + bw / 2, MY + 2.9, per[k], size=5.5)
T(42.4, MY - 2.0, "sizes  7, 7, 14, 28  =  56", size=6.8)

ax.add_patch(Circle((53, mcy), 1.9, fc="#FFFFFF", ec=MOD_EDGE, lw=1.7, zorder=6))
T(53, mcy, "x", size=12, weight="bold")
T(53, mcy + 3.6, "per band", size=6.2)
arrow(55.1, mcy, 58.0, mcy)

box(58.4, MY, 12, MH, BB_FILL, BB_EDGE, lw=1.2)
T(64.4, MY + MH - 2.3, "Inverse DWT", size=8.4, weight="bold")
T(64.4, MY + 6.6, "W transpose", size=7.4)
T(64.4, MY + 3.6, "exact reconstruction", size=6.4)
arrow(70.6, mcy, 73.6, mcy)

box(74, MY, 11.4, MH, MOD_FILL, MOD_EDGE, lw=1.5)
T(79.7, MY + MH - 2.2, "Gated feature", size=8.4, weight="bold")
for k in range(3):
    yy = MY + 8 - k * 2.7
    s = 0.5 + 0.42 * np.sin(2 * np.pi * tt * 7 + k) + 0.035 * np.sin(2 * np.pi * tt * 19 + 2 * k)
    ax.plot(75.5 + tt * 8.4, yy + s * 1.3 - 0.65, lw=1.0, color=CATC[k], zorder=6)
arrow(85.6, mcy, 90.6, mcy)
T(88.6, mcy + 2.4, "to forecast head", size=6.8)

# ---- gating network -----------------------------------------------------------
GX, GY, GW, GH = 31, 18, 40, 19
box(GX, GY, GW, GH, "#FFFFFF", MOD_EDGE, lw=1.5, r=1.0)
T(GX + 1.4, GY + GH - 2.4, "Gating network  (channel-conditioned, mean-one)",
  size=8.2, weight="bold", ha="left")
gy0 = 23.6; lx0 = 33.6
T(lx0 + 4.1, GY + GH - 5.4, "learnable logits", size=6.9)
T(lx0 + 4.1, GY + GH - 7.3, "theta  (3 x 4)", size=6.4)
for r in range(3):
    for c in range(4):
        ax.add_patch(Rectangle((lx0 + c * 2.05, gy0 + (2 - r) * 1.9), 1.9, 1.75,
                               fc=DAT_FILL, ec=DAT_EDGE, lw=0.6, zorder=6))
    T(lx0 - 0.4, gy0 + (2 - r) * 1.9 + 0.88, ["ch 1", "ch 2", "ch 3"][r], size=5.5, ha="right")
for c in range(4):
    T(lx0 + c * 2.05 + 0.95, gy0 - 1.3, labs[c], size=5.5)
arrow(lx0 + 8.5, gy0 + 2.8, lx0 + 10.6, gy0 + 2.8, ms=10)
T(lx0 + 9.55, gy0 + 6.6, "sigmoid", size=6.2)

nx0 = lx0 + 11.2
box(nx0, GY + 3.4, 12.2, 11, SIG_FILL, SIG_EDGE, lw=1.0, r=0.7)
T(nx0 + 6.1, GY + 12.2, "mean-one", size=7.0, weight="bold")
T(nx0 + 6.1, GY + 10.0, "rescale so that", size=6.3)
T(nx0 + 6.1, GY + 7.8, "sum n(j) g(c,j)", size=6.8)
T(nx0 + 6.1, GY + 6.0, "=  sum n(j)", size=6.8)
T(nx0 + 6.1, GY + 4.3, "n(j) = band size", size=5.8)
arrow(nx0 + 12.6, gy0 + 2.8, nx0 + 14.7, gy0 + 2.8, ms=10)

mx0 = nx0 + 15.3
T(mx0 + 4.1, GY + GH - 5.4, "gate g", size=7.0, weight="bold")
T(mx0 + 4.1, GY + GH - 7.3, "(3 x 4)", size=6.4)
demo = np.array([[1.29, 1.23, 1.02, 0.86],
                 [1.27, 1.24, 1.04, 0.85],
                 [1.30, 1.16, 0.99, 0.89]])
vmax = 0.34
for r in range(3):
    for c in range(4):
        v = demo[r, c]
        t_norm = 0.5 + (v - 1) / (2 * vmax)
        rgb = _diverging(t_norm)
        ax.add_patch(Rectangle((mx0 + c * 2.05, gy0 + (2 - r) * 1.9), 1.9, 1.75,
                               fc=rgb, ec="white", lw=0.7, zorder=6))
        T(mx0 + c * 2.05 + 0.95, gy0 + (2 - r) * 1.9 + 0.88, f"{v:.2f}", size=5.1)
for c in range(4):
    T(mx0 + c * 2.05 + 0.95, gy0 - 1.3, labs[c], size=5.5)
T(mx0 + 3.9, GY + 1.9, "below 1 = suppress    above 1 = amplify", size=5.4)
arrow(mx0 + 4.1, GY + GH + 0.3, 54, mcy - 2.3, color=MOD_EDGE, lw=1.4, ms=12)

# ---- invariance estimator strip ------------------------------------------------
IX, IY, IW, IH = 4, 6.4, 87, 8.8
box(IX, IY, IW, IH, SIG_FILL, SIG_EDGE, lw=1.2, r=0.8)
T(IX + 1.6, IY + IH - 1.9, "Cross-entity invariance estimator  (trains the gate)",
  size=8.2, weight="bold", ha="left")
steps = [(20, "band energy of\nthe gated feature"), (40, "group the batch\nby entity"),
         (61, "variance across entities\nOmega (per channel, band)"),
         (82, "objective\nL = MAE + lambda x Omega")]
for k, (sx, txt) in enumerate(steps):
    T(sx, IY + 3.1, txt, size=6.8)
    if k < 3:
        arrow(sx + 9, IY + 3.1, steps[k + 1][0] - 9, IY + 3.1, color=SIG_EDGE, ms=10)
arrow(79.7, MY - 0.3, 79.7, IY + IH + 0.3, color=SIG_EDGE, ls=(0, (3, 2)))
arrow(8, IY + IH + 0.3, 33.4, GY + 0.6, color=SIG_EDGE, ls=(0, (4, 2.5)))
T(19, 17.6, "gradient", size=6.5)

# ---- legend ---------------------------------------------------------------------
for k, (lab, fc, ec) in enumerate([("standard backbone", BB_FILL, BB_EDGE),
                                   ("added module", MOD_FILL, MOD_EDGE),
                                   ("training signal", SIG_FILL, SIG_EDGE),
                                   ("data / tensors", DAT_FILL, DAT_EDGE)]):
    x = 4 + k * 22
    ax.add_patch(FancyBboxPatch((x, 0.8), 2.6, 1.7, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=fc, ec=ec, lw=1.1, zorder=6))
    T(x + 3.3, 1.65, lab, size=7.2, ha="left")

out = os.path.join(FIG, "architecture_color.pdf")
fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(out.replace(".pdf", ".png"), dpi=220, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close(fig)
print("wrote", out)
