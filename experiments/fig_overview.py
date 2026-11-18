"""Overview figure for the methodology section: the whole study in one picture.

(a) what is measured and when. Real daily staffing series (hours per resident-day,
    RN + LPN + CNA) of nursing facilities, around the real decision date of the
    temporal analysis (1 January 2025). Both models are fitted only on history
    observed before the date; the estimate is scored before it and compared with
    the value realized after it.
(b) why the usual number misleads: 'error ladders' for two anchor splits, the rise
    in error when a model moves to a new group next to the part of it that
    retraining on that group recovers.
(c) when the estimate can be trusted: calendar rings showing which months of the
    deployment horizon the observed history covers.

Every number is read from experiments/results/ and the cached panels; nothing
is typed by hand. Facilities in (a) are illustrative draws from the panel.
Writes Manuscript/overview_headroom.pdf (and a .png preview).
"""
import calendar
import datetime as dt
import json
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Wedge

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from common import PANELS, load_panel  # noqa: E402
from src.oracle_experiment import build_meta  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.2, "axes.linewidth": 0.6, "savefig.dpi": 600,
})
# calm, flat palette on a white ground: no tints, no transparency
INK = "#2B333B"; MUTED = "#707A84"; RULE = "#BFC7CE"
SRC = "#4A78A6"                  # steel blue: source group
TGT = "#C2705A"                  # soft terracotta: target group
EST = "#5B9A7D"                  # sage: estimate, covered months
GAP = "#98A1AA"                  # grey: apparent gap
WARN = "#D6A13F"                 # soft ochre: months never observed
REF = "Target-trained reference (ERM-MLP)"


def J(n):
    with open(os.path.join(RES, n)) as fh:
        return json.load(fh)


ANCH = J("anchor_stats.json")
TEMP = pd.DataFrame(J("temporal_stats.json")["temporal"])
COV = pd.read_csv(os.path.join(RES, "coverage_table.csv"))


def tr_r(panel):
    g = TEMP[(TEMP.panel == panel) & (TEMP.variant == "standard") & (TEMP.metric == "mae_cat")]
    return float(g.pearson.iloc[0])


def ymd(v):
    v = int(v)
    return dt.date(v // 10000, (v // 100) % 100, v % 100)


def bracket(ax, x0, x1, y, text, color, below=True, fs=6.4, bold=False):
    h = 0.07 if below else -0.07
    ax.plot([x0, x0, x1, x1], [y + h, y, y, y + h], color=color, lw=0.8,
            solid_capstyle="butt", clip_on=False)
    ax.text(x0 + (x1 - x0) * 0.5, y - (0.06 if below else -0.06), text, ha="center",
            va="top" if below else "bottom", fontsize=fs, color=color,
            fontweight="semibold" if bold else "normal", clip_on=False)


# ----------------------------------------------------------------- panel (a)
def panel_a(ax):
    d = np.load(PANELS["Nursing"]["path"], allow_pickle=True)
    X = d["X"][:, :, :3].sum(2)
    dates = np.array([ymd(v) for v in d["dates"]])
    lo, hi = np.searchsorted(d["dates"], 20240901), np.searchsorted(d["dates"], 20250301)
    bnd = ymd(PANELS["Nursing"]["boundary"])
    seg = X[:, lo:hi]; t = dates[lo:hi]
    ok = np.where(np.isfinite(seg).all(1) & (np.nanstd(seg, 1) > 0))[0]
    # facilities with a visible weekly rhythm: strongest 7-day autocorrelation
    z = (seg[ok] - seg[ok].mean(1, keepdims=True)) / seg[ok].std(1, keepdims=True)
    ac7 = (z[:, :-7] * z[:, 7:]).mean(1)
    # the traces come from the real ownership anchor split: the source row is a
    # for-profit facility, the two target rows are non-profit facilities. Among the
    # most weekly facilities of each group, keep those whose level does not shift at
    # the decision date, so one line per row reads as a steady weekly rhythm
    own, _ = build_meta(load_panel("Nursing", 13402))
    b0 = int(np.searchsorted(t, ymd(PANELS["Nursing"]["boundary"])))
    shift = np.abs(z[:, b0:].mean(1) - z[:, :b0].mean(1))

    def steady(group, k):
        cand = [j for j in np.argsort(-ac7) if own[ok[j]] == group and shift[j] < 0.15]
        return [ok[cand[i]] for i in k]

    pick = steady("ForProfit", [0]) + steady("NonProfit", [0, 7])
    tn = mdates.date2num(t)
    ax.set_xlim(tn[0] - 2, tn[-1] + 1)
    ax.set_ylim(-0.25, 3.72)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", colors=MUTED, length=2.5, width=0.6, labelsize=6.4, pad=1.5)
    bn = mdates.date2num(bnd)
    # two dotted frames, one per era, instead of a shaded background
    for x0, x1, col in ((tn[0] - 1.5, bn - 2.0, MUTED), (bn + 2.0, tn[-1] + 0.5, RULE)):
        ax.add_patch(Rectangle((x0, 0.06), x1 - x0, 3.48, fill=False, ec=col, lw=0.8,
                               ls=(0, (1, 1.6)), zorder=1, clip_on=False))
    ax.plot([bn, bn], [0.02, 3.64], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.text(bn, 3.66, "decision date, 1 Jan 2025", ha="center", va="bottom", fontsize=6.6,
            color=INK, fontweight="semibold")
    ax.text(tn[0] + 1, 3.38, "observed history: labels already known", fontsize=6.2,
            color=MUTED, style="italic")
    ax.text(bn + 2.5, 3.38, "deployment: not yet seen", fontsize=6.2, color=MUTED,
            style="italic")
    rows = [("source group", 2.85, SRC, pick[0:1]),
            ("target group, training", 1.85, TGT, pick[1:2]),
            ("target group, test", 0.85, TGT, pick[2:3])]
    for lab, yc, col, ids in rows:
        s = seg[ids[0]]; s = (s - s.mean()) / (s.std() + 1e-9)
        ax.plot(tn, yc + 0.07 + 0.16 * s, color=col, lw=0.7, zorder=2)
        ax.text(tn[0] - 3.5, yc + 0.07, lab, ha="right", va="center", fontsize=6.6,
                color=col, fontweight="semibold", clip_on=False)
    # what each model sees
    bracket(ax, tn[0], bn - 1, 2.42, r"fit source model $f_{\mathcal{S}}$", SRC)
    bracket(ax, tn[0], bn - 1, 1.42, r"fit target-trained reference $f_{\mathcal{T}}$", TGT)
    # where both are scored
    bracket(ax, tn[0], bn - 1, 0.42, r"score both models $\rightarrow$ estimate $\hat H_T$",
            EST, bold=True)
    bracket(ax, bn + 1, tn[-1], 0.42, r"$\rightarrow$ realized $H_T$", INK, bold=True)
    txt = ",  ".join(f"{p.lower()} {tr_r(p):+.2f}" for p in ("Nursing", "Electricity", "Traffic"))
    ax.text(tn[0], -0.17, rf"agreement of $\hat H_T$ with $H_T$ (Pearson $r$): {txt}",
            fontsize=6.2, color=EST, va="center", clip_on=False)


# ----------------------------------------------------------------- panel (b)
def panel_b(ax):
    cases = [("Nursing|Ownership (For-profit -> Non-profit)|A", "nursing,\nfor-profit to\nnon-profit"),
             ("Electricity|Flat -> WorkWeek|A", "electricity,\nflat to\nworkweek clients")]
    ax.set_xlim(-0.5, 5.2)
    for sp in ("right", "top", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.set_xticks([])
    ax.tick_params(axis="y", colors=MUTED, length=2.5, width=0.6, labelsize=6.2)
    top = 0
    for i, (k, lab) in enumerate(cases):
        st = ANCH[k]; e = st["ERM-MLP"]
        app = e["apparent_gap_mae_cat"]; hd = st["headroom_per_seed"]["mae_cat"]["mean"]
        # heights follow the two quantities of Table 3 exactly (both are seed
        # means): the apparent gap lifts the error by app%, and the target-trained
        # reference then lowers it by the headroom, H% of the source model's error
        v = np.array([1.0, 1 + app / 100, (1 + app / 100) * (1 - hd / 100)])
        top = max(top, v.max())
        xo = 0.45 + i * 2.75; xs = xo + np.array([0.0, 0.7, 1.4])
        ax.plot(xs[:2], v[:2], color=GAP, lw=1.3, zorder=2)
        ax.plot(xs[1:], v[1:], color=TGT, lw=1.6, zorder=2)
        ax.scatter(xs[:2], v[:2], s=24, color=SRC, zorder=4, edgecolor="white", lw=0.7)
        ax.scatter(xs[2:], v[2:], s=24, color=TGT, zorder=4, edgecolor="white", lw=0.7)
        ax.text(xs[0] + 0.3 * (xs[1] - xs[0]) - 0.06, v[0] + 0.3 * (v[1] - v[0]),
                f"{app:+.0f}%",
                ha="right", va="center", fontsize=6.8, color=GAP, fontweight="semibold")
        hs = f"{hd:+.2f}%" if abs(hd) < 1 else f"{hd:+.1f}%"
        ax.text(xs[2] + 0.08, v[2] - (0.02 if hd > 1 else -0.02), hs, ha="left",
                va="top" if hd > 1 else "bottom", fontsize=6.8, color=TGT,
                fontweight="semibold")
        ax.text(xs[1], 0.93, lab, ha="center", va="top", fontsize=6.0, color=INK,
                linespacing=1.05)
    ax.set_ylim(0.62, top + 0.13)
    ax.set_yticks([1.0, 1.2, 1.4, 1.6])
    ax.set_ylabel("error / source error on source", fontsize=6.2, color=MUTED, labelpad=2)
    ax.text(0.04, 0.985, "apparent gap: the usual number", transform=ax.transAxes,
            fontsize=5.9, color=GAP, va="top", fontweight="semibold")
    ax.text(0.04, 0.935, "headroom: what target data buys", transform=ax.transAxes,
            fontsize=5.9, color=TGT, va="top", fontweight="semibold")


# ----------------------------------------------------------------- panel (c)
def month_profile(panel):
    spec = PANELS[panel]
    d = np.load(spec["path"], allow_pickle=True)["dates"]
    b = int(np.searchsorted(d, spec["boundary"]))
    mo = (np.asarray(d) // 100) % 100
    obs, dep = mo[:b], mo[b:]
    nyear = max(1.0, round(len(obs) / 365.25))
    per = np.array([calendar.monthrange(2023, m)[1] for m in range(1, 13)], float)
    fo = np.array([min(1.0, (obs == m).sum() / (per[m - 1] * nyear)) for m in range(1, 13)])
    dm = np.array([(dep == m).sum() > 0 for m in range(1, 13)])
    have = np.array([(obs == m).sum() > 0 for m in range(1, 13)])
    return fo, dm, have


def ring(ax, panel, title):
    fo, dm, have = month_profile(panel)
    cov = COV[(COV.panel == panel) & (COV.setting == "full history")].coverage.iloc[0]
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.62, 1.25); ax.set_aspect("equal"); ax.axis("off")
    for m in range(12):
        a0 = 90 - (m + 1) * 30 + 2.0; a1 = 90 - m * 30 - 2.0
        # inner ring: outline for every month, filled outward in proportion to the
        # share of that month present in the observed history (a single day of a
        # month shows as a sliver, not as a full segment)
        ax.add_patch(Wedge((0, 0), 0.74, a0, a1, width=0.20, fc="white", ec=RULE,
                           lw=0.6))
        if fo[m] > 0:
            ax.add_patch(Wedge((0, 0), 0.54 + 0.20 * max(fo[m], 0.08), a0, a1,
                               width=0.20 * max(fo[m], 0.08), fc=EST, ec=EST, lw=0.4))
        # outer ring: deployment months, sage if that month was observed, ochre if not
        if dm[m]:
            ax.add_patch(Wedge((0, 0), 0.97, a0, a1, width=0.17,
                               fc="white", ec=EST if have[m] else WARN, lw=0.9))
            if not have[m]:
                ax.add_patch(Wedge((0, 0), 0.97, a0, a1, width=0.17, fc=WARN, ec=WARN,
                                   lw=0.6))
        ang = np.deg2rad(90 - (m + 0.5) * 30)
        ax.text(1.12 * np.cos(ang), 1.12 * np.sin(ang), calendar.month_abbr[m + 1][0],
                ha="center", va="center", fontsize=5.4, color=MUTED)
    ax.text(0, 0.08, f"{100 * cov:.0f}%", ha="center", va="center", fontsize=7.0,
            fontweight="bold", color=EST if cov > 0.999 else WARN)
    ax.text(0, -0.17, "covered", ha="center", va="center", fontsize=4.9, color=MUTED)
    ax.text(0, -1.3, f"{title}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(0, -1.53, f"$r={tr_r(panel):+.2f}$", ha="center", va="center", fontsize=6.5,
            color=EST if cov > 0.999 else WARN, fontweight="semibold")


def main():
    fig = plt.figure(figsize=(7.2, 2.75), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[2.25, 1.05, 0.62], wspace=0.22, hspace=0.25,
                          left=0.115, right=0.995, top=0.86, bottom=0.08)
    axa = fig.add_subplot(gs[:, 0]); panel_a(axa)
    axb = fig.add_subplot(gs[:, 1]); panel_b(axb)
    axc = fig.add_subplot(gs[0, 2]); ring(axc, "Nursing", "nursing, 12 months")
    ring(fig.add_subplot(gs[1, 2]), "Solar", "solar, 8 months")
    ty = 0.955
    for x, letter, title in ((0.012, "a", "Measure headroom before the decision date"),
                             (axb.get_position().x0 - 0.03, "b",
                              "The rise in error is not the benefit"),
                             (axc.get_position().x0 - 0.005, "c", "Seasonal coverage")):
        fig.text(x, ty, letter, fontsize=10, fontweight="bold", color=INK, va="bottom")
        fig.text(x + 0.02, ty, title, fontsize=8.2, fontweight="semibold", color=INK,
                 va="bottom")
    fig.text(axc.get_position().x0 - 0.005, ty - 0.012,
             "inner ring: history observed\nouter ring: deployment months",
             fontsize=5.5, color=MUTED, va="top", linespacing=1.15)
    out = os.path.join(ROOT, "Manuscript", "overview_headroom.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.replace(".pdf", ".png"), dpi=230, bbox_inches="tight", pad_inches=0.03)
    print("wrote", out)


if __name__ == "__main__":
    main()
