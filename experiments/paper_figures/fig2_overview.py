"""Figure 2: overview of the study.

(a) What is available at a decision date: daily load of one source-group and two
    target-group electricity clients around the decision date 2014-01-01, with the
    observed history (where the estimate E is made) and the deployment period (where
    the realized value D is measured).
(b) The increase in error on the target group (apparent gap) and the benefit of
    retraining on it (headroom) for the six anchor splits (values of Table 3).
(c) Calendar coverage: share of each calendar month's days observed in the history
    available at the decision date, by panel; outlined cells are the deployment
    months; r is the agreement of estimate and realized headroom (Table 6).

Data: cache/electricity/panel_elec.npz (observed-era scaling).
Output: ../figures/fig2_overview.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "..", "..", "cache", "electricity", "panel_elec.npz")
OUT = os.path.join(HERE, "..", "..", "Manuscript", "figures", "fig2_overview.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SRC, TGT = "#0072B2", "#D55E00"          # validated categorical pair (Okabe-Ito)
INK, MUTED, GRID = "#222222", "#5F5F5F", "#E3E3E3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": "#8A8A8A", "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "savefig.dpi": 600,
})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel_label(ax, s, x=-0.02):
    ax.text(x, 1.06, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


# ------------------------------------------------------------------ panel (a) data
d = np.load(PANEL, allow_pickle=True)
dates = pd.to_datetime(d["dates"].astype(str))
X = d["X"].sum(2)                                   # daily total, scaled by the client's observed-era mean
grp = np.asarray(d["state"]).astype(str)
obs = dates < "2014-01-01"


def weekly_amp(i):
    s = pd.Series(X[i, obs], index=dates[obs])
    return s.groupby(s.index.dayofweek).mean().std()


def pick(group, k, seed=0):
    idx = np.where(grp == group)[0]
    amp = np.array([weekly_amp(i) for i in idx])
    order = idx[np.argsort(np.abs(amp - np.median(amp)))]   # typical clients, not extremes
    return order[:k]


src_id = pick("Flat", 1)[0]
tgt_ids = pick("WorkWeek", 2)
win = (dates >= "2013-11-01") & (dates < "2014-03-01")

# ------------------------------------------------------------------ figure layout
fig = plt.figure(figsize=(7.2, 3.9))
outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.95], hspace=0.62, left=0.07, right=0.97, top=0.9, bottom=0.11)
top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.45, 1.0], wspace=0.42)
bot = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[1.0, 1.0], wspace=0.08)
gs = [top[0], bot[0], bot[1], top[1]]

# (a) time series
ax = fig.add_subplot(gs[0])
t = dates[win]
ax.axvspan(t[0], pd.Timestamp("2014-01-01"), color="#F1F1F1", lw=0)
ax.plot(t, X[src_id, win], color=SRC, lw=0.9, label="Source-group client (flat profile)")
ax.plot(t, X[tgt_ids[0], win], color=TGT, lw=0.9, label="Target-group client (workweek profile)")
ax.axvline(pd.Timestamp("2014-01-01"), color=INK, lw=0.8, ls=(0, (3, 2)))
lo, hi = 0.45, 1.35
ax.set_ylim(lo, hi + 0.28 * (hi - lo))
yb = hi + 0.09 * (hi - lo)
for x0, x1, lab in ((t[0] + pd.Timedelta(days=1), pd.Timestamp("2013-12-27"), "observed history: estimate $E$"),
                    (pd.Timestamp("2014-01-06"), t[-1], "deployment: realized $D$")):
    ax.plot([x0, x1], [yb, yb], color=MUTED, lw=0.7)
    for xx in (x0, x1):
        ax.plot([xx, xx], [yb - 0.035 * (hi - lo), yb], color=MUTED, lw=0.7)
    ax.text(x0 + (x1 - x0) / 2, yb + 0.03 * (hi - lo), lab, ha="center", va="bottom", fontsize=6.5, color=MUTED)
ax.text(pd.Timestamp("2014-01-01"), lo + 0.02, " decision date", ha="left", va="bottom", fontsize=6.3, color=INK)
ax.set_ylabel("Daily load (relative)"); ax.set_yticks([0.6, 0.8, 1.0, 1.2])
ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
ax.set_xlabel("")
plt.setp(ax.get_xticklabels(), rotation=0)
ax.set_xticks([pd.Timestamp(x) for x in ("2013-11-01", "2013-12-01", "2014-01-01", "2014-02-01")])
ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2, frameon=False, fontsize=6.6, handlelength=1.4)
ax.set_title("Data at a decision date (electricity)", loc="left")
clean(ax)
panel_label(ax, "a", x=-0.13)

# (b) apparent gap and headroom, Table 3
names = ["Nursing, ownership", "Nursing, geography", "Electricity, F$\\to$W",
         "Electricity, W$\\to$F", "Traffic, F$\\to$W", "Solar, F$\\to$W"]
gap = np.array([27, 15, 63, -29, 19, 16], float)
head = np.array([0.07, 0.08, 7.1, -2.4, 0.9, 7.8])
y = np.arange(len(names))[::-1]
for j, (vals, title, col, xl) in enumerate(((gap, "Increase in error on the target group (apparent gap)", "#9A9A9A", (-45, 85)),
                                           (head, "Benefit of retraining on the target group (headroom $H_T$)", TGT, (-4.5, 10.5)))):
    ax = fig.add_subplot(gs[1 + j])
    ax.barh(y, vals, height=0.62, color=col, lw=0)
    ax.axvline(0, color="#8A8A8A", lw=0.6)
    for yy, v in zip(y, vals):
        txt = (f"{v:+.0f}" if j == 0 else (f"{v:+.2f}" if abs(v) < 0.5 else f"{v:+.1f}")).replace("-", "−")
        ax.text(v + (xl[1] - xl[0]) * (0.03 if v >= 0 else -0.03), yy, txt, va="center",
                ha="left" if v >= 0 else "right", fontsize=6.3, color=INK)
    ax.set_xlim(*xl)
    ax.set_yticks(y)
    ax.set_yticklabels(names if j == 0 else [""] * len(names), fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("relative to the source model on target entities (%)" if j else "relative to held-out source entities (%)", fontsize=6.8)
    ax.set_title(title, loc="left")
    clean(ax, grid_axis="x")
    if j == 0:
        panel_label(ax, "b", x=-0.42)

# (c) calendar coverage heatmap
SPAN = {"Nursing": ("2024-01-01", "2025-01-01", "2025-12-31"),
        "Electricity": ("2012-01-01", "2014-01-01", "2014-12-31"),
        "Traffic": ("2015-01-01", "2016-05-04", "2016-12-31"),
        "Solar": ("2006-01-01", "2006-09-02", "2006-12-31")}
R = {"Nursing": "+0.96", "Electricity": "+0.97", "Traffic": "+0.81", "Solar": "$-$0.07"}
rows = list(SPAN)
M = np.zeros((len(rows), 12))
DEP = np.zeros((len(rows), 12), bool)
for i, p in enumerate(rows):
    h0, b, e = (pd.Timestamp(x) for x in SPAN[p])
    hist = pd.date_range(h0, b - pd.Timedelta(days=1))
    dep = pd.date_range(b, e)
    for m in range(1, 13):
        days = pd.Series(hist[hist.month == m].day).unique()
        M[i, m - 1] = min(1.0, len(days) / pd.Period(f"2021-{m:02d}").days_in_month)
        DEP[i, m - 1] = (dep.month == m).any()
ax = fig.add_subplot(gs[3])
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1.5, aspect="auto")
for i in range(len(rows)):
    for m in range(12):
        if DEP[i, m]:
            ax.add_patch(Rectangle((m - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.3))
ax.set_xticks(range(12))
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i, p in enumerate(rows):
    ax.text(12.0, i, f"$r$ = {R[p]}", va="center", ha="left", fontsize=6.6, color=INK)
ax.set_title("Calendar coverage of the history", loc="left")
ax.set_xlabel("Calendar month (outlined: deployment months)", fontsize=7)
panel_label(ax, "c", x=-0.22)

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT.replace(".pdf", "_preview.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print("clients", src_id, tgt_ids, "->", OUT)
