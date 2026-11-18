"""
Two-regime validation figure (second dataset).

LEFT : the reducibility diagnostic across both datasets and all four leave-group-out
       splits, showing that the diagnostic separates an irreducible regime (nursing,
       and one electricity direction) from a reducible one (electricity Flat ->
       WorkWeek, +8.3%).
RIGHT: on the one split where headroom demonstrably exists, how much of it each
       method actually converts. An oracle trained on the target group captures the
       full 8.3%; every domain-generalization method captures a small fraction of it.
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


def main():
    nur = pd.read_csv(os.path.join(OUT, "oracle_results.csv"))
    ele = pd.read_csv(os.path.join(OUT, "oracle_electricity.csv"))
    bench = pd.read_csv(os.path.join(OUT, "electricity_benchmark.csv"))

    # ---------------- left: reducible gap across both datasets ----------------
    rows = []
    for name, g in nur.groupby("split"):
        short = "Nursing\nownership" if "Ownership" in name else "Nursing\ngeography"
        rows.append((short, g.reducible_gap_pct.mean(), g.reducible_gap_pct.std(), "Nursing"))
    for name, g in ele.groupby("split"):
        short = ("Electricity\nWorkWeek to Flat" if "WorkWeek ->" in name
                 else "Electricity\nFlat to WorkWeek")
        rows.append((short, g.reducible_gap_pct.mean(), g.reducible_gap_pct.std(), "Electricity"))
    R = pd.DataFrame(rows, columns=["split", "mean", "sd", "dataset"])
    R = R.sort_values("mean").reset_index(drop=True)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.0))

    colors = [OI["blue"] if m > 3 else FAINT for m in R["mean"]]
    y = np.arange(len(R))
    axL.barh(y, R["mean"], xerr=R["sd"], color=colors, height=0.6,
             error_kw=dict(ecolor=MUTED, lw=1.0, capsize=3))
    axL.axvline(0, color=INK, lw=1.0)
    axL.set_yticks(y); axL.set_yticklabels(R["split"], fontsize=8.5)
    axL.set_xlabel("reducible gap (%)   [ (transfer - oracle) / transfer ]")
    axL.set_title("The diagnostic separates two regimes", loc="left",
                  fontweight="semibold", fontsize=10.5)
    axL.axvspan(-12, 3, color="#F2F2F2", zorder=0)
    axL.text(-7.5, 1.5, "irreducible:\nno method can help", fontsize=8,
             color=MUTED, ha="center", va="center")
    axL.text(6.4, 2.55, "reducible:\nheadroom exists", fontsize=8,
             color=OI["blue"], ha="center", va="center", fontweight="semibold")
    soft_grid(axL, axis="x")

    # ---------------- right: how much of the headroom is converted ------------
    s = bench[bench.split == "Flat -> WorkWeek"]
    ref = s[s.method == "ERM-MLP"].mae_total.mean()
    caps = ((ref - s.groupby("method").mae_total.mean()) / ref * 100).sort_values()
    caps = caps[caps.index != "ERM-MLP"]
    order = caps.index.tolist()
    vals = caps.values

    yy = np.arange(len(order))
    cols = [OI["blue"] if "CD-ISG-TS" in m else FAINT for m in order]
    axR.barh(yy, vals, color=cols, height=0.62)
    axR.axvline(0, color=INK, lw=1.0)
    axR.axvline(8.26, color=OI["vermilion"], lw=1.4, ls="--")
    axR.text(8.26, len(order) - 0.4, "  oracle headroom 8.3%", color=OI["vermilion"],
             fontsize=8.6, va="top", fontweight="semibold")
    axR.set_yticks(yy); axR.set_yticklabels(order, fontsize=8.5)
    axR.set_xlabel("error reduction vs pooled ERM (%)")
    axR.set_title("None of the methods converts the available headroom", loc="left",
                  fontweight="semibold", fontsize=10.5)
    axR.set_xlim(min(vals.min() * 1.25, -1.5), 9.6)
    soft_grid(axR, axis="x")

    fig.suptitle("Validating the reducibility diagnostic on a second, unrelated panel",
                 x=0.008, ha="left", fontsize=11.6, fontweight="semibold", color=INK)
    fig.text(0.008, 0.925,
             "Left: four leave-group-out splits across two datasets. "
             "Right: on the one reducible split, measured gain of every method "
             "against the pooled baseline.",
             ha="left", fontsize=8.4, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, os.path.join(FIG, "two_regime_validation.pdf"))


if __name__ == "__main__":
    main()
