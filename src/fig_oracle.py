"""Oracle / irreducibility figure (600-dpi PDF), matching the paper style.
Shows that the large apparent cross-domain gap is essentially fully irreducible:
transfer (source-trained) error approx= oracle (target-trained) error."""
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from src.figstyle import set_style, OI, INK, MUTED, FAINT, save, soft_grid

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); FIG = os.path.join(ROOT, "figures")
set_style()

def main():
    df = pd.read_csv(os.path.join(OUT, "oracle_results.csv"))
    g = df.groupby("split").agg(
        source=("source_err", "mean"), transfer=("transfer", "mean"), oracle=("oracle", "mean"),
        transfer_sd=("transfer", "std"), oracle_sd=("oracle", "std"),
        app=("apparent_gap_pct", "mean"), red=("reducible_gap_pct", "mean")).reset_index()
    splits = list(g["split"])
    fig, axes = plt.subplots(1, len(splits), figsize=(9.4, 4.1), sharey=False)
    if len(splits) == 1:
        axes = [axes]
    C_SRC = "#9AA0A6"; C_TR = OI["vermilion"]; C_OR = OI["blue"]
    for ax, (_, r) in zip(axes, g.iterrows()):
        vals = [r["source"], r["transfer"], r["oracle"]]
        errs = [0, r["transfer_sd"], r["oracle_sd"]]
        cols = [C_SRC, C_TR, C_OR]
        labs = ["Source domain\n(in-domain)", "Transfer\n(source→target)",
                "Target-retrained\n(target→target)"]
        x = np.arange(3)
        ax.bar(x, vals, 0.62, yerr=errs, capsize=3, color=cols, edgecolor="white",
               linewidth=0.8, zorder=3, error_kw=dict(lw=1.0))
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.4,
                    color=INK, fontweight="medium")
        ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8.2)
        ax.set_ylim(0, max(vals) * 1.32)
        soft_grid(ax, "y")
        ax.set_title(r["split"], loc="left", fontsize=9.4, fontweight="semibold")
        # apparent-gap bracket (source -> transfer)
        yb = max(vals) * 1.16
        ax.annotate("", xy=(1, yb), xytext=(0, yb),
                    arrowprops=dict(arrowstyle="<->", color=C_TR, lw=1.2))
        ax.text(0.5, yb + 0.004, f"apparent shift +{r['app']:.0f}%", ha="center", va="bottom",
                fontsize=7.8, color=C_TR, fontweight="medium")
        # reducible-gap bracket (transfer -> oracle)
        yb2 = max(vals) * 1.05
        ax.annotate("", xy=(2, yb2), xytext=(1, yb2),
                    arrowprops=dict(arrowstyle="<->", color=C_OR, lw=1.2))
        ax.text(1.5, yb2 + 0.004, f"headroom {r['red']:+.0f}%", ha="center", va="bottom",
                fontsize=7.8, color=C_OR, fontweight="bold")
    axes[0].set_ylabel("Out-of-distribution MAE  (hrs / resident-day)")
    fig.suptitle("The cross-facility gap is large but irreducible: training in-domain gains nothing",
                 x=0.012, ha="left", fontsize=10.6, fontweight="semibold", y=1.06)
    fig.text(0.012, 1.0,
             "ERM trained on the source group transfers to an unseen target group as well as a model trained on the target group itself "
             "(reducible gap ≈ 0), so no domain-generalization method can close it.",
             ha="left", fontsize=7.6, color=MUTED)
    save(fig, os.path.join(FIG, "oracle_irreducibility.pdf"))

if __name__ == "__main__":
    main()
