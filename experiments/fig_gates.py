"""Gate figures, from results/gate_runs.csv and gate_icc_aligned.csv.

learned_gates_vs_icc.pdf (main text): left, mean learned gate over 11 runs
(five facility splits, two disjoint facility halves x three seeds) with the
across-run SD; right, gate suppression against the ICC computed with the
model's own transform (no bands are summed), one point per run and cell.
cross_facility_frequency_structure.pdf (supplement): aligned ICC and mean
relative energy by band and staff category.
"""
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.figstyle import CAT_COLOR, FAINT, INK, MUTED, save, set_style, soft_grid  # noqa

set_style()
CATS, BANDS = ["RN", "LPN", "CNA"], ["A3", "D3", "D2", "D1"]
PER = {"A3": ">16 d", "D3": "8-16 d", "D2": "4-8 d", "D1": "2-4 d"}


def main():
    R = pd.read_csv(os.path.join(HERE, "results", "gate_runs.csv"))
    I = pd.read_csv(os.path.join(HERE, "results", "gate_icc_aligned.csv"))
    icc = I.pivot(index="channel", columns="band", values="ICC").loc[CATS, BANDS].to_numpy()
    G = np.stack([R[[f"g_{c}_{b}" for b in BANDS]].to_numpy() for c in CATS], 1)  # [run,3,4]
    Gm, Gs = G.mean(0), G.std(0)

    fig, (axG, axI) = plt.subplots(1, 2, figsize=(8.8, 3.3),
                                   gridspec_kw=dict(width_ratios=[1, 1], wspace=0.38))
    dev = max(abs(Gm - 1).max(), 0.05)
    im = axG.imshow(Gm, cmap="RdBu_r", vmin=1 - dev, vmax=1 + dev, aspect="auto")
    for i in range(3):
        for j in range(4):
            axG.text(j, i, f"{Gm[i, j]:.2f}\n$\\pm${Gs[i, j]:.2f}", ha="center",
                     va="center", fontsize=7.4,
                     color="white" if abs(Gm[i, j] - 1) > 0.6 * dev else INK)
    axG.set_xticks(range(4)); axG.set_xticklabels([f"{b}\n{PER[b]}" for b in BANDS], fontsize=7.6)
    axG.set_yticks(range(3)); axG.set_yticklabels(CATS)
    cb = fig.colorbar(im, ax=axG, fraction=0.046, pad=0.04)
    cb.set_label("learned gate (mean over 11 runs)", fontsize=7.4); cb.ax.tick_params(labelsize=7)
    axG.set_title("Learned gates, mean $\\pm$ SD across runs", loc="left", fontsize=9.2,
                  fontweight="semibold")
    rs = []
    for k in range(len(R)):
        supp = 1 - G[k]
        for i, c in enumerate(CATS):
            axI.scatter(icc[i], supp[i], s=12, color=CAT_COLOR[c], alpha=0.35,
                        edgecolor="none", zorder=2)
        rs.append(np.corrcoef(icc.ravel(), supp.ravel())[0, 1])
    for i, c in enumerate(CATS):
        axI.scatter(icc[i], 1 - Gm[i], s=46, color=CAT_COLOR[c], edgecolor="white",
                    lw=0.7, zorder=4, label=c)
    axI.axhline(0, color=FAINT, lw=0.8)
    axI.set_xlabel("between-facility share of variance (ICC), model bands")
    axI.set_ylabel("gate suppression (1 $-$ gate)")
    axI.legend(title="staff category", loc="lower right", fontsize=7.2, title_fontsize=7.2)
    axI.set_title(f"Pearson r per run: {np.mean(rs):+.2f} (range {min(rs):+.2f} to {max(rs):+.2f})",
                  loc="left", fontsize=9.0, fontweight="semibold")
    soft_grid(axI)
    out = os.path.join(ROOT, "Manuscript", "learned_gates_vs_icc.pdf")
    save(fig, out); shutil.copy(out, os.path.join(HERE, "figures", os.path.basename(out)))

    E = I.pivot(index="channel", columns="band", values="mean_rel_energy").loc[CATS, BANDS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.1), gridspec_kw=dict(wspace=0.32))
    x = np.arange(4); w = 0.26
    for i, c in enumerate(CATS):
        a1.bar(x + (i - 1) * w, icc[i], w, color=CAT_COLOR[c], label=c)
        a2.bar(x + (i - 1) * w, E.loc[c].to_numpy(), w, color=CAT_COLOR[c], label=c)
    for a, t, yl in ((a1, "Between-facility share of variance (ICC)", "ICC"),
                     (a2, "Mean relative energy", "share of window energy")):
        a.set_xticks(x); a.set_xticklabels([f"{b}\n{PER[b]}" for b in BANDS], fontsize=7.6)
        a.set_title(t, loc="left", fontsize=9.2, fontweight="semibold"); a.set_ylabel(yl)
        soft_grid(a)
    a1.legend(fontsize=7.4, frameon=False)
    out = os.path.join(ROOT, "Manuscript", "cross_facility_frequency_structure.pdf")
    save(fig, out); shutil.copy(out, os.path.join(HERE, "figures", os.path.basename(out)))


if __name__ == "__main__":
    main()
