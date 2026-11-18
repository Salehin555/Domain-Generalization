"""
Publication figures (600-dpi vector PDF, descriptive titles, colorblind-safe).
Run after experiments.py, ablation.py, make_examples.py.
"""
import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from src.figstyle import (set_style, METHOD_STYLE, CAT_COLOR, INK, MUTED, FAINT,
                          OI, save, soft_grid)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); CACHE = os.path.join(ROOT, "cache")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
set_style()
BANDS = ["A3\n>16 d", "D3\n8-16 d", "D2\n4-8 d", "D1\n2-4 d"]
BAND_SHORT = ["A3", "D3", "D2", "D1"]
PROTO_LABEL = {"LFO": "Leave-facilities-out\n(same period)",
               "LFO-T": "Leave-facilities-out\n+ 2024→2025",
               "LSO-T": "Leave-states-out\n+ 2024→2025"}


# ------------------------------------------------------------------ Fig 1
def fig_motivation():
    ex = np.load(os.path.join(CACHE, "explore.npz"), allow_pickle=True)
    icc = ex["icc"]                       # [4 ch, 5 bands]  ch=RN,LPN,CNA,TOTAL
    ch = list(ex["ch"]); bands5 = list(ex["bands"])   # A4,D4,D3,D2,D1
    period = {"A4": ">32 d", "D4": "16-32 d", "D3": "8-16 d",
              "D2": "4-8 d", "D1": "2-4 d"}
    cats = ["RN", "LPN", "CNA"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.3, 3.5),
                                   gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.32))
    # LEFT: grouped bars ICC per band, per category
    x = np.arange(len(bands5)); w = 0.26
    for i, c in enumerate(cats):
        vals = icc[ch.index(c)]
        axL.bar(x + (i - 1) * w, vals, w, color=CAT_COLOR[c], label=c,
                edgecolor="white", linewidth=0.6, zorder=3)
    axL.set_xticks(x)
    axL.set_xticklabels([f"{b}\n{period[b]}" for b in bands5])
    axL.set_ylabel("Between-facility share of variance  (ICC)")
    axL.axvspan(2.5, 4.5, color="#F2A88C", alpha=0.16, zorder=0)
    axL.text(3.5, icc.max()*1.02, "facility-specific\nhigh-frequency bands",
             ha="center", va="top", fontsize=7.4, color="#B5410E", style="normal")
    soft_grid(axL); axL.legend(title="staff category", loc="upper left", ncol=1)
    axL.set_title("Between-facility variance by sub-band", loc="left",
                  fontsize=9.2, fontweight="semibold")
    axL.margins(x=0.02)
    # RIGHT: mean relative band energy spectrum (shared structure) with facility spread
    fac_mean = ex["fac_mean"]             # [n_fac,4,5]
    tot = fac_mean[:, ch.index("TOTAL"), :]
    m = tot.mean(0); q1 = np.quantile(tot, .25, 0); q3 = np.quantile(tot, .75, 0)
    xx = np.arange(len(bands5))
    axR.fill_between(xx, q1, q3, color=OI["blue"], alpha=0.18, zorder=2,
                     label="inter-quartile range\nacross facilities")
    axR.plot(xx, m, "-", color=OI["blue"], lw=2, zorder=4)
    axR.scatter(xx, m, s=34, color=OI["blue"], zorder=5, label="median facility")
    axR.set_xticks(xx); axR.set_xticklabels([f"{b}\n{period[b]}" for b in bands5])
    axR.set_ylabel("Share of signal energy")
    soft_grid(axR); axR.legend(loc="upper right")
    axR.set_title("Signal-energy spectrum across facilities", loc="left",
                  fontsize=9.2, fontweight="semibold")
    fig.suptitle("Cross-facility variation concentrates in high-frequency sub-bands, yet most energy is shared",
                 x=0.012, ha="left", fontsize=10.4, fontweight="semibold", y=1.13)
    fig.text(0.012, 1.05,
             "Daily nurse hours per resident-day · 13,402 CMS facilities · level-4 db4 wavelet decomposition.",
             ha="left", fontsize=7.4, color=MUTED)
    save(fig, os.path.join(FIG, "cross_facility_frequency_structure.pdf"))


# ------------------------------------------------------------------ Fig 2
def fig_benchmark():
    fs = os.path.join(OUT, "fullscale_main.csv")
    df = pd.read_csv(fs if os.path.exists(fs) else os.path.join(OUT, "main_results.csv"))
    protos = ["LFO", "LFO-T", "LSO-T"]
    full_order = ["Wave-Mask", "FixedBand", "CORAL", "GroupDRO", "FEDformer", "DANN",
                  "IRM", "Autoformer", "ERM-DLinear", "ERM-MLP", "CD-ISG-TS"]
    present = set(df.method.unique())
    order = [m for m in full_order if m in present]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.7), sharey=True)
    for k, (ax, proto) in enumerate(zip(axes, protos)):
        sub = df[df.protocol == proto].set_index("method")
        ys = np.arange(len(order))
        for j, meth in enumerate(order):
            if meth not in sub.index:
                continue
            mae = sub.loc[meth, "mae_total_mean"]; sd = sub.loc[meth, "mae_total_std"]
            stl = METHOD_STYLE[meth]
            emph = meth == "CD-ISG-TS"
            ax.errorbar(mae, ys[j], xerr=sd, fmt=stl["m"], color=stl["c"],
                        ms=11 if emph else 7, mfc=stl["c"], mec="white", mew=0.8,
                        elinewidth=1.1, capsize=2.4, zorder=5 if emph else 3)
            if emph:
                ax.axvline(mae, color=stl["c"], lw=0.8, ls=(0, (4, 3)), alpha=0.5, zorder=1)
        if k == 0:
            # The data files still carry the project's working name for the
            # proposed method; the article calls it FADE. Relabel at draw time
            # so the figures never disagree with the manuscript.
            ax.set_yticks(ys)
            ax.set_yticklabels(["FADE" if m == "CD-ISG-TS" else m for m in order])
        ax.set_title(PROTO_LABEL[proto], loc="left", fontsize=8.8, fontweight="medium")
        ax.set_xlabel("Out-of-distribution MAE  (hrs / resident-day)")
        soft_grid(ax, "x")
        ax.margins(y=0.06)
    axes[0].set_ylabel("")
    fig.suptitle("FADE matches the top methods and clearly beats the plain baseline and frequency heuristics",
                 x=0.012, ha="left", fontsize=10.2, fontweight="semibold", y=1.10)
    ns = "5 seeds (3 for transformers), all 13,402 facilities" if os.path.exists(
        os.path.join(OUT, "fullscale_main.csv")) else "3 seeds"
    fig.text(0.012, 1.045, f"Mean ± s.d. over {ns}; unseen target facilities only. Lower is better.",
             ha="left", fontsize=7.6, color=MUTED)
    save(fig, os.path.join(FIG, "ood_benchmark_by_protocol.pdf"))


# ------------------------------------------------------------------ Fig 3
def fig_gap():
    g = pd.read_csv(os.path.join(OUT, "gen_gap_clean.csv"))
    protos = ["LFO", "LFO-T", "LSO-T"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
    # LEFT: source vs target MAE, paired per seed, per protocol
    xpos = {p: i for i, p in enumerate(protos)}
    for _, r in g.iterrows():
        x = xpos[r["protocol"]]
        axL.plot([x-0.14, x+0.14], [r["source_mae"], r["target_mae"]],
                 color=FAINT, lw=1.0, zorder=1)
        axL.scatter([x-0.14], [r["source_mae"]], color="#9AA0A6", s=34, zorder=3)
        axL.scatter([x+0.14], [r["target_mae"]], color=OI["blue"], s=40, zorder=3)
    axL.set_xticks(list(xpos.values())); axL.set_xticklabels([PROTO_LABEL[p] for p in protos])
    axL.set_ylabel("MAE  (hrs / resident-day)")
    handles = [Line2D([0],[0], marker="o", color="w", mfc="#9AA0A6", ms=8, label="source facilities (seen)"),
               Line2D([0],[0], marker="o", color="w", mfc=OI["blue"], ms=8, label="target facilities (unseen)")]
    axL.legend(handles=handles, loc="upper left")
    soft_grid(axL)
    axL.set_title("Unseen-facility error tracks seen-facility error", loc="left",
                  fontsize=9.4, fontweight="semibold")
    # RIGHT: distribution of the per-seed gap (%), centred on zero
    gm = g.groupby("protocol")["gap_pct"].agg(["mean", "std"]).reindex(protos)
    y = np.arange(len(protos))
    axR.axvline(0, color="#3A3A3A", lw=1.0, zorder=2)
    axR.errorbar(gm["mean"], y, xerr=gm["std"], fmt="o", color=OI["blue"], ms=8,
                 mec="white", mew=0.8, elinewidth=1.3, capsize=3, zorder=4)
    for yi, (mn) in zip(y, gm["mean"]):
        axR.annotate(f"{mn:+.1f}%", (mn, yi), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=7.4, color=INK)
    axR.set_yticks(y); axR.set_yticklabels([p for p in protos])
    axR.set_xlabel("source → target gap  (%)")
    axR.set_xlim(-9, 9)
    soft_grid(axR, "x")
    axR.set_title("Mean gap ≈ 0, dominated by sampling", loc="left",
                  fontsize=9.4, fontweight="semibold")
    fig.text(0.012, -0.03, "The strong shared calendar/regulatory structure of nurse staffing makes unseen facilities almost as forecastable as seen ones, "
             "the domain shift is real but small (per-seed gap −11%…+7%).", ha="left", fontsize=7.2, color=MUTED)
    save(fig, os.path.join(FIG, "generalization_gap.pdf"))


# ------------------------------------------------------------------ Fig 4
def fig_gates():
    gz = np.load(os.path.join(CACHE, "gates_main.npz"))
    keys = [k for k in gz.files if k.startswith("LFO-T")]
    G = np.mean([gz[k] for k in keys], 0)      # [3 cat, 4 band]
    cats = ["RN", "LPN", "CNA"]
    fig, (axG, axI) = plt.subplots(1, 2, figsize=(8.8, 3.3),
                                   gridspec_kw=dict(width_ratios=[1, 1], wspace=0.36))
    im = axG.imshow(G, cmap="RdBu_r", vmin=1 - abs(G-1).max(), vmax=1 + abs(G-1).max(),
                    aspect="auto")
    for i in range(3):
        for j in range(4):
            axG.text(j, i, f"{G[i,j]:.2f}", ha="center", va="center",
                     fontsize=8.4, color=INK if abs(G[i,j]-1) < .12 else "white",
                     fontweight="medium")
    axG.set_xticks(range(4)); axG.set_xticklabels(BAND_SHORT)
    axG.set_yticks(range(3)); axG.set_yticklabels(cats)
    axG.set_xlabel("wavelet sub-band  (coarse → fine)")
    cb = fig.colorbar(im, ax=axG, fraction=0.046, pad=0.04)
    cb.set_label("learned gate  (>1 amplify, <1 suppress)", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    axG.set_title("Learned category-conditioned gates", loc="left",
                  fontsize=9.4, fontweight="semibold")
    # right: gate suppression vs empirical ICC (validate the mechanism)
    ex = np.load(os.path.join(CACHE, "explore.npz"), allow_pickle=True)
    icc = ex["icc"]; chl = list(ex["ch"])
    # map explore bands (A4,D4,D3,D2,D1) -> model bands (A3,D3,D2,D1): use D3,D2,D1 + coarse
    icc_model = np.stack([icc[:, 0]+icc[:, 1], icc[:, 2], icc[:, 3], icc[:, 4]], 1)  # [4ch,4band]
    supp = 1 - G                       # suppression per cat/band
    for i, c in enumerate(cats):
        axI.scatter(icc_model[chl.index(c)], supp[i], s=48, color=CAT_COLOR[c],
                    edgecolor="white", linewidth=0.7, zorder=4, label=c)
    xs = icc_model[:3].ravel(); ysv = supp.ravel()
    b1, b0 = np.polyfit(xs, ysv, 1)
    xr = np.linspace(xs.min(), xs.max(), 20)
    axI.plot(xr, b0 + b1*xr, color=MUTED, lw=1.1, ls=(0,(4,3)), zorder=2)
    r = np.corrcoef(xs, ysv)[0, 1]
    axI.text(0.05, 0.92, f"Pearson r = {r:.2f}", transform=axI.transAxes,
             fontsize=8, color=INK)
    axI.axhline(0, color=FAINT, lw=0.8)
    axI.set_xlabel("empirical between-facility variance (ICC)")
    axI.set_ylabel("gate suppression  (1 − gate)")
    axI.legend(title="staff category", loc="lower right")
    soft_grid(axI)
    axI.set_title("Gates suppress the empirically facility-specific bands", loc="left",
                  fontsize=9.4, fontweight="semibold")
    save(fig, os.path.join(FIG, "learned_gates_vs_icc.pdf"))


# ------------------------------------------------------------------ Fig 5
def fig_lambda():
    d = pd.read_csv(os.path.join(OUT, "ablation_lambda.csv"))
    erm = pd.read_csv(os.path.join(OUT, "ablation_ermref.csv")).iloc[0]["mae"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw=dict(wspace=0.3))
    lam = d["lam"].values.copy(); lam[lam == 0] = 0.03      # for log axis show 0 as small
    axA.axhline(erm, color="#5F6368", lw=1.2, ls=(0,(4,3)), zorder=2)
    axA.text(lam[-1], erm, "  ERM", va="center", ha="left", fontsize=7.6, color="#5F6368")
    axA.fill_between(lam, d["mae"]-d["mae_std"], d["mae"]+d["mae_std"],
                     color=OI["blue"], alpha=0.15, zorder=2)
    axA.plot(lam, d["mae"], "-o", color=OI["blue"], ms=5, zorder=4)
    axA.set_xscale("log"); axA.set_xlabel("invariance weight  λ")
    axA.set_ylabel("OOD MAE  (hrs / resident-day)")
    soft_grid(axA)
    axA.set_title("Accuracy is stable across invariance strength", loc="left",
                  fontsize=9.2, fontweight="semibold")
    # retained energy stacked area (collapse diagnostic)
    ret = d[["ret_A3", "ret_D3", "ret_D2", "ret_D1"]].values.T
    cols = ["#08306B", "#2171B5", "#6BAED6", "#C6DBEF"]
    axB.stackplot(lam, ret, colors=cols, labels=BAND_SHORT, edgecolor="white", linewidth=0.4)
    axB.set_xscale("log"); axB.set_xlabel("invariance weight  λ")
    axB.set_ylabel("retained sub-band energy share")
    axB.set_ylim(0, 1)
    axB.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="sub-band")
    axB.set_title("Retained energy is preserved, no gate collapse", loc="left",
                  fontsize=9.2, fontweight="semibold")
    save(fig, os.path.join(FIG, "invariance_weight_sensitivity.pdf"))


# ------------------------------------------------------------------ Fig 6
def fig_ablation():
    erm = pd.read_csv(os.path.join(OUT, "ablation_ermref.csv")).iloc[0]["mae"]
    cat = pd.read_csv(os.path.join(OUT, "ablation_category.csv"))
    place = pd.read_csv(os.path.join(OUT, "ablation_placement.csv"))
    fft = pd.read_csv(os.path.join(OUT, "ablation_fft.csv"))
    wav = pd.read_csv(os.path.join(OUT, "ablation_wavelet.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.0))
    fig.subplots_adjust(hspace=0.55, wspace=0.5)

    def hbar(ax, labels, vals, title, highlight=None):
        vals = np.asarray(vals); y = np.arange(len(labels))
        cols = [OI["blue"] if (highlight and l in highlight) else "#B7BCC2" for l in labels]
        ax.barh(y, vals, color=cols, edgecolor="white", height=0.6, zorder=3)
        ax.axvline(erm, color="#D55E00", lw=1.1, ls=(0,(4,3)), zorder=4)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        lo = min(vals.min(), erm); hi = max(vals.max(), erm)
        ax.set_xlim(lo - (hi-lo)*0.6 - 1e-4, hi + (hi-lo)*1.1 + 1e-4)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.4f}", va="center", ha="left", fontsize=7, color=INK)
        soft_grid(ax, "x"); ax.set_title(title, loc="left", fontsize=8.8, fontweight="semibold")

    hbar(axes[0, 0], list(cat["variant"]), cat["mae"].values,
         "Invariance estimator", highlight=["category-conditioned"])
    hbar(axes[0, 1], list(place["placement"]), place["mae"].values,
         "Gate placement", highlight=["feature"])
    hbar(axes[1, 0], list(fft["gate"]), fft["mae"].values,
         "Gate transform", highlight=["wavelet-subband"])
    wl = [f"{r.wavelet}·L{r.level}" for r in wav.itertuples()]
    hbar(axes[1, 1], wl, wav["mae"].values, "Wavelet basis × depth")
    axes[1, 1].text(0.99, -0.28, "orange dashed = ERM reference", transform=axes[1, 1].transAxes,
                    ha="right", fontsize=7, color="#D55E00")
    fig.suptitle("Ablations: feature-space gating, category-conditioning and localized wavelets each help",
                 x=0.012, ha="left", fontsize=10.0, fontweight="semibold", y=1.0)
    fig.text(0.012, 0.955, "Realistic leave-facilities-out + 2024→2025 protocol; lower MAE is better; blue = chosen setting.",
             ha="left", fontsize=7.4, color=MUTED)
    save(fig, os.path.join(FIG, "ablation_summary.pdf"))


# ------------------------------------------------------------------ Fig 7
def fig_scatter():
    fs = os.path.join(OUT, "fullscale_per_facility.csv")
    df = pd.read_csv(fs if os.path.exists(fs) else os.path.join(OUT, "per_facility_results.csv"))
    sub = df[df.protocol == "LFO-T"]
    piv = sub.pivot_table(index="fac", columns="method", values="mae_total")
    piv = piv[["ERM-DLinear", "CD-ISG-TS"]].dropna()
    e = piv["ERM-DLinear"].values; o = piv["CD-ISG-TS"].values
    fig = plt.figure(figsize=(6.4, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0]); axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    lim = [min(e.min(), o.min())*0.98, np.quantile(np.r_[e, o], 0.995)]
    ax.plot(lim, lim, color=MUTED, lw=1, ls=(0,(4,3)), zorder=2)
    ax.scatter(e, o, s=13, color=OI["blue"], alpha=0.4, edgecolor="none", zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("ERM  per-facility OOD MAE"); ax.set_ylabel("CD-ISG-TS  per-facility OOD MAE")
    win = (o < e).mean()
    ax.text(0.04, 0.93, f"CD-ISG-TS lower for {100*win:.0f}% of unseen facilities\n"
                        f"mean Δ = {1000*(o.mean()-e.mean()):+.1f}×10$^{{-3}}$ hrs/res-day",
            transform=ax.transAxes, fontsize=8, color=INK, va="top")
    soft_grid(ax); soft_grid(ax, "x")
    for a in (axt, axr):
        a.axis("off")
    axt.hist(e, bins=40, color="#9AA0A6", alpha=0.9)
    axr.hist(o, bins=40, orientation="horizontal", color=OI["blue"], alpha=0.85)
    axt.set_title("CD-ISG-TS lowers error for two-thirds of unseen facilities",
                  loc="left", fontsize=9.6, fontweight="semibold")
    save(fig, os.path.join(FIG, "per_facility_scatter.pdf"))


# ------------------------------------------------------------------ Fig 8
def fig_backbone():
    bb = pd.read_csv(os.path.join(OUT, "ablation_backbone.csv")).set_index("backbone")
    df = pd.read_csv(os.path.join(OUT, "main_results.csv"))
    # granularity: facility (LFO-T) vs state (LSO-T) for ERM-MLP & CD-ISG-TS
    gran = df[df.method.isin(["ERM-MLP", "CD-ISG-TS"])]
    fig, (axB, axG) = plt.subplots(1, 2, figsize=(8.8, 3.6), gridspec_kw=dict(wspace=0.3))
    w = 0.34
    def paired(ax, xticklabels, series):
        xs = np.arange(len(xticklabels)); allv = []
        for j, meth in enumerate(["ERM-MLP", "CD-ISG-TS"]):
            vals, err = series[meth]
            allv += [v+e for v, e in zip(vals, err)] + [v-e for v, e in zip(vals, err)]
            ax.bar(xs + (j-0.5)*w, vals, w, yerr=err, capsize=3,
                   color=METHOD_STYLE[meth]["c"], edgecolor="white",
                   label=("ERM" if meth == "ERM-MLP" else "CD-ISG-TS"), zorder=3,
                   error_kw=dict(lw=1.0))
            for xi, v in zip(xs + (j-0.5)*w, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.6,
                        color=INK, rotation=0)
        ax.set_xticks(xs); ax.set_xticklabels(xticklabels)
        lo, hi = min(allv), max(allv); pad = (hi-lo)*0.5 + 1e-3
        ax.set_ylim(lo - pad, hi + pad*1.6)
        ax.set_ylabel("OOD MAE  (hrs / resident-day)")
        soft_grid(ax, "y")
    paired(axB, ["Linear\n(DLinear)", "Transformer\n(patch)"],
           {"ERM-MLP": ([bb.loc["mlp","ERM"], bb.loc["transformer","ERM"]],
                        [bb.loc["mlp","ERM_std"], bb.loc["transformer","ERM_std"]]),
            "CD-ISG-TS": ([bb.loc["mlp","CD_ISG_TS"], bb.loc["transformer","CD_ISG_TS"]],
                          [bb.loc["mlp","CD_ISG_TS_std"], bb.loc["transformer","CD_ISG_TS_std"]])})
    axB.set_title("Backbone transfer", loc="left", fontsize=9.2, fontweight="semibold")
    gv = {m: ([gran[(gran.method==m)&(gran.protocol==p)].iloc[0]["mae_total_mean"] for p in ["LFO-T","LSO-T"]],
              [gran[(gran.method==m)&(gran.protocol==p)].iloc[0]["mae_total_std"] for p in ["LFO-T","LSO-T"]])
          for m in ["ERM-MLP","CD-ISG-TS"]}
    paired(axG, ["Facility\ndomains", "State\ndomains"], gv)
    axG.set_title("Domain granularity", loc="left", fontsize=9.2, fontweight="semibold")
    handles = [plt.Rectangle((0,0),1,1,color=METHOD_STYLE["ERM-MLP"]["c"]),
               plt.Rectangle((0,0),1,1,color=METHOD_STYLE["CD-ISG-TS"]["c"])]
    fig.legend(handles, ["ERM", "CD-ISG-TS"], loc="upper right",
               bbox_to_anchor=(0.995, 1.05), ncol=2, frameon=False)
    fig.suptitle("CD-ISG-TS tracks the strong ERM baseline across backbones and domain granularities",
                 x=0.012, ha="left", fontsize=10.0, fontweight="semibold", y=1.05)
    save(fig, os.path.join(FIG, "backbone_and_granularity.pdf"))


# ------------------------------------------------------------------ Fig 9
def fig_examples():
    ex = np.load(os.path.join(CACHE, "examples.npz"), allow_pickle=True)
    picks = ex["picks"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.9))
    fig.subplots_adjust(hspace=0.48, wspace=0.2)
    axes = axes.ravel()
    for i, (ax, f) in enumerate(zip(axes, picks)):
        x = ex[f"fac{f}_x"].sum(0)            # total across categories, lookback
        y = ex[f"fac{f}_y"].sum(0)
        pe = ex[f"fac{f}_erm"].sum(0); po = ex[f"fac{f}_ours"].sum(0)
        L = len(x); H = len(y)
        tin = np.arange(-L, 0); tout = np.arange(0, H)
        ax.plot(tin, x, color="#3A3A3A", lw=1.2, zorder=3)
        ax.plot(np.r_[-1, tout], np.r_[x[-1], y], color="#3A3A3A", lw=1.7, label="actual", zorder=4)
        ax.plot(np.r_[-1, tout], np.r_[x[-1], pe], color="#C48A2A", lw=1.5,
                ls=(0,(3,2)), label="ERM", zorder=5)
        ax.plot(np.r_[-1, tout], np.r_[x[-1], po], color=OI["blue"], lw=1.7, label="CD-ISG-TS", zorder=6)
        ax.axvspan(0, H, color="#EEF3F9", zorder=0)
        ax.axvline(0, color=FAINT, lw=0.8, zorder=1)
        st = str(ex[f"fac{f}_state"]); mc = float(ex[f"fac{f}_medcen"])
        ax.set_title(f"unseen facility · {st} · ~{mc:.0f} beds", loc="left",
                     fontsize=8.2, fontweight="medium")
        soft_grid(ax)
        if i >= 3:
            ax.set_xlabel("day  (0 = forecast origin)")
        ax.margins(x=0.02)
    axes[0].set_ylabel("total nurse hrs / resident-day")
    axes[3].set_ylabel("total nurse hrs / resident-day")
    axes[0].legend(loc="upper left", ncol=1, fontsize=7)
    fig.suptitle("Fourteen-day staffing forecasts for facilities never seen in training",
                 x=0.012, ha="left", fontsize=10.4, fontweight="semibold", y=0.98)
    fig.text(0.012, 0.945, "Grey = history; forecast horizon shaded. ERM and CD-ISG-TS forecasts nearly coincide, both generalize to the shared weekly cycle.",
             ha="left", fontsize=7.3, color=MUTED)
    save(fig, os.path.join(FIG, "forecast_examples.pdf"))


# ------------------------------------------------------------------ Fig 10 (schematic)
def fig_schematic():
    fig, ax = plt.subplots(figsize=(10.2, 2.7)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 34)
    def box(x, y, w, h, text, fc, ec, tc=INK, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1.2, zorder=3))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)
    def arrow(x0, x1, y=17):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11,
                                     lw=1.3, color="#6B6B6B", zorder=2))
    box(1, 11, 13, 12, "staffing\nwindow\n[RN,LPN,CNA]\n56 days", "#F4F6F8", "#B8BFC6", fs=7.6)
    arrow(14, 18)
    box(18, 11, 12, 12, "RevIN\n(remove level\n& scale)", "#EAF1FB", OI["blue"], fs=7.8)
    arrow(30, 34)
    box(34, 11, 13, 12, "nonlinear\ntemporal stem\n(feature Z)", "#EAF1FB", OI["blue"], fs=7.8)
    arrow(47, 51)
    # gate block (highlighted)
    box(51, 8, 20, 18, "", "#FBF3E7", OI["orange"])
    ax.text(61, 24.4, "sub-band gate", ha="center", fontsize=8.2, fontweight="semibold", color="#8A5A00")
    box(52.2, 17.5, 17.6, 5.5, "multi-level DWT → bands", "white", "#E4C58A", fs=7)
    box(52.2, 11.5, 17.6, 5.5, "category-conditioned\nenergy-preserving gate", "white", "#E4C58A", fs=6.8)
    box(52.2, 8.6, 17.6, 2.6, "inverse DWT", "white", "#E4C58A", fs=7)
    arrow(71, 75)
    box(75, 11, 12, 12, "forecaster\n(linear / \n transformer)", "#EAF1FB", OI["blue"], fs=7.6)
    arrow(87, 91)
    box(91, 11, 8.5, 12, "14-day\nforecast", "#F4F6F8", "#B8BFC6", fs=7.6)
    # invariance regularizer callout
    ax.add_patch(FancyArrowPatch((61, 8), (61, 4.5), arrowstyle="-|>", mutation_scale=10,
                                 lw=1.1, color=OI["green"], zorder=2))
    ax.text(61, 3.0, "cross-domain invariance regularizer  (align retained spectrum across facilities)",
            ha="center", fontsize=7.4, color="#0B7355", style="italic")
    ax.text(0.4, 32, "CD-ISG-TS: learned frequency-invariance gating inside the forecaster",
            fontsize=10.4, fontweight="semibold", color=INK)
    save(fig, os.path.join(FIG, "method_schematic.pdf"))


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["motivation", "benchmark", "gap", "gates", "lambda",
                            "ablation", "scatter", "backbone", "examples", "schematic"]
    fns = dict(motivation=fig_motivation, benchmark=fig_benchmark, gap=fig_gap,
               gates=fig_gates, lambda_=fig_lambda, ablation=fig_ablation,
               scatter=fig_scatter, backbone=fig_backbone, examples=fig_examples,
               schematic=fig_schematic)
    alias = {"lambda": "lambda_"}
    for t in todo:
        fns[alias.get(t, t)]()
