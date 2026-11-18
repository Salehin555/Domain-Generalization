"""
Full-scale benchmark: ALL facilities in the panel, 5 seeds for the compute-light
methods and 3 seeds for the heavy transformers, across the three DG protocols.
Adds the standard DG baselines CORAL, DANN, GroupDRO, and uses the FAITHFUL
Autoformer / FEDformer (Auto-Correlation / Frequency-Enhanced Block).

Model selection uses a capped source-validation set for speed; target evaluation is
on all unseen target facilities.

Outputs: outputs/fullscale_per_facility.csv, outputs/fullscale_main.csv,
         outputs/fullscale_significance.csv
"""
import numpy as np, pandas as pd, os, time
from scipy.stats import wilcoxon
from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); CACHE = os.path.join(ROOT, "cache")

NFAC = 13402                 # all facilities in the panel
L, H, STRIDE = 56, 14, 16
VAL_CAP = 800                # facilities used for model selection (speed)
EP, PAT = 24, 5

LIGHT_SEEDS = [0, 1, 2, 3, 4]
HEAVY_SEEDS = [0, 1, 2]

LIGHT = [
    ("ERM-DLinear", dict(method="dlinear_plain", epochs=EP, patience=PAT)),
    ("ERM-MLP",     dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")),
    ("Wave-Mask",   dict(method="wavemask", epochs=EP, patience=PAT, backbone="mlp")),
    ("IRM",         dict(method="irm", epochs=EP, patience=PAT, backbone="mlp", lam_irm=1.0)),
    ("CORAL",       dict(method="coral", epochs=EP, patience=PAT, backbone="mlp", lam_coral=1.0)),
    ("DANN",        dict(method="dann", epochs=EP, patience=PAT, backbone="mlp", lam_dann=0.5)),
    ("GroupDRO",    dict(method="groupdro", epochs=EP, patience=PAT, backbone="mlp")),
    ("FixedBand",   dict(method="fixedband", epochs=EP, patience=PAT, backbone="mlp", drop_fine=1)),
    ("CD-ISG-TS",   dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
                        lam_inv=0.5, gate_lr=0.01)),
]
HEAVY = [
    ("Autoformer", dict(method="autoformer", epochs=18, patience=4)),
    ("FEDformer",  dict(method="fedformer", epochs=18, patience=4)),
]

def masks(P, proto):
    return (None, None) if proto == "LFO" else (P.is_2024, P.is_2025)

def split(P, proto, seed):
    return P.split_states(seed) if proto == "LSO-T" else P.split_facilities(seed)

def main():
    t0 = time.time()
    P = Panel(os.path.join(CACHE, "panel.npz"), n_facilities=NFAC, L=L, H=H,
              stride=STRIDE, seed=0)
    print(f"panel: {P.Xw.shape}  facilities={P.nF}  built in {time.time()-t0:.0f}s", flush=True)
    rows = []
    for proto in ["LFO", "LFO-T", "LSO-T"]:
        tmask, vmask = masks(P, proto)
        for (label, cfg), seeds in ([(m, LIGHT_SEEDS) for m in LIGHT] +
                                    [(m, HEAVY_SEEDS) for m in HEAVY]):
            for seed in seeds:
                tr, va, te = split(P, proto, seed)
                rng = np.random.default_rng(seed); va_cap = va.copy(); rng.shuffle(va_cap)
                va_sel = va_cap[:VAL_CAP]
                t = time.time()
                model, diag = train_model(P, cfg, tr, va_sel, seed=seed,
                                          train_mask=tmask, val_mask=vmask)
                res = evaluate(model, P, te, time_mask=vmask)
                for r in res:
                    r.update(dict(protocol=proto, seed=seed, method=label))
                    rows.append(r)
                mae = np.mean([x["mae_total"] for x in res])
                print(f"  {proto:5s} s{seed} {label:12s} OOD_MAE={mae:.4f} "
                      f"n_te={len(res)} ({time.time()-t:.0f}s)", flush=True)
        # checkpoint after each protocol
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "fullscale_per_facility.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fullscale_per_facility.csv"), index=False)
    metrics = ["mae_total", "mae_RN", "mae_LPN", "mae_CNA", "rmse_total", "smape_total"]
    per_seed = df.groupby(["protocol", "method", "seed"])[metrics].mean().reset_index()
    agg = per_seed.groupby(["protocol", "method"])[metrics].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]; agg = agg.reset_index()
    agg.to_csv(os.path.join(OUT, "fullscale_main.csv"), index=False)

    sig = []
    for proto in df.protocol.unique():
        sub = df[df.protocol == proto]
        piv = sub.pivot_table(index=["fac", "seed"], columns="method", values="mae_total")
        if "CD-ISG-TS" not in piv: continue
        for m in piv.columns:
            if m == "CD-ISG-TS": continue
            a = piv[["CD-ISG-TS", m]].dropna()
            if len(a) < 10: continue
            try: p = wilcoxon(a["CD-ISG-TS"], a[m]).pvalue
            except ValueError: p = 1.0
            sig.append(dict(protocol=proto, baseline=m,
                            ours_minus_baseline=float((a["CD-ISG-TS"] - a[m]).mean()),
                            ours_win_pct=float((a["CD-ISG-TS"] < a[m]).mean()),
                            pvalue=p, n=len(a)))
    pd.DataFrame(sig).to_csv(os.path.join(OUT, "fullscale_significance.csv"), index=False)
    print("\n=== FULL-SCALE OOD MAE (mean over seeds) ===")
    print(agg.pivot_table(index="method", columns="protocol", values="mae_total_mean").round(4))
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
