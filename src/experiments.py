"""
Main benchmark (RO5): leave-facilities-out DG with training-domain model selection,
across three protocols of increasing domain shift, all methods, multiple seeds.

Protocols
  LFO    : leave-facilities-out, same 2024-2025 period          (near-zero shift)
  LFO-T  : leave-facilities-out + train 2024 / deploy 2025       (realistic shift)
  LSO-T  : leave-STATE-out       + train 2024 / deploy 2025      (largest shift)

Outputs: outputs/per_facility_results.csv, outputs/main_results.csv,
         outputs/significance.csv, outputs/gen_gap.csv, cache/gates_main.npz
"""
import numpy as np, pandas as pd, os, time, json
from scipy.stats import wilcoxon
from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); CACHE = os.path.join(ROOT, "cache")

NFAC, L, H, STRIDE = 1500, 56, 14, 12
SEEDS = [0, 1, 2]
EP, PAT = 26, 6

# (label, config).  cdisgts uses a small, val-robust invariance weight.
METHODS = [
    ("ERM-DLinear", dict(method="dlinear_plain", epochs=EP, patience=PAT)),
    ("ERM-MLP",     dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")),
    ("Autoformer",  dict(method="autoformer", epochs=20, patience=5)),
    ("FEDformer",   dict(method="fedformer", epochs=EP, patience=PAT)),
    ("Wave-Mask",   dict(method="wavemask", epochs=EP, patience=PAT, backbone="mlp")),
    ("IRM",         dict(method="irm", epochs=EP, patience=PAT, backbone="mlp", lam_irm=1.0)),
    ("FixedBand",   dict(method="fixedband", epochs=EP, patience=PAT, backbone="mlp", drop_fine=1)),
    ("CD-ISG-TS",   dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
                        lam_inv=0.5, gate_lr=0.01)),
]

def protocol_masks(P, name):
    if name == "LFO":
        return None, None
    return P.is_2024, P.is_2025            # LFO-T, LSO-T

def protocol_split(P, name, seed):
    if name == "LSO-T":
        return P.split_states(seed)
    return P.split_facilities(seed)

def main():
    P = Panel(os.path.join(CACHE, "panel.npz"), n_facilities=NFAC, L=L, H=H,
              stride=STRIDE, seed=0)
    print("panel:", P.Xw.shape, "facilities:", P.nF, flush=True)
    rows = []; gap_rows = []; gate_store = {}
    for proto in ["LFO", "LFO-T", "LSO-T"]:
        tmask, vmask = protocol_masks(P, proto)
        for seed in SEEDS:
            tr, va, te = protocol_split(P, proto, seed)
            emask = vmask  # evaluate target on deployment period (2025) for -T protocols
            for label, cfg in METHODS:
                t0 = time.time()
                model, diag = train_model(P, cfg, tr, va, seed=seed,
                                          train_mask=tmask, val_mask=vmask)
                res = evaluate(model, P, te, time_mask=emask)
                for r in res:
                    r.update(dict(protocol=proto, seed=seed, method=label,
                                  state=str(P.state[r["fac"]]),
                                  medcen=float(P.medcen[r["fac"]])))
                    rows.append(r)
                # generalization gap (train-facility vs test-facility) for ERM & ours
                if label in ("ERM-MLP", "CD-ISG-TS"):
                    tr_res = evaluate(model, P, tr, time_mask=tmask)
                    gap_rows.append(dict(protocol=proto, seed=seed, method=label,
                                         train_mae=np.mean([x["mae_total"] for x in tr_res]),
                                         test_mae=np.mean([x["mae_total"] for x in res])))
                if label == "CD-ISG-TS":
                    gate_store[f"{proto}_s{seed}"] = diag.get("gates")
                mae = np.mean([x["mae_total"] for x in res])
                print(f"  {proto:5s} s{seed} {label:12s} OOD_MAE={mae:.4f} "
                      f"val={diag['val_mae']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "per_facility_results.csv"), index=False)

    # aggregate: mean over facilities per (protocol,seed,method), then mean+-std over seeds
    metrics = ["mae_total", "mae_RN", "mae_LPN", "mae_CNA", "rmse_total", "smape_total"]
    per_seed = df.groupby(["protocol", "method", "seed"])[metrics].mean().reset_index()
    agg = per_seed.groupby(["protocol", "method"])[metrics].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    # worst-decile (per protocol/method/seed) then averaged
    w10 = df.groupby(["protocol", "method", "seed"])["mae_total"].quantile(0.9).reset_index()
    w10 = w10.groupby(["protocol", "method"])["mae_total"].mean().reset_index()
    w10.columns = ["protocol", "method", "mae_total_worst10_mean"]
    agg = agg.merge(w10, on=["protocol", "method"])
    agg.to_csv(os.path.join(OUT, "main_results.csv"), index=False)

    # paired significance vs ERM-DLinear and vs CD-ISG-TS, per protocol (pooled facilities)
    sig = []
    for proto in df.protocol.unique():
        sub = df[df.protocol == proto]
        piv = sub.pivot_table(index=["fac", "seed"], columns="method",
                              values="mae_total")
        for m in piv.columns:
            for ref in ["ERM-DLinear", "CD-ISG-TS"]:
                if m == ref:
                    continue
                a = piv[[ref, m]].dropna()
                if len(a) < 10:
                    continue
                try:
                    p = wilcoxon(a[ref], a[m]).pvalue
                except ValueError:
                    p = 1.0
                sig.append(dict(protocol=proto, ref=ref, method=m,
                                delta_mean=float((a[m] - a[ref]).mean()),
                                ref_better_frac=float((a[ref] < a[m]).mean()),
                                pvalue=p, n=len(a)))
    pd.DataFrame(sig).to_csv(os.path.join(OUT, "significance.csv"), index=False)
    pd.DataFrame(gap_rows).to_csv(os.path.join(OUT, "gen_gap.csv"), index=False)
    np.savez(os.path.join(CACHE, "gates_main.npz"),
             **{k: v for k, v in gate_store.items() if v is not None})
    print("\nSaved outputs/{per_facility_results,main_results,significance,gen_gap}.csv")
    print("\n=== OOD MAE (mean over seeds) ===")
    print(agg.pivot_table(index="method", columns="protocol", values="mae_total_mean").round(4))

if __name__ == "__main__":
    main()
