"""
Leave-group-out domain-generalization benchmark on the electricity panel.

The reducibility diagnostic (src/oracle_electricity.py) placed the
Flat -> WorkWeek direction in the REDUCIBLE regime (+8.3% reducible gap over five
seeds, min +5.4%), in contrast to the CMS nursing panel where every split was
irreducible. The diagnostic therefore makes a falsifiable PREDICTION: on this
split, and unlike on nursing, a domain-generalization method has room to improve
over a pooled baseline.

This script tests that prediction. Both directions are run, so the reverse
(WorkWeek -> Flat, measured irreducible at -4.3%) acts as a built-in control: the
diagnostic predicts NO improvement there, and if we observed a large improvement
in the irreducible direction the diagnostic would be falsified.

Methods: pooled baselines, the frequency heuristics, the domain-generalization
baselines, the static CD-ISG-TS gate, and the input-conditioned (adaptive) gate.

Output: outputs/electricity_benchmark.csv (per-entity errors, all seeds)
"""
import os
import numpy as np
import pandas as pd

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache", "electricity")
OUT = os.path.join(ROOT, "outputs")

PANEL = os.path.join(CACHE, "panel_elec.npz")
L, H, STRIDE = 56, 14, 18
EP, PAT = 24, 5
SEEDS = [0, 1, 2, 3, 4]

METHODS = [
    ("ERM-DLinear", dict(method="dlinear_plain", epochs=EP, patience=PAT)),
    ("ERM-MLP",     dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")),
    ("Wave-Mask",   dict(method="wavemask", epochs=EP, patience=PAT, backbone="mlp")),
    ("FixedBand",   dict(method="fixedband", epochs=EP, patience=PAT, backbone="mlp",
                         drop_fine=1)),
    ("IRM",         dict(method="irm", epochs=EP, patience=PAT, backbone="mlp",
                         lam_irm=1.0)),
    ("CORAL",       dict(method="coral", epochs=EP, patience=PAT, backbone="mlp",
                         lam_coral=1.0)),
    ("DANN",        dict(method="dann", epochs=EP, patience=PAT, backbone="mlp",
                         lam_dann=0.5)),
    ("GroupDRO",    dict(method="groupdro", epochs=EP, patience=PAT, backbone="mlp")),
    ("CD-ISG-TS (static)",   dict(method="cdisgts", epochs=EP, patience=PAT,
                                  backbone="mlp", lam_inv=0.5, gate_lr=0.01)),
    ("CD-ISG-TS (adaptive)", dict(method="cdisgts", epochs=EP, patience=PAT,
                                  backbone="mlp", lam_inv=0.5, gate_lr=0.01,
                                  adaptive=True)),
]


def main():
    P = Panel(PANEL, n_facilities=10 ** 6, L=L, H=H, stride=STRIDE, seed=0,
              n_channels=3, boundary_date=20140101)
    grp = np.asarray(P.state).astype(str)
    idx = np.arange(P.nF)
    print(f"electricity panel: nF={P.nF} C={P.C} windows={len(P.Xw)}", flush=True)

    SPLITS = {
        "Flat -> WorkWeek": (idx[grp == "Flat"], idx[grp == "WorkWeek"]),   # reducible
        "WorkWeek -> Flat": (idx[grp == "WorkWeek"], idx[grp == "Flat"]),   # irreducible
    }

    rows = []
    for split_name, (src_all, tgt_all) in SPLITS.items():
        n_val = max(10, int(0.15 * len(tgt_all)))
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            s = src_all.copy(); rng.shuffle(s)
            s_va, s_tr = s[:n_val], s[n_val:]
            t_te = tgt_all.copy()
            for mname, cfg in METHODS:
                m, _ = train_model(P, cfg, s_tr, s_va, seed=seed)
                per = evaluate(m, P, t_te)
                for r in per:
                    rows.append(dict(split=split_name, seed=seed, method=mname,
                                     fac=r["fac"], mae_total=r["mae_total"],
                                     rmse_total=r.get("rmse_total", np.nan),
                                     smape_total=r.get("smape_total", np.nan)))
                mae = np.mean([r["mae_total"] for r in per])
                print(f"  {split_name:18s} s{seed} {mname:22s} MAE={mae:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "electricity_benchmark.csv"), index=False)

    print("\n=== mean OOD MAE by split and method ===")
    piv = df.groupby(["split", "method"])["mae_total"].mean().unstack(0).round(4)
    print(piv)
    print("\nSaved outputs/electricity_benchmark.csv")


if __name__ == "__main__":
    main()
