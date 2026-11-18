"""Full-scale nursing benchmark (Supplement D).

Differences from src/experiments_fullscale.py:
  * DANN uses stable facility domain labels; GroupDRO is the faithful online
    algorithm; the batch-softmax heuristic is reported under its own name
  * TEMPORAL PROTOCOLS: model selection now uses OBSERVED-ERA windows of the
    source-validation facilities (train_mask = val_mask = observed era); the
    exploratory script early-stopped on deployment-era windows (val_mask = 2025)
  * both endpoints saved per facility (mae_cat primary, mae_total secondary)
  * the full FADE model (gate + periodicity) is included

Protocols
  LFO    leave-facilities-out, all windows (retrospective cross-entity)
  LFO-T  leave-facilities-out, train/select on 2024 windows, test on 2025
  LSO-T  leave-states-out,     train/select on 2024 windows, test on 2025

Usage: python experiments/fullscale.py <light|heavy> <protocol> [...]
Output: results/fullscale_<group>_<protocol>.csv (per facility, seed, method)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES                         # noqa: E402
from src.data import Panel                       # noqa: E402
from src.train import evaluate, train_model      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFAC, L, H, STRIDE = 13402, 56, 14, 16
VAL_CAP = 800
EP, PAT = 24, 5
MLP = dict(epochs=EP, patience=PAT, backbone="mlp")
LIGHT = [
    ("ERM-DLinear", dict(method="dlinear_plain", epochs=EP, patience=PAT)),
    ("ERM-MLP", dict(method="erm", **MLP)),
    ("Wave-Mask", dict(method="wavemask", **MLP)),
    ("IRM", dict(method="irm", lam_irm=1.0, **MLP)),
    ("CORAL", dict(method="coral", lam_coral=1.0, **MLP)),
    ("DANN", dict(method="dann", lam_dann=0.5, **MLP)),
    ("GroupDRO", dict(method="groupdro", dro_eta=0.01, **MLP)),
    ("Batch-softmax reweighting (former GroupDRO)",
     dict(method="dro_heuristic", dro_tau=0.1, **MLP)),
    ("FixedBand", dict(method="fixedband", drop_fine=1, **MLP)),
    ("FADE gate only (static)", dict(method="cdisgts", lam_inv=0.5, gate_lr=0.01, **MLP)),
    ("FADE (gate + periodicity)", dict(method="cdisgts", lam_inv=0.5, gate_lr=0.05,
                                       periodic=True, **MLP)),
]
# DG penalties with checkpoint selection restricted to post-warm-up epochs,
# so the selected model has actually been trained with its penalty
ACTIVE = [
    ("IRM (penalty active)", dict(method="irm", lam_irm=1.0, select_after_warmup=True, **MLP)),
    ("CORAL (penalty active)", dict(method="coral", lam_coral=1.0, select_after_warmup=True, **MLP)),
    ("DANN (penalty active)", dict(method="dann", lam_dann=0.5, select_after_warmup=True, **MLP)),
]
HEAVY = [
    ("Autoformer", dict(method="autoformer", epochs=18, patience=4)),
    ("FEDformer", dict(method="fedformer", epochs=18, patience=4)),
]
# The two transformers take roughly an hour per run on 13,402 facilities on the
# CPU used here, so they are run with one seed per protocol (disclosed in the
# paper); every other method uses five.
SEEDS = {"light": [0, 1, 2, 3, 4], "heavy": [0], "active": [0, 1, 2]}
KEEP = ["fac", "mae_cat", "mae_total", "mae_RN", "mae_LPN", "mae_CNA",
        "rmse_cat", "rmse_total", "smape_cat", "smape_total"]


def main():
    group, protos = sys.argv[1], sys.argv[2:]
    methods = {"light": LIGHT, "heavy": HEAVY, "active": ACTIVE}[group]
    P = Panel(os.path.join(ROOT, "cache", "panel.npz"), n_facilities=NFAC, L=L,
              H=H, stride=STRIDE, seed=0)
    for proto in protos:
        temporal = proto != "LFO"
        tmask = P.is_2024 if temporal else None       # training AND selection
        emask = P.is_2025 if temporal else None       # evaluation only
        rows, path = [], os.path.join(RES, f"fullscale_{group}_{proto}.csv")
        done = set()
        if os.path.exists(path):                      # resume after interruption
            old = pd.read_csv(path)
            rows = old.to_dict("records")
            done = set(zip(old.method, old.seed))
        for label, cfg in methods:
            for seed in SEEDS[group]:
                if (label, seed) in done:
                    continue
                tr, va, te = (P.split_states(seed) if proto == "LSO-T"
                              else P.split_facilities(seed))
                va_sel = va.copy(); np.random.default_rng(seed).shuffle(va_sel)
                va_sel = va_sel[:VAL_CAP]
                t = time.time()
                m, _ = train_model(P, cfg, tr, va_sel, seed=seed, train_mask=tmask,
                                   val_mask=tmask)
                res = evaluate(m, P, te, time_mask=emask)
                for r in res:
                    rows.append({**{k: r[k] for k in KEEP}, "protocol": proto,
                                 "seed": seed, "method": label})
                print(f"{proto:5s} s{seed} {label:44s} cat={np.mean([r['mae_cat'] for r in res]):.4f} "
                      f"tot={np.mean([r['mae_total'] for r in res]):.4f} "
                      f"({time.time() - t:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
