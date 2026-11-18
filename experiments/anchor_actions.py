"""What each development action buys on the benchmark anchors.

On the anchor manifests and seeds of anchor.py (common manifest, the target test
entities of the case study) every action a practitioner could take is scored
against the model in service (deployed model, 1x budget):

  train longer   deployed model at 2x, 4x, 8x budget, and trained to convergence
                 (300 epochs, patience 25, source-validation early stopping)
  no retraining  deployed model blended with seasonal naive, and combined with
                 seasonal naive, persistence and Holt-Winters (weights on the
                 source early-stopping entities, grid step 0.1)
  target data    target-only reference at the same budgets and at convergence;
                 the same reference combined with seasonal naive, persistence and
                 Holt-Winters (weights on the target early-stopping entities);
                 joint training (every source entity plus the target training
                 entities) at convergence
  size-matched   size-matched source model at convergence (headroom at convergence)

Budget m: every entity list is tiled to m x (deployed size) entries per epoch, so
all arms at the same m receive the same number of gradient steps (24 epochs,
patience 5). Holt-Winters forecasts for the target test and source early-stopping
entities are read from classical_cache_<panel>.pkl (classical.py); those for the
target early-stopping entities are computed here.
Output: results/anchor_actions_rows.csv, results/anchor_actions_entities.csv
"""
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, RES, allocate, fit, per_entity, predict, seasonal_naive  # noqa: E402
from anchor import anchors  # noqa: E402
from classical import entity_forecasts, simplex  # noqa: E402
from joint_finetune import ERM, KEEP, tiled  # noqa: E402

torch.set_num_threads(int(os.environ.get("EXP_THREADS", "3")))
CONV = dict(ERM, epochs=300, patience=25)
HW = "Holt-Winters, additive weekly season"
PERS = "Persistence (last value)"
MULTS = (1, 2, 4, 8)


def combo(parts_v, Yv, parts_t):
    """Convex weights on a 0.1 grid chosen on validation windows, applied to test."""
    keys = list(parts_v)
    best, bw = np.inf, None
    for w in simplex(len(keys)):
        e = float(np.abs(sum(wi * parts_v[k] for wi, k in zip(w, keys)) - Yv).mean())
        if e < best - 1e-12:
            best, bw = e, w
    return sum(wi * parts_t[k] for wi, k in zip(bw, keys)), dict(zip(keys, bw))


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar"]
    only = os.environ.get("EXP_SEEDS")              # e.g. "0" or "1,2": run a subset of seeds
    only = None if not only else [int(x) for x in only.split(",")]
    tag = "_".join(want) + ("" if only is None else "_s" + "".join(map(str, only)))
    rows, ents = [], []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if (panel, split) not in KEEP:
            continue
        if only is not None:
            seeds = [s for s in seeds if s in only]
        cache = pickle.load(open(os.path.join(RES, f"classical_cache_{panel}.pkl"), "rb"))
        allocs = {s: allocate(S, T, np.random.default_rng(s), floors=floors, fracs=fracs,
                              fallback=(panel == "Solar")) for s in seeds}
        need = sorted({int(e) for a in allocs.values() for e in a["t_va"] if int(e) not in cache})
        if need:
            warnings.filterwarnings("ignore")
            with Pool(int(os.environ.get("EXP_WORKERS", "3"))) as pool:
                for e, out, _, _ in pool.imap_unordered(entity_forecasts,
                                                        [(e, P.eval_arrays([e])[0]) for e in need]):
                    cache[e] = out
        g = lambda facs, nm: np.concatenate([cache[int(e)][nm] for e in facs])
        for seed in seeds:
            t0 = time.time()
            al = allocs[seed]
            used = set(al["s_va"].tolist())
            s_all = np.array([e for e in S if e not in used])
            N = len(s_all)
            pool_list = np.concatenate([s_all, al["t_tr"]])
            Xt, Yt, _ = P.eval_arrays(al["t_te"])
            Xv, Yv, _ = P.eval_arrays(al["s_va"])
            Xr, Yr, _ = P.eval_arrays(al["t_va"])
            fid = P.eval_arrays(al["t_te"])[2]

            def score(arm, pred, extra=None):
                e = per_entity(pred, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, arm=arm, n_dep=N,
                         n_tgt_train=len(al["t_tr"]), **(extra or {}))
                for mt in METRICS:
                    r[mt] = float(e[mt].mean())
                rows.append(r)
                ents.append(e.reset_index().assign(panel=panel, split=split, seed=seed, arm=arm))

            def with_refs(arm, model, val_X, val_Y, val_facs, val_side):
                """Score a model alone, blended and combined (weights on its own validation)."""
                pt = predict(model, P, al["t_te"])[2]
                pv = predict(model, P, val_facs)[2]
                score(arm, pt)
                sn_t, sn_v = seasonal_naive(Xt), seasonal_naive(val_X)
                pb, wb = combo({"model": pv, "snaive": sn_v}, val_Y, {"model": pt, "snaive": sn_t})
                score(arm + "+blend", pb, {"w_snaive": wb["snaive"]})
                pc, wc = combo({"model": pv, "snaive": sn_v, "pers": g(val_facs, PERS), "hw": g(val_facs, HW)},
                               val_Y, {"model": pt, "snaive": sn_t, "pers": g(al["t_te"], PERS),
                                       "hw": g(al["t_te"], HW)})
                score(arm + "+comb", pc, {f"w_{k}": v for k, v in wc.items()})

            for m in MULTS:
                dep = fit(P, ERM, tiled(s_all, m * N), al["s_va"], seed)
                with_refs(f"dep_{m}x", dep, Xv, Yv, al["s_va"], "source")
                ref = fit(P, ERM, tiled(al["t_tr"], m * N), al["t_va"], seed)
                with_refs(f"ref_{m}x", ref, Xr, Yr, al["t_va"], "target")
                src = fit(P, ERM, tiled(al["s_tr"], m * N), al["s_va"], seed)
                score(f"src_{m}x", predict(src, P, al["t_te"])[2])
            dep = fit(P, CONV, s_all, al["s_va"], seed)
            with_refs("dep_conv", dep, Xv, Yv, al["s_va"], "source")
            ref = fit(P, CONV, tiled(al["t_tr"], N), al["t_va"], seed)
            with_refs("ref_conv", ref, Xr, Yr, al["t_va"], "target")
            src = fit(P, CONV, tiled(al["s_tr"], N), al["s_va"], seed)
            score("src_conv", predict(src, P, al["t_te"])[2])
            joint = fit(P, CONV, pool_list, al["t_va"], seed)
            score("joint_conv", predict(joint, P, al["t_te"])[2])
            for nm, lab in ((HW, "holt_winters"), (PERS, "persistence")):
                score(lab, g(al["t_te"], nm))
            score("snaive", seasonal_naive(Xt))
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"anchor_actions_rows_{tag}.csv"), index=False)
            pd.concat(ents).to_csv(os.path.join(RES, f"anchor_actions_entities_{tag}.csv"), index=False)
            print(f"  {panel} {split} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
