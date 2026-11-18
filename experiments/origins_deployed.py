"""Retraining decisions against the DEPLOYED model at equal budget.

origins.py compared the target-trained reference with a size-matched source
model. A deployment decision compares it with the model already in service, which
is trained on every available source entity. This script repeats the several-
decision-date design of origins.py (same panels, origins, splits, seeds,
windows and scaling) and adds, for every split x origin x seed unit:

    dep      deployed model: every source entity outside s_va, early stopping on s_va
    refE     target-trained reference with its entity list tiled to the deployed
             size, so it receives exactly the deployed model's optimization budget
    dep2     deployed model trained for twice its budget (list tiled to 2x), the
             source-only "train longer" alternative
    blend    deployed model blended with seasonal naive, weight chosen on the
             source early-stopping entities (observed era), no retraining

The size-matched pair (src, ref) of origins.py is refitted as well; its rows
must reproduce origin_rows exactly, which checks reproducibility from the raw data.

Every arm is trained and selected on windows that end before the origin and scored
on the target test entities in the observed era (estimate) and in the deployment
period (realized). Output: results/origin_deployed_<panels>.csv
(one row per unit, mean errors per arm and era) and
results/origin_deployed_entities_<panels>.csv (per-entity errors).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA_GRID, ERM, METRICS, RES, fit, per_entity, predict,  # noqa: E402
                      seasonal_naive)
from temporal import N_SPLITS, SEEDS, match_source, split_target  # noqa: E402
from joint_finetune import tiled  # noqa: E402
from origins import ORIGINS, panel_at  # noqa: E402
from src.label_free_calibration import make_splits  # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402

torch.set_num_threads(int(os.environ.get("EXP_THREADS", "3")))


def pick_alpha(pred, X, Y):
    sn = seasonal_naive(X)
    best, ba = np.inf, 0.0
    for a in ALPHA_GRID:
        e = float(np.abs((1 - a) * pred + a * sn - Y).mean())
        if e < best - 1e-12:
            best, ba = e, a
    return ba


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar"]
    tag = "_".join(want)
    rows, ents = [], []
    for name in want:
        t0 = time.time()
        orig = ORIGINS[name]
        P0 = panel_at(name, orig[0], stride=18)
        stats, _ = entity_stats(P0)
        splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
        del P0
        for j, o in enumerate(orig):
            P = panel_at(name, o)
            d = np.asarray(P.dates)
            end = orig[j + 1] if j + 1 < len(orig) else int(d[-1]) + 1
            last = d[np.minimum(P.wstart + P.span - 1, len(d) - 1)]
            obs = P.is_2024.copy()
            dep_era = P.is_2025 & (last < end) & (P.wstart + P.span <= len(d))
            for sp, (A, B) in enumerate(splits):
                for sd in SEEDS:
                    rng = np.random.default_rng(sd)
                    t_tr, t_va, t_te = split_target(B, rng)
                    if len(t_tr) < 12:
                        continue
                    s_tr, s_va = match_source(A, B, len(t_tr), rng)
                    if len(s_tr) < 12:
                        continue
                    used = set(s_va.tolist())
                    s_all = np.array([e for e in A if e not in used])
                    N = len(s_all)
                    arms = {
                        "src": fit(P, ERM, s_tr, s_va, sd, obs, obs),
                        "ref": fit(P, ERM, t_tr, t_va, sd, obs, obs),
                        "dep": fit(P, ERM, s_all, s_va, sd, obs, obs),
                        "refE": fit(P, ERM, tiled(t_tr, N), t_va, sd, obs, obs),
                        "dep2": fit(P, ERM, tiled(s_all, 2 * N), s_va, sd, obs, obs),
                    }
                    Xv, Yv, Pv, _ = predict(arms["dep"], P, s_va, obs)
                    alpha = pick_alpha(Pv, Xv, Yv)
                    r = dict(panel=name, origin=o, origin_idx=j, deploy_end=end, split=sp,
                             seed=sd, n_src_train=len(s_tr), n_dep=N, n_tgt_train=len(t_tr),
                             n_tgt_val=len(t_va), n_tgt_test=len(t_te), alpha=alpha)
                    for era, msk in (("obs", obs), ("dep", dep_era)):
                        preds = {}
                        Xt = Yt = fid = None
                        for a, m in arms.items():
                            Xt, Yt, preds[a], fid = predict(m, P, t_te, msk)
                        preds["blend"] = (1 - alpha) * preds["dep"] + alpha * seasonal_naive(Xt)
                        preds["snaive"] = seasonal_naive(Xt)
                        for a, pr in preds.items():
                            e = per_entity(pr, Yt, fid)
                            for mt in METRICS:
                                r[f"{a}_{era}_{mt}"] = float(e[mt].mean())
                            ents.append(e.reset_index().assign(panel=name, origin=o, split=sp,
                                                               seed=sd, arm=a, era=era))
                    rows.append(r)
                    hm = lambda a, b, era: 100 * (r[f"{a}_{era}_mae_cat"] - r[f"{b}_{era}_mae_cat"]) / r[f"{a}_{era}_mae_cat"]
                    print(f"  {name} {o} split {sp:2d} seed {sd}: size-matched est {hm('src','ref','obs'):+6.2f} "
                          f"real {hm('src','ref','dep'):+6.2f} | deployed est {hm('dep','refE','obs'):+6.2f} "
                          f"real {hm('dep','refE','dep'):+6.2f} | N={N}", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, f"origin_deployed_{tag}.csv"), index=False)
            pd.concat(ents).to_csv(os.path.join(RES, f"origin_deployed_entities_{tag}.csv"), index=False)
            del P
        print(f"[{name} done {time.time() - t0:.0f}s]", flush=True)
    print("done")


if __name__ == "__main__":
    main()
