"""Sequential procedure at several decision dates: improve the deployed model first,
then measure the residual benefit of retraining on target data.

Same panels, decision dates, splits, seeds, windows and scaling as
origins_deployed.py. For every split x date x seed unit:

    depC    deployed model (every source entity outside s_va) trained until source-
            validation error stops improving (up to 150 epochs, patience 10)
    refC    target-trained reference, entity list tiled to the deployed size, trained
            the same way with early stopping on t_va
    depC+b  depC blended with seasonal naive, weight chosen on s_va (observed era)
    refC+b  refC blended with seasonal naive, weight chosen on t_va (observed era)

The improved model in service is depC+b; the residual benefit of retraining is the
reduction of refC+b relative to it, estimated on observed-era windows of the target
test entities and realized on their deployment-period windows.
Output: results/origin_seq_<panel>[_<origins>].csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ERM, METRICS, RES, fit, per_entity, predict, seasonal_naive  # noqa: E402
from temporal import N_SPLITS, SEEDS, match_source, split_target  # noqa: E402
from joint_finetune import tiled  # noqa: E402
from origins import ORIGINS, panel_at  # noqa: E402
from origins_deployed import pick_alpha  # noqa: E402
from src.label_free_calibration import make_splits  # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402

torch.set_num_threads(int(os.environ.get("EXP_THREADS", "2")))
CONV = dict(ERM, epochs=150, patience=10)


def main():
    name = sys.argv[1]
    which = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    orig = ORIGINS[name]
    tag = name + ("" if which is None else "_" + "_".join(map(str, which)))
    P0 = panel_at(name, orig[0], stride=18)
    stats, _ = entity_stats(P0)
    splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
    del P0
    rows = []
    for j, o in enumerate(orig):
        if which is not None and j not in which:
            continue
        t0 = time.time()
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
                mD = fit(P, CONV, s_all, s_va, sd, obs, obs)
                mR = fit(P, CONV, tiled(t_tr, N), t_va, sd, obs, obs)
                Xv, Yv, Pv, _ = predict(mD, P, s_va, obs)
                aD = pick_alpha(Pv, Xv, Yv)
                Xr, Yr, Pr, _ = predict(mR, P, t_va, obs)
                aR = pick_alpha(Pr, Xr, Yr)
                r = dict(panel=name, origin=o, origin_idx=j, split=sp, seed=sd, n_dep=N,
                         alpha_dep=aD, alpha_ref=aR)
                for era, msk in (("obs", obs), ("dep", dep_era)):
                    Xt, Yt, pD, fid = predict(mD, P, t_te, msk)
                    _, _, pR, _ = predict(mR, P, t_te, msk)
                    sn = seasonal_naive(Xt)
                    preds = {"depC": pD, "refC": pR, "depCb": (1 - aD) * pD + aD * sn,
                             "refCb": (1 - aR) * pR + aR * sn}
                    for a, pr in preds.items():
                        e = per_entity(pr, Yt, fid)
                        for mt in METRICS:
                            r[f"{a}_{era}_{mt}"] = float(e[mt].mean())
                rows.append(r)
                red = lambda a, b, e: 100 * (r[f"{a}_{e}_mae_cat"] - r[f"{b}_{e}_mae_cat"]) / r[f"{a}_{e}_mae_cat"]
                print(f"  {name} {o} split {sp:2d} seed {sd}: residual est {red('depCb', 'refCb', 'obs'):+6.2f} "
                      f"real {red('depCb', 'refCb', 'dep'):+6.2f}", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, f"origin_seq_{tag}.csv"), index=False)
        print(f"[{name} {o} done {time.time() - t0:.0f}s]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
