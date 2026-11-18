"""Sensitivity of headroom to the model family.

Aim: headroom is defined against one family (ERM-Stem). Does the
picture change with other strong families?

For every anchor split and allocation seed the entity manifest is exactly the
one anchor.py draws, so the ERM-Stem row reproduces the anchor table and the
other rows differ only in the model family. Headroom is computed within each
family: the source-trained and target-trained members share architecture,
budget and stopping rule, and training sets are size-matched.

    ERM-Stem                       the reference family of the article
    ERM-Stem + seasonal blend      each member blended with seasonal naive, the
                                   weight chosen on that member's OWN
                                   early-stopping entities (source validation for
                                   the source model, target validation for the
                                   target-trained model); tests whether headroom
                                   survives once the weekly pattern the base
                                   family underuses is supplied
    ERM-DLinear                    the same without the convolutional stem
    Patch transformer              patch-based transformer head (about 44k
                                   parameters, PatchTST-style), 25 times the
                                   capacity of ERM-Stem

Output: family_rows.csv (one row per panel, split, seed, family)
Usage:  python family.py [panel ...]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (METRICS, RES, allocate, blend_alpha, fit, per_entity,  # noqa: E402
                      predict, seasonal_naive)
from anchor import anchors  # noqa: E402

EP, PAT = 24, 5
FAMILIES = {
    "ERM-Stem": dict(method="erm", epochs=EP, patience=PAT, backbone="mlp"),
    "ERM-DLinear": dict(method="dlinear_plain", epochs=EP, patience=PAT),
    "Patch transformer": dict(method="erm", epochs=EP, patience=PAT, backbone="transformer"),
}


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar", "Nursing"]
    tag = "_".join(want)
    out = os.path.join(RES, f"family_rows_{tag}.csv")
    rows = []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        for seed in seeds:
            t0 = time.time()
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            for fam, cfg in FAMILIES.items():
                ms = fit(P, cfg, al["s_tr"], al["s_va"], seed)
                mt = fit(P, cfg, al["t_tr"], al["t_va"], seed)
                Xt, Yt, Ps, fid = predict(ms, P, al["t_te"])
                _, _, Pr, _ = predict(mt, P, al["t_te"])
                preds = {fam: (Ps, Pr, None, None)}
                if fam == "ERM-Stem":
                    Xv, Yv, Pv, _ = predict(ms, P, al["s_va"])
                    Xw, Yw, Pw, _ = predict(mt, P, al["t_va"])
                    a_s = blend_alpha(Pv, Xv, Yv, "mae_cat")
                    a_t = blend_alpha(Pw, Xw, Yw, "mae_cat")
                    sn = seasonal_naive(Xt)
                    preds["ERM-Stem + seasonal blend"] = ((1 - a_s) * Ps + a_s * sn,
                                                          (1 - a_t) * Pr + a_t * sn, a_s, a_t)
                for name, (tp, rp, a_s, a_t) in preds.items():
                    et, er = per_entity(tp, Yt, fid), per_entity(rp, Yt, fid)
                    r = dict(panel=panel, split=split, seed=seed, family=name,
                             n_train=len(al["s_tr"]), n_test=len(al["t_te"]),
                             alpha_source=a_s, alpha_target=a_t)
                    for m in METRICS:
                        tr, rf = float(et[m].mean()), float(er[m].mean())
                        r[f"transfer_{m}"] = tr
                        r[f"reference_{m}"] = rf
                        r[f"headroom_{m}"] = 100.0 * (tr - rf) / tr
                    rows.append(r)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  {panel} {split} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["panel", "split", "family"]).headroom_mae_cat.agg(["mean", "std"])
          .round(2).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
