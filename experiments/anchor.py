"""Anchor experiment on ONE COMMON MANIFEST.

For every anchor split and allocation seed, one entity manifest is drawn:
  s_tr, s_va   source training / early-stopping entities
  s_te         held-out source entities (source error for the apparent gap)
  t_tr, t_va   target entities for the target-trained reference, |t_tr| = |s_tr|
  t_te         target test entities: EVERY row below is scored on these
               entities and all their windows, with both endpoints

Manifest A (primary, size-matched): every source-trained method uses s_tr.
Manifest B (sensitivity, benchmark-style): source-trained methods use all
  source entities outside s_va (s_te included, as in the original benchmark). Its target-trained reference is
  unchanged, so manifest-B headroom is NOT size-matched and is reported only
  as context.

All methods share one budget (24 epochs, patience 5, source-validation early
stopping on mae_cat); none sees t_te or target labels.

Outputs: anchor_rows.csv (per split/seed/manifest/method), anchor_entities.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ALPHA_GRID, METRICS, RES, allocate, blend_alpha, fit,  # noqa
                      headroom, load_panel, per_entity, predict, seasonal_naive)

EP, PAT = 24, 5
MLP = dict(epochs=EP, patience=PAT, backbone="mlp")
METHODS = [
    ("ERM-MLP", dict(method="erm", **MLP)),
    ("ERM-DLinear", dict(method="dlinear_plain", epochs=EP, patience=PAT)),
    ("Wave-Mask", dict(method="wavemask", **MLP)),
    ("FixedBand", dict(method="fixedband", drop_fine=1, **MLP)),
    ("IRM", dict(method="irm", lam_irm=1.0, **MLP)),
    ("CORAL", dict(method="coral", lam_coral=1.0, **MLP)),
    ("DANN", dict(method="dann", lam_dann=0.5, **MLP)),
    ("GroupDRO", dict(method="groupdro", dro_eta=0.01, **MLP)),
    ("Batch-softmax reweighting (former GroupDRO)",
     dict(method="dro_heuristic", dro_tau=0.1, **MLP)),
    ("FADE gate only (static)", dict(method="cdisgts", lam_inv=0.5, gate_lr=0.01, **MLP)),
    ("FADE gate only (input-conditioned)",
     dict(method="cdisgts", lam_inv=0.5, gate_lr=0.01, adaptive=True, **MLP)),
    ("FADE periodicity only", dict(method="cdisgts", lam_inv=0.0, gate_lr=0.05,
                                   periodic=True, gate_init=12.0, **MLP)),
    ("FADE (gate + periodicity)", dict(method="cdisgts", lam_inv=0.5, gate_lr=0.05,
                                       periodic=True, **MLP)),
]

# the smaller cases run the component set only; for nursing the complete DG
# comparison is the full-scale benchmark (fullscale.py), and with thousands of
# facilities per group the benchmark-style manifest B adds nothing
COMPONENT = {"ERM-MLP", "FADE gate only (static)", "FADE periodicity only",
             "FADE (gate + periodicity)"}


def anchors(want):
    out = []
    P = load_panel("Electricity") if "Electricity" in want else None
    if P is not None:
        grp = np.asarray(P.state).astype(str); idx = np.arange(P.nF)
    if P is not None:
      out.append(("Electricity", P, "Flat -> WorkWeek", idx[grp == "Flat"],
                idx[grp == "WorkWeek"], (20, 10), (0.35, 0.15), [0, 1, 2, 3, 4]))
      out.append(("Electricity", P, "WorkWeek -> Flat", idx[grp == "WorkWeek"],
                idx[grp == "Flat"], (20, 10), (0.35, 0.15), [0, 1, 2, 3, 4]))
    # component-ablation cases (Figure 3): same manifest logic, smaller groups
    for nm in [x for x in ("Solar", "Traffic") if x in want]:
        Pc = load_panel(nm)
        gc = np.asarray(Pc.state).astype(str); ic = np.arange(Pc.nF)
        out.append((nm, Pc, "Flat -> WorkWeek", ic[gc == "Flat"], ic[gc == "WorkWeek"],
                    (20, 10), (0.35, 0.15), [0, 1, 2, 3, 4]))
    if "Nursing" not in want:
        return out
    from src.oracle_experiment import build_meta
    Pn = load_panel("Nursing", 13402)
    own, urban = build_meta(Pn)
    idn = np.arange(Pn.nF)
    out.append(("Nursing", Pn, "Ownership (For-profit -> Non-profit)",
                idn[own == "ForProfit"], idn[own == "NonProfit"], (800, 400),
                (0.0, 0.0), [0, 1, 2]))
    out.append(("Nursing", Pn, "Geography (Urban -> Rural)", idn[urban == "Y"],
                idn[urban == "N"], (800, 400), (0.0, 0.0), [0, 1, 2]))
    return out


def main():
    want = sys.argv[1:] or ["Electricity", "Nursing", "Solar", "Traffic"]
    rows, ents = [], []
    tag = "" if len(sys.argv) == 1 else "_" + "_".join(want)
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if panel not in want:
            continue
        for seed in seeds:
            t0 = time.time()
            al = allocate(S, T, np.random.default_rng(seed), floors=floors,
                          fracs=fracs, fallback=(panel == "Solar"))
            # manifest B: every source entity except the early-stopping set, as in
            # the original benchmark (source error is only reported for A)
            used = set(al["s_va"])
            s_all = np.array([e for e in S if e not in used])
            manifests = {"A": al["s_tr"], "B": s_all}

            def score(label, man, pred, Y, fid, extra=None):
                e = per_entity(pred, Y, fid)
                r = dict(panel=panel, split=split, seed=seed, manifest=man,
                         method=label, n_train=len(manifests.get(man, al["t_tr"])),
                         n_test=len(al["t_te"]), **(extra or {}))
                for mt in METRICS:
                    r[mt] = e[mt].mean()
                rows.append(r)
                ents.append(e.reset_index().assign(panel=panel, split=split,
                                                   seed=seed, manifest=man,
                                                   method=label))

            # target-trained reference and source error (manifest A models)
            ref = fit(P, METHODS[0][1], al["t_tr"], al["t_va"], seed)
            Xt, Yt, Rt, fid = predict(ref, P, al["t_te"])
            score("Target-trained reference (ERM-MLP)", "A", Rt, Yt, fid)
            for man, s_tr in manifests.items():
                for label, cfg in METHODS:
                    if panel in ("Solar", "Traffic", "Nursing") and label not in COMPONENT:
                        continue
                    if panel in ("Solar", "Traffic", "Nursing") and man == "B":
                        continue
                    m = fit(P, cfg, s_tr, al["s_va"], seed)
                    _, _, Pt, _ = predict(m, P, al["t_te"])
                    score(label, man, Pt, Yt, fid)
                    if label == "ERM-MLP":
                        Xs, Ys, Ps, fs = predict(m, P, al["s_te"])
                        e = per_entity(Ps, Ys, fs)
                        rows.append(dict(panel=panel, split=split, seed=seed,
                                         manifest=man, method="source error (s_te)",
                                         n_train=len(s_tr), n_test=len(al["s_te"]),
                                         **{mt: e[mt].mean() for mt in METRICS}))
                        # simple actions built on this source-trained ERM
                        Xv, Yv, Pv, _ = predict(m, P, al["s_va"])
                        a_star = blend_alpha(Pv, Xv, Yv, "mae_cat")
                        sn = seasonal_naive(Xt)
                        if man == "A":
                            score("Seasonal naive", man, sn, Yt, fid)
                            for a in ALPHA_GRID[1:-1]:
                                score(f"Fixed blend a={a:g}", man,
                                      (1 - a) * Pt + a * sn, Yt, fid, dict(alpha=a))
                        score("Blend, a chosen on source validation", man,
                              (1 - a_star) * Pt + a_star * sn, Yt, fid,
                              dict(alpha=a_star))
            print(f"  {panel} {split} seed {seed} done ({time.time() - t0:.0f}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"anchor_rows{tag}.csv"),
                                      index=False)
            pd.concat(ents).to_csv(os.path.join(RES, f"anchor_entities{tag}.csv"),
                                   index=False)
    df = pd.DataFrame(rows)
    A = df[df.manifest == "A"].groupby(["panel", "split", "method"])[list(METRICS)].mean()
    print(A.round(4).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
