"""Gate-interpretability evidence.

1. ICC ALIGNED TO THE MODEL. The earlier figure computed ICCs with a level-4
   db4 transform on 64-day windows and then ADDED the A4 and D4 ICCs to stand in
   for the model's coarse band. ICCs are variance ratios and cannot be added.
   Here the ICC is computed with exactly the model's transform (db4, level 3,
   periodization, 56-day non-overlapping windows, per-window standardisation),
   so every model band has its own directly estimated ICC:
       ICC_j = Var_between(facility means of relative energy in band j)
               / (Var_between + mean within-facility variance)

2. STABILITY. The static gate is trained (LFO-T protocol, observed-era training
   and selection) on five different facility splits (seeds 0-4), and on two
   DISJOINT halves of one training set (three seeds each). Reported: per-cell
   mean and SD, between-run correlation of the 12 gate values, and the
   gate-ICC correlation for every run.

3. WHAT THE NORMALISATION DOES. The gate is rescaled so that its
   coefficient-count-weighted mean is one (sum_j n_j g_j = sum_j n_j). This
   prevents uniform shutdown but does NOT preserve signal energy, which is
   sum_j g_j^2 E_j. The actual retained energy fraction on held-out windows is
   measured for every run.

4. ACCURACY on the same splits: gate model vs ERM-MLP, both endpoints.

Outputs: gate_icc_aligned.csv, gate_runs.csv, gate_stability_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import pywt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, fit, per_entity, predict              # noqa: E402
from src.data import Panel                                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = ["RN", "LPN", "CNA"]
BANDS = ["A3", "D3", "D2", "D1"]
EP, PAT = 26, 6
GATE = dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
            lam_inv=0.5, gate_lr=0.01)
ERM = dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")


def rel_energy(w):
    w = (w - w.mean()) / (w.std() + 1e-6)
    c = pywt.wavedec(w, "db4", mode="periodization", level=3)
    e = np.array([np.sum(x ** 2) for x in c])
    return e / (e.sum() + 1e-12)


def aligned_icc():
    X = np.load(os.path.join(ROOT, "cache", "panel.npz"))["X"]
    n, T, _ = X.shape
    nw = T // 56
    rows = []
    icc = np.zeros((3, 4))
    for c in range(3):
        E = np.stack([[rel_energy(X[i, k * 56:(k + 1) * 56, c]) for k in range(nw)]
                      for i in range(n)])                       # [n, nw, 4]
        vb = E.mean(1).var(0)
        vw = E.var(1).mean(0)
        icc[c] = vb / (vb + vw + 1e-12)
        for j, b in enumerate(BANDS):
            rows.append(dict(channel=CATS[c], band=b, mean_rel_energy=E[..., j].mean(),
                             var_between=vb[j], var_within=vw[j], ICC=icc[c, j]))
        print(f"  ICC {CATS[c]}: " + " ".join(f"{v:.3f}" for v in icc[c]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "gate_icc_aligned.csv"), index=False)
    return icc


def retained_energy(model, Xw):
    """Fraction of wavelet-domain energy kept after gating, on real windows."""
    gate = model.gate
    with torch.no_grad():
        x = model.stem(model.revin(torch.from_numpy(Xw)))   # the gate acts here
        coeff = torch.einsum("nl,bcl->bcn", gate.W, x)          # [B,C,n]
        g = gate.gates()                                          # [C, bands]
        gpos = g[:, gate.band_id]                                 # [C, n]
        kept = ((gpos[None] * coeff) ** 2).sum((0, 2))
        tot = (coeff ** 2).sum((0, 2))
    return (kept / tot).numpy()                                   # per channel


def main():
    icc = aligned_icc()
    P = Panel(os.path.join(ROOT, "cache", "panel.npz"), n_facilities=1500, L=56,
              H=14, stride=12, seed=0)
    obs, dep = P.is_2024, P.is_2025
    runs = []

    def one(tag, tr, va, te, seed):
        mg = fit(P, GATE, tr, va, seed, obs, obs)
        me = fit(P, ERM, tr, va, seed, obs, obs)
        G = mg.gate.gates().detach().numpy()                     # [3, 4]
        Xw, Y, Pg, fid = predict(mg, P, te, dep)
        _, _, Pe, _ = predict(me, P, te, dep)
        eg, ee = per_entity(Pg, Y, fid), per_entity(Pe, Y, fid)
        ret = retained_energy(mg, Xw[:4000])
        supp = (1 - G).ravel()
        r = dict(run=tag, seed=seed, n_train=len(tr),
                 gate_icc_pearson=float(np.corrcoef(icc.ravel(), supp)[0, 1]),
                 **{f"g_{c}_{b}": G[i, j] for i, c in enumerate(CATS)
                    for j, b in enumerate(BANDS)},
                 **{f"retained_energy_{c}": ret[i] for i, c in enumerate(CATS)},
                 gate_mae_cat=eg.mae_cat.mean(), erm_mae_cat=ee.mae_cat.mean(),
                 gate_mae_total=eg.mae_total.mean(), erm_mae_total=ee.mae_total.mean())
        runs.append(r)
        print(f"  {tag} s{seed} r(gate,ICC)={r['gate_icc_pearson']:+.2f} "
              f"retained={ret.round(3)} gate={eg.mae_cat.mean():.4f} erm={ee.mae_cat.mean():.4f}",
              flush=True)
        pd.DataFrame(runs).to_csv(os.path.join(RES, "gate_runs.csv"), index=False)

    for seed in range(5):
        tr, va, te = P.split_facilities(seed)
        one("split", tr, va, te, seed)
    tr, va, te = P.split_facilities(0)
    perm = np.random.default_rng(99).permutation(tr)
    for h, half in enumerate((perm[: len(perm) // 2], perm[len(perm) // 2:])):
        for seed in range(3):
            one(f"half{h}", half, va, te, seed)

    df = pd.DataFrame(runs)
    gcols = [c for c in df.columns if c.startswith("g_")]
    Gm = df[gcols].to_numpy()
    C = np.corrcoef(Gm)
    iu = np.triu_indices(len(Gm), 1)
    h0 = Gm[df.run == "half0"].mean(0); h1 = Gm[df.run == "half1"].mean(0)
    summ = dict(
        n_runs=len(df),
        mean_between_run_corr=float(C[iu].mean()),
        min_between_run_corr=float(C[iu].min()),
        disjoint_half_corr=float(np.corrcoef(h0, h1)[0, 1]),
        gate_icc_r_mean=float(df.gate_icc_pearson.mean()),
        gate_icc_r_min=float(df.gate_icc_pearson.min()),
        gate_icc_r_max=float(df.gate_icc_pearson.max()),
        mean_gate_cell_sd=float(Gm.std(0).mean()),
        max_abs_gate_deviation=float(np.abs(Gm - 1).max()),
        retained_energy_min=float(df[[c for c in df if c.startswith("retained")]].min().min()),
        retained_energy_mean=float(df[[c for c in df if c.startswith("retained")]].mean().mean()),
        gate_vs_erm_mae_cat_pct=float(100 * (df.gate_mae_cat.mean() - df.erm_mae_cat.mean())
                                      / df.erm_mae_cat.mean()),
        gate_vs_erm_mae_total_pct=float(100 * (df.gate_mae_total.mean() - df.erm_mae_total.mean())
                                        / df.erm_mae_total.mean()),
    )
    pd.DataFrame([summ]).to_csv(os.path.join(RES, "gate_stability_summary.csv"), index=False)
    print(pd.Series(summ).to_string())


if __name__ == "__main__":
    main()
