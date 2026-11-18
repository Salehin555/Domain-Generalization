"""
Why do all methods fail to convert the available headroom?

HYPOTHESIS
----------
On the electricity Flat -> WorkWeek split, the source group (Flat) has almost no
weekly cycle while the target group (WorkWeek) has a strong one. A model trained
only on Flat clients has therefore never seen a weekly pattern and cannot learn to
emit one, no matter what invariance penalty is applied. The transferable structure
is not a fixed pattern to memorize from the source; it is the CONDITIONAL rule
"continue whatever periodicity the input window already shows". The input window at
deployment does contain the target's weekly cycle, so a predictor that extrapolates
observed periodicity should recover part of the gap without ever seeing target
labels, while a trained regressor biased toward source dynamics will not.

DECISIVE, ZERO-COST TEST
------------------------
Seasonal-naive forecasting requires no training at all: y_hat[t+h] = y[t+h-7]. It
is therefore immune to source-domain bias by construction. If the hypothesis holds,
seasonal-naive should be competitive with, or better than, the trained models on
the reducible split, while being clearly worse on a panel that is irreducible.
If seasonal-naive is uniformly bad, the hypothesis is wrong and no amount of
periodicity machinery will help.

We report three references on the same test sets:
    ERM-MLP        : trained on the source group   (what the paper already reports)
    SeasonalNaive  : no training whatsoever
    Persistence    : last value carried forward, a periodicity-free control
"""
import os
import numpy as np
import pandas as pd

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

PANELS = {
    "Electricity": dict(path=os.path.join(ROOT, "cache", "electricity",
                                          "panel_elec.npz"), boundary=20140101),
    "Nursing": dict(path=os.path.join(ROOT, "cache", "panel.npz"),
                    boundary=20250101, nfac=1500),
}
ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")
SEEDS = [0, 1, 2]


def seasonal_naive(Xw, H, period=7):
    """y_hat[:, :, h] = last observed value at the same phase of the cycle."""
    L = Xw.shape[-1]
    idx = [L - period + (h % period) for h in range(H)]
    idx = [i - period * ((i - (L - 1)) // period + 1) if i > L - 1 else i for i in idx]
    return Xw[:, :, idx]


def persistence(Xw, H):
    return np.repeat(Xw[:, :, -1:], H, axis=2)


def per_entity_mae(pred, Yw, fid):
    rows = []
    pt, yt = pred.sum(1), Yw.sum(1)
    for f in np.unique(fid):
        m = fid == f
        rows.append(np.abs(pt[m] - yt[m]).mean())
    return float(np.mean(rows))


def main():
    rows = []
    for pname, spec in PANELS.items():
        P = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6),
                  L=56, H=14, stride=18, seed=0, n_channels=3,
                  boundary_date=spec["boundary"])
        grp = np.asarray(P.state).astype(str)
        idx = np.arange(P.nF)
        if pname == "Electricity":
            src, tgt = idx[grp == "Flat"], idx[grp == "WorkWeek"]
            label = "Flat -> WorkWeek (reducible)"
        else:
            # nursing: use the ownership analogue, an irreducible split
            uniq = [u for u in np.unique(grp) if u not in ("nan",)]
            half = len(uniq) // 2 or 1
            src = idx[np.isin(grp, uniq[:half])]
            tgt = idx[np.isin(grp, uniq[half:])]
            label = "ownership split (irreducible)"
        if len(src) < 20 or len(tgt) < 20:
            print(f"skip {pname}: groups too small"); continue

        Xw, Yw, fid = P.eval_arrays(tgt)
        sn = per_entity_mae(seasonal_naive(Xw, P.H), Yw, fid)
        pe = per_entity_mae(persistence(Xw, P.H), Yw, fid)

        erm = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            s = src.copy(); rng.shuffle(s)
            nv = max(8, int(0.15 * len(s)))
            m, _ = train_model(P, ERM, s[nv:], s[:nv], seed=seed)
            erm.append(np.mean([r["mae_total"] for r in evaluate(m, P, tgt)]))
        erm = float(np.mean(erm))

        print(f"\n### {pname}: {label}")
        print(f"  ERM-MLP (trained on source) : {erm:.4f}")
        print(f"  SeasonalNaive (no training) : {sn:.4f}   "
              f"({100*(erm-sn)/erm:+.1f}% vs ERM)")
        print(f"  Persistence  (no training)  : {pe:.4f}", flush=True)
        rows.append(dict(panel=pname, split=label, erm=erm,
                         seasonal_naive=sn, persistence=pe,
                         sn_gain_pct=100 * (erm - sn) / erm))

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mechanism_probe.csv"), index=False)
    print("\nSaved outputs/mechanism_probe.csv")


if __name__ == "__main__":
    main()
