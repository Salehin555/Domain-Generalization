"""
Robustness of FADE and of the reducibility estimate to
    (ii) lookback / horizon, and (iii) the batch design (G entities x K windows)
    used to estimate the invariance term Omega.

Wavelet family and depth are already covered by the existing ablation, so this
script fills the two gaps. Both quantities are reported, because a design choice
that changed the diagnostic's answer would matter far more than one that only
moved the model's error.

Lookback values are restricted to multiples of eight so that a level-3
decomposition remains exact under periodization.

Output: outputs/sensitivity_lookback.csv, outputs/sensitivity_batch.csv,
        outputs/sensitivity_diagnostic.csv
"""
import os
import numpy as np
import pandas as pd

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

NUR = os.path.join(ROOT, "cache", "panel.npz")
ELE = os.path.join(ROOT, "cache", "electricity", "panel_elec.npz")
EP, PAT = 22, 5
SEEDS = [0, 1, 2]

ERM = dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")
FADE = dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
            lam_inv=0.5, gate_lr=0.05, periodic=True)


def run_pair(P, tr, va, te, extra=None):
    """Return (ERM, FADE) mean MAE on the same split."""
    out = []
    for cfg in (ERM, FADE):
        c = dict(cfg)
        if extra:
            c.update(extra)
        e = []
        for s in SEEDS:
            m, _ = train_model(P, c, tr, va, seed=s)
            e.append(np.mean([r["mae_total"] for r in evaluate(m, P, te)]))
        out.append(float(np.mean(e)))
    return out


def sweep_lookback():
    rows = []
    print("### lookback / horizon (nursing development subset)", flush=True)
    for L in [32, 56, 112]:
        for H in [7, 14, 28]:
            P = Panel(NUR, n_facilities=1200, L=L, H=H, stride=18, seed=0,
                      n_channels=3, boundary_date=20250101)
            tr, va, te = P.split_facilities(seed=0)
            erm, fade = run_pair(P, tr, va, te)
            gain = 100 * (erm - fade) / erm
            rows.append(dict(L=L, H=H, erm=erm, fade=fade, gain_pct=gain))
            print(f"  L={L:3d} H={H:2d}  ERM={erm:.4f}  FADE={fade:.4f}  "
                  f"gain={gain:+.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_lookback.csv"),
                              index=False)


def sweep_batch():
    rows = []
    print("\n### batch design for Omega (nursing development subset)", flush=True)
    P = Panel(NUR, n_facilities=1200, L=56, H=14, stride=18, seed=0,
              n_channels=3, boundary_date=20250101)
    tr, va, te = P.split_facilities(seed=0)
    for G, K in [(16, 8), (32, 4), (32, 8), (32, 16), (64, 8)]:
        erm, fade = run_pair(P, tr, va, te, extra=dict(G=G, K=K))
        gain = 100 * (erm - fade) / erm
        rows.append(dict(G=G, K=K, erm=erm, fade=fade, gain_pct=gain))
        print(f"  G={G:3d} K={K:3d}  ERM={erm:.4f}  FADE={fade:.4f}  "
              f"gain={gain:+.2f}%", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_batch.csv"),
                              index=False)


def sweep_diagnostic():
    """Does the reducible gap itself move with lookback and horizon?"""
    rows = []
    print("\n### reducibility estimate vs lookback / horizon (electricity)",
          flush=True)
    for L, H in [(32, 14), (56, 7), (56, 14), (56, 28), (112, 14)]:
        P = Panel(ELE, n_facilities=10 ** 6, L=L, H=H, stride=18, seed=0,
                  n_channels=3, boundary_date=20140101)
        grp = np.asarray(P.state).astype(str); idx = np.arange(P.nF)
        src, tgt = idx[grp == "Flat"], idx[grp == "WorkWeek"]
        n_test = max(20, int(0.35 * len(tgt)))
        n_val = max(8, int(0.15 * len(tgt)))
        n_tr = min(len(src) - n_val, len(tgt) - n_test - n_val)
        gaps = []
        for s in SEEDS:
            rng = np.random.default_rng(s)
            a = src.copy(); rng.shuffle(a)
            b = tgt.copy(); rng.shuffle(b)
            s_va, s_tr = a[:n_val], a[n_val:n_val + n_tr]
            t_te, pool = b[:n_test], b[n_test:]
            t_va, t_tr = pool[:n_val], pool[n_val:n_val + n_tr]
            def err(tr_, va_, te_):
                m, _ = train_model(P, ERM, tr_, va_, seed=0)
                return float(np.mean([r["mae_total"] for r in evaluate(m, P, te_)]))
            transfer = err(s_tr, s_va, t_te)
            oracle = err(t_tr, t_va, t_te)
            gaps.append(100 * (transfer - oracle) / transfer)
        rows.append(dict(L=L, H=H, reducible_gap_pct=float(np.mean(gaps)),
                         sd=float(np.std(gaps))))
        print(f"  L={L:3d} H={H:2d}  reducible gap = {np.mean(gaps):+.2f}% "
              f"(sd {np.std(gaps):.2f})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "sensitivity_diagnostic.csv"),
                              index=False)


if __name__ == "__main__":
    sweep_lookback()
    sweep_batch()
    sweep_diagnostic()
    print("\nSaved outputs/sensitivity_*.csv")
