"""Predicting the reducible gap for a NEW group without any of its labels.

THE OBJECTION THIS ANSWERS
--------------------------
The oracle test trains a model on the target group, so it needs target labels.
At deployment those do not exist: a new facility has a history but no future. If
the diagnostic were only ever reachable through the oracle, it could never be run
when it actually matters, and the contribution would be retrospective only.

THE DISTINCTION THAT DISSOLVES IT
---------------------------------
Two different things are being conflated.

  target LABELS  the future values of the new group. Never available. Needed
                 only by the oracle.
  target INPUTS  the recent history of the new group. ALWAYS available, since
                 without it no forecast could be produced at all.

The spectral divergence screen reads only band-energy profiles, which come from
Panel.Xw (input windows) and never from Panel.Yw. So the screen is already
deployable. What the paper has not yet shown is that it can be turned from a
rank-ordering signal into a calibrated estimate of the gap itself.

THE PROPOSAL: source-internal calibration
-----------------------------------------
The oracle needs labels, but not the TARGET's labels: it needs labels for
whatever entities it is trained on. Inside the historical pool every entity is
fully labelled. So a practitioner can:

  Phase 1, offline, once, on data already in hand
      carve the historical pool into many pseudo source/target splits, run the
      oracle test on each, and regress the measured reducible gap on the
      training-free divergence. Costs compute, needs no new data.

  Phase 2, deployment, per new group, zero labels
      compute the divergence between the historical pool and the new group's
      INPUTS, and read the predicted gap off the calibration curve.

This converts the oracle from a per-deployment cost into a one-time in-house
calibration, and the curve is dataset-specific, which should beat the
cross-panel correlation of r=0.57 the paper currently reports.

WHAT WOULD FALSIFY IT
---------------------
Leave-one-out: every held-out split's measured gap is excluded from the fit that
predicts it. A shuffled-divergence control re-runs the identical pipeline with
the predictor broken, and must collapse. If the shuffled control predicts as
well, the divergence carries nothing and the idea is dead.

Output: outputs/label_free_calibration.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from src.data import Panel
from src.survey_reducibility import entity_stats, js_div
from src.train import train_model, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")
SEEDS = [0, 1]
KEYS = ["size", "vol", "weekly", "trend"]

PANELS = {
    "Electricity": dict(path=os.path.join(ROOT, "cache", "electricity",
                                          "panel_elec.npz"), boundary=20140101),
    "Solar": dict(path=os.path.join(ROOT, "cache", "panels", "panel_solar.npz"),
                  boundary=20060902),
    "Traffic": dict(path=os.path.join(ROOT, "cache", "panels",
                                      "panel_traffic.npz"), boundary=20160504),
    "Nursing": dict(path=os.path.join(ROOT, "cache", "panel.npz"),
                    boundary=20250101, nfac=1200),
}


def make_splits(stats, n_splits, rng, min_grp=25):
    """Diverse group pairs along random directions in descriptor space.

    Quartile cuts on the four named descriptors give only eight splits, too few
    to fit anything. Random directions with a varying cut severity span a much
    wider range of group separation, which is what the calibration curve needs
    in order to cover the divergences a real deployment might present.
    """
    V = np.stack([stats[k] for k in KEYS], 1).astype(np.float64)
    V = (V - V.mean(0)) / (V.std(0) + 1e-8)
    out, tries = [], 0
    while len(out) < n_splits and tries < n_splits * 25:
        tries += 1
        w = rng.normal(size=len(KEYS))
        w /= np.linalg.norm(w) + 1e-12
        s = V @ w
        q = float(rng.choice([10, 15, 20, 25, 30, 35]))
        lo, hi = np.percentile(s, [q, 100 - q])
        A = np.where(s <= lo)[0]
        B = np.where(s >= hi)[0]
        if len(A) < min_grp or len(B) < min_grp:
            continue
        if rng.random() < 0.5:
            A, B = B, A
        out.append((A, B))
    return out


def measure_gap(P, src_all, tgt_all, seed):
    """Reducible gap. Uses labels, but only of entities already in the pool."""
    rng = np.random.default_rng(seed)
    n_test = max(15, int(0.35 * len(tgt_all)))
    n_val = max(6, int(0.15 * len(tgt_all)))
    n_train = min(len(src_all) - n_val, len(tgt_all) - n_test - n_val)
    if n_train < 12:
        return None
    s = src_all.copy(); rng.shuffle(s)
    s_va, s_tr = s[:n_val], s[n_val:n_val + n_train]
    t = tgt_all.copy(); rng.shuffle(t)
    t_te, pool = t[:n_test], t[n_test:]
    t_va, t_tr = pool[:n_val], pool[n_val:n_val + n_train]

    def run(tr, va, te):
        m, _ = train_model(P, ERM, tr, va, seed=seed)
        return float(np.mean([r["mae_total"] for r in evaluate(m, P, te)]))

    tra = run(s_tr, s_va, t_te)
    ora = run(t_tr, t_va, t_te)
    return 100.0 * (tra - ora) / tra


def loo_predict(x, y):
    """Leave-one-out linear prediction: each y is predicted without ever using it."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    pred = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, a = np.polyfit(x[m], y[m], 1)
        pred[i] = b * x[i] + a
    return pred


def report(df):
    from scipy import stats as ss
    print("\n" + "=" * 74)
    print("LEAVE-ONE-OUT: predicting a held-out group's gap from INPUTS ALONE")
    print("=" * 74)
    for name, g in df.groupby("panel"):
        if len(g) < 6:
            print(f"\n{name}: only {len(g)} splits, skipping")
            continue
        x, y = g.spec_div.values, g.reducible_gap_pct.values
        pred = loo_predict(x, y)
        r, p = ss.pearsonr(pred, y)
        mae = float(np.mean(np.abs(pred - y)))
        base = float(np.mean(np.abs(y - np.mean(y))))
        rc = np.random.default_rng(0)
        cr, cm = [], []
        for _ in range(300):
            p2 = loo_predict(rc.permutation(x), y)
            cr.append(ss.pearsonr(p2, y)[0]); cm.append(np.mean(np.abs(p2 - y)))
        acc = float(np.mean((pred >= 3.0) == (y >= 3.0)))
        print(f"\n{name}  (n={len(g)} splits, actual gap {y.min():+.1f}% .. {y.max():+.1f}%)")
        print(f"   LOO predicted vs actual  r={r:+.3f} (p={p:.2g})   MAE={mae:.2f} pp")
        print(f"   shuffled-divergence ctrl r={np.mean(cr):+.3f} +/-{np.std(cr):.3f}"
              f"   MAE={np.mean(cm):.2f} pp")
        print(f"   predict-the-mean baseline                  MAE={base:.2f} pp")
        print(f"   decision accuracy at a 3% threshold: {100*acc:.0f}%")


def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    want = sys.argv[2:] or ["Electricity", "Solar"]
    rows = []
    for name in want:
        spec = PANELS[name]
        P = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6), L=56, H=14,
                  stride=18, seed=0, n_channels=3, boundary_date=spec["boundary"])
        stats, prof = entity_stats(P)
        splits = make_splits(stats, n_splits, np.random.default_rng(0))
        print(f"\n### {name}: {P.nF} entities, {len(splits)} splits", flush=True)
        for k, (A, B) in enumerate(splits):
            # TRAINING-FREE, INPUT-ONLY: mean band-energy profile of each group
            sd = js_div(prof[A].mean(0), prof[B].mean(0))
            gaps = [g for g in (measure_gap(P, A, B, s) for s in SEEDS)
                    if g is not None]
            if not gaps:
                continue
            gap = float(np.mean(gaps))
            rows.append(dict(panel=name, split=k, n_src=len(A), n_tgt=len(B),
                             spec_div=sd, reducible_gap_pct=gap))
            print(f"  split {k:2d}  n=({len(A):3d},{len(B):3d})  "
                  f"JSdiv={sd:.5f}  gap={gap:+6.2f}%", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "label_free_calibration.csv")
    df.to_csv(path, index=False)
    report(df)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
