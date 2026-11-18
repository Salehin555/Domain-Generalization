"""Can the gate be made to earn its keep on accuracy?

HYPOTHESIS
----------
The paper's own survey says solar carries 6.2% of reducible headroom concentrated
in the VOLATILITY and TREND splits (averaging 9.9%), not in the weekly splits
(2.0%). The periodicity branch is tuned to a weekly period and cannot reach it,
and the sub-band gate -- static or input-conditioned on the feature -- does not
reach it either.

Why the feature gate cannot: under absolute-error loss the optimal point forecast
is shrunk toward the conditional mean by an amount that depends on how
predictable the entity is. A model fitted on a low-volatility source group learns
a low-shrinkage rule; deployed on a high-volatility target it tracks too eagerly.
That mis-calibration lives in the EMITTED forecast, and re-weighting an internal
feature before a forecast head can be partly undone by the head.

So the proposed mechanism rescales the forecast's own sub-bands, conditioned on a
label-free descriptor of the presented window (its relative sub-band energy
profile). Zero-initialized, hence exactly the identity at initialization.

NOTE ON WHAT WOULD NOT WORK
---------------------------
Matching the forecast's spectrum to the input's spectrum would HURT: forecasts
are optimally smoother than realizations, which is exactly what the manuscript's
inconclusive spectral test observed. The mechanism here is free to shrink or
amplify per band and must learn the direction from source data alone.

Splits are constructed by the same protocol as survey_reducibility.py so the
measured headroom applies. Output: outputs/gate_calibration_v2.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from src.data import Panel
from src.survey_reducibility import entity_stats
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

EP, PAT = 24, 5
SEEDS = [0, 1, 2]

VARIANTS = [
    ("ERM-MLP (backbone)", dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")),
    ("static gate", dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
                         lam_inv=0.5, gate_lr=0.01)),
    ("calibration gate", dict(method="fade_calib", epochs=EP, patience=PAT,
                              backbone="mlp", lam_inv=0.0, gate_lr=0.05)),
    ("calib (shrink-only)", dict(method="fade_calib", epochs=EP, patience=PAT,
                                 backbone="mlp", lam_inv=0.0, gate_lr=0.05,
                                 shrink_only=True)),
]

# (panel, statistic, direction) chosen where the survey measured real headroom
# that the periodicity branch does not serve.
PANELS = {
    "Solar": dict(path=os.path.join(ROOT, "cache", "panels", "panel_solar.npz"),
                  boundary=20060902),
    "Electricity": dict(path=os.path.join(ROOT, "cache", "electricity", "panel_elec.npz"),
                        boundary=20140101),
    "Traffic": dict(path=os.path.join(ROOT, "cache", "panels", "panel_traffic.npz"),
                    boundary=20160504),
    "Nursing": dict(path=os.path.join(ROOT, "cache", "panel.npz"),
                    boundary=20250101, nfac=1500),
}

CASES = [
    ("Solar", "vol", "high->low"),      # survey: +11.28%
    ("Solar", "vol", "low->high"),      # survey: +10.97%
    ("Solar", "trend", "low->high"),    # survey:  +9.48%
    ("Electricity", "trend", "high->low"),   # survey: +18.35%
    ("Electricity", "vol", "low->high"),     # survey:  +8.47%
    ("Traffic", "vol", "high->low"),         # survey:  +4.22%
    ("Nursing", "vol", "high->low"),         # no-harm check: +0.33%
]


def make_split(P, stat_name, direction):
    stats, _ = entity_stats(P)
    v = stats[stat_name]
    idx = np.arange(P.nF)
    q1, q3 = np.percentile(v, [25, 75])
    A, B = idx[v <= q1], idx[v >= q3]
    if len(A) < 30 or len(B) < 30:
        return None
    return (B, A) if direction == "high->low" else (A, B)


def run_case(P, src_all, tgt_all, seed, cfg):
    """Same sizing rule as survey_reducibility.diagnose, transfer direction only."""
    rng = np.random.default_rng(seed)
    n_test = max(20, int(0.35 * len(tgt_all)))
    n_val = max(8, int(0.15 * len(tgt_all)))
    n_train = min(len(src_all) - n_val, len(tgt_all) - n_test - n_val)
    if n_train < 15:
        n_test = max(10, int(0.30 * len(tgt_all)))
        n_val = max(4, int(0.12 * len(tgt_all)))
        n_train = min(len(src_all) - n_val, len(tgt_all) - n_test - n_val)
        if n_train < 12:
            return None
    s = src_all.copy(); rng.shuffle(s)
    s_va, s_tr = s[:n_val], s[n_val:n_val + n_train]
    t = tgt_all.copy(); rng.shuffle(t)
    t_te = t[:n_test]
    m, _ = train_model(P, cfg, s_tr, s_va, seed=seed)
    return float(np.mean([r["mae_total"] for r in evaluate(m, P, t_te)]))


def main():
    rows = []
    cache = {}
    for panel, stat, direction in CASES:
        if panel not in cache:
            spec = PANELS[panel]
            cache[panel] = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6),
                                 L=56, H=14, stride=18, seed=0, n_channels=3,
                                 boundary_date=spec["boundary"])
        P = cache[panel]
        sp = make_split(P, stat, direction)
        if sp is None:
            print(f"  SKIP {panel} {stat} {direction}: groups too small", flush=True)
            continue
        src_all, tgt_all = sp
        label = f"{panel} {stat}: {direction}"
        print(f"\n### {label}  (src={len(src_all)}, tgt={len(tgt_all)})", flush=True)
        for vname, cfg in VARIANTS:
            errs = []
            for seed in SEEDS:
                e = run_case(P, src_all, tgt_all, seed, cfg)
                if e is not None:
                    errs.append(e)
                    rows.append(dict(case=label, panel=panel, variant=vname,
                                     seed=seed, mae=e))
            if errs:
                print(f"  {vname:22s} {np.mean(errs):.4f} +/- {np.std(errs):.4f}",
                      flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "gate_calibration_v2.csv"), index=False)

    print("\n" + "=" * 70)
    print("GAIN AGAINST BACKBONE (%), positive = better")
    print("=" * 70)
    m = df.groupby(["case", "variant"]).mae.mean().unstack()
    base = m["ERM-MLP (backbone)"]
    for v in ["static gate", "calibration gate", "calib (shrink-only)"]:
        if v in m:
            m[v + " gain%"] = 100 * (base - m[v]) / base
    cols = [c for c in m.columns if c.endswith("gain%")]
    print(m[cols].round(2).to_string())
    print("\nSaved outputs/gate_calibration_v2.csv")


if __name__ == "__main__":
    sys.exit(main())
