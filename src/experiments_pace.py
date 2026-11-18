"""
Does PACE convert the headroom the diagnostic identifies?

The mechanism probe showed a double dissociation: on the reducible electricity
split a zero-training seasonal-naive forecaster beats every trained model by
16%, while on the irreducible nursing split it is 22% worse, and a non-periodic
naive control (persistence) is worse on both. The missing structure is therefore
specifically the extrapolation of periodicity observed in the input window.

PACE adds exactly that as an input-gated branch. This script tests two things that
must BOTH hold for the method to be worth reporting:

  1. it converts headroom where the diagnostic says headroom exists
     (electricity Flat -> WorkWeek, reducible);
  2. it does no harm where the diagnostic says there is none
     (nursing, irreducible), because the branch should switch itself off.

A method that only satisfies (1) would be a periodicity trick, not a
domain-generalization contribution.

Output: outputs/pace_results.csv
"""
import os
import numpy as np
import pandas as pd

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

EP, PAT = 24, 5
SEEDS = [0, 1, 2, 3, 4]

METHODS = [
    ("ERM-DLinear", dict(method="dlinear_plain", epochs=EP, patience=PAT)),
    ("ERM-MLP", dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")),
    ("IRM", dict(method="irm", epochs=EP, patience=PAT, backbone="mlp", lam_irm=1.0)),
    ("CORAL", dict(method="coral", epochs=EP, patience=PAT, backbone="mlp",
                   lam_coral=1.0)),
    ("CD-ISG-TS", dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
                       lam_inv=0.5, gate_lr=0.01)),
    ("PACE (proposed)", dict(method="pace", epochs=EP, patience=PAT,
                             backbone="mlp", gate_lr=0.05)),
    ("PACE + sub-band gate", dict(method="pace", epochs=EP, patience=PAT,
                                  backbone="mlp", gate_lr=0.05, with_gate=True)),
]

CASES = [
    dict(name="Electricity: Flat -> WorkWeek (reducible)",
         path=os.path.join(ROOT, "cache", "electricity", "panel_elec.npz"),
         boundary=20140101, nfac=10 ** 6, src="Flat", tgt="WorkWeek"),
    dict(name="Traffic: weekly low -> high (mildly reducible)",
         path=os.path.join(ROOT, "cache", "panels", "panel_traffic.npz"),
         boundary=20160504, nfac=10 ** 6, src="Flat", tgt="WorkWeek"),
]


def main():
    rows = []
    for case in CASES:
        P = Panel(case["path"], n_facilities=case["nfac"], L=56, H=14, stride=18,
                  seed=0, n_channels=3, boundary_date=case["boundary"])
        grp = np.asarray(P.state).astype(str)
        idx = np.arange(P.nF)
        src_all, tgt_all = idx[grp == case["src"]], idx[grp == case["tgt"]]
        print(f"\n### {case['name']}  (src n={len(src_all)}, tgt n={len(tgt_all)})",
              flush=True)
        n_val = max(8, int(0.15 * len(src_all)))
        for mname, cfg in METHODS:
            errs = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                s = src_all.copy(); rng.shuffle(s)
                m, _ = train_model(P, cfg, s[n_val:], s[:n_val], seed=seed)
                per = evaluate(m, P, tgt_all)
                errs.append(np.mean([r["mae_total"] for r in per]))
                for r in per:
                    rows.append(dict(case=case["name"], method=mname, seed=seed,
                                     fac=r["fac"], mae_total=r["mae_total"]))
            print(f"  {mname:24s} MAE={np.mean(errs):.4f} +/- {np.std(errs):.4f}",
                  flush=True)

    # nursing no-harm check
    P = Panel(os.path.join(ROOT, "cache", "panel.npz"), n_facilities=1500, L=56,
              H=14, stride=18, seed=0, n_channels=3, boundary_date=20250101)
    tr, va, te = P.split_facilities(seed=0)
    print("\n### Nursing (irreducible): no-harm check", flush=True)
    for mname, cfg in [m for m in METHODS if m[0] in
                       ("ERM-MLP", "CD-ISG-TS", "PACE (proposed)")]:
        errs = []
        for seed in SEEDS[:3]:
            m, _ = train_model(P, cfg, tr, va, seed=seed)
            errs.append(np.mean([r["mae_total"] for r in evaluate(m, P, te)]))
            for r in evaluate(m, P, te):
                rows.append(dict(case="Nursing (irreducible)", method=mname,
                                 seed=seed, fac=r["fac"], mae_total=r["mae_total"]))
        print(f"  {mname:24s} MAE={np.mean(errs):.4f} +/- {np.std(errs):.4f}",
              flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "pace_results.csv"), index=False)
    print("\nSaved outputs/pace_results.csv")


if __name__ == "__main__":
    main()
