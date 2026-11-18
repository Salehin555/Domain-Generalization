"""
Unified CD-ISG-TS: does one architecture serve both regimes?

The module now carries two frequency-domain mechanisms:
  (a) the category-conditioned sub-band gate, trained under the cross-domain
      invariance objective, which reads WHICH bands transfer;
  (b) an input-driven periodicity branch, which EXTENDS the rhythm the presented
      window already carries.

The diagnostic predicts which one should matter where. On an irreducible panel
there is nothing to exploit, so (b) should switch itself off and the module should
match its backbone. On a reducible panel the transferable structure is periodic, so
(b) should fire and convert it.

This script runs the full component ablation on every panel, so the contribution of
each mechanism is separated rather than asserted:
    ERM-MLP                 backbone only
    + gate                  the original CD-ISG-TS
    + periodicity           the branch alone
    + both (unified)        the proposed module

Output: outputs/unified_results.csv
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

VARIANTS = [
    ("ERM-MLP (backbone)", dict(method="erm", epochs=EP, patience=PAT,
                                backbone="mlp")),
    ("+ sub-band gate", dict(method="cdisgts", epochs=EP, patience=PAT,
                             backbone="mlp", lam_inv=0.5, gate_lr=0.01)),
    ("+ periodicity branch", dict(method="cdisgts", epochs=EP, patience=PAT,
                                  backbone="mlp", lam_inv=0.0, gate_lr=0.05,
                                  periodic=True, gate_init=12.0)),
    ("Unified CD-ISG-TS", dict(method="cdisgts", epochs=EP, patience=PAT,
                               backbone="mlp", lam_inv=0.5, gate_lr=0.05,
                               periodic=True)),
]

CASES = [
    dict(name="Electricity (reducible)",
         path=os.path.join(ROOT, "cache", "electricity", "panel_elec.npz"),
         boundary=20140101, nfac=10 ** 6, src="Flat", tgt="WorkWeek"),
    dict(name="Solar (reducible)",
         path=os.path.join(ROOT, "cache", "panels", "panel_solar.npz"),
         boundary=20060902, nfac=10 ** 6, src="Flat", tgt="WorkWeek"),
    dict(name="Traffic (mild)",
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
        print(f"\n### {case['name']}  (src={len(src_all)}, tgt={len(tgt_all)})",
              flush=True)
        n_val = max(6, int(0.15 * len(src_all)))
        for vname, cfg in VARIANTS:
            errs = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                s = src_all.copy(); rng.shuffle(s)
                m, _ = train_model(P, cfg, s[n_val:], s[:n_val], seed=seed)
                per = evaluate(m, P, tgt_all)
                errs.append(np.mean([r["mae_total"] for r in per]))
                for r in per:
                    rows.append(dict(case=case["name"], variant=vname, seed=seed,
                                     fac=r["fac"], mae_total=r["mae_total"]))
            print(f"  {vname:24s} {np.mean(errs):.4f} +/- {np.std(errs):.4f}",
                  flush=True)

    # nursing, the irreducible panel: the no-harm requirement
    P = Panel(os.path.join(ROOT, "cache", "panel.npz"), n_facilities=1500, L=56,
              H=14, stride=18, seed=0, n_channels=3, boundary_date=20250101)
    tr, va, te = P.split_facilities(seed=0)
    print("\n### Nursing (irreducible): no-harm requirement", flush=True)
    for vname, cfg in VARIANTS:
        errs = []
        for seed in SEEDS[:3]:
            m, _ = train_model(P, cfg, tr, va, seed=seed)
            per = evaluate(m, P, te)
            errs.append(np.mean([r["mae_total"] for r in per]))
            for r in per:
                rows.append(dict(case="Nursing (irreducible)", variant=vname,
                                 seed=seed, fac=r["fac"], mae_total=r["mae_total"]))
        print(f"  {vname:24s} {np.mean(errs):.4f} +/- {np.std(errs):.4f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "unified_results.csv"), index=False)
    print("\nSaved outputs/unified_results.csv")


if __name__ == "__main__":
    main()
