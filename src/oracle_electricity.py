"""
Reducibility diagnostic on the SECOND dataset (UCI electricity).

This runs the IDENTICAL procedure as src/oracle_experiment.py (the CMS nursing
version): for a leave-group-out split we compare

    TRANSFER : ERM trained on the SOURCE group, evaluated on the TARGET group
    ORACLE   : ERM trained on the TARGET group itself, evaluated on the TARGET group

reducible gap = (TRANSFER - ORACLE) / TRANSFER

PRE-COMMITTED PROTOCOL
----------------------
This diagnostic is run BEFORE any benchmark on this dataset, and its outcome is
reported whatever it is. The purpose is to test whether the diagnostic correctly
PREDICTS the regime:

  * If the reducible gap is near zero, electricity is in the same irreducible
    regime as nursing, and the diagnostic predicts that CD-ISG-TS and every other
    domain-generalization method will tie with a well-tuned pooled baseline.
  * If the reducible gap is clearly positive, transferable cross-domain structure
    exists, and the diagnostic predicts that a domain-generalization method has
    room to help.

Either outcome is reported. Both directions of the group split are run so the
result cannot depend on which group is designated the source.

Output: outputs/oracle_electricity.csv
"""
import os
import numpy as np
import pandas as pd

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache", "electricity")
OUT = os.path.join(ROOT, "outputs")

PANEL = os.path.join(CACHE, "panel_elec.npz")
ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")
SEEDS = [0, 1, 2, 3, 4]


def run(P, tr, va, te):
    m, _ = train_model(P, ERM, tr, va, seed=0)
    return np.array([x["mae_total"] for x in evaluate(m, P, te)])


def main():
    P = Panel(PANEL, n_facilities=10 ** 6, L=56, H=14, stride=18, seed=0,
              n_channels=3, boundary_date=20140101)
    grp = np.asarray(P.state).astype(str)
    idx = np.arange(P.nF)
    print(f"electricity panel: nF={P.nF}  C={P.C}  windows={len(P.Xw)}")
    for g in np.unique(grp):
        print(f"   group {g}: n={(grp == g).sum()}")

    SPLITS = {
        "Load profile (WorkWeek -> Flat)": (idx[grp == "WorkWeek"], idx[grp == "Flat"]),
        "Load profile (Flat -> WorkWeek)": (idx[grp == "Flat"], idx[grp == "WorkWeek"]),
    }

    rows = []
    for name, (src_all, tgt_all) in SPLITS.items():
        # SIZE MATCHING (important on this dataset).
        # With only ~79 entities per group the learning curve is still steep, so
        # giving the transfer model more training entities than the oracle would
        # confound "domain" with "amount of data" and bias the reducible gap
        # downward, making the data look more irreducible than it is. On the CMS
        # panel both groups held thousands of facilities and the curve was flat,
        # so this did not matter; here we equalize the number of TRAINING entities
        # (and of validation entities) given to the transfer and oracle models.
        n_tgt = len(tgt_all)
        n_test = max(20, int(0.35 * n_tgt))
        n_val = max(10, int(0.15 * n_tgt))
        n_train = min(len(src_all) - n_val, n_tgt - n_test - n_val)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            s = src_all.copy(); rng.shuffle(s)
            s_va, s_tr = s[:n_val], s[n_val:n_val + n_train]
            t = tgt_all.copy(); rng.shuffle(t)
            t_te = t[:n_test]; t_pool = t[n_test:]
            t_va, t_tr = t_pool[:n_val], t_pool[n_val:n_val + n_train]
            assert len(s_tr) == len(t_tr) == n_train, "training sizes must match"

            src_err = run(P, s_tr, s_va, s_va).mean()
            transfer = run(P, s_tr, s_va, t_te).mean()
            oracle = run(P, t_tr, t_va, t_te).mean()
            rows.append(dict(split=name, seed=seed, source_err=src_err,
                             transfer=transfer, oracle=oracle,
                             apparent_gap_pct=100 * (transfer - src_err) / src_err,
                             reducible_gap_pct=100 * (transfer - oracle) / transfer))
            print(f"  {name:34s} s{seed}  source={src_err:.4f} transfer={transfer:.4f} "
                  f"oracle={oracle:.4f} | apparent {100*(transfer-src_err)/src_err:+.1f}% "
                  f"| reducible {100*(transfer-oracle)/transfer:+.1f}%", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "oracle_electricity.csv"), index=False)
    print("\n=== summary (mean over seeds) ===")
    print(df.groupby("split")[["source_err", "transfer", "oracle",
                               "apparent_gap_pct", "reducible_gap_pct"]].mean().round(4))
    print("\nSaved outputs/oracle_electricity.csv")


if __name__ == "__main__":
    main()
