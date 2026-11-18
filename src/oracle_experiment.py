"""
Oracle / irreducibility experiment (headline result).

For a leave-group-out split (train on a source ownership/geography group, deploy on an
unseen target group), we compare:
    TRANSFER : ERM trained on the SOURCE group, evaluated on the TARGET group
    ORACLE   : ERM trained on the TARGET group itself, evaluated on the TARGET group

If TRANSFER approx= ORACLE, the source->target gap is IRREDUCIBLE (a difference in
intrinsic predictability, not in transferable structure) -- so no domain-generalization
method can close it. We also report the naive cross-group gap (source-domain error vs
target-domain error) to show how large the apparent shift looks before the oracle test.

Groups: Ownership (For-profit -> Non-profit) and Geography (Urban -> Rural), from CMS
Provider Information merged by CCN. 3 seeds.

Output: outputs/oracle_results.csv
"""
import numpy as np, pandas as pd, os
from src.data import Panel
from src.train import train_model, evaluate
from src.ownership_probe import build_meta

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache"); OUT = os.path.join(ROOT, "outputs")
ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")
N_TEST = 800

def run(P, tr, va, te):
    m, _ = train_model(P, ERM, tr, va, seed=0)
    return np.array([x["mae_total"] for x in evaluate(m, P, te)])

def main():
    P = Panel(os.path.join(CACHE, "panel.npz"), n_facilities=13402, L=56, H=14, stride=18, seed=0)
    own, urban = build_meta(P)
    idx = np.arange(P.nF)
    SPLITS = {
        "Ownership\n(For-profit -> Non-profit)": (idx[own == "ForProfit"], idx[own == "NonProfit"]),
        "Geography\n(Urban -> Rural)":           (idx[urban == "Y"],       idx[urban == "N"]),
    }
    rows = []
    for name, (src_all, tgt_all) in SPLITS.items():
        for seed in [0, 1, 2]:
            rng = np.random.default_rng(seed)
            s = src_all.copy(); rng.shuffle(s); s_va = s[:400]; s_tr = s[400:]
            t = tgt_all.copy(); rng.shuffle(t)
            t_te = t[:N_TEST]; t_pool = t[N_TEST:]; t_va = t_pool[:400]; t_tr = t_pool[400:]
            src_err = run(P, s_tr, s_va, s_va).mean()            # source-domain error (val)
            transfer = run(P, s_tr, s_va, t_te).mean()           # source-trained on target
            oracle = run(P, t_tr, t_va, t_te).mean()             # target-trained on target
            rows.append(dict(split=name.replace("\n", " "), seed=seed,
                             source_err=src_err, transfer=transfer, oracle=oracle,
                             apparent_gap_pct=100*(transfer-src_err)/src_err,
                             reducible_gap_pct=100*(transfer-oracle)/transfer))
            print(f"  {name.splitlines()[0]:11s} s{seed}  source={src_err:.4f} transfer={transfer:.4f} "
                  f"oracle={oracle:.4f} | apparent gap {100*(transfer-src_err)/src_err:+.1f}% "
                  f"| reducible {100*(transfer-oracle)/transfer:+.1f}%", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "oracle_results.csv"), index=False)
    print("\n=== summary (mean over seeds) ===")
    print(df.groupby("split")[["source_err", "transfer", "oracle",
                               "apparent_gap_pct", "reducible_gap_pct"]].mean().round(4))
    print("\nSaved outputs/oracle_results.csv")

if __name__ == "__main__":
    main()
