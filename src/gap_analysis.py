"""Clean source-vs-target FACILITY generalization gap, controlling for evaluation
period (for the -T protocols both source and target are scored on 2025), so the gap
isolates the cross-facility component rather than the temporal one."""
import numpy as np, pandas as pd, os
from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P = Panel(os.path.join(ROOT, "cache", "panel.npz"), n_facilities=1500, L=56, H=14, stride=12, seed=0)
TM, VM = P.is_2024, P.is_2025
rows = []
for proto in ["LFO", "LFO-T", "LSO-T"]:
    tmask = None if proto == "LFO" else TM
    emask = None if proto == "LFO" else VM       # score BOTH source & target on same period
    for seed in [0, 1, 2]:
        split = P.split_states(seed) if proto == "LSO-T" else P.split_facilities(seed)
        tr, va, te = split
        m, _ = train_model(P, dict(method="erm", epochs=26, patience=6, backbone="mlp"),
                           tr, va, seed=seed, train_mask=tmask, val_mask=VM)
        src = np.mean([x["mae_total"] for x in evaluate(m, P, va, time_mask=emask)])
        tgt = np.mean([x["mae_total"] for x in evaluate(m, P, te, time_mask=emask)])
        rows.append(dict(protocol=proto, seed=seed, source_mae=src, target_mae=tgt,
                         gap_pct=100 * (tgt - src) / src))
        print(f"{proto:6s} s{seed} source={src:.4f} target={tgt:.4f} gap={100*(tgt-src)/src:+.1f}%",
              flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "outputs", "gen_gap_clean.csv"), index=False)
print(df.groupby("protocol")[["source_mae", "target_mae", "gap_pct"]].mean().round(4))
