"""Domain-generalization baselines with source-validation tuning.

The case study ran IRM, CORAL, DANN and GroupDRO with fixed penalty weights and
each training entity as a domain. Here, on the four benchmark anchors (common
manifest, five seeds), each method is run over a small grid of its penalty weight
(GroupDRO: step size) and two domain definitions:

    entity    every training entity is a domain (as in the case study)
    cluster   four domains from k-means on the standardized observed-era entity
              statistics of the survey (level, variation, weekly amplitude,
              low-frequency share), fitted on the source training entities only

Early stopping and checkpoint selection use only checkpoints trained with the
penalty switched on (select_after_warmup), and the configuration is chosen by
source-validation error, so no target data enters any choice. Each configuration
is also scored on the target test entities, so the best configuration chosen in
hindsight can be reported as an upper reference.
Output: results/dg_tuned_rows.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import METRICS, RES, allocate, per_entity, predict  # noqa: E402
from anchor import MLP, anchors  # noqa: E402
from joint_finetune import KEEP  # noqa: E402
from src.survey_reducibility import entity_stats  # noqa: E402
from src.train import train_model  # noqa: E402

torch.set_num_threads(int(os.environ.get("EXP_THREADS", "3")))
GRID = {
    "IRM": ("irm", "lam_irm", [0.1, 1.0, 10.0]),
    "CORAL": ("coral", "lam_coral", [0.1, 1.0, 10.0]),
    "DANN": ("dann", "lam_dann", [0.1, 0.5, 2.0]),
    "GroupDRO": ("groupdro", "dro_eta", [0.001, 0.01, 0.1]),
}


def clusters(stats, train, k=4, seed=0):
    Z = np.column_stack([stats[c] for c in ("size", "vol", "weekly", "trend")])
    mu, sd = Z[train].mean(0), Z[train].std(0) + 1e-9
    Z = (Z - mu) / sd
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z[train])
    return km.predict(Z)


def main():
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar"]
    tag = "_".join(want)
    rows = []
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if (panel, split) not in KEEP:
            continue
        stats, _ = entity_stats(P)
        for seed in seeds:
            t0 = time.time()
            al = allocate(S, T, np.random.default_rng(seed), floors=floors, fracs=fracs,
                          fallback=(panel == "Solar"))
            dmap = clusters(stats, al["s_tr"], seed=seed)

            def run(label, cfg, extra):
                m, diag = train_model(P, cfg, al["s_tr"], al["s_va"], seed=seed)
                _, Yt, Pt, fid = predict(m, P, al["t_te"])
                e = per_entity(Pt, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, method=label,
                         val_mae=float(diag["val_mae"]), **extra)
                for mt in METRICS:
                    r[mt] = float(e[mt].mean())
                rows.append(r)

            run("ERM", dict(method="erm", **MLP), dict(domains="-", weight=np.nan))
            for label, (meth, key, vals) in GRID.items():
                for dom in ("entity", "cluster"):
                    for v in vals:
                        cfg = dict(method=meth, select_after_warmup=True, **MLP)
                        cfg[key] = v
                        if dom == "cluster":
                            cfg["domain_map"] = dmap
                        run(label, cfg, dict(domains=dom, weight=v))
            pd.DataFrame(rows).to_csv(os.path.join(RES, f"dg_tuned_rows_{tag}.csv"), index=False)
            print(f"  {panel} {split} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
