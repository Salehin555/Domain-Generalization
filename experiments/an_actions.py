"""Analysis: actions on the benchmark anchors (anchor_actions.py).

Per anchor and arm: error reduction relative to the deployed model at 1x budget,
per seed then averaged (mean, seed SD), and a paired entity-bootstrap 95% interval
of the ratio of seed-averaged errors. Headroom by budget: size-matched
(src_m vs ref_m), deployed (dep_m vs ref_m), and the deployed model's own gain.
Also checks the 1x/2x/4x arms against budget_curve_rows.csv.
Outputs: results/actions_table.csv, actions_budget.csv
"""
import glob
import os

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
B = 2000


def main():
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "anchor_actions_rows_*.csv"))])
    ents = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "anchor_actions_entities_*.csv"))])
    out = []
    rng = np.random.default_rng(0)
    for (p, sp), g in rows.groupby(["panel", "split"]):
        base = g[g.arm == "dep_1x"].set_index("seed")
        ge = ents[(ents.panel == p) & (ents.split == sp)]
        e0 = ge[ge.arm == "dep_1x"].groupby("fac").mae_cat.mean()
        for arm, ga in g.groupby("arm"):
            ga = ga.set_index("seed")
            for mt in ("mae_cat", "mae_total"):
                red = 100 * (base[mt] - ga[mt]) / base[mt]
                r = dict(panel=p, split=sp, arm=arm, metric=mt, reduction=red.mean(), seed_sd=red.std(ddof=1))
                if mt == "mae_cat":
                    e1 = ge[ge.arm == arm].groupby("fac").mae_cat.mean().reindex(e0.index)
                    a, b = e0.values, e1.values
                    bs = []
                    for _ in range(B):
                        i = rng.integers(0, len(a), len(a))
                        bs.append(100 * (a[i].mean() - b[i].mean()) / a[i].mean())
                    r["lo"], r["hi"] = np.percentile(bs, [2.5, 97.5])
                out.append(r)
    tab = pd.DataFrame(out)
    tab.to_csv(os.path.join(RES, "actions_table.csv"), index=False)
    t = tab[tab.metric == "mae_cat"].pivot_table(index="arm", columns=["panel", "split"], values="reduction")
    print(t.round(1).to_string())

    bud = []
    for (p, sp), g in rows.groupby(["panel", "split"]):
        for m in ("1x", "2x", "4x", "8x", "conv"):
            for mt in ("mae_cat", "mae_total"):
                pv = g.pivot_table(index="seed", columns="arm", values=mt)
                sm = 100 * (pv[f"src_{m}"] - pv[f"ref_{m}"]) / pv[f"src_{m}"]
                dh = 100 * (pv[f"dep_{m}"] - pv[f"ref_{m}"]) / pv[f"dep_{m}"]
                dg = 100 * (pv["dep_1x"] - pv[f"dep_{m}"]) / pv["dep_1x"]
                r = dict(panel=p, split=split_name(sp), budget=m, metric=mt, deployed_gain=dg.mean(),
                         size_matched=sm.mean(), size_matched_sd=sm.std(ddof=1), vs_deployed=dh.mean())
                if m == "conv":
                    r["joint_vs_deployed"] = (100 * (pv["dep_conv"] - pv["joint_conv"]) / pv["dep_conv"]).mean()
                    r["joint_vs_target"] = (100 * (pv["ref_conv"] - pv["joint_conv"]) / pv["ref_conv"]).mean()
                bud.append(r)
    bud = pd.DataFrame(bud)
    bud.to_csv(os.path.join(RES, "actions_budget.csv"), index=False)
    print(bud[bud.metric == "mae_cat"].round(1).to_string())

    old = pd.read_csv(os.path.join(RES, "budget_curve_rows.csv"))
    chk = []
    for _, r in old.iterrows():
        arm = {"src": "src", "ref": "ref", "dep": "dep"}.get(r.arm)
        if arm is None:
            continue
        n = rows[(rows.panel == r.panel) & (rows.split == r.split) & (rows.seed == r.seed) &
                 (rows.arm == f"{arm}_{int(r.mult)}x")]
        if len(n):
            chk.append(abs(n.mae_cat.iloc[0] - r.mae_cat) / r.mae_cat)
    print("reproduction vs budget_curve_rows: n", len(chk), "max rel diff", max(chk) if chk else None)


def split_name(s):
    return s


if __name__ == "__main__":
    main()
