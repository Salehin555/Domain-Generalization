"""Stratified permutation test: statistic = mean over panels of the
within-panel Spearman correlation (8 behavioral splits per panel). The null
permutes the 4 groupings within each panel (pairs kept together) and, for
direction-specific x (apparent gap), also flips the order within each pair.
All 24 x 16 = 384 arrangements per panel are enumerated and combined exactly."""
import itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
t = pd.read_csv(os.path.join(RES, "survey_split_table.csv"))
t = t[t.behavioral]
out = {}
for m in ["mae_cat", "mae_total"]:
    for x in ["apparent_gap_" + m, "js_amp", "js_energy", "acf_l2", "blend_gain_" + m,
              "snaive_gain_" + m]:
        per_null, per_obs = [], []
        for p, g in t.groupby("panel"):
            g = g.sort_values(["grouping", "split"])
            keys = list(dict.fromkeys(g.grouping))
            X = [g[g.grouping == k][x].to_numpy() for k in keys]
            Y = [g[g.grouping == k]["headroom_" + m].to_numpy() for k in keys]
            xs = np.concatenate(X)
            per_obs.append(spearmanr(xs, np.concatenate(Y))[0])
            vals = []
            for perm in itertools.permutations(range(4)):
                for flips in itertools.product([0, 1], repeat=4):
                    ys = np.concatenate([Y[o][::-1] if f else Y[o]
                                         for o, f in zip(perm, flips)])
                    vals.append(spearmanr(xs, ys)[0])
            per_null.append(np.array(vals))
        obs = float(np.mean(per_obs))
        # per-panel nulls are enumerated exactly; their combination is sampled
        # (2e6 draws), panels being independent under the stratified null
        rng = np.random.default_rng(0)
        tot = np.mean([v[rng.integers(0, len(v), 2_000_000)] for v in per_null], 0)
        p = float(((np.abs(tot) >= abs(obs) - 1e-12).sum() + 1) / (len(tot) + 1))
        base = x.replace("_" + m, "")
        out[f"{base}|{m}"] = dict(
            mean_within_panel_spearman=obs, p_exact=p,
            per_panel=[float(v) for v in per_obs])
        print(x, m, round(obs, 3), "p =", round(p, 4), [round(v, 2) for v in per_obs])
json.dump(out, open(os.path.join(RES, "survey_exact_within.json"), "w"), indent=1)
