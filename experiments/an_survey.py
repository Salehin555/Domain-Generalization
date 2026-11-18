"""Statistics for the 36-split survey.

Dependence the analysis respects
  * the two directions of a grouping (A->B, B->A) come from the same entities,
    and the symmetric screens take the SAME value on both; they are one unit;
  * different groupings within a panel share entities.

Uncertainty: ENTITY-LEVEL JOINT BOOTSTRAP within each panel. One multinomial
resample of the panel's entities is drawn per replicate and applied to every
split and seed at once (errors of test entities, errors of source-test
entities, and group-mean screen features), so dependence through shared
entities is carried into every interval. Models are not retrained; intervals
reflect evaluation-sample variability given the trained models, while seed
variability is already averaged in each split value.

Association tests: PAIR-PRESERVING PERMUTATION. The 16 behavioral groupings
(4 per panel) are the exchangeable units; a replicate permutes whole groupings
(keeping each pair's two headroom values together, randomly ordered within the
pair) either across all panels (pooled) or within panel (stratified).

Outputs: survey_split_table.csv (36 rows, both endpoints), survey_stats.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PANELS = ["Nursing", "Traffic", "Solar", "Electricity"]
SCREENS = ["js_amp", "js_energy", "acf_l2"]
MET = ["mae_cat", "mae_total"]
NB, NPERM = 1000, 20000


def js_rows(p, q, eps=1e-9):
    p = p / (p.sum(-1, keepdims=True) + eps); q = q / (q.sum(-1, keepdims=True) + eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log((a + eps) / (b + eps)), -1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load():
    S = pd.read_csv(os.path.join(RES, "survey_splits.csv"))
    E = pd.read_csv(os.path.join(RES, "survey_entities.csv"))
    Q = pd.read_csv(os.path.join(RES, "survey_source_entities.csv"))
    F = {p: np.load(os.path.join(RES, f"survey_features_{p}.npz"), allow_pickle=True)
         for p in PANELS}
    return S, E, Q, F


def split_table(S):
    g = S.groupby(["panel", "split"])
    cols = {c: "mean" for c in S.columns if c.startswith(("headroom_", "apparent_",
                                                          "snaive_gain", "blend_gain"))}
    cols.update({s: "first" for s in SCREENS})
    cols.update(dict(n_A="first", n_B="first", n_src_train="first",
                     n_tgt_test="first", seed="count"))
    t = g.agg(cols).rename(columns={"seed": "n_seeds"}).reset_index()
    t["behavioral"] = t.split != "random control"
    t["grouping"] = t.split.str.split(":").str[0]
    return t


def boot_panel(p, S, E, Q, F, rng):
    """NB bootstrap replicates of split-level (headroom, apparent, screens)."""
    f = F[p]
    names = list(f["split_names"])
    n = f["acf"].shape[0]
    sp = S[S.panel == p].reset_index(drop=True)
    out = {k: np.zeros((NB, len(names))) for k in
           [f"h_{m}" for m in MET] + [f"a_{m}" for m in MET] + SCREENS}
    # dense per-row matrices over entities
    rows = []
    for _, r in sp.iterrows():
        e = E[(E.panel == p) & (E.split == r.split) & (E.seed == r.seed)]
        q = Q[(Q.panel == p) & (Q.split == r.split) & (Q.seed == r.seed)]
        rows.append((names.index(r.split), e, q))
    W = np.array([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(NB)],
                 float)                                             # [NB, n]
    for key in SCREENS:
        pass
    for j, nm in enumerate(names):
        A, B = f[f"A_{j}"], f[f"B_{j}"]
        wa, wb = W[:, A], W[:, B]
        ma = lambda X: (wa @ X[A]) / (wa.sum(1, keepdims=True) + 1e-12)
        mb = lambda X: (wb @ X[B]) / (wb.sum(1, keepdims=True) + 1e-12)
        out["js_amp"][:, j] = js_rows(ma(f["amp4"]), mb(f["amp4"]))
        out["js_energy"][:, j] = js_rows(ma(f["eng4"]), mb(f["eng4"]))
        out["acf_l2"][:, j] = np.linalg.norm(ma(f["acf"]) - mb(f["acf"]), axis=1)
    acc = {k: np.zeros((NB, len(names))) for k in out if k.startswith(("h_", "a_"))}
    cnt = np.zeros(len(names))
    for j, e, q in rows:
        cnt[j] += 1
        we, wq = W[:, e.fac.to_numpy()], W[:, q.fac.to_numpy()]
        for m in MET:
            tr = we @ e[f"transfer_{m}"].to_numpy()
            rf = we @ e[f"reference_{m}"].to_numpy()
            so = (wq @ q[f"source_{m}"].to_numpy()) / (wq.sum(1) + 1e-12)
            trm = tr / (we.sum(1) + 1e-12)
            acc[f"h_{m}"][:, j] += 100 * (tr - rf) / tr
            acc[f"a_{m}"][:, j] += 100 * (trm - so) / so
    for k in acc:
        out[k] = acc[k] / cnt
    return names, out


def pair_perm(t, xcol, ycol, stratified, rng, nperm=NPERM):
    """Pooled Spearman over the behavioral splits; null permutes whole groupings."""
    b = t[t.behavioral].copy()
    units = b.groupby(["panel", "grouping"])
    keys = list(units.groups)
    x = np.concatenate([units.get_group(k)[xcol].to_numpy() for k in keys])
    Y = [units.get_group(k)[ycol].to_numpy() for k in keys]
    obs = spearmanr(x, np.concatenate(Y))[0]
    pan = np.array([k[0] for k in keys])
    hits = 0
    for _ in range(nperm):
        if stratified:
            order = np.arange(len(keys))
            for p in np.unique(pan):
                ix = np.flatnonzero(pan == p)
                order[ix] = rng.permutation(ix)
        else:
            order = rng.permutation(len(keys))
        y = np.concatenate([Y[o][rng.permutation(len(Y[o]))] for o in order])
        hits += abs(spearmanr(x, y)[0]) >= abs(obs) - 1e-12
    return float(obs), float((hits + 1) / (nperm + 1))


def ci(v):
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    S, E, Q, F = load()
    t = split_table(S)
    t.to_csv(os.path.join(RES, "survey_split_table.csv"), index=False)
    rng = np.random.default_rng(0)
    out = {}
    boots = {}
    for p in PANELS:
        names, bo = boot_panel(p, S, E, Q, F, rng)
        boots[p] = (names, bo)
    for m in MET:
        hc = f"headroom_{m}"; ac = f"apparent_gap_{m}"
        for p in PANELS:
            tp = t[(t.panel == p)]
            beh = tp[tp.behavioral]
            names, bo = boots[p]
            bi = [names.index(s) for s in beh.split]
            ri = names.index("random control")
            out[f"{p}_{m}"] = dict(
                mean=float(beh[hc].mean()), min=float(beh[hc].min()),
                max=float(beh[hc].max()),
                mean_ci=ci(bo[f"h_{m}"][:, bi].mean(1)),
                random_control=float(tp[~tp.behavioral][hc].iloc[0]),
                random_control_ci=ci(bo[f"h_{m}"][:, ri]),
                apparent_min=float(beh[ac].min()), apparent_max=float(beh[ac].max()),
                n_behavioral=int(len(beh)))
        # pooled associations with joint-bootstrap CIs
        beh = t[t.behavioral]
        def pooled(xc, xb):
            reps = []
            for k in range(NB):
                xs, ys = [], []
                for p in PANELS:
                    names, bo = boots[p]
                    bi = [names.index(s) for s in beh[beh.panel == p].split]
                    xs.append(bo[xb][k, bi]); ys.append(bo[f"h_{m}"][k, bi])
                reps.append(spearmanr(np.concatenate(xs), np.concatenate(ys))[0])
            return ci(np.array(reps))
        assoc = {}
        for name, col, bkey in [("apparent_gap", ac, f"a_{m}")] + \
                               [(s, s, s) for s in SCREENS]:
            r_pool, p_pool = pair_perm(t, col, hc, False, rng)
            _, p_strat = pair_perm(t, col, hc, True, rng)
            per_panel = {p: float(spearmanr(beh[beh.panel == p][col],
                                            beh[beh.panel == p][hc])[0]) for p in PANELS}
            assoc[name] = dict(spearman=r_pool, ci=pooled(col, bkey),
                               p_pair_perm_pooled=p_pool,
                               p_pair_perm_within_panel=p_strat,
                               per_panel=per_panel)
        # grouping-level (n=16) versions and panel difference test
        gl = beh.groupby(["panel", "grouping"]).agg(h=(hc, "mean"),
                                                    **{s: (s, "first") for s in SCREENS}).reset_index()
        assoc["grouping_level_n"] = int(len(gl))
        for s in SCREENS:
            assoc[f"grouping_level_spearman_{s}"] = float(spearmanr(gl[s], gl.h)[0])
        kw = kruskal(*[g.h.values for _, g in gl.groupby("panel")])
        out[f"assoc_{m}"] = assoc
        out[f"kruskal_grouping_level_{m}"] = dict(H=float(kw.statistic),
                                                   p=float(kw.pvalue), n_units=int(len(gl)))
        # symmetric-screen check: both directions share the screen value
        sym = beh.groupby(["panel", "grouping"])[SCREENS].nunique().max().max()
        out["screens_identical_across_directions"] = bool(sym == 1)
        # direction asymmetry of headroom: |h(A->B) - h(B->A)| per grouping
        asym = beh.groupby(["panel", "grouping"])[hc].agg(lambda v: float(np.ptp(v)))
        out[f"direction_asymmetry_{m}"] = dict(mean=float(asym.mean()),
                                               max=float(asym.max()))
    with open(os.path.join(RES, "survey_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if "mae_cat" in k}, indent=1)[:6000])


if __name__ == "__main__":
    main()
