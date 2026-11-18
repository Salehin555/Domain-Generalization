"""Screen validation and decision-policy analysis.

Split collections (see screen.py): DISCOVERY (split rng 0) and CONFIRMATION
(split rng 1), 26 candidates per panel each; headroom per split = mean over two
allocation seeds.

1. DISCOVERY. The statistic is chosen from the eight pre-registered candidates
   by the largest minimum-over-panels leave-one-out
   Pearson r). Selection-adjusted significance: max-statistic permutation test in
   which EACH REPLICATE PERMUTES EVERY PANEL'S HEADROOM VECTOR ONCE and scores
   ALL candidates against that same permuted vector (a separate permutation per
   candidate would break their dependence).
2. FROZEN CONFIRMATION. On discovery data only, fit per panel a line
   headroom = a + b * statistic. On the confirmation splits nothing is refitted:
   report (i) direct Spearman rank correlation of the statistic with headroom,
   with an entity-level bootstrap CI; (ii) the absolute error of the FROZEN line
   against the frozen discovery-mean baseline; (iii) agreement with headroom at
   the prespecified 3% analysis threshold against the majority-class rate.
   Leave-one-out fits on the confirmation data are reported separately and
   labelled as within-sample cross-validated fits.
3. LEAVE-ONE-PANEL-OUT. Statistic and threshold quantile chosen on the other
   three panels' DISCOVERY splits; evaluated on the held-out panel's
   CONFIRMATION splits.
4. DECISION POLICIES on the confirmation splits, budget = number of splits that
   can be investigated (full diagnostic, two model fits each):
      screen      investigate the top-k splits by the frozen statistic
      random      expected value of investigating k splits at random
      apparent    top-k by apparent gap (a common heuristic)
      hindsight   top-k by realised headroom (unattainable reference)
   Action after investigating: retrain on target-group data if measured
   headroom >= 3% (value = headroom). Separately, the label-free action "apply
   the source-validated seasonal blend" is evaluated everywhere, on screened
   splits only, and on diagnosed splits only. Captured value, missed
   opportunities (splits >= 3% not investigated) and fits saved are reported.

Outputs: screen_stats.json, screen_confirmation_table.csv, screen_lopo.csv,
         policy_table.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PANELS = ["Nursing", "Electricity", "Solar", "Traffic"]
CANDS = ["js_mean", "js_fine", "js_perchan", "mmd_profile", "w1_profile", "acf_l2",
         "acf_weighted", "diff_ratio"]
THR = 3.0
NPERM, NB = 2000, 1000


def loo_predict(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    out = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        b, a = np.polyfit(x[m], y[m], 1)
        out[i] = b * x[i] + a
    return out


def loo_r(x, y):
    return ss.pearsonr(loo_predict(x, y), y)[0] if len(y) >= 6 else np.nan


def split_means(df, m="mae_cat"):
    keep = CANDS + ["js_amp", "js_energy", "n_A", "n_B"]
    g = df.groupby(["panel", "split"])
    t = g[keep].first()
    for c in [f"headroom_{m}", f"apparent_gap_{m}", f"blend_gain_{m}",
              f"snaive_gain_{m}"]:
        t[c.replace(f"_{m}", "")] = g[c].mean()
    return t.reset_index()


def boot_spearman(panel, sset, stat, t, rng):
    """Entity-level bootstrap CI for Spearman(stat, headroom) on one split set."""
    f = np.load(os.path.join(RES, f"screen_features_{panel}.npz"))
    E = pd.read_csv(os.path.join(RES, f"screen_{sset}_entities.csv"))
    E = E[E.panel == panel]
    n = f["acf"].shape[0]
    splits = t.split.to_numpy()
    reps = []
    for _ in range(NB):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        xs, ys = [], []
        for k in splits:
            A, B = f[f"{sset}_A_{k}"], f[f"{sset}_B_{k}"]
            if stat == "acf_l2":
                ma = (w[A] @ f["acf"][A]) / (w[A].sum() + 1e-12)
                mb = (w[B] @ f["acf"][B]) / (w[B].sum() + 1e-12)
                xs.append(np.linalg.norm(ma - mb))
            else:
                xs.append(t.loc[t.split == k, stat].iloc[0])
            hs = []
            for _, e in E[E.split == k].groupby("seed"):
                we = w[e.fac.to_numpy()]
                tr = we @ e.transfer_mae_cat.to_numpy(); rf = we @ e.reference_mae_cat.to_numpy()
                hs.append(100 * (tr - rf) / tr)
            ys.append(np.mean(hs))
        reps.append(ss.spearmanr(xs, ys)[0])
    reps = np.array(reps)
    return [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]


def main():
    D = split_means(pd.read_csv(os.path.join(RES, "screen_discovery_splits.csv")))
    C = split_means(pd.read_csv(os.path.join(RES, "screen_confirmation_splits.csv")))
    out = dict(n_discovery={p: int((D.panel == p).sum()) for p in PANELS},
               n_confirmation={p: int((C.panel == p).sum()) for p in PANELS})

    # ------------------------------------------------------------ 1. discovery
    crit = {c: min(loo_r(D[D.panel == p][c], D[D.panel == p].headroom)
                   for p in PANELS) for c in CANDS}
    best = max(crit, key=crit.get)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(NPERM):
        perm = {p: rng.permutation(D[D.panel == p].headroom.to_numpy()) for p in PANELS}
        null.append(max(min(loo_r(D[D.panel == p][c].to_numpy(), perm[p])
                            for p in PANELS) for c in CANDS))
    null = np.array(null)
    out["discovery"] = dict(criterion=crit, selected=best, bar=float(crit[best]),
                            p_selection_adjusted=float(((null >= crit[best]).sum() + 1)
                                                       / (NPERM + 1)),
                            null_mean=float(null.mean()),
                            null_95=float(np.percentile(null, 95)),
                            spearman={p: {c: float(ss.spearmanr(D[D.panel == p][c],
                                                                 D[D.panel == p].headroom)[0])
                                          for c in CANDS + ["js_amp", "js_energy"]}
                                      for p in PANELS})

    # --------------------------------------------------- 2. frozen confirmation
    rows = []
    for p in PANELS:
        d, c = D[D.panel == p], C[C.panel == p]
        for stat in sorted({best, "acf_l2", "js_amp"}):
            b, a = np.polyfit(d[stat], d.headroom, 1)
            pred = a + b * c[stat]
            base = np.full(len(c), d.headroom.mean())
            y = c.headroom.to_numpy()
            maj = max((y >= THR).mean(), 1 - (y >= THR).mean())
            rho, prho = ss.spearmanr(c[stat], y)
            rows.append(dict(
                panel=p, statistic=stat, n=len(c), spearman=rho, spearman_p=prho,
                spearman_ci=boot_spearman(p, "confirmation", stat, c, rng)
                if stat in ("acf_l2", "js_amp") else None,
                frozen_mae=float(np.abs(pred - y).mean()),
                baseline_mae=float(np.abs(base - y).mean()),
                frozen_agree=float(((pred >= THR) == (y >= THR)).mean()),
                majority=float(maj),
                loo_r_within_confirmation=float(loo_r(c[stat], y)),
                discovery_slope=float(b)))
    CT = pd.DataFrame(rows)
    CT.to_csv(os.path.join(RES, "screen_confirmation_table.csv"), index=False)
    out["confirmation"] = CT.to_dict("records")

    # ------------------------------------------------ 3. leave-one-panel-out
    lo = []
    for held in PANELS:
        tr = D[D.panel != held]; te = C[C.panel == held]
        sc = {s: np.mean([ss.spearmanr(g[s], g.headroom)[0]
                          for _, g in tr.groupby("panel")]) for s in CANDS}
        sel = max(sc, key=sc.get)
        best_q, best_acc = None, -1
        for q in np.arange(0.10, 0.91, 0.05):
            acc = np.mean([((g[sel] >= np.quantile(g[sel], q)) ==
                            (g.headroom >= THR)).mean() for _, g in tr.groupby("panel")])
            if acc > best_acc:
                best_acc, best_q = acc, q
        pred = te[sel] >= np.quantile(te[sel], best_q)
        truth = te.headroom >= THR
        lo.append(dict(held_out=held, selected=sel, rho=float(ss.spearmanr(te[sel], te.headroom)[0]),
                       rho_acf_l2=float(ss.spearmanr(te.acf_l2, te.headroom)[0]),
                       quantile=float(best_q), threshold_acc=float((pred == truth).mean()),
                       majority=float(max(truth.mean(), 1 - truth.mean())),
                       frac_above=float(truth.mean()), n=len(te)))
    LO = pd.DataFrame(lo)
    LO.to_csv(os.path.join(RES, "screen_lopo.csv"), index=False)
    out["lopo"] = LO.to_dict("records")

    # ------------------------------------------------------ 4. decision policy
    pol = []
    for p in PANELS:
        c = C[C.panel == p].reset_index(drop=True)
        n = len(c); h = c.headroom.to_numpy(); bl = c.blend_gain.to_numpy()
        opp = h >= THR
        potential = h[opp].sum()
        for frac in (0.25, 0.50):
            k = max(1, int(round(frac * n)))
            orders = dict(screen=np.argsort(-c[best].to_numpy()),
                          apparent=np.argsort(-c.apparent_gap.to_numpy()),
                          hindsight=np.argsort(-h))
            for name, o in orders.items():
                sel = np.zeros(n, bool); sel[o[:k]] = True
                pol.append(dict(panel=p, budget_frac=frac, k=k, n=n, policy=name,
                                captured_headroom=float(h[sel & opp].sum()),
                                potential_headroom=float(potential),
                                captured_frac=float(h[sel & opp].sum() / potential)
                                if potential > 0 else np.nan,
                                missed_opportunities=int((opp & ~sel).sum()),
                                opportunities=int(opp.sum()), fits_saved=2 * (n - k),
                                blend_gain_if_blend_on_selected=float(bl[sel].mean()),
                                blend_gain_if_blend_on_diagnosed=float(
                                    np.where(sel & opp, bl, 0).mean())))
            pol.append(dict(panel=p, budget_frac=frac, k=k, n=n, policy="random",
                            captured_headroom=float(potential * k / n),
                            potential_headroom=float(potential),
                            captured_frac=float(k / n) if potential > 0 else np.nan,
                            missed_opportunities=float(opp.sum() * (1 - k / n)),
                            opportunities=int(opp.sum()), fits_saved=2 * (n - k),
                            blend_gain_if_blend_on_selected=float(bl.mean()),
                            blend_gain_if_blend_on_diagnosed=float(
                                np.where(opp, bl, 0).mean() * k / n)))
        pol.append(dict(panel=p, budget_frac=1.0, k=n, n=n, policy="blend everywhere, no diagnostic",
                        blend_gain_if_blend_on_selected=float(bl.mean()),
                        blend_harm_splits=int((bl < 0).sum()), fits_saved=2 * n))
    PT = pd.DataFrame(pol)
    PT.to_csv(os.path.join(RES, "policy_table.csv"), index=False)
    out["policy_note"] = "captured/potential headroom summed over confirmation splits"
    with open(os.path.join(RES, "screen_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out["discovery"], indent=1, default=float)[:1500])
    print(CT.to_string()); print(LO.to_string())
    print(PT.round(3).to_string())


if __name__ == "__main__":
    main()
