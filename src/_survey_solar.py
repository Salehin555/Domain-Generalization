"""
Reducibility survey: run the diagnostic over MANY leave-group-out splits, under one
uniform split-generation protocol, on every panel available.

WHY
---
Two splits on two datasets is an anecdote. To claim anything general about when
cross-entity shift is reducible, and to test whether reducibility can be PREDICTED
without paying for the oracle, we need many splits with known ground truth. This
script generates them systematically.

UNIFORM SPLIT PROTOCOL
----------------------
For each panel and each behavioral statistic s (computed on the TRAINING period
only, so no deployment information leaks into the domain definition):
    group A = bottom quartile of s,  group B = top quartile of s
and we run the diagnostic in BOTH directions. Statistics used:
    size    : mean level of the series          (analogue of facility bed count)
    vol     : coefficient of variation          (intrinsic difficulty)
    weekly  : relative weekly spectral amplitude(the frequency-domain property the gate acts on)
    trend   : share of energy in the slow bands (spectral shape)
Plus a RANDOM split as a negative control, where the reducible gap must be ~0 by
construction; if it is not, the procedure itself is biased.

Training sets are SIZE-MATCHED between transfer and oracle throughout, which
matters whenever the learning curve is still steep (see oracle_electricity.py).

CHEAP PREDICTORS
----------------
For every split we also record statistics that cost no model training, to test
whether the reducible gap can be predicted without running the oracle at all:
    spec_div    : Jensen-Shannon divergence between the groups' mean band-energy profiles
    diff_ratio  : ratio of intrinsic difficulty (target / source)
    sep         : standardized separation of the grouping statistic between groups
    n_ratio     : group size ratio
Whether any of these predicts the measured gap is an empirical question this
script is designed to answer honestly, including in the negative.

Output: outputs/reducibility_survey_solar.csv
"""
import os
import numpy as np
import pandas as pd

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")
SEEDS = [0, 1, 2]

PANELS = [
    dict(name="Nursing", path=os.path.join(ROOT, "cache", "panel.npz"),
         nfac=1500, boundary=20250101, C=3),
    dict(name="Electricity", path=os.path.join(ROOT, "cache", "electricity",
                                               "panel_elec.npz"),
         nfac=10 ** 6, boundary=20140101, C=3),
    dict(name="Traffic", path=os.path.join(ROOT, "cache", "panels",
                                           "panel_traffic.npz"),
         nfac=10 ** 6, boundary=20160504, C=3),
    dict(name="Solar", path=os.path.join(ROOT, "cache", "panels",
                                         "panel_solar.npz"),
         nfac=10 ** 6, boundary=20060902, C=3),
]


# ----------------------------------------------------------------- statistics
def entity_stats(P):
    """Cheap per-entity descriptors from the windows themselves (training era only)."""
    X = P.Xw[P.is_2024] if P.is_2024.any() else P.Xw          # [N,C,L]
    f = P.fid[P.is_2024] if P.is_2024.any() else P.fid
    n = P.nF
    tot = X.sum(1)                                            # [N,L] total across channels
    lev = np.zeros(n); vol = np.zeros(n); wk = np.zeros(n); tr = np.zeros(n)
    prof = np.zeros((n, 4))
    freq = np.fft.rfftfreq(tot.shape[1], 1.0)
    k7 = int(np.argmin(abs(freq - 1 / 7.0)))
    edges = np.linspace(0, len(freq), 5).astype(int)
    for i in range(n):
        m = f == i
        if not m.any():
            continue
        t = tot[m]
        lev[i] = t.mean()
        vol[i] = t.std(1).mean() / (abs(t.mean()) + 1e-8)
        c = t - t.mean(1, keepdims=True)
        A = np.abs(np.fft.rfft(c, axis=1))
        wk[i] = A[:, k7].mean() / (c.std(1).mean() * t.shape[1] + 1e-8)
        band = np.array([A[:, edges[b]:edges[b + 1]].sum() for b in range(4)])
        band = band / (band.sum() + 1e-8)
        prof[i] = band
        tr[i] = band[0]
    return dict(size=lev, vol=vol, weekly=wk, trend=tr), prof


def js_div(p, q, eps=1e-9):
    p = p / (p.sum() + eps); q = q / (q.sum() + eps); m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log((a + eps) / (b + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ----------------------------------------------------------------- diagnostic
def run(P, tr, va, te):
    m, _ = train_model(P, ERM, tr, va, seed=0)
    return float(np.mean([x["mae_total"] for x in evaluate(m, P, te)]))


def diagnose(P, src_all, tgt_all, seed):
    rng = np.random.default_rng(seed)
    n_test = max(20, int(0.35 * len(tgt_all)))
    n_val = max(8, int(0.15 * len(tgt_all)))
    n_train = min(len(src_all) - n_val, len(tgt_all) - n_test - n_val)
    if n_train < 15:
        # Small-panel fallback (solar has only 35 entities per quartile group).
        # The absolute floors above would starve training there, so fall back to
        # proportional sizes. Larger panels never reach this branch, so their
        # numbers are unchanged.
        n_test = max(10, int(0.30 * len(tgt_all)))
        n_val = max(4, int(0.12 * len(tgt_all)))
        n_train = min(len(src_all) - n_val, len(tgt_all) - n_test - n_val)
        if n_train < 12:
            return None
    s = src_all.copy(); rng.shuffle(s)
    s_va, s_tr = s[:n_val], s[n_val:n_val + n_train]
    t = tgt_all.copy(); rng.shuffle(t)
    t_te, pool = t[:n_test], t[n_test:]
    t_va, t_tr = pool[:n_val], pool[n_val:n_val + n_train]
    src = run(P, s_tr, s_va, s_va)
    tra = run(P, s_tr, s_va, t_te)
    ora = run(P, t_tr, t_va, t_te)
    return dict(source_err=src, transfer=tra, oracle=ora,
                apparent_gap_pct=100 * (tra - src) / src,
                reducible_gap_pct=100 * (tra - ora) / tra)


def main():
    rows = []
    for spec in [s for s in PANELS if s["name"]=="Solar"]:
        P = Panel(spec["path"], n_facilities=spec["nfac"], L=56, H=14, stride=18,
                  seed=0, n_channels=spec["C"], boundary_date=spec["boundary"])
        print(f"\n### {spec['name']}: nF={P.nF} windows={len(P.Xw)}", flush=True)
        stats, prof = entity_stats(P)
        idx = np.arange(P.nF)

        splits = {}
        for sname, v in stats.items():
            q1, q3 = np.percentile(v, [25, 75])
            A, B = idx[v <= q1], idx[v >= q3]
            if len(A) >= 30 and len(B) >= 30:
                splits[f"{sname}: low->high"] = (A, B)
                splits[f"{sname}: high->low"] = (B, A)
        rng = np.random.default_rng(0)
        p = rng.permutation(idx)
        h = len(p) // 2
        splits["random control"] = (p[:h], p[h:])

        for split_name, (A, B) in splits.items():
            # cheap, training-free predictors
            pa, pb = prof[A].mean(0), prof[B].mean(0)
            sd = js_div(pa, pb)
            dr = float(np.mean(stats["vol"][B]) / (np.mean(stats["vol"][A]) + 1e-8))
            key = split_name.split(":")[0]
            vv = stats.get(key, stats["vol"])
            sep = float(abs(vv[B].mean() - vv[A].mean()) /
                        (np.std(np.concatenate([vv[A], vv[B]])) + 1e-8))
            nr = len(B) / max(len(A), 1)

            for seed in SEEDS:
                d = diagnose(P, A, B, seed)
                if d is None:
                    continue
                rows.append(dict(panel=spec["name"], split=split_name, seed=seed,
                                 n_src=len(A), n_tgt=len(B), spec_div=sd,
                                 diff_ratio=dr, sep=sep, n_ratio=nr, **d))
            done = [r for r in rows if r["split"] == split_name
                    and r["panel"] == spec["name"]]
            if done:
                print(f"  {split_name:22s} apparent={np.mean([r['apparent_gap_pct'] for r in done]):+6.1f}%  "
                      f"reducible={np.mean([r['reducible_gap_pct'] for r in done]):+6.2f}%", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "reducibility_survey_solar.csv"), index=False)
    print("\n=== survey summary (mean over seeds) ===")
    g = df.groupby(["panel", "split"])[["apparent_gap_pct", "reducible_gap_pct"]].mean()
    print(g.round(2).to_string())
    print("\nSaved outputs/reducibility_survey_solar.csv")


if __name__ == "__main__":
    main()
