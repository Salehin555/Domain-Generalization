"""Classical references and forecast combinations.

Aim: contextualize the seasonal-blend finding with classical
seasonal/trend baselines and simple forecast combinations.

Every row is scored on the SAME target test entities and windows as the
case-study table, because the entity allocation (seed, floors, fractions) is
exactly the one anchor.py uses. Rows are written as anchor_rows_classical.csv
and anchor_entities_classical.csv, so an_anchor.py computes their error
reductions and paired entity-bootstrap intervals together with every other
method.

Local classical models are fitted on each 56-day input window alone, per
channel, and forecast its 14-day horizon. They use no training entities and no
target labels, so the source/target distinction does not arise for them.

    Persistence                       last observed value repeated
    Holt-Winters                      additive weekly season, no trend (ETS(A,N,A))
    STL + ETS                         STL with period 7; damped additive-trend ETS
                                      on the seasonally adjusted series
    Theta                             weekly deseasonalized Theta method
    SARIMA(1,0,0)(0,1,1)_7            one fixed weekly order for every series; no
                                      automatic order selection was available

Combinations: convex weights over {source-trained ERM-Stem, seasonal naive,
persistence} and over the same set plus Holt-Winters, chosen on a 0.1 simplex
grid to minimize the primary metric on SOURCE-VALIDATION entities only, exactly
as the single-weight blend is chosen.

A fit that raises or returns non-finite values falls back to seasonal naive for
that series; every fallback is counted in classical_fallbacks.csv.

Usage: python classical.py [panel ...]   (default: Electricity Traffic Solar)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H, PERIOD = 14, 7
CLASSICAL = ["Persistence (last value)",
             "Holt-Winters, additive weekly season",
             "STL + ETS, damped trend",
             "Theta, weekly deseasonalized",
             "SARIMA(1,0,0)(0,1,1)7, fixed order"]
COMBOS = {
    "Combination: source model, seasonal naive, persistence":
        ["model", "snaive", "Persistence (last value)"],
    "Combination: source model, seasonal naive, persistence, Holt-Winters":
        ["model", "snaive", "Persistence (last value)", "Holt-Winters, additive weekly season"],
}
WORKERS = int(os.environ.get("EXP_WORKERS", "6"))


# ------------------------------------------------------------------ forecasters
def _snaive(y):
    return np.array([y[len(y) - PERIOD + (h % PERIOD)] for h in range(H)], float)


def _one(name, y):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.forecasting.stl import STLForecast
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.forecasting.theta import ThetaModel
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if name == "Persistence (last value)":
        return np.repeat(y[-1], H).astype(float)
    if np.ptp(y) < 1e-9:                        # constant series: every model is flat
        return np.repeat(y[-1], H).astype(float)
    if name == "Holt-Winters, additive weekly season":
        return ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=PERIOD,
                                    initialization_method="estimated").fit().forecast(H)
    if name == "STL + ETS, damped trend":
        return STLForecast(y, ETSModel, period=PERIOD,
                           model_kwargs=dict(error="add", trend="add",
                                             damped_trend=True)).fit().forecast(H)
    if name == "Theta, weekly deseasonalized":
        return ThetaModel(y, period=PERIOD).fit().forecast(H)
    if name == "SARIMA(1,0,0)(0,1,1)7, fixed order":
        return SARIMAX(y, order=(1, 0, 0), seasonal_order=(0, 1, 1, PERIOD)).fit(
            disp=False).forecast(H)
    raise ValueError(name)


def entity_forecasts(args):
    """All classical forecasts for every window and channel of one entity."""
    ent, X = args                                # X: [n_windows, C, L]
    warnings.filterwarnings("ignore")
    out = {m: np.zeros((X.shape[0], X.shape[1], H), np.float32) for m in CLASSICAL}
    fb = {m: 0 for m in CLASSICAL}
    for w in range(X.shape[0]):
        for c in range(X.shape[1]):
            y = X[w, c].astype(float)
            for m in CLASSICAL:
                try:
                    f = np.asarray(_one(m, y), float)
                    if f.shape != (H,) or not np.all(np.isfinite(f)):
                        raise ValueError("non-finite")
                except Exception:
                    f = _snaive(y); fb[m] += 1
                out[m][w, c] = f
    return ent, out, fb, X.shape[0] * X.shape[1]


# ------------------------------------------------------------------ helpers
def simplex(k, step=0.1):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)], float) / n


def pooled_mae(pred, Y):
    return float(np.abs(pred - Y).mean())


def gather(cache, facs, name):
    return np.concatenate([cache[e][name] for e in facs])


def main():
    from common import (METRICS, RES, allocate, fit, per_entity, predict,
                          seasonal_naive)
    from anchor import anchors
    want = sys.argv[1:] or ["Electricity", "Traffic", "Solar"]
    erm_cfg = dict(method="erm", epochs=24, patience=5, backbone="mlp")
    ref_rows = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in
                          ("anchor_rows_Electricity.csv", "anchor_rows_Solar_Traffic.csv")])
    rows, ents, fbs, checks = [], [], [], []
    tag = "_".join(want)
    for panel, P, split, S, T, floors, fracs, seeds in anchors(want):
        if panel == "Nursing":
            continue
        t0 = time.time()
        allocs = {s: allocate(S, T, np.random.default_rng(s), floors=floors, fracs=fracs,
                              fallback=(panel == "Solar")) for s in seeds}
        need = sorted({int(e) for a in allocs.values() for e in np.r_[a["t_te"], a["s_va"]]})
        cpath = os.path.join(RES, f"classical_cache_{panel}.pkl")
        cache = pickle.load(open(cpath, "rb")) if os.path.exists(cpath) else {}
        todo = [e for e in need if e not in cache]
        if todo:
            jobs = [(e, P.eval_arrays([e])[0]) for e in todo]
            n_series = 0; fb_tot = {m: 0 for m in CLASSICAL}
            with Pool(WORKERS) as pool:
                for k, (e, out, fb, ns) in enumerate(pool.imap_unordered(entity_forecasts, jobs)):
                    cache[e] = out; n_series += ns
                    for m in CLASSICAL:
                        fb_tot[m] += fb[m]
                    if (k + 1) % 25 == 0:
                        print(f"  {panel}: {k + 1}/{len(jobs)} entities", flush=True)
            pickle.dump(cache, open(cpath, "wb"))
            for m in CLASSICAL:
                fbs.append(dict(panel=panel, method=m, series=n_series, fallbacks=fb_tot[m]))
        print(f"  {panel}: classical forecasts ready ({time.time() - t0:.0f}s)", flush=True)

        for seed in seeds:
            al = allocs[seed]
            m = fit(P, erm_cfg, al["s_tr"], al["s_va"], seed)
            Xt, Yt, Pt, fid = predict(m, P, al["t_te"])
            Xv, Yv, Pv, _ = predict(m, P, al["s_va"])
            # determinism check against the case-study run on the same manifest
            e0 = per_entity(Pt, Yt, fid)["mae_cat"].mean()
            ref = ref_rows[(ref_rows.panel == panel) & (ref_rows.split == split) &
                           (ref_rows.seed == seed) & (ref_rows.manifest == "A") &
                           (ref_rows.method == "ERM-MLP")].mae_cat
            checks.append(dict(panel=panel, split=split, seed=seed, retrained=e0,
                               case_study=float(ref.iloc[0]) if len(ref) else np.nan))

            def score(label, pred, n_train, extra=None):
                e = per_entity(pred, Yt, fid)
                r = dict(panel=panel, split=split, seed=seed, manifest="A", method=label,
                         n_train=n_train, n_test=len(al["t_te"]), **(extra or {}))
                for mt in METRICS:
                    r[mt] = e[mt].mean()
                rows.append(r)
                ents.append(e.reset_index().assign(panel=panel, split=split, seed=seed,
                                                   manifest="A", method=label))

            comp_t = {"model": Pt, "snaive": seasonal_naive(Xt)}
            comp_v = {"model": Pv, "snaive": seasonal_naive(Xv)}
            for name in CLASSICAL:
                comp_t[name] = gather(cache, al["t_te"], name)
                comp_v[name] = gather(cache, al["s_va"], name)
                score(name, comp_t[name], 0)
            for label, parts in COMBOS.items():
                best, bw = np.inf, None
                for w in simplex(len(parts)):
                    e = pooled_mae(sum(wi * comp_v[p] for wi, p in zip(w, parts)), Yv)
                    if e < best - 1e-12:
                        best, bw = e, w
                pred = sum(wi * comp_t[p] for wi, p in zip(bw, parts))
                score(label, pred, len(al["s_tr"]),
                      {f"w_{p.split(' ')[0].split(',')[0].lower()}": float(wi)
                       for wi, p in zip(bw, parts)})
            print(f"  {panel} {split} seed {seed} scored", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"anchor_rows_classical_{tag}.csv"),
                                  index=False)
        pd.concat(ents).to_csv(os.path.join(RES, f"anchor_entities_classical_{tag}.csv"),
                               index=False)
    if fbs:
        pd.DataFrame(fbs).to_csv(os.path.join(RES, f"classical_fallbacks_{tag}.csv"),
                                 index=False)
    ck = pd.DataFrame(checks)
    ck.to_csv(os.path.join(RES, f"classical_determinism_{tag}.csv"), index=False)
    print(ck.assign(diff=(ck.retrained - ck.case_study).abs()).to_string())
    print("done", flush=True)


if __name__ == "__main__":
    main()
