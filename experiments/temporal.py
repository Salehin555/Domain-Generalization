"""PROSPECTIVE TEMPORAL experiments (Table 4, Suppl. F).

A. Observed-history estimate vs deployment headroom. Both models are trained
   and early-stopped on observed-era windows only; headroom is then measured
   (i) on the target test entities' observed-era windows  -> the estimate
   (ii) on the same entities' deployment-era windows       -> the realised value
B. Target-GROUP history curve. The target-trained reference is restricted to
   windows ending by day k, for a pool of target-group entities, and evaluated
   on DIFFERENT target entities in the deployment era. This measures how much
   history a new GROUP needs, not how much one individual facility needs.

Nursing sensitivity variants for A:
   standard     analysis cohort (>=97% coverage over both years)
   clean        same cohort; windows whose forecast horizon contains an imputed
                day, or whose inputs contain a day imputed from an observation
                after the forecast origin, are excluded from training,
                selection and scoring
   prospective  cohort selected on 2024 coverage only; deployment days that are
                missing are never filled across long gaps, and windows touching
                them are excluded

Usage: python experiments/temporal.py [A|B] [panel ...]
Outputs: temporal_A.csv, temporal_B.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ERM, METRICS, RES, fit, headroom, load_panel,    # noqa: E402
                      per_entity, predict)
from src.data import Panel                                  # noqa: E402
from src.label_free_calibration import make_splits          # noqa: E402
from src.survey_reducibility import entity_stats            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFAC = {"Nursing": 1200}
N_SPLITS, SEEDS = 12, [0, 1]


def split_target(tgt_all, rng):
    n_test = max(15, int(0.35 * len(tgt_all)))
    n_val = max(6, int(0.15 * len(tgt_all)))
    t = tgt_all.copy(); rng.shuffle(t)
    return t[n_test + n_val:], t[n_test:n_test + n_val], t[:n_test]


def match_source(src_all, tgt_all, n_train, rng):
    n_val = max(6, int(0.15 * len(tgt_all)))
    s = src_all.copy(); rng.shuffle(s)
    return s[n_val:n_val + n_train], s[:n_val]


def clean_window_mask(P):
    """Nursing only: True for windows with no imputed target day and no input
    day imputed from an observation after the forecast origin."""
    mk = np.load(os.path.join(RES, "nursing_imputation_masks.npz"))
    imp, right = mk["imputed"], mk["right"]
    ok = np.ones(len(P.fid), bool)
    n_tgt = n_post = 0
    for w in range(len(P.fid)):
        f = P.fac[P.fid[w]]; st = int(P.wstart[w]); o = st + P.L - 1
        if not imp[f, st:st + P.L + P.H].any():
            continue
        t_bad = imp[f, st + P.L:st + P.L + P.H].any()
        seg = slice(st, st + P.L)
        p_bad = bool((imp[f, seg] & (right[f, seg] > o)).any())
        n_tgt += t_bad; n_post += p_bad
        ok[w] = not (t_bad or p_bad)
    return ok, dict(windows=len(ok), imputed_target_windows=int(n_tgt),
                    post_origin_input_windows=int(n_post),
                    excluded_windows=int((~ok).sum()))


def valid_window_mask(P):
    return ~(np.isnan(P.Xw).any(axis=(1, 2)) | np.isnan(P.Yw).any(axis=(1, 2)))


def run_A(name, variant="standard"):
    if variant == "prospective":
        P = Panel(os.path.join(ROOT, "cache", "panel_prospective.npz"),
                  n_facilities=NFAC.get(name, 10 ** 6), L=56, H=14, stride=18,
                  seed=0, n_channels=3, boundary_date=20250101)
        ok = valid_window_mask(P); audit = dict(excluded_windows=int((~ok).sum()),
                                                windows=len(ok))
        # entity_stats must not see NaN windows
        P.Xw = np.nan_to_num(P.Xw); P.Yw_nan = ~ok
    else:
        P = load_panel(name, NFAC.get(name, 10 ** 6))
        ok, audit = (clean_window_mask(P) if variant == "clean"
                     else (np.ones(len(P.fid), bool), {}))
    obs, dep = P.is_2024 & ok, P.is_2025 & ok
    stats, _ = entity_stats(P)
    splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
    rows = []
    for k, (A, B) in enumerate(splits):
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            t_tr, t_va, t_te = split_target(B, rng)
            if len(t_tr) < 12:
                continue
            s_tr, s_va = match_source(A, B, len(t_tr), rng)
            if len(s_tr) < 12:
                continue
            m_s = fit(P, ERM, s_tr, s_va, sd, obs, obs)
            m_r = fit(P, ERM, t_tr, t_va, sd, obs, obs)
            r = dict(panel=name, variant=variant, split=k, seed=sd,
                     n_src=len(s_tr), n_tgt_train=len(t_tr), n_tgt_test=len(t_te))
            for era, msk in (("obs", obs), ("dep", dep)):
                _, Y, Ps, fid = predict(m_s, P, t_te, msk)
                _, _, Pr, _ = predict(m_r, P, t_te, msk)
                es, er = per_entity(Ps, Y, fid), per_entity(Pr, Y, fid)
                for mt in METRICS:
                    r[f"transfer_{era}_{mt}"] = es[mt].mean()
                    r[f"reference_{era}_{mt}"] = er[mt].mean()
                    r[f"headroom_{era}_{mt}"] = headroom(es[mt].mean(), er[mt].mean())
            rows.append(r)
            print(f"  [A {variant}] {name} split {k:2d} seed {sd} "
                  f"est={r['headroom_obs_mae_cat']:+6.2f} dep={r['headroom_dep_mae_cat']:+6.2f}",
                  flush=True)
    return rows, dict(panel=name, variant=variant, **audit)


def run_B(name):
    P = load_panel(name, NFAC.get(name, 10 ** 6))
    obs, dep = P.is_2024, P.is_2025
    bidx = int(np.searchsorted(P.dates, {"Nursing": 20250101, "Electricity": 20140101,
                                         "Traffic": 20160504, "Solar": 20060902}[name]))
    ladder = sorted({d for d in [P.span + 7, P.span + 35, P.span + 91, P.span + 175,
                                 P.span + 259, bidx] if d <= bidx})
    stats, _ = entity_stats(P)
    splits = make_splits(stats, N_SPLITS, np.random.default_rng(0))
    rows = []
    for k, (A, B) in enumerate(splits):
        if k % 3:
            continue
        rng = np.random.default_rng(0)
        t_tr, t_va, t_te = split_target(B, rng)
        if len(t_tr) < 12:
            continue
        s_tr, s_va = match_source(A, B, len(t_tr), rng)
        if len(s_tr) < 12:
            continue
        m_s = fit(P, ERM, s_tr, s_va, 0, obs, obs)
        _, Y, Ps, fid = predict(m_s, P, t_te, dep)
        es = per_entity(Ps, Y, fid)
        for d in ladder:
            km = ((P.wstart + P.span) <= d) & obs
            if km.sum() < 4 * len(t_tr):
                continue
            m_r = fit(P, ERM, t_tr, t_va, 0, km, km)
            _, _, Pr, _ = predict(m_r, P, t_te, dep)
            er = per_entity(Pr, Y, fid)
            r = dict(panel=name, split=k, days=d, n_tgt_train=len(t_tr))
            for mt in METRICS:
                r[f"headroom_{mt}"] = headroom(es[mt].mean(), er[mt].mean())
            rows.append(r)
            print(f"  [B] {name} split {k} days={d} h={r['headroom_mae_cat']:+.2f}",
                  flush=True)
    return rows


def main():
    part = sys.argv[1]
    panels = sys.argv[2:] or ["Nursing", "Electricity", "Traffic", "Solar"]
    if part == "A":
        rows, audits = [], []
        jobs = [(p, "standard") for p in panels]
        if "Nursing" in panels:
            jobs += [("Nursing", "clean"), ("Nursing", "prospective")]
        for p, v in jobs:
            t0 = time.time()
            r, a = run_A(p, v)
            rows += r; audits.append(a)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "temporal_A.csv"), index=False)
            pd.DataFrame(audits).to_csv(os.path.join(RES, "temporal_A_window_audit.csv"),
                                        index=False)
            print(f"[{p} {v} done {time.time() - t0:.0f}s]", flush=True)
    else:
        rows = []
        for p in panels:
            rows += run_B(p)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "temporal_B.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
