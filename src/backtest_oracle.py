"""Can the reducibility diagnostic be run when it actually matters?

THE OBJECTION THIS ANSWERS
--------------------------
The reducible gap is (R_transfer - R_oracle) / R_transfer, and R_oracle is the error
of a model TRAINED ON THE TARGET GROUP. That needs target labels. At deployment a
new group has a history but no future, so the objection is that the diagnostic can
never be computed at the moment a practitioner would want it, which would make the
whole contribution retrospective.

THE OBSERVATION THAT DISSOLVES IT
---------------------------------
In forecasting the label IS the series' own future, so labels self-reveal. An entity
with L+H = 70 days of observed history already supplies one genuine (input, label)
pair. Nothing about the oracle requires data from the FUTURE; it requires data that
is LABELED, and the target's own observed past is labeled. So shift the whole
measurement backwards in time and it costs no unavailable data.

This is not a new trick in this paper. The periodicity branch already sets its blend
weight from a label-free in-window backtest. This script applies the same move to the
diagnostic itself.

TWO EXPERIMENTS, ANSWERING TWO DIFFERENT COLD-START CASES
---------------------------------------------------------
A) TEMPORAL VALIDITY. The target entities exist in the pool and are labeled up to
   today, but the deployment period is the future. Measure the gap entirely inside
   the observed era and ask whether it predicts the gap actually realized in the
   deployment era. If it does, a practitioner standing at the boundary can run the
   diagnostic with data in hand and trust the answer.

       gap_backtest : transfer and oracle both trained on observed-era windows,
                      evaluated on held-out target entities' OBSERVED-era windows.
                      Computable at decision time.
       gap_deploy   : the same two models evaluated on the same entities'
                      DEPLOYMENT-era windows. The ground truth, not computable
                      at decision time.

   The claim under test is that gap_backtest tracks gap_deploy across splits. The
   falsifier is a correlation near zero, or a bias large enough to flip the go/no-go
   call at the 3 percent operating point.

B) COLD-START LEARNING CURVE. The target entity is genuinely new and has only k days
   of history. Restrict the oracle's training windows to the first k days and sweep k.
   The question is how much history must accumulate before the gap estimate settles.
   An answer of "a few months" converts "the oracle can never be run" into "the oracle
   can be run after N weeks", which is a measured statement rather than a concession.

   HONEST CONFOUND, stated because it changes how the curve must be read: a
   small-k oracle is trained on less data, so it is worse, so the measured gap shrinks
   for a reason that has nothing to do with the target being irreducible. The oracle's
   own error is therefore reported at every k. A gap estimate that rises to a plateau
   as k grows is the expected and benign shape; what would falsify the method is a
   curve that never settles, or one whose plateau differs from the full-data gap.

Output: outputs/backtest_oracle_temporal.csv, outputs/backtest_oracle_curve.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from src.data import Panel
from src.survey_reducibility import entity_stats
from src.label_free_calibration import PANELS, make_splits
from src.train import train_model, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")
PANEL_ORDER = ["Nursing", "Electricity", "Solar", "Traffic"]


# --------------------------------------------------------------------- masks
def mask_upto(P, day):
    """Windows whose input AND horizon both end on or before `day`.

    This is exactly the set of windows an observer standing at `day` has already
    seen the labels for, which is what makes the oracle trainable without any
    future data.
    """
    return (P.wstart + P.span) <= day


def boundary_index(P, boundary_date):
    return int(np.searchsorted(P.dates, boundary_date))


def split_target(tgt_all, rng):
    """Carve the target group into oracle-train / oracle-val / evaluation entities."""
    n_test = max(15, int(0.35 * len(tgt_all)))
    n_val = max(6, int(0.15 * len(tgt_all)))
    t = tgt_all.copy()
    rng.shuffle(t)
    t_te, pool = t[:n_test], t[n_test:]
    t_va, t_tr = pool[:n_val], pool[n_val:]
    return t_tr, t_va, t_te


def mae(model, P, facs, time_mask):
    rows = evaluate(model, P, facs, time_mask=time_mask)
    return float(np.mean([r["mae_total"] for r in rows]))


# ------------------------------------------------------- experiment A: time
def temporal_split(P, src_all, tgt_all, seed, obs_mask, dep_mask):
    """Gap measured inside the observed era vs gap realized in the deployment era."""
    rng = np.random.default_rng(seed)
    t_tr, t_va, t_te = split_target(tgt_all, rng)
    if len(t_tr) < 12:
        return None

    # size-match the transfer model's training entities to the oracle's
    s = src_all.copy()
    rng.shuffle(s)
    n_val = max(6, int(0.15 * len(tgt_all)))
    s_va, s_tr = s[:n_val], s[n_val:n_val + len(t_tr)]
    if len(s_tr) < 12:
        return None

    # both models see ONLY observed-era windows - nothing here needs the future
    m_tra, _ = train_model(P, ERM, s_tr, s_va, seed=seed,
                           train_mask=obs_mask, val_mask=obs_mask)
    m_ora, _ = train_model(P, ERM, t_tr, t_va, seed=seed,
                           train_mask=obs_mask, val_mask=obs_mask)

    # computable at decision time
    tra_obs = mae(m_tra, P, t_te, obs_mask)
    ora_obs = mae(m_ora, P, t_te, obs_mask)
    # the ground truth a practitioner could not have seen
    tra_dep = mae(m_tra, P, t_te, dep_mask)
    ora_dep = mae(m_ora, P, t_te, dep_mask)

    return dict(
        gap_backtest=100.0 * (tra_obs - ora_obs) / tra_obs,
        gap_deploy=100.0 * (tra_dep - ora_dep) / tra_dep,
        transfer_obs=tra_obs, oracle_obs=ora_obs,
        transfer_dep=tra_dep, oracle_dep=ora_dep,
        n_src=len(s_tr), n_tgt_train=len(t_tr), n_tgt_test=len(t_te),
    )


# ------------------------------------------------- experiment B: how much history
def curve_split(P, src_all, tgt_all, seed, ladder, obs_mask, eval_mask):
    """Sweep how many days of the target's own history the oracle is allowed."""
    rng = np.random.default_rng(seed)
    t_tr, t_va, t_te = split_target(tgt_all, rng)
    if len(t_tr) < 12:
        return []

    s = src_all.copy()
    rng.shuffle(s)
    n_val = max(6, int(0.15 * len(tgt_all)))
    s_va, s_tr = s[:n_val], s[n_val:n_val + len(t_tr)]
    if len(s_tr) < 12:
        return []

    # the transfer model is unchanged across the ladder: in reality the source
    # history is fully available, only the TARGET is new
    m_tra, _ = train_model(P, ERM, s_tr, s_va, seed=seed,
                           train_mask=obs_mask, val_mask=obs_mask)
    tra = mae(m_tra, P, t_te, eval_mask)

    rows = []
    for k in ladder:
        km = mask_upto(P, k) & obs_mask
        n_win = int(km.sum())
        if n_win < 4 * len(t_tr):
            # too few windows for even one grouped batch per entity; record and skip
            rows.append(dict(days=k, windows_per_entity=n_win / max(len(t_tr), 1),
                             gap_pct=np.nan, oracle=np.nan, transfer=tra,
                             skipped=True))
            continue
        m_ora, _ = train_model(P, ERM, t_tr, t_va, seed=seed,
                               train_mask=km, val_mask=km)
        ora = mae(m_ora, P, t_te, eval_mask)
        rows.append(dict(days=k, windows_per_entity=n_win / max(len(t_tr), 1),
                         gap_pct=100.0 * (tra - ora) / tra, oracle=ora,
                         transfer=tra, skipped=False))
    return rows


# ---------------------------------------------------------------------- main
def main():
    n_splits = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    seeds = [0, 1]
    want = sys.argv[2:] or ["Nursing", "Solar"]

    temporal_rows, curve_rows = [], []
    for name in want:
        spec = PANELS[name]
        P = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6), L=56, H=14,
                  stride=18, seed=0, n_channels=3,
                  boundary_date=spec["boundary"])
        obs_mask = P.is_2024          # everything observed by the decision boundary
        dep_mask = P.is_2025          # the deployment era, unavailable at that moment
        if obs_mask.sum() < 50 or dep_mask.sum() < 50:
            print(f"{name}: not enough windows either side of the boundary, skipping")
            continue

        bidx = boundary_index(P, spec["boundary"])
        # ladder of "days of history the new entity has accumulated"
        ladder = [d for d in [P.span + 7, P.span + 35, P.span + 91, P.span + 175,
                              P.span + 259, bidx] if d <= bidx]
        ladder = sorted(set(ladder))

        stats, _ = entity_stats(P)
        splits = make_splits(stats, n_splits, np.random.default_rng(0))
        print(f"\n### {name}: {P.nF} entities, {len(splits)} splits, "
              f"boundary at day {bidx}, ladder {ladder}", flush=True)

        for k, (A, B) in enumerate(splits):
            for sd in seeds:
                r = temporal_split(P, A, B, sd, obs_mask, dep_mask)
                if r is None:
                    continue
                temporal_rows.append(dict(panel=name, split=k, seed=sd, **r))
                print(f"  [A] split {k:2d} seed {sd}  backtest={r['gap_backtest']:+6.2f}%"
                      f"  deploy={r['gap_deploy']:+6.2f}%", flush=True)

            # the curve is the expensive one, so run it on a subset of splits
            if k % 3 == 0:
                for row in curve_split(P, A, B, 0, ladder, obs_mask, dep_mask):
                    curve_rows.append(dict(panel=name, split=k, **row))
                    if not row["skipped"]:
                        print(f"  [B] split {k:2d} days={row['days']:4d} "
                              f"({row['windows_per_entity']:.1f} win/entity)  "
                              f"gap={row['gap_pct']:+6.2f}%", flush=True)

    os.makedirs(OUT, exist_ok=True)
    dt = pd.DataFrame(temporal_rows)
    dc = pd.DataFrame(curve_rows)
    dt.to_csv(os.path.join(OUT, "backtest_oracle_temporal.csv"), index=False)
    dc.to_csv(os.path.join(OUT, "backtest_oracle_curve.csv"), index=False)
    report(dt, dc)


def report(dt, dc):
    from scipy import stats as ss
    print("\n" + "=" * 78)
    print("A. CAN THE DIAGNOSTIC BE RUN AT DECISION TIME?")
    print("   backtest gap (observed era only) vs deploy gap (the unavailable truth)")
    print("=" * 78)
    if dt.empty:
        print("   no rows")
    else:
        print(f"{'panel':<13}{'n':>4}{'r':>8}{'rho':>8}{'bias pp':>10}"
              f"{'MAE pp':>9}{'agree@3%':>10}")
        print("-" * 62)
        for p, g in dt.groupby("panel"):
            x, y = g.gap_backtest.values, g.gap_deploy.values
            if len(g) < 4:
                print(f"{p:<13}{len(g):>4}  too few splits")
                continue
            r = ss.pearsonr(x, y)[0]
            rho = ss.spearmanr(x, y)[0]
            agree = 100 * np.mean((x >= 3.0) == (y >= 3.0))
            print(f"{p:<13}{len(g):>4}{r:>+8.3f}{rho:>+8.3f}"
                  f"{np.mean(x - y):>+10.2f}{np.mean(np.abs(x - y)):>9.2f}"
                  f"{agree:>9.0f}%")
        print("\n   A high r with small bias means a practitioner standing at the")
        print("   boundary would have reached the same conclusion as the oracle that")
        print("   needed the future. That is the whole claim.")

    print("\n" + "=" * 78)
    print("B. HOW MUCH OF ITS OWN HISTORY DOES A NEW ENTITY NEED?")
    print("=" * 78)
    if dc.empty:
        print("   no rows")
        return
    live = dc[~dc.skipped.astype(bool)]
    for p, g in live.groupby("panel"):
        full = g[g.days == g.days.max()]
        ref = float(full.gap_pct.mean()) if len(full) else np.nan
        print(f"\n{p}  (full-history gap = {ref:+.2f}%)")
        print(f"   {'days':>6}{'win/entity':>12}{'gap':>9}{'vs full':>10}{'oracle MAE':>12}")
        for d, gg in g.groupby("days"):
            m = float(gg.gap_pct.mean())
            print(f"   {int(d):>6}{gg.windows_per_entity.mean():>12.1f}"
                  f"{m:>+9.2f}{m - ref:>+10.2f}{gg.oracle.mean():>12.4f}")


if __name__ == "__main__":
    main()
