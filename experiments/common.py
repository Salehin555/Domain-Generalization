"""Shared helpers for the experiment scripts.

Every experiment script scores models with BOTH endpoints and saves per-entity
errors, so that uncertainty can be computed by resampling entities afterwards.

    mae_cat   PRIMARY. Mean absolute error over the C channels and the H horizon
              steps, per entity, then averaged over entities. This is the metric
              the manuscript's equation defines, and the one training (L1 loss
              over all channels) and model selection (source-validation mae_cat)
              already optimise.
    mae_total SECONDARY. Absolute error of the channel-summed forecast, i.e. error
              in total staffing (nursing) or total load/flow/output. An
              over-forecast in one channel can cancel an under-forecast in
              another, so mae_total <= C * mae_cat and the two can rank models
              differently.

All results are written under experiments/results/; nothing in outputs/ is
overwritten.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

from src.data import Panel                       # noqa: E402
from src.train import train_model                # noqa: E402

METRICS = ("mae_cat", "mae_total")
L, H, PERIOD = 56, 14, 7
SN_IDX = np.array([(L - PERIOD) + (h % PERIOD) for h in range(H)])
ALPHA_GRID = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
ERM = dict(method="erm", epochs=22, patience=5, backbone="mlp")

PANELS = {
    "Nursing": dict(path=os.path.join(ROOT, "cache", "panel.npz"),
                    boundary=20250101),
    "Electricity": dict(path=os.path.join(ROOT, "cache", "electricity",
                                          "panel_elec.npz"), boundary=20140101),
    "Traffic": dict(path=os.path.join(ROOT, "cache", "panels",
                                      "panel_traffic.npz"), boundary=20160504),
    "Solar": dict(path=os.path.join(ROOT, "cache", "panels", "panel_solar.npz"),
                  boundary=20060902),
}


def load_panel(name, nfac=10 ** 6, stride=18):
    spec = PANELS[name]
    return Panel(spec["path"], n_facilities=nfac, L=L, H=H, stride=stride, seed=0,
                 n_channels=3, boundary_date=spec["boundary"])


def predict(model, P, facs, time_mask=None, batch=2048):
    """Raw-space inputs, targets, predictions and entity ids for `facs`."""
    Xw, Yw, fid = P.eval_arrays(facs, time_mask=time_mask)
    out = np.zeros_like(Yw)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(Xw), batch):
            out[i:i + batch] = model(torch.from_numpy(Xw[i:i + batch])).numpy()
    return Xw, Yw, out, fid


def seasonal_naive(Xw):
    return Xw[:, :, SN_IDX]


def per_entity(pred, Yw, fid, keep=None):
    """Per-entity errors for both endpoints, aggregated exactly as
    src.train.evaluate does (mean over that entity's windows).

    keep: optional bool[n_windows]; windows outside it are not scored (used for
    the imputed-target sensitivity analysis)."""
    if keep is not None:
        pred, Yw, fid = pred[keep], Yw[keep], fid[keep]
    ae = np.abs(pred - Yw)                                  # [N,C,H]
    cat = ae.mean(axis=(1, 2))
    tot = np.abs(pred.sum(1) - Yw.sum(1)).mean(axis=1)
    df = pd.DataFrame(dict(fac=fid, mae_cat=cat, mae_total=tot))
    return df.groupby("fac")[list(METRICS)].mean()


def blend_alpha(pred_v, X_v, Y_v, metric="mae_cat"):
    """Blend weight chosen on SOURCE-VALIDATION entities only."""
    sn = seasonal_naive(X_v)
    best, best_a = np.inf, 0.0
    for a in ALPHA_GRID:
        e = per_entity((1 - a) * pred_v + a * sn, Y_v,
                       np.zeros(len(Y_v), int))[metric].iloc[0]
        if e < best - 1e-12:
            best, best_a = e, a
    return best_a


def fit(P, cfg, tr, va, seed, train_mask=None, val_mask=None):
    m, _ = train_model(P, cfg, tr, va, seed=seed, train_mask=train_mask,
                       val_mask=val_mask)
    return m


def allocate(src_all, tgt_all, rng, floors=(20, 8), fracs=(0.35, 0.15),
             min_train=15, fallback=True, need_source_test=True):
    """Size-matched entity allocation shared by every diagnostic.

    Target group -> t_te (evaluation), t_va, t_tr (target-trained reference).
    Source group -> s_te (held-out source evaluation, same size as t_te where
    possible), s_va, s_tr, with |s_tr| == |t_tr| and |s_va| == |t_va|.
    Source error is measured on s_te, never on the early-stopping set s_va.
    """
    def sizes(ft, fv, pt, pv):
        n_test = max(ft, int(pt * len(tgt_all)))
        n_val = max(fv, int(pv * len(tgt_all)))
        n_train = len(tgt_all) - n_test - n_val
        return n_test, n_val, n_train

    n_test, n_val, n_train = sizes(floors[0], floors[1], fracs[0], fracs[1])
    if n_train < min_train and fallback:
        n_test, n_val, n_train = sizes(10, 4, 0.30, 0.12)
    s_room = len(src_all) - n_val - (n_test if need_source_test else 0)
    n_train = min(n_train, s_room)
    if n_train < 12:
        return None
    s = src_all.copy(); rng.shuffle(s)
    t = tgt_all.copy(); rng.shuffle(t)
    n_ste = min(n_test, len(src_all) - n_val - n_train) if need_source_test else 0
    s_te = s[:n_ste]
    s_va = s[n_ste:n_ste + n_val]
    s_tr = s[n_ste + n_val:n_ste + n_val + n_train]
    t_te = t[:n_test]
    t_va = t[n_test:n_test + n_val]
    t_tr = t[n_test + n_val:n_test + n_val + n_train]
    assert len(s_tr) == len(t_tr) == n_train
    return dict(s_te=s_te, s_va=s_va, s_tr=s_tr, t_te=t_te, t_va=t_va, t_tr=t_tr)


def headroom(transfer, reference):
    return 100.0 * (transfer - reference) / transfer
