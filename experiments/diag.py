"""One diagnostic run on one split, shared by the survey and screen experiments.

For a source group A and target group B:
  source-trained model  ERM trained on s_tr (early stopping on s_va)
  target-trained ref.   ERM trained on t_tr (early stopping on t_va), |t_tr|=|s_tr|
  target test set       t_te, used to evaluate BOTH models and both simple actions
  source test set       s_te, held out from training and early stopping

Simple actions evaluated on the same t_te windows, needing no target labels:
  seasonal naive        repeat the last observed week
  source-val blend      (1-a)*source-trained + a*seasonal naive, a chosen on s_va

Screens are computed from TRAINING-ERA INPUT windows only (Panel.Xw[is_2024]).
"""
import numpy as np
import pandas as pd

from common import (ERM, METRICS, allocate, blend_alpha, fit, headroom,
                      per_entity, predict, seasonal_naive)


# ------------------------------------------------------------------ features
def entity_features(P):
    """Per-entity input descriptors used by the screens (observed era only).

    For each entity, every observed-era input window is summed over channels,
    mean-centred, and Fourier-transformed (rfft, L=56 -> 29 frequency bins,
    split into four equal-width bin groups).
      amp4[i]  normalised profile of summed |Fourier amplitudes| per bin group
      eng4[i]  same with squared amplitudes (spectral energy)
      acf[i]   autocorrelation of the standardised window at lags 1..21,
               averaged over the entity's windows
    """
    X = P.Xw[P.is_2024] if P.is_2024.any() else P.Xw
    f = P.fid[P.is_2024] if P.is_2024.any() else P.fid
    n, Lw = P.nF, X.shape[-1]
    tot = X.sum(1)
    nb = len(np.fft.rfftfreq(Lw, 1.0))
    e4 = np.linspace(0, nb, 5).astype(int)
    amp4 = np.zeros((n, 4)); eng4 = np.zeros((n, 4)); acf = np.zeros((n, 21))
    for i in range(n):
        m = f == i
        if not m.any():
            continue
        c = tot[m] - tot[m].mean(1, keepdims=True)
        A = np.abs(np.fft.rfft(c, axis=1))
        a = np.array([A[:, e4[b]:e4[b + 1]].sum() for b in range(4)])
        e = np.array([(A[:, e4[b]:e4[b + 1]] ** 2).sum() for b in range(4)])
        amp4[i] = a / (a.sum() + 1e-8)
        eng4[i] = e / (e.sum() + 1e-8)
        z = c / (c.std(1, keepdims=True) + 1e-8)
        for lag in range(1, 22):
            acf[i, lag - 1] = float((z[:, :-lag] * z[:, lag:]).mean())
    return dict(amp4=amp4, eng4=eng4, acf=acf)


def js_div(p, q, eps=1e-9):
    p = p / (p.sum() + eps); q = q / (q.sum() + eps); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + eps) / (b + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def core_screens(F, A, B):
    """The three screens reported in the paper. All are SYMMETRIC in (A, B)."""
    return dict(
        js_amp=js_div(F["amp4"][A].mean(0), F["amp4"][B].mean(0)),
        js_energy=js_div(F["eng4"][A].mean(0), F["eng4"][B].mean(0)),
        acf_l2=float(np.linalg.norm(F["acf"][A].mean(0) - F["acf"][B].mean(0))),
    )


# ------------------------------------------------------------------ one split
def run_split(P, A, B, seed, floors=(20, 8), cfg=ERM, time_masks=(None, None)):
    """Returns (split_row, entity_rows) or None if the groups are too small."""
    rng = np.random.default_rng(seed)
    al = allocate(A, B, rng, floors=floors)
    if al is None:
        return None
    tmask, emask = time_masks
    m_src = fit(P, cfg, al["s_tr"], al["s_va"], seed, tmask, tmask)
    m_ref = fit(P, cfg, al["t_tr"], al["t_va"], seed, tmask, tmask)

    Xv, Yv, Pv, _ = predict(m_src, P, al["s_va"], tmask)
    alpha = blend_alpha(Pv, Xv, Yv, "mae_cat")

    Xt, Yt, Pt, fid = predict(m_src, P, al["t_te"], emask)
    _, _, Rt, _ = predict(m_ref, P, al["t_te"], emask)
    sn = seasonal_naive(Xt)
    e_tr = per_entity(Pt, Yt, fid)
    e_rf = per_entity(Rt, Yt, fid)
    e_sn = per_entity(sn, Yt, fid)
    e_bl = per_entity((1 - alpha) * Pt + alpha * sn, Yt, fid)
    Xs, Ys, Ps, fs = predict(m_src, P, al["s_te"], emask)
    e_so = per_entity(Ps, Ys, fs)

    row = dict(n_src_train=len(al["s_tr"]), n_tgt_test=len(al["t_te"]),
               n_src_test=len(al["s_te"]), alpha=alpha)
    for mt in METRICS:
        so, tr, rf = e_so[mt].mean(), e_tr[mt].mean(), e_rf[mt].mean()
        sn_, bl = e_sn[mt].mean(), e_bl[mt].mean()
        row.update({f"source_{mt}": so, f"transfer_{mt}": tr,
                    f"reference_{mt}": rf, f"snaive_{mt}": sn_,
                    f"blend_{mt}": bl,
                    f"apparent_gap_{mt}": 100 * (tr - so) / so,
                    f"headroom_{mt}": headroom(tr, rf),
                    f"snaive_gain_{mt}": headroom(tr, sn_),
                    f"blend_gain_{mt}": headroom(tr, bl)})
    ent = pd.concat({"transfer": e_tr, "reference": e_rf, "snaive": e_sn,
                     "blend": e_bl}, axis=1)
    ent.columns = [f"{a}_{b}" for a, b in ent.columns]
    ent = ent.reset_index()
    src = e_so.add_prefix("source_").reset_index()
    return row, ent, src
