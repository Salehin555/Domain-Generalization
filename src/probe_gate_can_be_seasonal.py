"""Is the seasonal branch expressible as a sub-band gate? A no-training probe.

The question: can mechanism 2 be folded into the gate, so the paper claims one
gating mechanism that both diagnoses and delivers the gain?

Two things have to be true for that to be honest rather than cosmetic.

  (a) The seasonal-naive predictor has to be reachable by re-weighting sub-bands.
      A gate can only SCALE content that is already present; the seasonal branch
      copies observed values forward into the horizon, which is a different kind
      of operation. This probe measures how much of SN is recovered by keeping
      the weekly-carrying sub-bands of the input and extending them.

  (b) The existing gate would have to survive the change, since the RQ2
      interpretability result (gate vs ICC) depends on it acting on the feature.

This script only answers (a), numerically, with no model training.
"""
import os

import numpy as np
import pywt

from src.data import Panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L, H, P_ = 56, 14, 7


def seasonal_naive(x):
    """What mechanism 2 actually injects: repeat the matching phase of the cycle."""
    idx = [(L - P_) + (h % P_) for h in range(H)]
    return x[..., idx]


def band_extend(x, keep):
    """Best a gate-style operator can do: keep some sub-bands, extend periodically.

    Reconstruct the input from a subset of its wavelet sub-bands, then continue
    that reconstruction into the horizon by repeating its last period. This is the
    closest thing to 'seasonal-naive as a sub-band gate'.
    """
    w = pywt.Wavelet("db4")
    coeffs = pywt.wavedec(x, w, mode="periodization", level=3, axis=-1)
    out = [c if i in keep else np.zeros_like(c) for i, c in enumerate(coeffs)]
    rec = pywt.waverec(out, w, mode="periodization", axis=-1)[..., :L]
    idx = [(L - P_) + (h % P_) for h in range(H)]
    return rec[..., idx]


def main():
    Pn = Panel(os.path.join(ROOT, "cache", "electricity", "panel_elec.npz"),
               n_facilities=10 ** 6, L=L, H=H, stride=18, seed=0, n_channels=3,
               boundary_date=20140101)
    grp = np.asarray(Pn.state).astype(str)
    tgt = np.arange(Pn.nF)[grp == "WorkWeek"]
    m = np.isin(Pn.fid, tgt)
    X, Y = Pn.Xw[m], Pn.Yw[m]
    # normalize per window per channel, as RevIN does, so errors are comparable
    mu = X.mean(-1, keepdims=True)
    sd = X.std(-1, keepdims=True) + 1e-8
    Xn, Yn = (X - mu) / sd, (Y - mu) / sd

    sn = seasonal_naive(Xn)
    e_sn = np.abs(sn - Yn).mean()
    print(f"windows={len(Xn)}  seasonal-naive MAE (normalized) = {e_sn:.4f}\n")

    print(f"{'sub-bands kept':<28} {'MAE':>8} {'vs SN':>9} {'corr w/ SN':>11}")
    print("-" * 60)
    combos = [((0,), "approx only (trend)"),
              ((0, 1), "approx + D3"),
              ((0, 1, 2), "approx + D3 + D2"),
              ((0, 1, 2, 3), "all bands (= identity)"),
              ((1,), "D3 only"),
              ((1, 2), "D3 + D2"),
              ((2,), "D2 only"),
              ((3,), "D1 only (fastest)")]
    for keep, label in combos:
        be = band_extend(Xn, set(keep))
        e = np.abs(be - Yn).mean()
        c = np.corrcoef(be.ravel(), sn.ravel())[0, 1]
        print(f"{label:<28} {e:8.4f} {100*(e_sn-e)/e_sn:+8.1f}% {c:11.3f}")

    print("\nReading: 'all bands' IS seasonal-naive by construction, so it must")
    print("match exactly. If no strict SUBSET gets close, then what the branch")
    print("contributes is the forward copy itself, not a choice of sub-bands,")
    print("and a gate over sub-bands cannot stand in for it.")


if __name__ == "__main__":
    main()
