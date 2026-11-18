"""Why the spectral screen fails on solar and traffic, and what could replace it.

The measured reducible gaps are already in outputs/label_free_calibration.csv and
the splits that produced them are deterministic (seed 0), so alternative screens
can be evaluated against the same ground truth WITHOUT retraining anything. That
makes it cheap to test many candidates instead of guessing one.

WHY THE CURRENT SCREEN MIGHT BE THE WRONG STATISTIC
---------------------------------------------------
The Jensen-Shannon divergence compares how two groups distribute ENERGY across
sub-bands. But a reducible gap does not require that the groups merely differ; it
requires that they differ in a way a forecaster could LEARN. Two groups that
differ only in noise level have a large spectral divergence and no reducible gap
at all, because noise is not learnable. So an energy statistic should be expected
to fire on differences that carry no headroom.

That suggests weighting or replacing energy with something closer to
predictability. The candidates below span that idea:

  js_mean        current screen: JS between mean 4-band energy profiles
  js_perchan     same, but per channel, so cross-channel structure survives
  js_fine        same, with 8 bands instead of 4
  mmd_profile    MMD between the per-entity profile CLOUDS, not their means, so
                 a difference in spread is visible and not just a shift
  w1_profile     per-band Wasserstein-1 between the entity distributions
  acf_l2         L2 distance between group-mean autocorrelation functions. This
                 is the direct measure of temporal dependence, which is what a
                 forecaster actually fits
  acf_weighted   JS on band energies, each band weighted by how predictable that
                 band is (its own autocorrelation), so unlearnable noise bands
                 stop contributing
  diff_ratio     ratio of intrinsic difficulty, target over source

HONESTY ABOUT MULTIPLICITY
--------------------------
Eight candidates are scored on the same splits, so the best one is selected
post-hoc and its leave-one-out score is optimistically biased. That is reported
explicitly. A candidate is only interesting here if it works on the two panels
the current screen FAILS on while not destroying the two it succeeds on; that
pattern is much harder to produce by chance than a single good number.

Output: outputs/screen_statistics_probe.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as ss

from src.data import Panel
from src.survey_reducibility import entity_stats, js_div
from src.label_free_calibration import PANELS, make_splits, loo_predict, KEYS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


# --------------------------------------------------------------- descriptors
def per_entity_features(P):
    """All per-entity quantities the candidate screens need. Inputs only."""
    X = P.Xw[P.is_2024] if P.is_2024.any() else P.Xw          # [N,C,L]
    f = P.fid[P.is_2024] if P.is_2024.any() else P.fid
    n, L = P.nF, X.shape[-1]
    tot = X.sum(1)                                            # [N,L]
    freq = np.fft.rfftfreq(L, 1.0)
    e4 = np.linspace(0, len(freq), 5).astype(int)
    e8 = np.linspace(0, len(freq), 9).astype(int)

    prof4 = np.zeros((n, 4))
    prof8 = np.zeros((n, 8))
    profC = np.zeros((n, 3 * 4))
    acf = np.zeros((n, 21))
    band_ac = np.zeros((n, 4))
    diff = np.zeros(n)

    for i in range(n):
        m = f == i
        if not m.any():
            continue
        t = tot[m]
        c = t - t.mean(1, keepdims=True)
        A = np.abs(np.fft.rfft(c, axis=1))
        b4 = np.array([A[:, e4[b]:e4[b + 1]].sum() for b in range(4)])
        prof4[i] = b4 / (b4.sum() + 1e-8)
        b8 = np.array([A[:, e8[b]:e8[b + 1]].sum() for b in range(8)])
        prof8[i] = b8 / (b8.sum() + 1e-8)

        xc = X[m]                                             # [nw,C,L]
        pc = []
        for ch in range(xc.shape[1]):
            cc = xc[:, ch] - xc[:, ch].mean(1, keepdims=True)
            Ac = np.abs(np.fft.rfft(cc, axis=1))
            bb = np.array([Ac[:, e4[b]:e4[b + 1]].sum() for b in range(4)])
            pc.append(bb / (bb.sum() + 1e-8))
        profC[i] = np.concatenate(pc)

        # autocorrelation of the centered total series, averaged over windows
        sd = c.std(1, keepdims=True) + 1e-8
        z = c / sd
        for lag in range(1, 22):
            acf[i, lag - 1] = float((z[:, :-lag] * z[:, lag:]).mean())
        diff[i] = float(t.std(1).mean() / (abs(t.mean()) + 1e-8))

        # per-band predictability: autocorrelation of each band-filtered signal
        for b in range(4):
            Fb = np.zeros_like(np.fft.rfft(c, axis=1))
            Fb[:, e4[b]:e4[b + 1]] = np.fft.rfft(c, axis=1)[:, e4[b]:e4[b + 1]]
            sb = np.fft.irfft(Fb, n=L, axis=1)
            sbz = sb / (sb.std(1, keepdims=True) + 1e-8)
            band_ac[i, b] = float(abs((sbz[:, :-1] * sbz[:, 1:]).mean()))

    return dict(prof4=prof4, prof8=prof8, profC=profC, acf=acf,
                band_ac=band_ac, diff=diff)


# ------------------------------------------------------------------- screens
def mmd(Pa, Pb):
    """Unbiased-ish RBF MMD between two clouds of profile vectors."""
    Z = np.vstack([Pa, Pb])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0]) + 1e-12
    K = np.exp(-d2 / med)
    na = len(Pa)
    Kaa, Kbb, Kab = K[:na, :na], K[na:, na:], K[:na, na:]
    return float(Kaa.mean() + Kbb.mean() - 2 * Kab.mean())


def screens(F, A, B):
    out = {}
    out["js_mean"] = js_div(F["prof4"][A].mean(0), F["prof4"][B].mean(0))
    out["js_fine"] = js_div(F["prof8"][A].mean(0), F["prof8"][B].mean(0))
    pa, pb = F["profC"][A].mean(0), F["profC"][B].mean(0)
    out["js_perchan"] = float(sum(
        js_div(pa[4 * k:4 * k + 4], pb[4 * k:4 * k + 4]) for k in range(3)))
    out["mmd_profile"] = mmd(F["prof4"][A], F["prof4"][B])
    out["w1_profile"] = float(sum(
        ss.wasserstein_distance(F["prof4"][A][:, b], F["prof4"][B][:, b])
        for b in range(4)))
    out["acf_l2"] = float(np.linalg.norm(F["acf"][A].mean(0) - F["acf"][B].mean(0)))
    w = 0.5 * (F["band_ac"][A].mean(0) + F["band_ac"][B].mean(0))
    qa = F["prof4"][A].mean(0) * w; qb = F["prof4"][B].mean(0) * w
    out["acf_weighted"] = js_div(qa / (qa.sum() + 1e-12), qb / (qb.sum() + 1e-12))
    out["diff_ratio"] = float(F["diff"][B].mean() / (F["diff"][A].mean() + 1e-8))
    return out


def main():
    truth = pd.read_csv(os.path.join(OUT, "label_free_calibration.csv"))
    rows = []
    for name in ["Solar", "Electricity", "Traffic", "Nursing"]:
        sub = truth[truth.panel == name]
        if sub.empty:
            continue
        spec = PANELS[name]
        P = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6), L=56, H=14,
                  stride=18, seed=0, n_channels=3, boundary_date=spec["boundary"])
        stats, _ = entity_stats(P)
        splits = make_splits(stats, 26, np.random.default_rng(0))
        F = per_entity_features(P)
        print(f"{name}: {len(sub)} measured splits", flush=True)
        for _, r in sub.iterrows():
            A, B = splits[int(r["split"])]
            s = screens(F, A, B)
            # the regenerated split must be the one that produced this gap
            assert abs(s["js_mean"] - r["spec_div"]) < 1e-9, "split mismatch"
            rows.append(dict(panel=name, split=int(r["split"]),
                             gap=r["reducible_gap_pct"], **s))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "screen_statistics_probe.csv"), index=False)

    cands = [c for c in df.columns if c not in ("panel", "split", "gap")]
    print("\n" + "=" * 78)
    print("LEAVE-ONE-OUT r OF EACH CANDIDATE SCREEN (predicting the measured gap)")
    print("=" * 78)
    hdr = f"{'screen':<14}" + "".join(f"{p:>13}" for p in
                                      ["Nursing", "Electricity", "Solar", "Traffic"])
    print(hdr); print("-" * len(hdr))
    table = {}
    for c in cands:
        line = f"{c:<14}"
        table[c] = {}
        for pan in ["Nursing", "Electricity", "Solar", "Traffic"]:
            g = df[df.panel == pan]
            if len(g) < 6:
                line += f"{'--':>13}"; continue
            pred = loo_predict(g[c].values, g.gap.values)
            r = ss.pearsonr(pred, g.gap.values)[0]
            table[c][pan] = r
            line += f"{r:>+13.3f}"
        print(line)
    print("\nA candidate is only interesting if it rescues Solar and Traffic")
    print("without breaking Nursing and Electricity.")
    ok = [c for c, v in table.items()
          if v.get("Solar", -1) > 0.4 and v.get("Traffic", -1) > 0.4]
    print(f"\ncandidates positive on BOTH failing panels (r>0.4): {ok or 'NONE'}")


if __name__ == "__main__":
    main()
