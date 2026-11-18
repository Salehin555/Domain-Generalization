"""
Sensitivity checks of the training-free screen. Each block answers one question.

S1  predictor specification + sensitivity: how the Jensen-Shannon divergence is
    computed (per channel vs aggregated, DWT level, normalization) and how much
    the correlation with the reducible gap depends on those choices.
S2  hypothesis-class evidence: spectral comparison of predictions against targets,
    testing directly whether a source-trained backbone fails to emit the target's
    rhythm and whether the periodicity branch restores it.
S3  screening threshold: precision/recall of the training-free screen as a
    decision rule, with a recommended operating point.

Output: outputs/screen_sens_*.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")

PANELS = [
    ("Nursing", os.path.join(ROOT, "cache", "panel.npz"), 20250101, 1500),
    ("Electricity", os.path.join(ROOT, "cache", "electricity", "panel_elec.npz"),
     20140101, 10 ** 6),
    ("Traffic", os.path.join(ROOT, "cache", "panels", "panel_traffic.npz"),
     20160504, 10 ** 6),
    ("Solar", os.path.join(ROOT, "cache", "panels", "panel_solar.npz"),
     20060902, 10 ** 6),
]


def js(p, q, eps=1e-9):
    p = p / (p.sum() + eps); q = q / (q.sum() + eps); m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + eps) / (b + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def band_profile(X, f, n, nbands, per_channel):
    """Mean relative band-energy profile per entity, at a given band count."""
    tot = X if per_channel else X.sum(1, keepdims=True)
    C = tot.shape[1]
    freq = np.fft.rfftfreq(tot.shape[-1], 1.0)
    edges = np.linspace(0, len(freq), nbands + 1).astype(int)
    prof = np.zeros((n, C * nbands))
    for i in range(n):
        m = f == i
        if not m.any():
            continue
        c = tot[m] - tot[m].mean(-1, keepdims=True)
        A = np.abs(np.fft.rfft(c, axis=-1))
        b = np.stack([A[..., edges[k]:edges[k + 1]].sum(-1)
                      for k in range(nbands)], -1)      # [w, C, nbands]
        b = b.mean(0)
        b = b / (b.sum() + 1e-9)
        prof[i] = b.ravel()
    return prof


def q3_predictor_sensitivity():
    """Recompute the screen under different specifications, against the measured gap."""
    surv = pd.read_csv(os.path.join(OUT, "reducibility_survey.csv"))
    truth = (surv.groupby(["panel", "split"])["reducible_gap_pct"]
             .mean().reset_index())
    rows = []
    for nbands in [4, 8]:
        for per_channel in [True, False]:
            recs = []
            for name, path, bd, nf in PANELS:
                P = Panel(path, n_facilities=nf, L=56, H=14, stride=18, seed=0,
                          n_channels=3, boundary_date=bd)
                X = P.Xw[P.is_2024] if P.is_2024.any() else P.Xw
                f = P.fid[P.is_2024] if P.is_2024.any() else P.fid
                prof = band_profile(X, f, P.nF, nbands, per_channel)
                # rebuild the same quartile splits the survey used
                tot = (P.Xw.sum(1))
                lev = np.array([tot[P.fid == i].mean() if (P.fid == i).any() else 0
                                for i in range(P.nF)])
                vol = np.array([tot[P.fid == i].std(1).mean() /
                                (abs(tot[P.fid == i].mean()) + 1e-8)
                                if (P.fid == i).any() else 0 for i in range(P.nF)])
                idx = np.arange(P.nF)
                for sname, v in [("size", lev), ("vol", vol)]:
                    q1, q3 = np.percentile(v, [25, 75])
                    A, B = idx[v <= q1], idx[v >= q3]
                    if len(A) < 30 or len(B) < 30:
                        continue
                    d = js(prof[A].mean(0), prof[B].mean(0))
                    for direction in ["low->high", "high->low"]:
                        recs.append((name, f"{sname}: {direction}", d))
            df = pd.DataFrame(recs, columns=["panel", "split", "spec"])
            mg = df.merge(truth, on=["panel", "split"], how="inner")
            if len(mg) < 6:
                continue
            r, p = pearsonr(mg.spec, mg.reducible_gap_pct)
            rs, ps = spearmanr(mg.spec, mg.reducible_gap_pct)
            rows.append(dict(bands=nbands, per_channel=per_channel, n=len(mg),
                             pearson=r, p_pearson=p, spearman=rs, p_spearman=ps))
            print(f"  bands={nbands} per_channel={per_channel} n={len(mg)} "
                  f"pearson={r:+.3f} (p={p:.3f}) spearman={rs:+.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "screen_sens_predictor_sens.csv"),
                              index=False)


def q10_threshold():
    """Precision/recall of the screen used as a go / no-go rule."""
    surv = pd.read_csv(os.path.join(OUT, "reducibility_survey.csv"))
    g = (surv.groupby(["panel", "split"])
         .agg(red=("reducible_gap_pct", "mean"), spec=("spec_div", "first"))
         .reset_index())
    g = g[g.split != "random control"]
    rows = []
    for thr_red in [3.0, 5.0]:
        pos = g.red >= thr_red
        for q in np.linspace(0.1, 0.9, 9):
            t = g.spec.quantile(q)
            pred = g.spec >= t
            tp = int((pred & pos).sum()); fp = int((pred & ~pos).sum())
            fn = int((~pred & pos).sum())
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            rows.append(dict(headroom_threshold=thr_red, screen_quantile=round(q, 2),
                             screen_value=t, precision=prec, recall=rec, f1=f1,
                             n_flagged=int(pred.sum())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "screen_sens_threshold.csv"), index=False)
    for thr in df.headroom_threshold.unique():
        b = df[df.headroom_threshold == thr].sort_values("f1", ascending=False).iloc[0]
        print(f"  headroom>={thr}%: best screen at quantile {b.screen_quantile} "
              f"(value {b.screen_value:.4f}) precision={b.precision:.2f} "
              f"recall={b.recall:.2f} F1={b.f1:.2f}", flush=True)


def q6_spectral_evidence():
    """Does a source-trained model fail to emit the target's rhythm, and does the
    periodicity branch restore it? Compare weekly spectral power of predictions
    against the realized target."""
    P = Panel(os.path.join(ROOT, "cache", "electricity", "panel_elec.npz"),
              n_facilities=10 ** 6, L=56, H=14, stride=18, seed=0, n_channels=3,
              boundary_date=20140101)
    grp = np.asarray(P.state).astype(str); idx = np.arange(P.nF)
    src, tgt = idx[grp == "Flat"], idx[grp == "WorkWeek"]
    nv = max(6, int(0.15 * len(src)))
    rng = np.random.default_rng(0); s = src.copy(); rng.shuffle(s)
    Xw, Yw, fid = P.eval_arrays(tgt)

    def weekly_power(A):
        c = A - A.mean(-1, keepdims=True)
        F = np.abs(np.fft.rfft(c, axis=-1))
        fr = np.fft.rfftfreq(A.shape[-1], 1.0)
        k = int(np.argmin(abs(fr - 1 / 7.0)))
        return float((F[..., k] / (np.linalg.norm(F, axis=-1) + 1e-9)).mean())

    import torch
    rows = []
    for name, cfg in [("ERM-MLP", dict(method="erm", epochs=24, patience=5,
                                       backbone="mlp")),
                      ("FADE (gate only)", dict(method="cdisgts", epochs=24,
                                                patience=5, backbone="mlp",
                                                lam_inv=0.5, gate_lr=0.01)),
                      ("FADE (both mechanisms)", dict(method="cdisgts", epochs=24,
                                                      patience=5, backbone="mlp",
                                                      lam_inv=0.5, gate_lr=0.05,
                                                      periodic=True))]:
        m, _ = train_model(P, cfg, s[nv:], s[:nv], seed=0)
        m.eval()
        preds = np.zeros_like(Yw)
        with torch.no_grad():
            for i in range(0, len(Xw), 1024):
                preds[i:i + 1024] = m(torch.from_numpy(Xw[i:i + 1024])).numpy()
        rows.append(dict(model=name, weekly_power_pred=weekly_power(preds.sum(1))))
        print(f"  {name:24s} weekly power of forecast = "
              f"{rows[-1]['weekly_power_pred']:.4f}", flush=True)
    rows.append(dict(model="Realized target", weekly_power_pred=weekly_power(Yw.sum(1))))
    print(f"  {'Realized target':24s} weekly power of target   = "
          f"{rows[-1]['weekly_power_pred']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "screen_sens_spectral.csv"), index=False)


if __name__ == "__main__":
    print("=== S1: predictor specification sensitivity ===", flush=True)
    q3_predictor_sensitivity()
    print("\n=== S3: screening threshold calibration ===", flush=True)
    q10_threshold()
    print("\n=== S2: spectral evidence for the hypothesis-class limit ===", flush=True)
    q6_spectral_evidence()
    print("\nSaved outputs/screen_sens_*.csv")
