"""
Ablation & sensitivity studies (RO6 / Step 8), on the realistic LFO-T protocol
(leave-facilities-out, train 2024 -> deploy 2025).

Covers:
  A. invariance-weight lambda sensitivity + retained sub-band energy (collapse
     diagnostic, caution #2)
  B. category-conditioned gate vs marginal (RQ2)
  C. gate placement: on the intermediate FEATURE vs on the raw INPUT (design check)
  D. wavelet basis and decomposition depth
  E. localized wavelet gate vs global-FFT learnable mask
  F. backbone transfer: MLP vs Transformer, ERM vs CD-ISG-TS (RQ4)
  G. computational overhead vs ERM

Outputs: outputs/ablation_*.csv, cache/lambda_gates.npz
"""
import numpy as np, pandas as pd, os, time, torch
from src.data import Panel
from src.train import train_model, evaluate

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs"); CACHE = os.path.join(ROOT, "cache")
NFAC, L, H, STRIDE, SEEDS = 1200, 56, 14, 12, [0, 1]
EP, PAT = 26, 6

P = Panel(os.path.join(CACHE, "panel.npz"), n_facilities=NFAC, L=L, H=H, stride=STRIDE, seed=0)
TM, VM = P.is_2024, P.is_2025

def run(cfg, seed):
    tr, va, te = P.split_facilities(seed)
    m, d = train_model(P, cfg, tr, va, seed=seed, train_mask=TM, val_mask=VM)
    res = evaluate(m, P, te, time_mask=VM)
    return np.array([x["mae_total"] for x in res]), d, m

def mean_over_seeds(cfg):
    vals = []; diags = []
    for s in SEEDS:
        r, d, _ = run(cfg, s); vals.append(r.mean()); diags.append(d)
    return np.mean(vals), np.std(vals), diags

def band_retained_energy(gates):
    """gates [C,nbands] energy-preserving -> retained relative energy per band."""
    blen = np.array([7, 7, 14, 28.0])  # A3,D3,D2,D1 coefficient counts for L=56, db, level3
    e = (gates ** 2) * blen[None]
    return (e / e.sum(1, keepdims=True)).mean(0)

def main():
    rows_lam, rows_gen, rows_cat, rows_place, rows_wav, rows_fft, rows_bb, rows_cost = \
        [], [], [], [], [], [], [], []
    lam_gates = {}

    # ---- baseline references ----
    erm_m, erm_s, _ = mean_over_seeds(dict(method="erm", epochs=EP, patience=PAT, backbone="mlp"))
    print(f"ERM-MLP  {erm_m:.4f}+-{erm_s:.4f}", flush=True)

    # A. lambda sensitivity + retained energy -------------------------------
    for lam in [0.0, 0.1, 0.5, 2.0, 8.0, 30.0, 100.0]:
        vals, gts = [], []
        for s in SEEDS:
            r, d, _ = run(dict(method="cdisgts", epochs=EP, patience=PAT, backbone="mlp",
                               lam_inv=lam, gate_lr=0.01), s)
            vals.append(r.mean()); gts.append(d["gates"])
        g = np.mean(gts, 0)
        ret = band_retained_energy(g)
        rows_lam.append(dict(lam=lam, mae=np.mean(vals), mae_std=np.std(vals),
                             ret_A3=ret[0], ret_D3=ret[1], ret_D2=ret[2], ret_D1=ret[3],
                             gate_min=g.min(), gate_max=g.max()))
        lam_gates[f"lam{lam}"] = g
        print(f"  lam={lam:6g} mae={np.mean(vals):.4f} retained[A3,D3,D2,D1]={np.round(ret,3)}",
              flush=True)

    # B. category-conditioned vs marginal -----------------------------------
    for marg in [False, True]:
        m_, s_, _ = mean_over_seeds(dict(method="cdisgts", epochs=EP, patience=PAT,
                                         backbone="mlp", lam_inv=0.5, gate_lr=0.01, marginal=marg))
        rows_cat.append(dict(variant="marginal" if marg else "category-conditioned",
                             mae=m_, mae_std=s_))
        print(f"  cat-cond={'marginal' if marg else 'per-category'} mae={m_:.4f}", flush=True)

    # C. gate placement: feature (stem) vs raw input ------------------------
    for stem in [True, False]:
        m_, s_, _ = mean_over_seeds(dict(method="cdisgts", epochs=EP, patience=PAT,
                                         backbone="mlp", lam_inv=0.5, gate_lr=0.01, use_stem=stem))
        rows_place.append(dict(placement="feature" if stem else "raw-input",
                               mae=m_, mae_std=s_))
        print(f"  placement={'feature' if stem else 'raw-input'} mae={m_:.4f}", flush=True)

    # D. wavelet basis & depth ----------------------------------------------
    for wav in ["haar", "db2", "db4", "sym4"]:
        for lvl in [2, 3]:
            try:
                m_, s_, _ = mean_over_seeds(dict(method="cdisgts", epochs=EP, patience=PAT,
                                                 backbone="mlp", lam_inv=0.5, gate_lr=0.01,
                                                 wavelet=wav, level=lvl))
                rows_wav.append(dict(wavelet=wav, level=lvl, mae=m_, mae_std=s_))
                print(f"  wavelet={wav} level={lvl} mae={m_:.4f}", flush=True)
            except Exception as e:
                print(f"  wavelet={wav} level={lvl} FAILED {e}", flush=True)

    # E. wavelet gate vs global-FFT mask ------------------------------------
    for fft in [False, True]:
        m_, s_, _ = mean_over_seeds(dict(method="cdisgts", epochs=EP, patience=PAT,
                                         backbone="mlp", lam_inv=0.5, gate_lr=0.01, use_fft=fft))
        rows_fft.append(dict(gate="global-FFT" if fft else "wavelet-subband",
                             mae=m_, mae_std=s_))
        print(f"  gate={'FFT' if fft else 'wavelet'} mae={m_:.4f}", flush=True)

    # F. backbone transfer: MLP vs Transformer, ERM vs ours -----------------
    for bb in ["mlp", "transformer"]:
        ep = EP if bb == "mlp" else 18
        em, es, _ = mean_over_seeds(dict(method="erm", epochs=ep, patience=5, backbone=bb))
        om, os_, _ = mean_over_seeds(dict(method="cdisgts", epochs=ep, patience=5, backbone=bb,
                                          lam_inv=0.5, gate_lr=0.01))
        rows_bb.append(dict(backbone=bb, ERM=em, ERM_std=es, CD_ISG_TS=om, CD_ISG_TS_std=os_))
        print(f"  backbone={bb} ERM={em:.4f} ours={om:.4f}", flush=True)

    # G. computational overhead ---------------------------------------------
    for label, cfg in [("ERM-MLP", dict(method="erm", epochs=3, patience=3, backbone="mlp")),
                       ("CD-ISG-TS", dict(method="cdisgts", epochs=3, patience=3, backbone="mlp",
                                          lam_inv=0.5, gate_lr=0.01))]:
        tr, va, te = P.split_facilities(0)
        t0 = time.time()
        _, _, model = run(cfg, 0)
        train_t = time.time() - t0
        nparam = sum(p.numel() for p in model.parameters())
        Xw, _, _ = P.eval_arrays(te[:50], time_mask=VM)
        xb = torch.from_numpy(Xw[:512])
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(5):
                model(xb)
        inf_t = (time.time() - t0) / 5
        rows_cost.append(dict(method=label, n_params=nparam,
                              train_time_3ep_s=train_t, infer_512_s=inf_t))
        print(f"  cost {label} params={nparam} infer512={inf_t*1000:.1f}ms", flush=True)

    pd.DataFrame(rows_lam).to_csv(os.path.join(OUT, "ablation_lambda.csv"), index=False)
    pd.DataFrame(rows_cat).to_csv(os.path.join(OUT, "ablation_category.csv"), index=False)
    pd.DataFrame(rows_place).to_csv(os.path.join(OUT, "ablation_placement.csv"), index=False)
    pd.DataFrame(rows_wav).to_csv(os.path.join(OUT, "ablation_wavelet.csv"), index=False)
    pd.DataFrame(rows_fft).to_csv(os.path.join(OUT, "ablation_fft.csv"), index=False)
    pd.DataFrame(rows_bb).to_csv(os.path.join(OUT, "ablation_backbone.csv"), index=False)
    pd.DataFrame(rows_cost).to_csv(os.path.join(OUT, "ablation_cost.csv"), index=False)
    pd.DataFrame([dict(method="ERM-MLP", mae=erm_m, mae_std=erm_s)]).to_csv(
        os.path.join(OUT, "ablation_ermref.csv"), index=False)
    np.savez(os.path.join(CACHE, "lambda_gates.npz"), **lam_gates)
    print("Saved ablation_*.csv", flush=True)

if __name__ == "__main__":
    main()
