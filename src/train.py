"""
Training / evaluation engine for the leave-facilities-out DG protocol.

Handles all methods through one config:
  erm        : RevIN + backbone (mandatory baseline to beat)
  cdisgts    : ours -- learnable category-conditioned sub-band gate + invariance reg
  fixedband  : fixed low-pass heuristic (domain-specific == high-freq)
  wavemask   : Wave-Mask/Mix input-level wavelet augmentation on ERM
  irm        : ERM + IRMv1 invariant-learning penalty across facilities in a batch

Model selection uses source-val only (no target access). Reports per-target-facility
MAE/RMSE/sMAPE for each staff category and the total, plus gate diagnostics.
"""
import numpy as np, torch, torch.nn as nn, os
from src.models import (ERM, CDISGTS, FADECalib, PACE, SubBandGate, FixedBand, DLinearPlain,
                        AutoformerLite, FEDformerLite, invariance_penalty,
                        build_wavelet_matrix)
from src.baselines_extra import (AutoformerFull, FEDformerFull, DomainDiscriminator,
                                 grad_reverse, coral_penalty)

torch.set_num_threads(int(os.environ.get("EXP_THREADS", max(1, os.cpu_count() - 2))))

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)

def build_model(cfg, L, H, C):
    m = cfg["method"]; bb = cfg.get("backbone", "mlp")
    stem = cfg.get("use_stem", True)
    if m in ("erm", "wavemask", "irm", "coral", "dann", "groupdro", "dro_heuristic"):
        return ERM(L, H, C, backbone=bb, use_stem=stem)
    if m == "dlinear_plain":
        return DLinearPlain(L, H, C)
    if m == "autoformer":
        return AutoformerFull(L, H, C)
    if m == "fedformer":
        return FEDformerFull(L, H, C)
    if m == "autoformer_lite":
        return AutoformerLite(L, H, C)
    if m == "fedformer_lite":
        return FEDformerLite(L, H, C)
    if m == "fixedband":
        return FixedBand(L, H, C, backbone=bb, wavelet=cfg.get("wavelet", "db4"),
                         level=cfg.get("level", 3), drop_fine=cfg.get("drop_fine", 1),
                         use_stem=stem)
    if m == "pace":
        g = None
        if cfg.get("with_gate", False):
            g = SubBandGate(L, C, cfg.get("wavelet", "db4"), cfg.get("level", 3),
                            adaptive=cfg.get("adaptive", False))
        return PACE(L, H, C, backbone=bb, period=cfg.get("period", 7),
                    use_stem=stem, gate=g)
    if m in ("cdisgts", "fade_calib"):
        cls = FADECalib if m == "fade_calib" else CDISGTS
        extra = dict(shrink_only=cfg.get("shrink_only", False)) \
            if m == "fade_calib" else {}
        return cls(L, H, C, backbone=bb, **extra, wavelet=cfg.get("wavelet", "db4"),
                       level=cfg.get("level", 3), marginal=cfg.get("marginal", False),
                       use_fft=cfg.get("use_fft", False),
                       gate_init=cfg.get("gate_init", 2.5), use_stem=stem,
                       adaptive=cfg.get("adaptive", False),
                       periodic=cfg.get("periodic", False),
                       period=cfg.get("period", 7),
                       aug_strength=cfg.get("aug_strength", None))
    raise ValueError(m)

# ---- Wave-Mask/Mix augmentation (Arabi et al., 2024), input-level ------------
class WaveAug:
    def __init__(self, L, wavelet="db4", level=3, p=0.5, mask_ratio=0.15):
        W, band_id, _ = build_wavelet_matrix(L, wavelet, level)
        self.W = torch.from_numpy(W); self.band_id = torch.from_numpy(band_id)
        self.p = p; self.mask_ratio = mask_ratio
    def __call__(self, x):                       # x [B,C,L]
        if torch.rand(()) > self.p:
            return x
        coeff = torch.einsum("nl,bcl->bcn", self.W, x)       # [B,C,n]
        B, C, n = coeff.shape
        # Wave-Mask: randomly zero a fraction of detail coefficients
        mask = (torch.rand(B, C, n) > self.mask_ratio).float()
        # Wave-Mix: blend with a shuffled sample's coefficients
        perm = torch.randperm(B)
        lam = 0.5
        mixed = torch.where(torch.rand(B, C, n) < 0.5 * self.p, coeff[perm], coeff)
        coeff = (lam * (coeff * mask) + (1 - lam) * mixed)
        return torch.einsum("nl,bcn->bcl", self.W, coeff)

def irm_penalty(pred, y, fac_ids, crit):
    """IRMv1 penalty: variance of the per-facility gradient wrt a dummy scale."""
    scale = torch.ones((), requires_grad=True, device=pred.device)
    pen = 0.0; nu = 0
    for f in torch.unique(fac_ids):
        m = fac_ids == f
        loss_f = crit(pred[m] * scale, y[m])
        g = torch.autograd.grad(loss_f, scale, create_graph=True)[0]
        pen = pen + g.pow(2); nu += 1
    return pen / max(nu, 1)

def evaluate(model, panel, fac_subset, batch=1024, time_mask=None):
    model.eval()
    Xw, Yw, fid = panel.eval_arrays(fac_subset, time_mask=time_mask)
    preds = np.zeros_like(Yw)
    with torch.no_grad():
        for i in range(0, len(Xw), batch):
            xb = torch.from_numpy(Xw[i:i+batch])
            preds[i:i+batch] = model(xb).numpy()
    # per-facility metrics
    rows = []
    for f in np.unique(fid):
        m = fid == f
        p = preds[m]; y = Yw[m]                         # [nw,3,H]
        ae = np.abs(p - y)
        pt = p.sum(1); yt = y.sum(1)                    # total across categories
        smape = np.mean(2*np.abs(pt-yt)/(np.abs(pt)+np.abs(yt)+1e-6))
        # PRIMARY endpoint: category-averaged MAE, the mean absolute
        # error over channels and horizon, which is what the paper's equation
        # defines and what training (L1 over all channels) and model selection use.
        # SECONDARY endpoint: total-staffing error on the channel-summed series,
        # in which an over-forecast in one channel can offset an under-forecast in
        # another. Both are always recorded.
        smape_c = np.mean(2*ae/(np.abs(p)+np.abs(y)+1e-6))
        rows.append(dict(fac=int(f),
                         mae_RN=ae[:,0].mean(), mae_LPN=ae[:,1].mean(),
                         mae_CNA=ae[:,2].mean(),
                         mae_total=np.abs(pt-yt).mean(),
                         rmse_total=np.sqrt(((pt-yt)**2).mean()),
                         smape_total=smape,
                         mae_cat=ae.mean(),
                         rmse_cat=np.sqrt((ae**2).mean()),
                         smape_cat=smape_c))
    return rows

def train_model(panel, cfg, train_fac, val_fac, seed=0, verbose=False,
                train_mask=None, val_mask=None):
    set_seed(seed)
    L, H, C = panel.L, panel.H, 3
    model = build_model(cfg, L, H, C)
    # Optional warm start (fine-tuning experiments). Absent by default, so every
    # other configuration is unaffected.
    if cfg.get("init_state") is not None:
        model.load_state_dict(cfg["init_state"])
    # gate parameters get a higher LR and NO weight decay so the gate can move out
    # of the sigmoid's flat region and strongly suppress domain-variant sub-bands;
    # weight decay would pull all gates uniformly toward 0.5 (non-selective).
    # The input-conditioned band_map belongs with the gate, not the backbone: it is
    # zero-initialized, so the backbone's weight decay would actively hold it at
    # zero, and the backbone learning rate is far too small to move it against the
    # gate logits. Grouping it here (large step, no weight decay) is what lets the
    # adaptive gate train at all.
    def _is_gate(n):
        return (("gate_logit" in n) or ("band_map" in n) or ("blend_" in n)
                or ("shrink_" in n))
    gate_p = [p for n, p in model.named_parameters() if _is_gate(n)]
    rest_p = [p for n, p in model.named_parameters() if not _is_gate(n)]
    groups = [dict(params=rest_p, lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-4))]
    if gate_p:
        groups.append(dict(params=gate_p, lr=cfg.get("gate_lr", 0.05), weight_decay=0.0))
    opt = torch.optim.Adam(groups)
    crit = nn.L1Loss()
    aug = WaveAug(L, cfg.get("wavelet", "db4"), cfg.get("level", 3)) \
        if cfg["method"] == "wavemask" else None
    G, K = cfg.get("G", 32), cfg.get("K", 8)
    epochs = cfg.get("epochs", 20)
    lam = cfg.get("lam_inv", 0.0); lam_irm = cfg.get("lam_irm", 0.0)
    lam_coral = cfg.get("lam_coral", 1.0); lam_dann = cfg.get("lam_dann", 1.0)
    disc = None
    # Optional coarse training domains. domain_map[e] gives the domain of entity e;
    # absent, every training entity is its own domain.
    dmap = None
    if cfg.get("domain_map") is not None:
        train_arr0 = np.asarray(train_fac)
        raw = np.asarray(cfg["domain_map"])[train_arr0]
        _, contig = np.unique(raw, return_inverse=True)
        lut = np.full(panel.nF, -1, np.int64)
        lut[train_arr0] = contig
        dmap = torch.from_numpy(lut)
        n_dom = int(contig.max()) + 1
    if cfg["method"] == "dann":
        # Each training facility keeps ONE stable domain label for the whole run
        # (ids renumbered inside every batch would make output k mean a different
        # facility in every batch, leaving the discriminator no consistent task). The discriminator classifies over all training domains.
        train_arr = np.asarray(train_fac)
        dom_lut = np.full(panel.nF, -1, np.int64)
        dom_lut[train_arr] = np.arange(len(train_arr))
        dom_lut = torch.from_numpy(dom_lut)
        disc = DomainDiscriminator(C * L, len(train_arr) if dmap is None else n_dom)
        opt.add_param_group({"params": disc.parameters(), "lr": cfg.get("lr", 1e-3)})
    if cfg["method"] == "dro_heuristic":
        cfg = {**cfg, "dro": True, "dro_tau": cfg.get("dro_tau", 0.1)}
    dro_q = None
    if cfg["method"] == "groupdro":
        # Faithful online GroupDRO (Sagawa et al., 2020, Alg. 1; the
        # authors' loss.py). A persistent weight q_g per group (group = training
        # facility) is updated multiplicatively, q_g <- q_g exp(eta * loss_g), and
        # renormalized over ALL groups; the training loss is sum_g q_g loss_g with
        # absent groups contributing zero, exactly as in the reference code.
        train_arr = np.asarray(train_fac)
        dro_lut = np.full(panel.nF, -1, np.int64)
        dro_lut[train_arr] = np.arange(len(train_arr))
        dro_lut = torch.from_numpy(dro_lut)
        n_g = len(train_arr) if dmap is None else n_dom
        dro_q = torch.ones(n_g) / n_g
        dro_eta = cfg.get("dro_eta", 0.01)
    best_val, best_state, best_ep, patience = 1e9, None, -1, 0
    if cfg.get("init_state") is not None:
        # a warm-started model is itself a candidate: fine-tuning may stop at epoch
        # -1 and return the starting model unchanged if no epoch improves on it
        vr0 = evaluate(model, panel, val_fac, time_mask=val_mask)
        best_val = float(np.mean([r["mae_cat"] for r in vr0]))
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    warmup = cfg.get("warmup_frac", 0.4)
    diag = {}
    for ep in range(epochs):
        # ramp the invariance penalty in only after the representation has settled,
        # so the gate does not over-suppress before useful features exist.
        ramp = 0.0 if ep < warmup * epochs else min(1.0, (ep - warmup * epochs) /
                                                     max(1.0, 0.3 * epochs))
        lam_ep = lam * ramp; lam_irm_ep = lam_irm * ramp
        model.train()
        for Xb, Yb, fb in panel.grouped_batches(train_fac, G, K, seed=seed*97+ep,
                                                time_mask=train_mask):
            xb = torch.from_numpy(Xb); yb = torch.from_numpy(Yb)
            fbi = torch.from_numpy(fb)
            dbi = fbi if dmap is None else dmap[fbi]
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad()
            if cfg["method"] == "cdisgts" and lam_ep > 0:
                pred, rel = model(xb, return_energy=True)
                loss = crit(pred, yb) + lam_ep * invariance_penalty(rel, fbi)
            elif cfg["method"] == "irm" and lam_irm_ep > 0:
                pred = model(xb)
                loss = crit(pred, yb) + lam_irm_ep * irm_penalty(pred, yb, dbi, crit)
            elif cfg["method"] == "coral":
                pred = model(xb)
                feats = model.encode(xb)
                loss = crit(pred, yb) + lam_coral * ramp * coral_penalty(feats, dbi)
            elif cfg["method"] == "dann":
                pred = model(xb)
                feats = model.encode(xb)
                dlab = dom_lut[fbi] if dmap is None else dbi                       # stable domain labels
                dlogits = disc(grad_reverse(feats, lamb=ramp))
                loss = crit(pred, yb) + lam_dann * nn.functional.cross_entropy(
                    dlogits, dlab)
            elif cfg["method"] == "groupdro":
                pred = model(xb)
                ae_s = (pred - yb).abs().mean(dim=(1, 2))      # per-sample L1
                gid = dro_lut[fbi] if dmap is None else dbi
                ug, inv = torch.unique(gid, return_inverse=True)
                gl = torch.zeros(len(ug)).index_add_(0, inv, ae_s) /                     torch.bincount(inv, minlength=len(ug)).float()
                with torch.no_grad():
                    dro_q[ug] = dro_q[ug] * torch.exp(dro_eta * gl.detach())
                    dro_q.div_(dro_q.sum())
                loss = (dro_q[ug] * gl).sum()
            else:
                pred = model(xb)
                loss = crit(pred, yb)
            # batch-softmax reweighting heuristic (NOT GroupDRO), method "dro_heuristic";
            # reported under its own name as an additional baseline.
            if cfg.get("dro", False):
                tau = cfg.get("dro_tau", 0.05)
                ae = (pred - yb).abs().mean(dim=(1, 2))          # per-sample loss
                uf = torch.unique(fbi)
                fl = torch.stack([ae[fbi == f].mean() for f in uf])
                w = torch.softmax(fl.detach() / tau, 0)
                loss = loss + (w * fl).sum() - fl.mean()         # add DRO term (mean-centered)
            loss.backward(); opt.step()
        # ---- source-val model selection ----
        # Optional: select only among checkpoints trained with the
        # method's penalty switched on, i.e. after the warm-up. Without this,
        # early stopping can return a pre-penalty checkpoint, which on the full
        # nursing panel made CORAL and DANN identical truncated ERM runs.
        if cfg.get("select_after_warmup", False) and ep < warmup * epochs:
            continue
        vr = evaluate(model, panel, val_fac, time_mask=val_mask)
        vmae = np.mean([r["mae_cat"] for r in vr])
        if vmae < best_val - 1e-5:
            best_val, best_state, best_ep = vmae, {k: v.detach().clone()
                                                   for k, v in model.state_dict().items()}, ep
            patience = 0
        else:
            patience += 1
            if patience >= cfg.get("patience", 5):
                break
        if verbose:
            print(f"   ep{ep:02d} val_mae={vmae:.4f} best={best_val:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    # gate diagnostics (retained energy per band) for collapse check + interpretability
    if cfg["method"] == "cdisgts":
        with torch.no_grad():
            g = model.gate.gates().cpu().numpy()          # [C, n_bands]
        diag["gates"] = g
        diag["best_epoch"] = best_ep
    diag["val_mae"] = best_val
    return model, diag
