"""Train ERM and CD-ISG-TS once (realistic LFO-T protocol) and save 14-day
forecast examples for several unseen test facilities -> cache/examples.npz,
for the qualitative forecast figure."""
import numpy as np, torch, os
from src.data import Panel
from src.train import train_model

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache")
P = Panel(os.path.join(CACHE, "panel.npz"), n_facilities=1500, L=56, H=14, stride=12, seed=0)
TM, VM = P.is_2024, P.is_2025
tr, va, te = P.split_facilities(0)

def predict(model, idxs):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(P.Xw[idxs])).numpy()

erm, _ = train_model(P, dict(method="erm", epochs=28, patience=7, backbone="mlp"),
                     tr, va, seed=0, train_mask=TM, val_mask=VM)
ours, _ = train_model(P, dict(method="cdisgts", epochs=28, patience=7, backbone="mlp",
                              lam_inv=0.5, gate_lr=0.01), tr, va, seed=0,
                      train_mask=TM, val_mask=VM)

# choose test facilities spanning the error range
per = {}
for f in te:
    idxs = P._pool(f, VM)
    if len(idxs) < 3:
        continue
    p = predict(erm, idxs); y = P.Yw[idxs]
    per[f] = np.abs(p.sum(1) - y.sum(1)).mean()
fs = sorted(per, key=per.get)
picks = [fs[int(q * (len(fs) - 1))] for q in [0.15, 0.4, 0.6, 0.8, 0.9, 0.97]]

ex = {}
dates = P.dates
for f in picks:
    idxs = P._pool(f, VM)
    # pick a mid window
    j = idxs[len(idxs) // 2]
    x = P.Xw[j]                       # [3,L]
    y = P.Yw[j]                       # [3,H]
    pe = predict(erm, np.array([j]))[0]
    po = predict(ours, np.array([j]))[0]
    st = int(P.wstart[j])
    ex[f"fac{f}_state"] = str(P.state[f])
    ex[f"fac{f}_medcen"] = float(P.medcen[f])
    ex[f"fac{f}_x"] = x
    ex[f"fac{f}_y"] = y
    ex[f"fac{f}_erm"] = pe
    ex[f"fac{f}_ours"] = po
    ex[f"fac{f}_dates"] = dates[st:st + P.span]
ex["picks"] = np.array(picks)
np.savez(os.path.join(CACHE, "examples.npz"), **ex)
print("saved examples for facilities", picks)
