"""Run the smaller supplementary experiments (src/) under the conventions of the
experiment scripts, without editing their source files.

  python experiments/supplementary_runs.py <mae_cat|mae_total> <module> [function ...]

Three patches are applied before the module is imported:
  1. METRIC. src.train.evaluate is wrapped so that the key the src/ scripts
     read ("mae_total") holds the requested endpoint; the original channel-summed
     value stays available as "mae_total_orig".
  2. MODEL SELECTION IN TEMPORAL RUNS. Whenever a training mask is supplied,
     early stopping uses the same (observed-era) mask; several src/ scripts
     passed the deployment-era mask as the validation mask.
  3. OUTPUT LOCATION. Every CSV/NPZ/figure the module writes under outputs/,
     cache/ or figures/ is redirected to experiments/results/supplementary_<metric>/,
     so no other result file is overwritten.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
metric, modname, funcs = sys.argv[1], sys.argv[2], sys.argv[3:]
DEST = os.path.join(HERE, "results", f"supplementary_{metric}")
os.makedirs(DEST, exist_ok=True)

import src.train as T                                   # noqa: E402

_eval, _train = T.evaluate, T.train_model


def evaluate(*a, **k):
    rows = _eval(*a, **k)
    for r in rows:
        r["mae_total_orig"] = r["mae_total"]
        r["mae_total"] = r[metric] if metric != "mae_total" else r["mae_total_orig"]
    return rows


def train_model(panel, cfg, train_fac, val_fac, seed=0, verbose=False,
                train_mask=None, val_mask=None):
    if train_mask is not None:
        val_mask = train_mask
    return _train(panel, cfg, train_fac, val_fac, seed=seed, verbose=verbose,
                  train_mask=train_mask, val_mask=val_mask)


T.evaluate, T.train_model = evaluate, train_model


def redirect(path):
    """Only WRITES pass through here, so every path under these folders moves."""
    p = os.path.abspath(str(path))
    for sub in ("outputs", "cache", "figures"):
        if p.startswith(os.path.join(ROOT, sub) + os.sep):
            return os.path.join(DEST, os.path.basename(p))
    return path


_csv = pd.DataFrame.to_csv
pd.DataFrame.to_csv = lambda self, path=None, *a, **k: _csv(self, redirect(path), *a, **k)
_savez = np.savez
np.savez = lambda path, *a, **k: _savez(redirect(path), *a, **k)
try:
    import matplotlib.figure as mf
    _sf = mf.Figure.savefig
    mf.Figure.savefig = lambda self, path, *a, **k: _sf(self, redirect(path), *a, **k)
except Exception:
    pass

mod = importlib.import_module(modname)
if funcs:
    for f in funcs:
        getattr(mod, f)()
elif hasattr(mod, "main"):
    mod.main()
print("done", flush=True)
