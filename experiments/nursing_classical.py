"""Persistence and Holt-Winters forecasts for the nursing anchor (ownership split).

anchor_actions.py combines the model with seasonal naive, persistence and
Holt-Winters and reads the classical forecasts from classical_cache_<panel>.pkl.
For nursing that cache is built here for exactly the entities the anchor needs
(target test, source early-stopping and target early-stopping entities of seeds
0, 1, 2), with the forecasters and fallback rule of classical.py. Only the two
forecasters used by anchor_actions.py are computed.
Output: results/classical_cache_Nursing.pkl
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, allocate  # noqa: E402
from anchor import anchors  # noqa: E402
from classical import H, _one, _snaive  # noqa: E402
from joint_finetune import KEEP  # noqa: E402

METHODS = ["Persistence (last value)", "Holt-Winters, additive weekly season"]


def forecasts(args):
    ent, X = args
    warnings.filterwarnings("ignore")
    out = {m: np.zeros((X.shape[0], X.shape[1], H), np.float32) for m in METHODS}
    fb = 0
    for w in range(X.shape[0]):
        for c in range(X.shape[1]):
            y = X[w, c].astype(float)
            for m in METHODS:
                try:
                    f = np.asarray(_one(m, y), float)
                    if f.shape != (H,) or not np.all(np.isfinite(f)):
                        raise ValueError("non-finite")
                except Exception:
                    f = _snaive(y); fb += 1
                out[m][w, c] = f
    return ent, out, fb


def main():
    path = os.path.join(RES, "classical_cache_Nursing.pkl")
    cache = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
    for panel, P, split, S, T, floors, fracs, seeds in anchors(["Nursing"]):
        if (panel, split) not in KEEP:
            continue
        need = set()
        for s in seeds:
            al = allocate(S, T, np.random.default_rng(s), floors=floors, fracs=fracs, fallback=False)
            for k in ("t_te", "s_va", "t_va"):
                need.update(int(e) for e in al[k])
        todo = sorted(e for e in need if e not in cache)
        print(f"{len(need)} entities needed, {len(todo)} to compute", flush=True)
        t0, fbs = time.time(), 0
        with Pool(int(os.environ.get("EXP_WORKERS", "4"))) as pool:
            for i, (e, out, fb) in enumerate(pool.imap_unordered(
                    forecasts, [(e, P.eval_arrays([e])[0]) for e in todo], chunksize=4)):
                cache[e] = out; fbs += fb
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
                    pickle.dump(cache, open(path, "wb"))
        pickle.dump(cache, open(path, "wb"))
        print(f"done: {len(cache)} entities, {fbs} fallbacks to seasonal naive, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
