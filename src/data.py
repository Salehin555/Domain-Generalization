"""
Windowing, domain-generalization splits, and facility-grouped batching.

Forecasting task (RO5): from a lookback of L=56 days of per-resident-day staffing
for the 3 staff categories [RN, LPN, CNA], predict the next H=14 days.

Splits
------
leave-facilities-out : disjoint train / source-val / target-test facility sets;
                       model selection uses source-val ONLY (training-domain
                       validation, no target access).
leave-state-out      : entire states held out for test (coarser domain granularity).

Facility-grouped batches: each batch draws G facilities x K windows so the
cross-facility invariance penalty is well estimated within every batch.
"""
import numpy as np

CATS = ["RN", "LPN", "CNA"]

class Panel:
    def __init__(self, path, n_facilities=2000, L=56, H=14, stride=12, seed=0,
                 n_channels=3, boundary_date=20250101):
        """n_channels and boundary_date are parameterized so the same pipeline can
        be run on a second panel dataset (see src/build_electricity.py). The
        defaults reproduce the CMS nurse-staffing setup exactly."""
        d = np.load(path, allow_pickle=True)
        C = n_channels
        X = d["X"][:, :, :C].astype(np.float32)          # e.g. RN,LPN,CNA (HPRD)
        state = d["state"]; medcen = d["medcensus"]; prov = d["provnum"]
        nF = X.shape[0]
        rng = np.random.default_rng(seed)

        # stratified sample of the working set: by size tertile x a spread of states
        size_bin = np.digitize(medcen, np.quantile(medcen, [1/3, 2/3]))
        sel = []
        per = n_facilities
        idx_all = np.arange(nF)
        # proportional sample across (state,size) strata to keep diversity
        order = rng.permutation(nF)
        sel = order[:min(n_facilities, nF)]
        self.fac = np.sort(sel)
        X = X[self.fac]; state = state[self.fac]; medcen = medcen[self.fac]; prov = prov[self.fac]

        self.L, self.H = L, H
        self.state = state; self.medcen = medcen; self.prov = prov
        self.nF = len(self.fac)

        # build windows
        span = L + H
        starts = np.arange(0, X.shape[1] - span + 1, stride)
        nw = len(starts); N = self.nF * nw
        # preallocate (memory-safe at 13k+ facilities) and fill
        self.C = C
        self.Xw = np.empty((N, C, L), np.float32)
        self.Yw = np.empty((N, C, H), np.float32)
        self.fid = np.empty(N, np.int64)
        self.wstart = np.empty(N, np.int64)
        p = 0
        for i in range(self.nF):
            s = X[i]                                     # [T,3]
            for st in starts:
                self.Xw[p] = s[st:st + L].T
                self.Yw[p] = s[st + L:st + span].T
                self.fid[p] = i; self.wstart[p] = st
                p += 1
        self.span = span
        self.dates = d["dates"]
        # per-facility window index lists
        self.by_fac = [np.where(self.fid == i)[0] for i in range(self.nF)]
        # windows fully inside 2024 (train) vs fully inside 2025 (deploy)
        # dates are YYYYMMDD ints; 2025 starts at the first index with date>=20250101
        boundary = int(np.searchsorted(self.dates, boundary_date))
        self.is_2024 = (self.wstart + span) <= boundary
        self.is_2025 = self.wstart >= boundary

    # ---- splits over facilities --------------------------------------------
    def split_facilities(self, seed=0, fracs=(0.60, 0.15, 0.25)):
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(self.nF)
        n1 = int(fracs[0] * self.nF); n2 = int((fracs[0] + fracs[1]) * self.nF)
        return perm[:n1], perm[n1:n2], perm[n2:]         # train, val, test (facility idx)

    def split_by_size(self, seed=0):
        """Leave-size-out: train on SMALL+MEDIUM facilities, test on LARGE facilities
        (bed count is a factor known to correlate with real staffing-behaviour
        differences -> a genuinely larger domain gap). Val drawn from source sizes."""
        rng = np.random.default_rng(3000 + seed)
        q1, q2 = np.quantile(self.medcen, [1/3, 2/3])
        large = np.where(self.medcen >= q2)[0]           # target = big facilities
        source = np.where(self.medcen < q2)[0]
        rng.shuffle(source)
        nval = int(0.18 * len(source))
        return source[nval:], source[:nval], large       # train, val, test

    def split_states(self, seed=0, test_frac=0.25, val_frac=0.15):
        rng = np.random.default_rng(2000 + seed)
        states = np.unique(self.state)
        rng.shuffle(states)
        n_test = max(1, int(test_frac * len(states)))
        n_val = max(1, int(val_frac * len(states)))
        test_st = set(states[:n_test]); val_st = set(states[n_test:n_test + n_val])
        test = np.array([i for i in range(self.nF) if self.state[i] in test_st])
        val = np.array([i for i in range(self.nF) if self.state[i] in val_st])
        train = np.array([i for i in range(self.nF)
                          if self.state[i] not in test_st and self.state[i] not in val_st])
        return train, val, test

    def _pool(self, f, time_mask):
        pool = self.by_fac[f]
        if time_mask is not None:
            pool = pool[time_mask[pool]]
        return pool

    # ---- batching ----------------------------------------------------------
    def grouped_batches(self, fac_subset, G=32, K=8, seed=0, shuffle=True, time_mask=None):
        """Yield (Xb[Bs,C,L], Yb[Bs,C,H], fidb[Bs]) with G facilities x K windows.
        time_mask (optional bool[N]) restricts to windows in a time period."""
        rng = np.random.default_rng(seed)
        facs = np.array(fac_subset)
        if shuffle:
            facs = rng.permutation(facs)
        for i in range(0, len(facs), G):
            grp = facs[i:i + G]
            idxs = []
            for f in grp:
                pool = self._pool(f, time_mask)
                if len(pool) == 0:
                    continue
                take = rng.choice(pool, size=min(K, len(pool)), replace=len(pool) < K)
                idxs.append(take)
            if not idxs:
                continue
            idxs = np.concatenate(idxs)
            yield self.Xw[idxs], self.Yw[idxs], self.fid[idxs]

    def eval_arrays(self, fac_subset, time_mask=None):
        """All windows of the given facilities (for deterministic evaluation)."""
        idxs = np.concatenate([self._pool(f, time_mask) for f in fac_subset])
        return self.Xw[idxs], self.Yw[idxs], self.fid[idxs]
