"""Split-level association and decision analysis of the observed-history estimate.

Analysis of saved results only (results/temporal_A.csv); no model is trained.

Unit. The temporal experiment has two allocation seeds per split. Seeds share the
split's entities and eras, so they are not independent deployment replications.
Here each split is one unit: estimate and realized headroom are first averaged over
its seeds. Association uncertainty comes from a bootstrap over splits (2,000
replicates). Splits of one panel still share entities, so these intervals are
optimistic about generalization to new panels.

Decision. At decision time only the observed-era estimate E is known; the realized
deployment-era headroom D is observed afterwards. For a threshold t (percent) the
rule "retrain if E >= t" is compared with "never retrain" and "always retrain".
A retraining is called beneficial when D >= t. Reported per panel and t:
  confusion counts (TP beneficial retrain, FP unnecessary retrain, FN missed
  beneficial retrain, TN), the accuracy of the rule, and the accuracy of the
  majority class of D >= t on that panel (a hindsight reference);
  gain: mean over splits of D * 1[retrain], the deployment-era error reduction the
  rule obtains relative to never retraining, in percent of the source model's
  error, with no cost charged for retraining;
  regret: gain of the hindsight rule 1[D > 0] minus gain of the rule.
Costs of measurement and retraining are not modeled; see the article.

Output: results/temporal_decision.csv, results/temporal_assoc_split.csv
"""
import os

import numpy as np
import pandas as pd
import scipy.stats as ss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
THRS = (1.0, 2.0, 3.0, 5.0)
B = 2000


def main():
    A = pd.read_csv(os.path.join(RES, "temporal_A.csv"))
    A = A[A.variant == "standard"]
    S = (A.groupby(["panel", "split"])
          .agg(E=("headroom_obs_mae_cat", "mean"), D=("headroom_dep_mae_cat", "mean"),
               nseed=("seed", "nunique")).reset_index())
    rng = np.random.default_rng(0)
    assoc, dec = [], []
    for p, g in S.groupby("panel"):
        e, d = g.E.to_numpy(), g.D.to_numpy()
        n = len(g)
        bp, bs = [], []
        for _ in range(B):
            i = rng.integers(0, n, n)
            if np.ptp(e[i]) > 0 and np.ptp(d[i]) > 0:
                bp.append(ss.pearsonr(e[i], d[i])[0]); bs.append(ss.spearmanr(e[i], d[i])[0])
        assoc.append(dict(
            panel=p, n_splits=n, seeds_per_split=int(g.nseed.min()),
            pearson=ss.pearsonr(e, d)[0], pearson_lo=np.percentile(bp, 2.5),
            pearson_hi=np.percentile(bp, 97.5),
            spearman=ss.spearmanr(e, d)[0], spearman_lo=np.percentile(bs, 2.5),
            spearman_hi=np.percentile(bs, 97.5),
            bias=float(np.mean(e - d)), mae=float(np.mean(np.abs(e - d))),
            slope=float(np.polyfit(e, d, 1)[0]), intercept=float(np.polyfit(e, d, 1)[1])))
        oracle = float(np.mean(d * (d > 0)))
        for t in THRS:
            act, ben = e >= t, d >= t
            tp, fp = int((act & ben).sum()), int((act & ~ben).sum())
            fn, tn = int((~act & ben).sum()), int((~act & ~ben).sum())
            gain = float(np.mean(d * act))
            dec.append(dict(panel=p, thr=t, n_splits=n, tp=tp, fp=fp, fn=fn, tn=tn,
                            acc=(tp + tn) / n, majority=max(ben.mean(), 1 - ben.mean()),
                            gain_rule=gain, gain_always=float(np.mean(d)), gain_never=0.0,
                            gain_oracle=oracle, regret=oracle - gain))
    pd.DataFrame(assoc).to_csv(os.path.join(RES, "temporal_assoc_split.csv"), index=False)
    pd.DataFrame(dec).to_csv(os.path.join(RES, "temporal_decision.csv"), index=False)
    print(pd.DataFrame(assoc).round(3).to_string())
    print(pd.DataFrame(dec).round(3).to_string())


if __name__ == "__main__":
    main()
