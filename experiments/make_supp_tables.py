"""Write the LaTeX tables of the decision-policy analyses for the supplement.

Usage: python make_supp_tables.py OUTDIR
Writes: tab_actions.tex, tab_dg.tex, tab_dec_costs.tex, tab_dec_multi.tex, tab_coverage.tex
"""
import os
import sys

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
ANCH = [("Electricity", "Flat -> WorkWeek", "E-F$\\to$W"), ("Electricity", "WorkWeek -> Flat", "E-W$\\to$F"),
        ("Traffic", "Flat -> WorkWeek", "Traffic"), ("Solar", "Flat -> WorkWeek", "Solar"),
        ("Nursing", "Ownership (For-profit -> Non-profit)", "Nursing")]
PANELS = ("Electricity", "Traffic", "Solar", "Nursing")


def sg(x, d=1):
    if pd.isna(x):
        return "--"
    if abs(x) < 0.5 * 10 ** (-d):
        return "0." + "0" * d
    s = f"{x:+.{d}f}"
    return "$" + s.replace("-", "-") + "$" if s[0] == "-" else s


def w(name, s):
    open(os.path.join(OUT, name), "w", encoding="utf-8").write(s)


def actions():
    t = pd.read_csv(os.path.join(RES, "actions_table.csv"))
    t = t[t.metric == "mae_cat"]
    arms = [("Blend with seasonal naive", "dep_1x+blend"), ("Combination incl. Holt-Winters", "dep_1x+comb"),
            ("Train 2$\\times$", "dep_2x"), ("Train 4$\\times$", "dep_4x"), ("Train 8$\\times$", "dep_8x"),
            ("Train to convergence", "dep_conv"), ("\\quad + combination", "dep_conv+comb"),
            ("Holt-Winters alone", "holt_winters"), ("Seasonal naive alone", "snaive"),
            ("Retrain on target, same budget", "ref_1x"), ("\\quad + combination", "ref_1x+comb"),
            ("Retrain on target, 4$\\times$", "ref_4x"), ("Retrain on target to convergence", "ref_conv"),
            ("\\quad + combination", "ref_conv+comb"), ("Joint training to convergence", "joint_conv")]
    rows = []
    for lab, a in arms:
        cells = []
        for p, sp, _ in ANCH:
            r = t[(t.panel == p) & (t.split == sp) & (t.arm == a)].iloc[0]
            d = 2 if p == "Nursing" else 1                  # nursing values are all below 0.1%
            cells.append(f"{sg(r.reduction, d)} ({r.seed_sd:.{d}f}) [{sg(r.lo, d)}, {sg(r.hi, d)}]")
        rows.append(lab + " & " + " & ".join(cells) + " \\\\")
        if a in ("snaive",):
            rows.append("\\midrule")
    s = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Actions against the model in service, with "
         "uncertainty} (anchor splits, primary metric). Reduction in target-test error relative to the "
         "deployed model at the standard budget: mean over seeds (five; three for nursing) of per-seed ratios, seed SD in "
         "parentheses, and paired entity-bootstrap 95\\% interval of the ratio of seed-averaged errors "
         "(2,000 resamples; models fixed); nursing values to two decimals. Budgets are multiples of the deployed model's gradient steps.}\n"
         "\\label{tab:Sact}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
         "\\textbf{Action} & " + " & ".join("\\textbf{" + a[2] + "}" for a in ANCH) + " \\\\\n\\midrule\n"
         "\\multicolumn{6}{@{}l}{\\emph{No target data}} \\\\\n" + "\n".join(rows[:10]) +
         "\n\\multicolumn{6}{@{}l}{\\emph{Labeled target entities}} \\\\\n" + "\n".join(rows[10:]) +
         "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    w("tab_actions.tex", s)


def dg():
    s = pd.read_csv(os.path.join(RES, "dg_summary.csv"))
    s = s[s.metric == "mae_cat"]
    rows = []
    for m in ("IRM", "CORAL", "DANN", "GroupDRO"):
        for dom in ("entity", "cluster"):
            cells = []
            for p, sp, _ in ANCH:
                r = s[(s.panel == p) & (s.split == sp) & (s.method == m) & (s.domains == dom)].iloc[0]
                cells.append(f"{sg(r.selected, 2)} ({r.selected_sd:.2f}) & {sg(r.hindsight, 2)}")
            rows.append(f"{m} & {dom} & " + " & ".join(cells) + " \\\\")
    s = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Domain-generalization baselines tuned on source "
         "validation} (anchor splits, primary metric, five seeds; three for nursing). Error reduction relative to ERM-Stem "
         "trained in the same run (\\%). Sel.: setting and checkpoint chosen on source validation among "
         "checkpoints trained with the penalty active, mean with seed SD in parentheses. Hind.: best setting "
         "on the target test set (not deployable). Grids: IRM and CORAL weight 0.1, 1, 10; DANN 0.1, 0.5, 2; "
         "GroupDRO step size 0.001, 0.01, 0.1. Domains: each training entity, or four clusters of the "
         "source entities on their observed-era level, variation, weekly amplitude and low-frequency share.}\n"
         "\\label{tab:Sdg}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrrrrrrrrrr@{}}\n\\toprule\n"
         " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{a[2]}}}}}" for a in ANCH) + " \\\\\n"
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\\cmidrule(l){11-12}\n"
         "\\textbf{Method} & \\textbf{Domains} & " + " & ".join(["\\textbf{Sel.} & \\textbf{Hind.}"] * len(ANCH)) +
         " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    w("tab_dg.tex", s)


def dec():
    rows = []
    for mt, lab in (("mae_cat", "Primary"), ("mae_total", "Total")):
        d = pd.read_csv(os.path.join(RES, f"decision_costs_{mt}.csv"))
        for p in PANELS + ("All panels",):
            for _, r in d[d.panel == p].iterrows():
                rows.append(f"{lab if (p == 'Electricity' and r.cost == 0) else ''} & "
                            f"{p if r.cost == 0 else ''} & {r.cost:.0f} & {sg(r.always, 2)} & {sg(r.rule, 2)} & "
                            f"{sg(r.oracle, 2)} & {r.regret_rule:.2f} & {int(r.unnecessary)}/{int(r.missed)} & "
                            f"{r.retrain_rate * 100:.0f}\\% & {sg(r.voi, 2)} \\\\")
            rows.append("\\midrule")
    rows = rows[:-1]
    s = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Retraining decisions against the deployed model} "
         "(four panels; {NUNITS} split $\\times$ date units, seeds averaged). Net benefit in percent of the "
         "deployed model's error; never retraining is 0. Regret: oracle minus rule. Unn./miss.: unnecessary "
         "and missed retrainings of the rule. Retrain: share of units the rule retrains. Gross value: rule minus the "
         "better of never and always, before the cost of the measurement.}\n\\label{tab:Sdec}\n\\footnotesize\n"
         "\\begin{tabular}{@{}llrrrrrrrr@{}}\n\\toprule\n\\textbf{Metric} & \\textbf{Panel} & $c$ & "
         "\\textbf{Always} & \\textbf{Rule} & \\textbf{Oracle} & \\textbf{Regret} & \\textbf{Unn./miss.} & "
         "\\textbf{Retrain} & \\textbf{Gross value} \\\\\n\\midrule\n" + "\n".join(rows) +
         "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    n_units = len(pd.read_csv(os.path.join(RES, "decision_units_mae_cat.csv")))
    w("tab_dec_costs.tex", s.replace("{NUNITS}", str(n_units)))

    lo = pd.read_csv(os.path.join(RES, "decision_lopo_mae_cat.csv"))
    asy = pd.read_csv(os.path.join(RES, "decision_asym_mae_cat.csv"))
    cal = pd.read_csv(os.path.join(RES, "decision_calibration_mae_cat.csv"))
    ma = pd.read_csv(os.path.join(RES, "decision_multi_mae_cat.csv"))
    r1 = []
    for _, r in ma[ma.cost.isin([0, 1, 3, 5])].iterrows():
        r1.append(f"{r.panel} & {r.cost:.0f} & {sg(r.fixed_retrain, 2)} & {sg(r.fixed_longer, 2)} & "
                  f"{sg(r.fixed_blend, 2)} & {sg(r.rule, 2)} & {sg(r.oracle, 2)} & "
                  f"{r.share_retrain * 100:.0f}/{r.share_longer * 100:.0f}/{r.share_blend * 100:.0f} \\\\")
    s2 = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Choosing among four actions} (four panels, "
          "primary metric, realized net benefit in the deployment period). Fixed policies apply one action to "
          "every unit: retrain on target (cost $c$), train the deployed model twice as long (cost $c/4$), or "
          "blend it with seasonal naive (cost 0); keeping the model is 0. The rule picks, per unit, the action "
          "with the largest net benefit estimated on observed history; the oracle picks the best in hindsight. "
          "Shares: units on which the rule retrains / trains longer / blends (\\%).}\n\\label{tab:Smulti}\n"
          "\\footnotesize\n\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n\\textbf{Panel} & $c$ & \\textbf{Retrain} & "
          "\\textbf{Longer} & \\textbf{Blend} & \\textbf{Rule} & \\textbf{Oracle} & \\textbf{Shares} \\\\\n"
          "\\midrule\n" + "\n".join(r1) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    w("tab_dec_multi.tex", s2)

    r2 = []
    for c in (0, 1, 2, 3, 5):
        g = lo[lo.cost == c].set_index("left_out")
        r2.append(f"{c} & " + " & ".join(f"{sg(g.loc[p, 'rule_minus_always'], 2)} / {sg(g.loc[p, 'voi'], 2)}"
                                           for p in PANELS) + " \\\\")
    r3 = []
    for _, r in asy[asy.panel == "All panels"].iterrows():
        r3.append(f"{r.cost:.0f} & {r.extra_false:.0f} & {sg(r.always, 2)} & {sg(r.rule, 2)} & {sg(r.oracle, 2)} \\\\")
    r4 = []
    for _, r in cal.iterrows():
        prec = "--" if pd.isna(r.precision) else f"{r.precision * 100:.0f}\\%"
        r4.append(f"{r.panel} & {r.threshold:.0f}\\% & {int(r.n_selected)} & {prec} & {r.recall * 100:.0f}\\% & "
                  f"{sg(r.mean_E_sel, 2)} & {sg(r.mean_D_sel, 2)} \\\\")
    s3 = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Robustness of the decision analysis} (four "
          "panels, primary metric). (a) Leaving one panel out of the pooled analysis: rule minus always / gross "
          "decision value. (b) Asymmetric costs: each unnecessary retraining is charged an extra penalty "
          "on top of $c$ (pooled). (c) Calibration at decision thresholds: units with estimate at or above "
          "the threshold, share of them whose realized headroom also reaches it (precision), share of units "
          "above the threshold that are selected (recall), and mean estimate and realized value of the "
          "selected units.}\n\\label{tab:Srob}\n\\footnotesize\n"
          "\\begin{tabular}{@{}lrrrr@{}}\n\\multicolumn{5}{@{}l}{(a) Leave one panel out} \\\\\n\\toprule\n"
          "$c$ & \\textbf{Without electricity} & \\textbf{Without traffic} & \\textbf{Without solar} & "
          "\\textbf{Without nursing} \\\\\n"
          "\\midrule\n" + "\n".join(r2) + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{6pt}\n"
          "\\begin{tabular}{@{}rrrrr@{}}\n\\multicolumn{5}{@{}l}{(b) Asymmetric costs, pooled} \\\\\n\\toprule\n"
          "$c$ & \\textbf{Extra penalty} & \\textbf{Always} & \\textbf{Rule} & \\textbf{Oracle} \\\\\n\\midrule\n" +
          "\n".join(r3) + "\n\\bottomrule\n\\end{tabular}\n\n\\vspace{6pt}\n"
          "\\begin{tabular}{@{}lrrrrrr@{}}\n\\multicolumn{7}{@{}l}{(c) Calibration at thresholds} \\\\\n\\toprule\n"
          "\\textbf{Panel} & \\textbf{Threshold} & \\textbf{Selected} & \\textbf{Precision} & \\textbf{Recall} & "
          "\\textbf{Mean $E$} & \\textbf{Mean $D$} \\\\\n\\midrule\n" + "\n".join(r4) +
          "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    w("tab_dec_robust.tex", s3)


def coverage():
    d = pd.read_csv(os.path.join(RES, "coverage_measures.csv"))
    a = pd.read_csv(os.path.join(RES, "coverage_assoc.csv")).set_index("measure")
    rows = []
    for _, r in d.iterrows():
        rows.append(f"{r.panel} & {r.setting} & {r.years:.2f} & {r.month * 100:.0f}\\% & {r.month14 * 100:.0f}\\% & "
                    f"{r.lag365 * 100:.0f}\\% & {sg(r.r, 2)} & {r.abs_bias:.2f} \\\\")
    foot = " & ".join(f"{sg(a.loc[k, 'spearman_r'], 2)}" for k in ("years", "month", "month14", "lag365"))
    s = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Calendar-coverage measures} (size-matched "
         "estimates, primary metric). Settings: full and truncated histories at the main decision date, and the "
         "decision dates of Table~\\ref{tab:S17}. Years: length of the observed history. Month: share of "
         "deployment days whose calendar month occurs at least once in the history; $\\ge$14 d: at least 14 "
         "observed days of that month; Lag 1 y: the same date one year earlier lies in the history. $r$ and "
         "$|$Bias$|$: agreement of estimate and realized headroom. Last row: Spearman correlation of each "
         "measure with $r$ over the 26 settings.}\n\\label{tab:Scov}\n\\footnotesize\n"
         "\\begin{tabular}{@{}llrrrrrr@{}}\n\\toprule\n\\textbf{Panel} & \\textbf{Setting} & \\textbf{Years} & "
         "\\textbf{Month} & \\textbf{$\\ge$14 d} & \\textbf{Lag 1 y} & $r$ & $|$\\textbf{Bias}$|$ \\\\\n\\midrule\n" +
         "\n".join(rows) + "\n\\midrule\n\\multicolumn{2}{@{}l}{Spearman with $r$} & " + foot +
         " & & \\\\\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    w("tab_coverage.tex", s)


if __name__ == "__main__":
    actions(); dg(); dec(); coverage()
    print("ok")
