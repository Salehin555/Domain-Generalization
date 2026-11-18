"""LaTeX tables for the coverage-gated rules and the sequential procedure.

Usage: python make_seq_tables.py OUTDIR
Writes: tab_multi_main.tex, tab_seq_main.tex (main text) and tab_seq_supp.tex (supplement, all costs, both metrics)
"""
import os
import sys

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = sys.argv[1]
PANELS = ("Electricity", "Traffic", "Solar", "Nursing", "All panels")
NAME = {"All panels": "Pooled"}


def f(x):
    if abs(x) < 0.005:
        return "0.00"
    s = f"{x:+.2f}"
    return f"${s}$" if s[0] == "-" else s


def main():
    fa = pd.read_csv(os.path.join(RES, "gated_four_action_mae_cat.csv"))
    rows = []
    for c in (1.0, 3.0, 5.0):
        for p in PANELS:
            r = fa[(fa.panel == p) & (fa.cost == c)].iloc[0]
            rows.append(f"{NAME.get(p, p)} & {c:.0f} & {f(r.always_retrain)} & {f(r.always_longer)} & "
                        f"{f(r.always_blend)} & {f(r.rule)} & {f(r.rule_gated)} & {f(r.oracle)} \\\\")
        rows.append("\\midrule")
    pooled = fa[fa.panel == "All panels"].set_index("cost")
    note = "; ".join(f"$c={c:.0f}$: rule [{pooled.loc[c, 'rule_minus_blend_lo']:+.2f}, {pooled.loc[c, 'rule_minus_blend_hi']:+.2f}], "
                     f"gated [{pooled.loc[c, 'gated_minus_blend_lo']:+.2f}, {pooled.loc[c, 'gated_minus_blend_hi']:+.2f}]"
                     for c in (1.0, 3.0, 5.0))
    s = ("\\begin{table*}[!t]\n\\centering\n\\caption{\\textbf{Choosing among four actions} (four panels, primary "
         "metric, 112 split $\\times$ date units). Realized net benefit in the deployment period, in percent of the "
         "deployed model's error; keeping the model is 0. Fixed policies apply one action to every unit: retrain on "
         "target data (cost $c$), train the deployed model twice as long (cost $c/4$), or blend it with seasonal naive "
         "(cost 0). The rule picks, per unit, the action with the largest net benefit estimated on observed history. "
         "The gated rule is the same rule, except that when the observed history does not cover the deployment "
         "months it ignores the estimate and blends; it was formulated after the solar failure and is evaluated on "
         "the same data, so it is exploratory. Pooled 95\\% bootstrap intervals over splits of the difference from "
         "always blending: " + note.replace("$+", "$+") + ".}\n\\label{tab:multi}\n\\footnotesize\n"
         "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n & & \\multicolumn{3}{c}{\\textbf{Fixed policy}} & & & \\\\\n"
         "\\cmidrule(lr){3-5}\n\\textbf{Panel} & $c$ & \\textbf{Retrain} & \\textbf{Train longer} & \\textbf{Blend} & "
         "\\textbf{Rule} & \\textbf{Gated rule} & \\textbf{Oracle} \\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
         "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    open(os.path.join(OUT, "tab_multi_main.tex"), "w", encoding="utf-8").write(s)

    sq = pd.read_csv(os.path.join(RES, "seq_costs_mae_cat.csv"))
    asc = pd.read_csv(os.path.join(RES, "seq_assoc_mae_cat.csv")).set_index("panel")
    rows = []
    for p in PANELS:
        a = asc.loc[p]
        cells = [f"{f(a.improve_D)}", f"{f(a.resid_E)}", f"{f(a.resid_D)}", f"{f(a.r_resid)}", f"{f(a.bias_resid)}"]
        for c in (1.0, 3.0):
            r = sq[(sq.panel == p) & (sq.cost == c)].iloc[0]
            cells += [f(r.always), f(r.rule), f(r.oracle)]
        rows.append(f"{NAME.get(p, p)} & " + " & ".join(cells) + " \\\\")
    s = ("\\begin{table*}[!t]\n\\centering\n\\caption{\\textbf{Sequential procedure: improve the model in service "
         "first, then decide on retraining} (four panels, primary metric, 112 split $\\times$ date units). "
         "Improvement: reduction in deployment-period error obtained without target data by training the deployed "
         "model until source-validation error stops improving and blending it with seasonal naive, relative to the "
         "deployed model at the standard budget. Residual headroom: further reduction from a target-trained reference "
         "treated the same way, relative to the improved model, estimated on observed history ($E$) and realized "
         "($D$). Net benefits of retraining are relative to the improved model; never retraining is 0.}\n"
         "\\label{tab:seq}\n\\footnotesize\n\\begin{tabular}{@{}lrrrrrrrrrrr@{}}\n\\toprule\n"
         " & & \\multicolumn{4}{c}{\\textbf{Residual headroom}} & \\multicolumn{3}{c}{\\textbf{Cost $c=1$}} & "
         "\\multicolumn{3}{c}{\\textbf{Cost $c=3$}} \\\\\n\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\\cmidrule(l){10-12}\n"
         "\\textbf{Panel} & \\textbf{Improvement} & \\textbf{Mean $E$} & \\textbf{Mean $D$} & $r$ & \\textbf{Bias} & "
         "\\textbf{Always} & \\textbf{Rule} & \\textbf{Oracle} & \\textbf{Always} & \\textbf{Rule} & \\textbf{Oracle} "
         "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    open(os.path.join(OUT, "tab_seq_main.tex"), "w", encoding="utf-8").write(s)

    rows = []
    for mt, lab in (("mae_cat", "Primary"), ("mae_total", "Total")):
        sq = pd.read_csv(os.path.join(RES, f"seq_costs_{mt}.csv"))
        for p in PANELS:
            for _, r in sq[sq.panel == p].iterrows():
                rows.append(f"{lab if (p == 'Electricity' and r.cost == 0) else ''} & {NAME.get(p, p) if r.cost == 0 else ''} & "
                            f"{r.cost:.0f} & {f(r.always)} & {f(r.rule)} & {f(r.rule_gated)} & {f(r.oracle)} & "
                            f"{int(r.unnecessary)}/{int(r.missed)} & {f(r.total_improve)} & {f(r.total_rule)} & "
                            f"{f(r.total_gate_retrain)} & {f(r.total_gate_full)} \\\\")
            rows.append("\\midrule")
    s = ("\\begin{table}[!ht]\n\\centering\n\\caption{\\textbf{Sequential procedure at every cost} (four panels, "
         "112 split $\\times$ date units). Left block: net benefit of retraining relative to the improved model (trained "
         "until validation error stops improving and blended with seasonal naive), in percent of the improved model's "
         "error; never retraining is 0. Gated: the rule ignores the estimate when the observed history does not cover "
         "the deployment months (exploratory). Unn./miss.: unnecessary and missed retrainings of the rule. Right block: "
         "overall net reduction relative to the deployed model at the standard budget, computed per unit and then "
         "averaged. Because the improvement $I$ and the residual reduction $R$ have different denominators, they "
         "combine as $I+(1-I)R$, and the retraining cost becomes $c(1-I)$ in deployed-model units. Improve: first step "
         "only. Rule: improve, then retrain if the residual estimate is at least $c$. Gate R: the coverage gate blocks "
         "only the retraining step. Gate F: the gate is applied before the first action, so units without coverage "
         "only blend the deployed model with seasonal naive.}\n"
         "\\label{tab:Sseq}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}llrrrrrrrrrr@{}}\n\\toprule\n"
         " & & & \\multicolumn{5}{c}{\\textbf{Retraining, vs improved model}} & "
         "\\multicolumn{4}{c}{\\textbf{Overall, vs deployed model}} \\\\\n\\cmidrule(lr){4-8}\\cmidrule(l){9-12}\n"
         "\\textbf{Metric} & \\textbf{Panel} & $c$ & \\textbf{Always} & \\textbf{Rule} & \\textbf{Gated} & "
         "\\textbf{Oracle} & \\textbf{Unn./miss.} & \\textbf{Improve} & \\textbf{Rule} & \\textbf{Gate R} & "
         "\\textbf{Gate F} \\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
         "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    open(os.path.join(OUT, "tab_seq_supp.tex"), "w", encoding="utf-8").write(s)
    print("ok")


if __name__ == "__main__":
    main()
