"""Rows of the main-text decision tables (deployed-model decisions, four actions,
sequential procedure), printed in the LaTeX layout of the article, from the result files
written by an_decisions.py and an_sequential.py.
usage: python main_table_rows.py [dec|multi|seq]
"""
import os
import sys

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PANELS = ["Electricity", "Traffic", "Solar", "Nursing", "All panels"]
NAME = {"All panels": "Pooled"}


def f(x, d=2):
    if abs(x) < 0.005 and d == 2:
        return "0.00"
    s = f"{x:+.{d}f}"
    return f"${s.replace('-', '-')}$" if s[0] == "-" else f"${s}$"


def g(x, d=2):
    """Plain signed number without math mode for positives (table style of the article)."""
    if abs(x) < 0.005:
        return "0.00"
    s = f"{x:+.{d}f}"
    return f"${s}$" if s[0] == "-" else s


def dec():
    a = pd.read_csv(os.path.join(RES, "decision_assoc_mae_cat.csv")).set_index("panel")
    c = pd.read_csv(os.path.join(RES, "decision_costs_mae_cat.csv"))
    for p in PANELS:
        cells = [NAME.get(p, p), str(int(a.loc[p, "n_units"])), f(a.loc[p, "r_dep"]), f(a.loc[p, "bias_dep"])]
        for cost in (1.0, 3.0):
            r = c[(c.panel == p) & (c.cost == cost)].iloc[0]
            cells += [f(r.always), f(r.rule), f(r.oracle), f"{int(r.unnecessary)}/{int(r.missed)}", f(r.voi)]
        print(" & ".join(cells) + " \\\\")
    for cost in (1.0, 3.0):
        r = c[(c.panel == "All panels") & (c.cost == cost)].iloc[0]
        print(f"c={cost:.0f}: equal panels {r.rma_equal_panel:+.2f}; pooled {r.rule_minus_always:+.2f} "
              f"[{r.rma_lo:+.2f}, {r.rma_hi:+.2f}] ([{r.rma_panel_lo:+.2f}, {r.rma_panel_hi:+.2f}])")


def multi():
    fa = pd.read_csv(os.path.join(RES, "gated_four_action_mae_cat.csv"))
    for cost in (1.0, 3.0, 5.0):
        for p in PANELS:
            r = fa[(fa.panel == p) & (fa.cost == cost)].iloc[0]
            print(f"{NAME.get(p, p)} & {cost:.0f} & {g(r.always_retrain)} & {g(r.always_longer)} & {g(r.always_blend)} & "
                  f"{g(r.rule)} & {g(r.rule_gated)} & {g(r.oracle)} \\\\")
        print("\\midrule")
    po = fa[fa.panel == "All panels"].set_index("cost")
    print(po.filter(regex="minus|lo|hi").round(2).to_string())


def seq():
    u = pd.read_csv(os.path.join(RES, "seq_assoc_mae_cat.csv")).set_index("panel")
    c = pd.read_csv(os.path.join(RES, "seq_costs_mae_cat.csv"))
    for p in PANELS:
        r = u.loc[p]
        cells = [NAME.get(p, p), g(r.improve_D), g(r.resid_E), g(r.resid_D), g(r.r_resid), g(r.bias_resid)]
        for cost in (1.0, 3.0):
            k = c[(c.panel == p) & (c.cost == cost)].iloc[0]
            cells += [g(k.always), g(k.rule), g(k.oracle)]
        print(" & ".join(cells) + " \\\\")


if __name__ == "__main__":
    {"dec": dec, "multi": multi, "seq": seq}[sys.argv[1] if len(sys.argv) > 1 else "dec"]()
