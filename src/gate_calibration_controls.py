"""Controls that decide whether the calibration gate is a real mechanism.

A gain over the backbone is not by itself evidence of the proposed mechanism. It
could be extra capacity, or a better-tuned output layer, and either would be a
much weaker claim. Three controls separate those:

  1. SHUFFLED DESCRIPTOR. Feed the gate another entity's window profile instead
     of the presented window's own. Same parameter count, same optimizer, same
     everything -- only the conditioning is destroyed. If the gain survives, the
     mechanism is capacity, not calibration, and the claim collapses.

  2. CONSTANT DESCRIPTOR. Replace the profile with its batch mean, which reduces
     the gate to a static per-band output rescaling. This separates "the forecast
     needed rescaling at all" from "the rescaling had to depend on the window".

  3. RANDOM-CONTROL SPLIT. Run on a randomly grouped split, where the survey
     measured no headroom. A mechanism that converts genuine cross-group
     structure should show little or nothing here; one that merely improves the
     output layer would gain just as much.

Output: outputs/gate_calibration_controls.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from src.data import Panel
from src.models import FADECalib
from src.gate_calibration_test import PANELS, make_split, run_case, EP, PAT, SEEDS
from src.train import train_model, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

BASE = dict(method="erm", epochs=EP, patience=PAT, backbone="mlp")
CALIB = dict(method="fade_calib", epochs=EP, patience=PAT, backbone="mlp",
             lam_inv=0.0, gate_lr=0.05)


class _Shuffled(FADECalib):
    """Control 1: the descriptor comes from a different entity's window."""

    def _rho_in(self, xn):
        rho = super()._rho_in(xn)
        return rho[torch.randperm(rho.shape[0], device=rho.device)]


class _Constant(FADECalib):
    """Control 2: the descriptor carries no per-window information."""

    def _rho_in(self, xn):
        rho = super()._rho_in(xn)
        return rho.mean(0, keepdim=True).expand_as(rho)


def run_with(cls, P, src_all, tgt_all, seed):
    """run_case, but forcing a specific model class."""
    import src.train as T
    orig = T.FADECalib
    T.FADECalib = cls
    try:
        return run_case(P, src_all, tgt_all, seed, CALIB)
    finally:
        T.FADECalib = orig


def main():
    rows = []
    cases = [("Solar", "vol", "high->low"),
             ("Solar", "vol", "low->high"),
             ("Electricity", "trend", "high->low")]
    cache = {}
    for panel, stat, direction in cases:
        if panel not in cache:
            spec = PANELS[panel]
            cache[panel] = Panel(spec["path"], n_facilities=spec.get("nfac", 10 ** 6),
                                 L=56, H=14, stride=18, seed=0, n_channels=3,
                                 boundary_date=spec["boundary"])
        P = cache[panel]
        sp = make_split(P, stat, direction)
        if sp is None:
            continue
        src_all, tgt_all = sp
        label = f"{panel} {stat}: {direction}"
        print(f"\n### {label}", flush=True)
        for vname, cls in [("backbone", None), ("calibration gate", FADECalib),
                           ("ctrl: shuffled descriptor", _Shuffled),
                           ("ctrl: constant descriptor", _Constant)]:
            errs = []
            for seed in SEEDS:
                if cls is None:
                    e = run_case(P, src_all, tgt_all, seed, BASE)
                else:
                    e = run_with(cls, P, src_all, tgt_all, seed)
                if e is not None:
                    errs.append(e)
                    rows.append(dict(case=label, variant=vname, seed=seed, mae=e))
            if errs:
                print(f"  {vname:28s} {np.mean(errs):.4f} +/- {np.std(errs):.4f}",
                      flush=True)

    # Control 3: random split on solar, where the survey found no headroom
    P = cache.get("Solar")
    if P is not None:
        rng = np.random.default_rng(0)
        p = rng.permutation(np.arange(P.nF))
        h = len(p) // 2
        label = "Solar random control"
        print(f"\n### {label}", flush=True)
        for vname, cfg in [("backbone", BASE), ("calibration gate", CALIB)]:
            errs = []
            for seed in SEEDS:
                e = run_case(P, p[:h], p[h:], seed, cfg)
                if e is not None:
                    errs.append(e)
                    rows.append(dict(case=label, variant=vname, seed=seed, mae=e))
            if errs:
                print(f"  {vname:28s} {np.mean(errs):.4f} +/- {np.std(errs):.4f}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "gate_calibration_controls.csv"), index=False)
    print("\n" + "=" * 70)
    print("GAIN vs BACKBONE (%)")
    print("=" * 70)
    m = df.groupby(["case", "variant"]).mae.mean().unstack()
    for v in m.columns:
        if v != "backbone":
            m[v] = 100 * (m["backbone"] - m[v]) / m["backbone"]
    print(m.drop(columns=["backbone"]).round(2).to_string())
    print("\nSaved outputs/gate_calibration_controls.csv")


if __name__ == "__main__":
    sys.exit(main())
