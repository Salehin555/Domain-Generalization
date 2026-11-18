"""Check that every data figure in a manuscript folder is what its script produces
from the current saved results.

Each figure script is run with its output redirected to a scratch folder (its
module-level ROOT is replaced; copies into experiments/figures are suppressed),
and the regenerated figure is compared with the manuscript's copy after
rasterizing both at 100 dpi. Nothing in the manuscript folders is overwritten.

Usage: python check_figures.py <manuscript_dir> <scratch_dir>
"""
import importlib
import os
import sys

import fitz
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCRIPTS = {"fig_survey": ["reducibility_survey.pdf"],
           "fig_gates": ["learned_gates_vs_icc.pdf", "cross_facility_frequency_structure.pdf"],
           "fig_components_all": ["component_ablation.pdf"],
           "fig_overview": ["overview_headroom.pdf"]}


def raster(path):
    p = fitz.open(path)[0].get_pixmap(dpi=100)
    return np.frombuffer(p.samples, np.uint8).reshape(p.h, p.w, p.n).astype(int)


def main():
    man, tmp = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(tmp, "Manuscript"), exist_ok=True)
    os.environ.pop("FIG_OUT", None)
    for mod_name, figs in SCRIPTS.items():
        mod = importlib.import_module(mod_name)
        mod.ROOT = tmp
        if hasattr(mod, "shutil"):
            mod.shutil.copy = lambda *a, **k: None
        mod.main()
        for f in figs:
            a, b = raster(os.path.join(tmp, "Manuscript", f)), raster(os.path.join(man, f))
            same_shape = a.shape == b.shape
            diff = float(np.abs(a - b).mean()) if same_shape else float("nan")
            print(f"{f:42s} shape_match={same_shape} mean_abs_pixel_diff={diff:.4f}")


if __name__ == "__main__":
    main()
