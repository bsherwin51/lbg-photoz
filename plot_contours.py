"""
Plot Fisher forecast parameter contours.

This script loads a Fisher matrix from an existing forecast run and
produces GetDist triangle contours for the chosen parameters.

Inputs
------
It expects the following files in:

    outputs/<run_name>/

    fisher_total_<run_name>.npy
    params_<run_name>.npy

These are typically produced by the Fisher forecast script.

Usage
-----
Plot all parameters:

    python plot_fisher_contours.py <run_name>

Plot only specific parameters:

    python plot_fisher_contours.py <run_name> Omega_M sigma8

Output
------
A triangle contour plot is saved to:

    outputs/<run_name>/fisher_<run_name>_contours.pdf
"""

import sys
import numpy as np
from pathlib import Path
from getdist import plots as getdist_plots
from derivkit import ForecastKit


if len(sys.argv) < 2:
    raise ValueError("Usage: python plot_fisher_contours.py"
                     " <run_name> [param1 param2 ...]")

run_name = sys.argv[1]
plot_params = sys.argv[2:]

outdir = Path("outputs") / run_name

fisher = np.load(outdir / f"fisher_total_{run_name}.npy")
param_names = list(np.load(outdir / f"params_{run_name}.npy", allow_pickle=True))

# If no specific params passed: plot all
if not plot_params:
    plot_params = param_names

label_map = {
    "Omega_M": r"\Omega_m",
    "sigma8": r"\sigma_8",
    "Omega_b": r"\Omega_b",
    "h": r"h",
    "n_s": r"n_s",
    "w0": r"w_0",
    "wa": r"w_a",
    "m_nu": r"\sum m_\nu",
    "bu": r"b_u",
    "bg": r"b_g",
    "br": r"b_r",
    "bu_int": r"b_u^{\mathrm{int}}",
    "bg_int": r"b_g^{\mathrm{int}}",
    "br_int": r"b_r^{\mathrm{int}}",
}

labels = [label_map.get(p, p) for p in param_names]

fk = ForecastKit(
    function=None,
    theta0=np.zeros(len(param_names)),
    cov=np.eye(len(param_names)),
)

gnd = fk.getdist_fisher_gaussian(
    fisher=fisher,
    names=param_names,
    labels=labels,
    label=run_name,
)
red = "#f21901"
plotter = getdist_plots.get_subplot_plotter(width_inch=4.0)
plotter.settings.linewidth_contour = 1.5
plotter.settings.linewidth = 1.5

plotter.triangle_plot(
    [gnd],
    params=plot_params,
    filled=False,
    contour_colors=[red]
)

outfile = outdir / f"fisher_{run_name}_contours.pdf"
plotter.export(str(outfile))

print("Saved plot to", outfile)
