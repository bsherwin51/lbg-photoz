"""
Compute Fisher matrices for LBG forecasts.

This script builds the Fisher information matrix for a given forecast run
defined in the YAML configuration. It uses the LBG forecasting pipeline to
generate the signal vector and covariance, then computes derivatives using
DerivKit's ForecastKit.

Workflow
--------
1. Load configuration from config.yaml.
2. Construct the cosmology and LBG tracer mappers for the chosen run.
3. Load or generate the default covariance matrix.
4. Evaluate the model signal vector at the fiducial parameters.
5. Compute the likelihood Fisher matrix using numerical derivatives.
6. Add Gaussian prior information from the YAML configuration.
7. Save Fisher matrices and run metadata to:

    outputs/<run_name>/

Files produced
--------------
cov_used.npy
fisher_like_<run>.npy
fisher_prior_<run>.npy
fisher_total_<run>.npy
params_<run>.npy
theta0_<run>.npy

Usage
-----
Run with a configuration file and run name:

    python run_fisher.py config.yaml lcdm

Optional derivative settings:

    python run_fisher.py config.yaml lcdm --method finite
    python run_fisher.py config.yaml lcdm --method finite --extrapolation ridders
    python run_fisher.py config.yaml lcdm --method finite --stepsize 0.01

Generate only the covariance matrix:

    python run_fisher.py config.yaml lcdm --cov-only
"""

from __future__ import annotations

# Prevent OpenMP / BLAS oversubscription
# not sure if you had this problem but it is a problem for me on mac
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from pathlib import Path
from derivkit import ForecastKit

import lbg_run_core as core


def main(cfg_path: str, run_name: str, args=None):
    """Compute Fisher matrix for a given run and save outputs."""
    cfg_path = Path(cfg_path).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = core.load_cfg(str(cfg_path))

    # Inject forecast src
    core.inject_forecast_src(cfg, cfg_dir)

    # Output dir
    outdir = core.resolve_outdir(cfg, cfg_dir, run_name)

    params, theta0, cosmo_params = core.build_run_vectors(cfg, run_name)

    print("Run:", run_name, flush=True)
    print("Params:", params, flush=True)

    cov = core.load_or_make_default_cov(cfg, cfg_dir)
    np.save(outdir / "cov_used.npy", cov)

    if args and args.cov_only:
        print("Covariance only. Done.", flush=True)
        return

    data0 = core.create_signal(theta0, params, cosmo_params, cfg)

    if cov.shape != (data0.size, data0.size):
        raise ValueError("Covariance shape mismatch.")

    print("Computing Fisher...", flush=True)

    def model_with_print(th):
        print(f"Evaluating model at theta = {th}", flush=True)
        return core.create_signal(th, params, cosmo_params, cfg)

    fk = ForecastKit(
        function=model_with_print,
        theta0=theta0,
        cov=cov,
    )

    fisher_kwargs = {}
    if args and args.method:
        fisher_kwargs["method"] = args.method
    if args and args.extrapolation:
        fisher_kwargs["extrapolation"] = args.extrapolation
    if args and args.stepsize:
        fisher_kwargs["stepsize"] = args.stepsize

    F_like = fk.fisher(**fisher_kwargs)
    np.save(outdir / f"fisher_like_{run_name}.npy", F_like)
    print("Likelihood Fisher done.", flush=True)

    F_prior = core.build_prior_fisher(cfg, params)
    F_total = F_like + F_prior

    np.save(outdir / f"fisher_prior_{run_name}.npy", F_prior)
    np.save(outdir / f"fisher_total_{run_name}.npy", F_total)

    print("Priors added (always on).", flush=True)

    np.save(outdir / f"params_{run_name}.npy", np.array(params, dtype=object))
    np.save(outdir / f"theta0_{run_name}.npy", theta0)

    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("run", nargs="?", default="lcdm")

    parser.add_argument("--method", default=None)
    parser.add_argument("--extrapolation", default=None)
    parser.add_argument("--stepsize", type=float, default=None)

    parser.add_argument("--cov-only", action="store_true")

    args = parser.parse_args()
    main(args.config, args.run, args)
