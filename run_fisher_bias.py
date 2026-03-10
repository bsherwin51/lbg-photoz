"""Run Fisher-bias projections for LBG forecasts.

This script:
1) Loads an existing Fisher run from outputs/<run>/ (needs fisher_total_<run>.npy, cov_used.npy, params_<run>.npy, theta0_<run>.npy)
2) Builds an unbiased data vector at the fiducial parameters
3) For each configured systematic scenario (dz, stretch, f_interlopers, ...), builds a biased data vector
4) Prints debug stats proving whether the bias changed the data vector
5) Projects the systematic-induced data shift into parameter shifts using DerivKit's Fisher-bias machinery
6) Optionally produces GetDist triangle plots

Notes:
- This script computes derivatives around the *fiducial systematics* (correct for standard Fisher-bias).
- It uses two mapper sets: one fiducial for derivatives, one biased for constructing the biased data vector.
"""

from __future__ import annotations

# Prevent OpenMP / BLAS oversubscription
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from derivkit import ForecastKit

import lbg_run_core as core


def _plot_pairs(params: List[str]) -> List[List[str]]:
    pairs: List[List[str]] = []
    if "Omega_M" in params and "sigma8" in params:
        pairs.append(["Omega_M", "sigma8"])
    if "w0" in params and "wa" in params:
        pairs.append(["w0", "wa"])
    return pairs


def _debug_mapper_state(mappers: list, tag: str, band: str, attr: str) -> None:
    """Print a short summary of mapper attribute values for debugging."""
    vals = []
    for m in mappers:
        if band == "all" or getattr(m, "drop_band", None) == band:
            try:
                vals.append((getattr(m, "drop_band", None), getattr(m, attr)))
            except Exception as e:  # noqa: BLE001
                vals.append((getattr(m, "drop_band", None), f"<ERR {type(e).__name__}: {e}>"))
    print(f"[DEBUG] {tag} mapper.{attr} for band={band}: {vals}", flush=True)


def _debug_data_diff(data_unbiased: np.ndarray, data_biased: np.ndarray) -> None:
    diff = data_biased - data_unbiased

    max_abs_diff = float(np.max(np.abs(diff)))
    max_abs_data = float(np.max(np.abs(data_unbiased)) + 1e-300)
    rel = float(max_abs_diff / max_abs_data)
    print(
        "data diff stats:",
        "max|diff|=", max_abs_diff,
        "max|diff|/max|data|=", rel,
        flush=True,
    )


def main(args) -> None:
    t0 = time.time()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg_dir = cfg_path.parent
    cfg = core.load_cfg(str(cfg_path))

    core.inject_forecast_src(cfg, cfg_dir)

    cfg_fb = cfg.get("fisher_bias", {}) or {}
    scenario_set = cfg_fb.get("scenario_set", "dz")
    method = cfg_fb.get("method", args.method or "finite")
    make_plots = bool(cfg_fb.get("make_plots", True))

    runs = cfg_fb.get("runs", None)
    if not runs:
        runs = [args.run]

    # Optional: force a single run
    if args.only_run is not None:
        runs = [args.only_run]

    # Optional: single-scenario override
    if args.single_param is not None:
        if args.single_band is None or args.single_value is None:
            raise ValueError(
                "If --single-param is set, you must also set --single-band and --single-value."
            )
        scenarios = [{
            "name": f"{args.single_param}_{args.single_band}_{str(args.single_value).replace('.', 'p')}",
            "param": args.single_param,
            "band": args.single_band,
            "val": float(args.single_value),
        }]
    else:
        scenarios = core.parse_scenarios(cfg, scenario_set)

    if make_plots:
        from getdist import plots as getdist_plots  # noqa: WPS433

    from lbg_desc_forecast.forecaster import Forecaster  # noqa: WPS433

    for run in runs:
        outdir = core.resolve_outdir(cfg, cfg_dir, run)

        # load existing Fisher artifacts
        params, theta0, cov, fisher = core.load_run_artifacts(outdir, run, fisher_kind="total")
        cosmo_params = list(cfg["runs"][run]["cosmo_params"])

        # fiducial state
        theta_dict0 = core.theta_to_dict(theta0, params)
        cosmo = core.build_cosmo(theta_dict0, cosmo_params)

        # Two mapper sets:
        # - fiducial: used for derivatives / forward model
        # - biased: used only to create data_biased
        mappers_fid = core.configure_mappers(theta_dict0, cfg)
        mappers_biased = core.configure_mappers(theta_dict0, cfg)

        forecaster_fid = Forecaster(mappers_fid, cosmo)

        print(f"\n=== Run {run} ===", flush=True)
        print("Building unbiased signal...", flush=True)
        data_unbiased = forecaster_fid.create_signal(add_noise=False)

        # forward model for derivatives (around fiducial systematics)
        def forward_model(th: np.ndarray) -> np.ndarray:
            thd = core.theta_to_dict(th, params)

            # update cosmology
            for p in cosmo_params:
                setattr(forecaster_fid.cosmo, p, thd[p])

            # update biases
            fid = cfg["defaults"]["theta0"]
            for mapper in mappers_fid:
                band = mapper.drop_band
                if band == "u":
                    mapper.g_bias = thd.get("bu", fid["bu"])
                    mapper.g_bias_inter = thd.get("bu_int", fid["bu_int"])
                elif band == "g":
                    mapper.g_bias = thd.get("bg", fid["bg"])
                    mapper.g_bias_inter = thd.get("bg_int", fid["bg_int"])
                elif band == "r":
                    mapper.g_bias = thd.get("br", fid["br"])
                    mapper.g_bias_inter = thd.get("br_int", fid["br_int"])


            forecaster_fid.invalidate_cache()

            return forecaster_fid.create_signal(add_noise=False)

        fk = ForecastKit(function=forward_model, theta0=theta0, cov=cov)

        # store results per run
        shifts: Dict[str, Any] = {}

        # debug: show baseline attribute values for each scenario param before looping
        unique_params = sorted({sc["param"] for sc in scenarios})
        for scp in unique_params:
            _debug_mapper_state(mappers_biased, tag="BASELINE", band="all", attr=scp)

        for sc in scenarios:
            sc_name = sc["name"]
            sc_param = sc["param"]
            sc_val = float(sc["val"])
            sc_band = sc["band"]

            print(f"\n-- scenario: {sc_name} ({sc_param}={sc_val} band={sc_band})", flush=True)

            # Debug: show mapper attribute BEFORE applying systematic
            _debug_mapper_state(mappers_biased, tag="BEFORE", band=sc_band, attr=sc_param)

            # Apply systematic to biased mapper set
            touched = False
            for mapper in mappers_biased:
                if sc_band == "all" or mapper.drop_band == sc_band:
                    try:
                        before_val = getattr(mapper, sc_param)
                    except Exception as e:  # noqa: BLE001
                        before_val = f"<ERR {type(e).__name__}: {e}>"

                    try:
                        setattr(mapper, sc_param, sc_val)
                        after_val = getattr(mapper, sc_param)
                    except Exception as e:  # noqa: BLE001
                        after_val = f"<ERR {type(e).__name__}: {e}>"

                    print(
                        "[DEBUG] setattr:",
                        "band=", getattr(mapper, "drop_band", None),
                        "param=", sc_param,
                        "before=", before_val,
                        "after=", after_val,
                        flush=True,
                    )
                    touched = True
                    if sc_band != "all":
                        # For a single-band scenario, only touch one mapper
                        break

            if not touched:
                print(f"WARNING: no mapper found for drop_band={sc_band}", flush=True)

            # Debug: show mapper attribute AFTER applying systematic
            _debug_mapper_state(mappers_biased, tag="AFTER", band=sc_band, attr=sc_param)

            # IMPORTANT: rebuild forecaster AFTER mutating mapper attributes (in case Forecaster caches)
            forecaster_biased = Forecaster(mappers_biased, cosmo)

            # Build biased signal
            print("[DEBUG] Building biased signal...", flush=True)
            data_biased = forecaster_biased.create_signal(add_noise=False)

            # Debug: prove whether bias changed the data vector
            _debug_data_diff(data_unbiased=data_unbiased, data_biased=data_biased)

            # Reset biased mappers back to fiducial for next scenario
            mappers_biased = core.configure_mappers(theta_dict0, cfg)

            # Fisher-bias projection
            dn = fk.delta_nu(data_unbiased=data_unbiased, data_biased=data_biased)
            _, delta_theta = fk.fisher_bias(fisher_matrix=fisher, delta_nu=dn, method=method)

            cov_params = np.linalg.pinv(fisher)
            sigma_theta = np.sqrt(np.diag(cov_params))
            delta_sigma = delta_theta / sigma_theta

            # Print parameter shifts
            print("Parameter shifts (sigma units):", flush=True)
            for p, d in zip(params, delta_sigma):
                print(f"{p:12s} : {d: .3e}", flush=True)

            # Debug: specific pair print
            if "Omega_M" in params and "sigma8" in params:
                idx = [params.index("Omega_M"), params.index("sigma8")]
                print("[DEBUG] theta0 pair:", theta0[idx], flush=True)
                print("[DEBUG] delta_theta pair:", delta_theta[idx], flush=True)
                print("[DEBUG] theta0+shift pair:", (theta0 + delta_theta)[idx], flush=True)
                print("[DEBUG] delta_sigma pair:", delta_sigma[idx], flush=True)

            # Save to dict
            shifts[sc_name] = {
                "param": sc_param,
                "band": sc_band,
                "val": sc_val,
                "params": list(params),
                "delta_theta": np.array(delta_theta, dtype=float),
                "delta_sigma": np.array(delta_sigma, dtype=float),
            }

            # Plotting (optional) — per scenario
            if make_plots:
                plot_pairs = _plot_pairs(params)
                if plot_pairs:
                    plot_dir = outdir / "fisher_bias_plots" / sc_name
                    plot_dir.mkdir(parents=True, exist_ok=True)

                    for pair in plot_pairs:
                        idx = [params.index(pair[0]), params.index(pair[1])]
                        Fsub = fisher[np.ix_(idx, idx)]

                        # Note: these ForecastKit objects are just for GetDist helper,
                        # function is unused for gaussian Fisher plots.
                        fk0 = ForecastKit(function=lambda th: np.array([]), theta0=theta0[idx], cov=np.eye(1))
                        fk1 = ForecastKit(function=lambda th: np.array([]), theta0=(theta0 + delta_theta)[idx], cov=np.eye(1))

                        g0 = fk0.getdist_fisher_gaussian(
                            fisher=Fsub,
                            names=pair,
                            labels=[rf"\mathrm{{{p}}}" for p in pair],
                            label="Unbiased",
                        )
                        g1 = fk1.getdist_fisher_gaussian(
                            fisher=Fsub,
                            names=pair,
                            labels=[rf"\mathrm{{{p}}}" for p in pair],
                            label="Biased",
                        )

                        plotter = getdist_plots.get_subplot_plotter(width_inch=4.5)
                        plotter.settings.figure_legend_frame = False
                        plotter.triangle_plot(
                            [g0, g1],
                            params=pair,
                            legend_labels=["Unbiased", "Biased"],
                            filled=[False, False],
                            contour_colors=["#e1af00", "#f21901"],
                        )

                        fname = f"{pair[0]}_{pair[1]}.png"
                        plotter.export(str(plot_dir / fname))

        np.save(outdir / f"bias_shifts_{run}.npy", shifts, allow_pickle=True)
        print(f"\nSaved bias shifts to: {outdir / f'bias_shifts_{run}.npy'}", flush=True)

    print(f"\nDone in {time.time() - t0:.2f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("run", nargs="?", default="lcdm")

    # quick single-scenario override
    parser.add_argument("--single-param", default=None)
    parser.add_argument("--single-band", default=None)
    parser.add_argument("--single-value", type=float, default=None)

    # optional override run
    parser.add_argument("--only-run", default=None)

    parser.add_argument("--method", default="finite")

    args = parser.parse_args()
    main(args)
