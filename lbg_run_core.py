"""
Core helpers for LBG Fisher / Fisher-bias scripts.

This module centralizes the plumbing used by the command-line scripts:
- Read the YAML config (load_cfg) and add the local lbg-desc-forecast src to sys.path (inject_forecast_src).
- Build the parameter vector ordering for a named run (build_run_vectors) and convert between vector/dict (theta_to_dict).
- Construct the forecast objects: cosmology (build_cosmo) + per-band mappers with fiducial systematics, Petri interloper fractions,
  and galaxy bias / interloper-bias parameters (configure_mappers).
- Generate model outputs: unbiased signal vectors (create_signal), biased signal vectors for a chosen systematic in a chosen band
  (create_biased_signal), and the covariance matrix (create_covariance).
- Load existing Fisher artifacts from outputs/<run>/ (load_run_artifacts), and optionally build diagonal Gaussian prior Fishers
  from the YAML (build_prior_fisher).
- Expand a compact YAML “scenarios:” block into a flat list of (param, band, value) cases to loop over (parse_scenarios).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml


def load_cfg(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def inject_forecast_src(cfg: dict, cfg_dir: Path) -> None:
    """Inject lbg-desc-forecast src into sys.path and print which one is used."""
    src = Path(cfg["paths"]["lbg_desc_forecast_src"])
    if not src.is_absolute():
        src = (cfg_dir / src).resolve()
    sys.path.insert(0, str(src))

    import lbg_desc_forecast  # noqa: F401


def resolve_outdir(cfg: dict, cfg_dir: Path, run_name: str) -> Path:
    """Resolve output directory (same logic as your fisher script)."""
    outdir = Path(cfg["paths"]["outdir"])
    if not outdir.is_absolute():
        outdir = (cfg_dir / outdir).resolve()
    outdir = outdir / run_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def theta_to_dict(theta: np.ndarray, params: List[str]) -> Dict[str, float]:
    """Convert theta to dict."""
    return {p: float(v) for p, v in zip(params, theta)}


def build_run_vectors(cfg: dict, run_name: str) -> Tuple[List[str], np.ndarray, List[str]]:
    """Build params list, theta0 vector, and cosmo_params list for a run."""
    run_cfg = cfg["runs"][run_name]

    cosmo_params = list(run_cfg["cosmo_params"])
    bias_params = run_cfg.get("bias_params", [])  # optional

    params = cosmo_params + bias_params

    fid = dict(cfg["defaults"]["theta0"])

    nd = cfg["defaults"].get("theta_round_decimals", None)
    if nd is not None:
        for k, v in fid.items():
            if isinstance(v, (int, float)):
                fid[k] = float(np.round(v, nd))

    theta0 = np.array([fid[p] for p in params], dtype=float)
    return params, theta0, cosmo_params


def build_cosmo(theta_dict: Dict[str, float], cosmo_params: List[str]):
    """Build cosmology."""
    from lbg_desc_forecast.cosmo_factories import MainCosmology
    kwargs = {k: theta_dict[k] for k in cosmo_params}
    return MainCosmology(**kwargs)


def configure_mappers(theta_dict: Dict[str, float], cfg: dict):
    """Configure mappers (clean systematics + Petri interlopers + biases)."""
    from lbg_desc_forecast.default_lbg import get_lbg_mappers

    cfg_common = cfg["common"]
    year = int(cfg_common["year"])
    contamination = float(cfg_common["contamination"])
    petri = cfg_common["petri_y10"]

    mappers = get_lbg_mappers(year, contamination=contamination)

    # fallback fiducial values
    fid = cfg["defaults"]["theta0"]

    for mapper in mappers:
        mapper.dz = 0.0
        mapper.stretch = 1.0
        mapper.dz_interlopers = 0.0
        mapper.stretch_interlopers = 1.0

        band = mapper.drop_band
        mapper.f_interlopers = float(petri[band]["f_interlopers"])

        if band == "u":
            mapper.g_bias = theta_dict.get("bu", fid["bu"])
            mapper.g_bias_inter = theta_dict.get("bu_int", fid["bu_int"])
        elif band == "g":
            mapper.g_bias = theta_dict.get("bg", fid["bg"])
            mapper.g_bias_inter = theta_dict.get("bg_int", fid["bg_int"])
        elif band == "r":
            mapper.g_bias = theta_dict.get("br", fid["br"])
            mapper.g_bias_inter = theta_dict.get("br_int", fid["br_int"])

    return mappers


def create_signal(theta: np.ndarray, params: List[str], cosmo_params: List[str], cfg: dict) -> np.ndarray:
    """Create the signal (unbiased data vector)."""
    from lbg_desc_forecast.forecaster import Forecaster

    theta_dict = theta_to_dict(theta, params)
    cosmo = build_cosmo(theta_dict, cosmo_params)
    mappers = configure_mappers(theta_dict, cfg)

    forecaster = Forecaster(mappers, cosmo)
    return forecaster.create_signal(add_noise=False)


def create_covariance(theta0: np.ndarray, params: List[str], cosmo_params: List[str], cfg: dict) -> np.ndarray:
    """Create the covariance matrix."""
    from lbg_desc_forecast.forecaster import Forecaster

    theta_dict = theta_to_dict(theta0, params)
    cosmo = build_cosmo(theta_dict, cosmo_params)
    mappers = configure_mappers(theta_dict, cfg)

    forecaster = Forecaster(mappers, cosmo)
    print("Generating covariance matrix...", flush=True)
    forecaster.create_cov()
    return forecaster.cov


def create_biased_signal(
    theta0: np.ndarray,
    params: List[str],
    cosmo_params: List[str],
    cfg: dict,
    *,
    param_name: str,
    bias_val: float,
    band: Optional[str] = None,
) -> np.ndarray:
    """Create a biased data vector by setting mapper.<param_name> = bias_val (optionally for one band)."""
    from lbg_desc_forecast.forecaster import Forecaster

    theta_dict = theta_to_dict(theta0, params)
    cosmo = build_cosmo(theta_dict, cosmo_params)
    mappers = configure_mappers(theta_dict, cfg)

    for mapper in mappers:
        if band is None or mapper.drop_band == band:
            setattr(mapper, param_name, bias_val)

    forecaster = Forecaster(mappers, cosmo)
    return forecaster.create_signal(add_noise=False)


def build_prior_fisher(cfg: dict, params: List[str]) -> np.ndarray:
    """Diagonal Gaussian priors: 1/sigma^2 on the diagonal (0 if not specified)."""
    sigmas = cfg["defaults"]["priors"]["sigma"]
    diag = np.zeros(len(params), dtype=float)

    for i, p in enumerate(params):
        s = sigmas.get(p, None)
        if s is not None:
            diag[i] = 1.0 / (float(s) ** 2)

    return np.diag(diag)


def load_or_make_default_cov(cfg: dict, cfg_dir: Path) -> np.ndarray:
    """
    Load the default covariance matrix (LCDM) from:
        <outdir>/lcdm/cov_lcdm.npy
    If missing, create it and save it there.
    """
    outroot = Path(cfg["paths"]["outdir"]).expanduser()
    if not outroot.is_absolute():
        outroot = (cfg_dir / outroot).resolve()

    lcdm_dir = outroot / "lcdm"
    lcdm_dir.mkdir(parents=True, exist_ok=True)
    cov_path = lcdm_dir / "cov_lcdm.npy"

    if cov_path.exists():
        print("Loaded default covariance.", flush=True)
        return np.load(cov_path)

    params, theta0, cosmo_params = build_run_vectors(cfg, "lcdm")
    cov = create_covariance(theta0, params, cosmo_params, cfg)

    np.save(cov_path, cov)
    print("Saved default covariance.", flush=True)
    return cov


def load_run_artifacts(outdir: Path, run_name: str, fisher_kind: str = "total"):
    """
    Load params/theta0/cov and a Fisher matrix from an existing fisher run.
    fisher_kind: "total" or "like"
    """
    params = list(np.load(outdir / f"params_{run_name}.npy", allow_pickle=True))
    theta0 = np.load(outdir / f"theta0_{run_name}.npy")
    cov = np.load(outdir / "cov_used.npy")

    if fisher_kind == "like":
        F = np.load(outdir / f"fisher_like_{run_name}.npy")
    else:
        F = np.load(outdir / f"fisher_total_{run_name}.npy")

    return params, theta0, cov, F


def parse_scenarios(cfg: dict, scenario_set: str):
    """
    Simple schema only (your YAML):

      scenarios:
        dz: { values: [...], bands: [u,g,r] }
        dz_interlopers: { values: [...], bands: [u,g,r] }
        stretch: { values: [...], bands: [u,g,r] }
        stretch_interlopers: { values: [...], bands: [u,g,r] }
        f_interlopers:
          custom:
            u: [...]
            g: [...]
            r: [...]

    scenario_set:
      - "all" expands all keys under cfg["scenarios"]
      - or one of the keys above (e.g. "dz")
    """
    sc = cfg.get("scenarios")
    if not isinstance(sc, dict) or not sc:
        raise KeyError("YAML must contain top-level 'scenarios:' dict.")

    keys = list(sc.keys())
    use_keys = keys if scenario_set == "all" else [scenario_set]
    if scenario_set != "all" and scenario_set not in sc:
        raise KeyError(f"Missing scenario-set '{scenario_set}'. Available: {keys} (or use 'all').")

    expanded = []

    for param in use_keys:
        spec = sc[param]

        # special: f_interlopers custom per band
        if param == "f_interlopers":
            if not (isinstance(spec, dict) and isinstance(spec.get("custom"), dict)):
                raise ValueError("scenarios.f_interlopers must be {custom: {band: [vals...]}}")
            for band, vals in spec["custom"].items():
                for v in vals:
                    expanded.append(
                        {"name": f"f_int_{band}_{v}", "param": "f_interlopers", "val": float(v), "band": str(band)}
                    )
            continue

        # generic: values + bands
        if not (isinstance(spec, dict) and "values" in spec and "bands" in spec):
            raise ValueError(f"Scenario '{param}' must be dict with keys 'values' and 'bands'.")

        for band in spec["bands"]:
            for v in spec["values"]:
                tag = f"{v}".replace(".", "p")  # keep filenames clean
                expanded.append({"name": f"{param}_{band}_{tag}", "param": param, "val": float(v), "band": str(band)})

    return expanded
