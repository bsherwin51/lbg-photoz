"""Helper utilities for Fisher-matrix plotting and prior-manipulation scripts.

This module centralizes shared configuration, parameter ordering, label formatting,
YAML loading, and Fisher-matrix validation utilities used by the plotting and
prior-removal scripts in this directory.

What this module provides
-------------------------
- Default file locations for Fisher matrices and parameter YAML files
- Canonical parameter ordering for cosmology, galaxy-bias, and IA sectors
- Display labels for GetDist / plotting
- Helpers to build parameter names and fiducial vectors from YAML files
- Helpers to extract sector-specific parameter subsets
- Fisher-to-covariance conversion and shape validation

Typical use
-----------
Import the helpers you need from another script, for example:

    from helpers import (
        choose_default_fisher,
        build_parameter_definition,
        fisher_to_cov,
        validate_fisher_shape,
    )

This file is intended to be imported, not executed directly.
"""

from pathlib import Path

import numpy as np
import yaml

script_dir = Path(__file__).resolve().parent
params_dir = script_dir / "params"

DEFAULT_FISHER_Y1 = script_dir / "srd+mu_y1_3x2pt_fisher_matrix_all_ellssrd.npy"
DEFAULT_FISHER_Y10 = script_dir / "srd+mu_y10_3x2pt_fisher_matrix_all_ellssrd.npy"

DEFAULT_COSMO_YAML = params_dir / "cosmological_parameters_mu.yaml"
DEFAULT_IA_YAML = params_dir / "intrinsic_alignment_parameters.yaml"
DEFAULT_GALAXY_BIAS_YAML = params_dir / "galaxy_bias_parameters.yaml"

LABEL_MAP = {
    "omega_m": r"\Omega_{\rm m}",
    "sigma_8": r"\sigma_8",
    "n_s": r"n_{\rm s}",
    "w_0": r"w_0",
    "w_a": r"w_a",
    "omega_b": r"\Omega_{\rm b}",
    "h": r"h",
    "m_nu": r"\sum m_\nu",
    "a_0": r"A_0",
    "beta": r"\beta",
    "eta_low_z": r"\eta_{\rm low\!-\!z}",
    "eta_high_z": r"\eta_{\rm high\!-\!z}",
}

COSMO_ORDER = [
    "omega_m",
    "sigma_8",
    "n_s",
    "w_0",
    "w_a",
    "omega_b",
    "h",
    "m_nu",
]

IA_ORDER = [
    "a_0",
    "beta",
    "eta_low_z",
    "eta_high_z",
]


def normalize_year(year: str) -> str:
    """Normalize forecast year labels to '1' or '10'."""
    year_clean = year.lower().replace("y", "")
    if year_clean not in {"1", "10"}:
        raise ValueError(f"Unsupported year '{year}'. Use y1 or y10.")
    return year_clean


def choose_default_fisher(year: str) -> Path:
    """Return the default Fisher-matrix path for the requested forecast year."""
    year_clean = normalize_year(year)
    if year_clean == "1":
        return DEFAULT_FISHER_Y1
    if year_clean == "10":
        return DEFAULT_FISHER_Y10
    raise ValueError(f"Unsupported year '{year}'.")


def get_n_gbias(year: str) -> int:
    """Return the number of galaxy-bias parameters for the chosen year."""
    year_clean = normalize_year(year)
    return 5 if year_clean == "1" else 10


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r") as file_handle:
        return yaml.safe_load(file_handle)


def get_label(name: str) -> str:
    """Return the plotting label for a parameter name."""
    if name in LABEL_MAP:
        return LABEL_MAP[name]
    if name.startswith("b_"):
        idx = name.split("_", 1)[1]
        return rf"b_{{{idx}}}"
    return name


def build_parameter_names(
    year: str,
    include_cosmo: bool = True,
    include_gbias: bool = True,
    include_ia: bool = True,
) -> list[str]:
    """Build the ordered parameter-name list for the selected model sectors."""
    names: list[str] = []

    if include_cosmo:
        names.extend(COSMO_ORDER)

    if include_gbias:
        n_bias = get_n_gbias(year)
        names.extend([f"b_{i}" for i in range(1, n_bias + 1)])

    if include_ia:
        names.extend(IA_ORDER)

    return names


def build_parameter_definition(
    cosmo_path: Path,
    ia_path: Path | None,
    galaxy_bias_path: Path | None,
    forecast_year: str,
    include_cosmo: bool = True,
    include_ia: bool = True,
    include_galaxy_bias: bool = True,
) -> tuple[list[str], list[str], np.ndarray]:
    """Build ordered parameter names, labels, and fiducial values from YAML inputs."""
    names: list[str] = []
    theta0: list[float] = []

    if include_cosmo:
        cosmo_cfg = load_yaml(cosmo_path)
        cosmo_fid = cosmo_cfg["fiducial_values"]

        missing = [param for param in COSMO_ORDER if param not in cosmo_fid]
        if missing:
            raise KeyError(f"Missing cosmology fiducials: {missing}")

        names.extend(COSMO_ORDER)
        theta0.extend([float(cosmo_fid[param]) for param in COSMO_ORDER])

    if include_galaxy_bias:
        if galaxy_bias_path is None:
            raise ValueError("include_galaxy_bias=True but galaxy_bias_path is None")

        gb_cfg = load_yaml(galaxy_bias_path)

        if "fiducial_values" not in gb_cfg:
            raise KeyError(f"No 'fiducial_values' section found in {galaxy_bias_path}")

        gb_fid_all = gb_cfg["fiducial_values"]
        year_key = normalize_year(forecast_year)

        if not isinstance(gb_fid_all, dict):
            raise TypeError(
                f"'fiducial_values' in {galaxy_bias_path} must be a dictionary"
            )

        if year_key in gb_fid_all:
            gb_fid = gb_fid_all[year_key]
        elif f"y{year_key}" in gb_fid_all:
            gb_fid = gb_fid_all[f"y{year_key}"]
        else:
            gb_fid = gb_fid_all

        n_bias = get_n_gbias(forecast_year)
        gb_order = [f"b_{i}" for i in range(1, n_bias + 1)]

        missing = [param for param in gb_order if param not in gb_fid]
        if missing:
            raise KeyError(
                f"Missing galaxy bias fiducials for year {forecast_year}: {missing}"
            )

        names.extend(gb_order)
        theta0.extend([float(gb_fid[param]) for param in gb_order])

    if include_ia:
        if ia_path is None:
            raise ValueError("include_ia=True but ia_path is None")

        ia_cfg = load_yaml(ia_path)
        ia_fid = ia_cfg["fiducial_values"]

        missing = [param for param in IA_ORDER if param not in ia_fid]
        if missing:
            raise KeyError(f"Missing IA fiducials: {missing}")

        names.extend(IA_ORDER)
        theta0.extend([float(ia_fid[param]) for param in IA_ORDER])

    labels = [get_label(param) for param in names]
    return names, labels, np.array(theta0, dtype=float)


def get_sector_names(full_names: list[str], sector: str) -> list[str]:
    """Return the ordered parameter names belonging to one model sector."""
    if sector == "cosmo":
        return [param for param in COSMO_ORDER if param in full_names]

    if sector == "gbias":
        return [param for param in full_names if param.startswith("b_")]

    if sector == "ia":
        return [param for param in IA_ORDER if param in full_names]

    raise ValueError(f"Unknown sector '{sector}'. Choose from 'cosmo', 'gbias', 'ia'.")


def fisher_to_cov(fisher: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Convert a Fisher matrix to a covariance matrix using a pseudo-inverse."""
    return np.linalg.pinv(fisher, rcond=rcond)


def validate_fisher_shape(fisher: np.ndarray, param_names: list[str]) -> None:
    """Validate that a Fisher matrix is square and matches the parameter count."""
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"Fisher matrix must be square, got shape {fisher.shape}")

    if fisher.shape[0] != len(param_names):
        raise ValueError(
            f"Fisher shape is {fisher.shape}, but parameter list has length {len(param_names)}"
        )
