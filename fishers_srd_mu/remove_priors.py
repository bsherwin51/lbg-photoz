"""Remove selected Gaussian prior contributions from a Fisher matrix.

This script loads a Fisher matrix, reconstructs the parameter ordering from the
chosen forecast setup, reads prior widths from the cosmology / intrinsic-alignment /
galaxy-bias YAML files, and subtracts the diagonal Fisher contribution of selected
Gaussian priors.

This is useful when you want to compare constraints with and without specific priors
without recomputing the full Fisher matrix from scratch.

Typical use
-----------
Use the default Fisher matrix for Year 1 and remove the neutrino-mass prior:
    python remove_priors.py --year y1 --remove-prior m_nu

Use the default Fisher matrix for Year 10 and remove multiple priors:
    python remove_priors.py --year y10 --remove-prior m_nu w_0 wa

Use a specific Fisher matrix file:
    python remove_priors.py --year y1 --fisher path/to/fisher.npy --remove-prior m_nu

Write the result to a custom output file:
    python remove_priors.py --year y1 --remove-prior m_nu --output fisher_no_mnu_prior.npy

Exclude IA or galaxy-bias parameters from the assumed parameter ordering:
    python remove_priors.py --year y1 --remove-prior m_nu --no-ia
    python remove_priors.py --year y1 --remove-prior m_nu --no-gbias

Notes
-----
- The script assumes the priors are Gaussian and contribute only to the diagonal
  of the Fisher matrix.
- The prior widths are read from the YAML configuration files.
- If --output is not given, the script writes a new file next to the input Fisher
  file using the suffix pattern:
      _noprior_<param1>_<param2>...
"""

import argparse
from pathlib import Path

import numpy as np

from fishers_srd_mu.helpers import (
    COSMO_ORDER,
    DEFAULT_COSMO_YAML,
    DEFAULT_GALAXY_BIAS_YAML,
    DEFAULT_IA_YAML,
    IA_ORDER,
    build_parameter_names,
    choose_default_fisher,
    get_n_gbias,
    load_yaml,
    normalize_year,
)


def extract_prior_sigmas(
    cosmo_yaml: Path,
    ia_yaml: Path,
    gbias_yaml: Path,
    year: str,
    include_cosmo: bool = True,
    include_gbias: bool = True,
    include_ia: bool = True,
) -> dict[str, float]:
    """Collect prior widths for all included parameters from the YAML files."""
    prior_sigmas: dict[str, float] = {}
    year_key = normalize_year(year)

    if include_cosmo:
        cfg = load_yaml(cosmo_yaml)
        sigmas = cfg.get("sigmas", {})
        for param in COSMO_ORDER:
            if param in sigmas:
                prior_sigmas[param] = float(sigmas[param])

    if include_gbias:
        cfg = load_yaml(gbias_yaml)
        sigmas_all = cfg.get("sigmas", {})

        if isinstance(sigmas_all, dict):
            if year_key in sigmas_all and isinstance(sigmas_all[year_key], dict):
                gbias_sigmas = sigmas_all[year_key]
            elif f"y{year_key}" in sigmas_all and isinstance(sigmas_all[f"y{year_key}"], dict):
                gbias_sigmas = sigmas_all[f"y{year_key}"]
            else:
                gbias_sigmas = sigmas_all
        else:
            gbias_sigmas = {}

        for i in range(1, get_n_gbias(year) + 1):
            name = f"b_{i}"
            if name in gbias_sigmas:
                prior_sigmas[name] = float(gbias_sigmas[name])

    if include_ia:
        cfg = load_yaml(ia_yaml)
        sigmas = cfg.get("sigmas", {})
        for param in IA_ORDER:
            if param in sigmas:
                prior_sigmas[param] = float(sigmas[param])

    return prior_sigmas


def build_prior_fisher_diag(
    param_names: list[str],
    prior_sigmas: dict[str, float],
) -> np.ndarray:
    """Build the diagonal Fisher contribution corresponding to the available priors."""
    diag = np.zeros(len(param_names), dtype=float)

    for i, name in enumerate(param_names):
        if name in prior_sigmas:
            sigma = float(prior_sigmas[name])
            if sigma <= 0:
                raise ValueError(f"Prior sigma for '{name}' must be positive, got {sigma}")
            diag[i] = 1.0 / sigma**2

    return diag


def remove_selected_priors_from_fisher(
    fisher: np.ndarray,
    param_names: list[str],
    prior_sigmas: dict[str, float],
    remove_params: str | list[str],
    check_positive_diag: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Remove the selected prior contributions from the Fisher diagonal."""
    if isinstance(remove_params, str):
        remove_params = [remove_params]

    remove_params = list(remove_params)

    unknown = [param for param in remove_params if param not in param_names]
    if unknown:
        raise KeyError(
            f"These requested parameters are not in the Fisher ordering: {unknown}\n"
            f"Available parameters: {param_names}"
        )

    missing_prior = [param for param in remove_params if param not in prior_sigmas]
    if missing_prior:
        raise KeyError(
            f"No prior sigma found in YAML for: {missing_prior}\n"
            f"Available prior sigmas for: {sorted(prior_sigmas.keys())}"
        )

    fisher_new = np.array(fisher, dtype=float, copy=True)
    removed_info: dict[str, float] = {}

    for param in remove_params:
        index = param_names.index(param)
        sigma = float(prior_sigmas[param])
        delta = 1.0 / sigma**2

        fisher_new[index, index] -= delta
        removed_info[param] = delta

        if check_positive_diag and fisher_new[index, index] <= 0:
            raise ValueError(
                f"After removing the prior on '{param}', the diagonal entry became non-positive:\n"
                f"F_new[{param},{param}] = {fisher_new[index, index]:.6e}\n"
                f"This usually means the input Fisher was dominated by that prior or the setup is inconsistent."
            )

    return fisher_new, removed_info


def make_output_name(input_path: Path, removed: list[str]) -> Path:
    """Build the default output filename for a no-prior Fisher matrix."""
    stem = input_path.stem
    suffix = "_noprior_" + "_".join(removed)
    return input_path.with_name(stem + suffix + input_path.suffix)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove selected Gaussian prior contributions from a Fisher matrix."
    )

    parser.add_argument(
        "--year",
        type=str,
        required=True,
        choices=["y1", "y10"],
        help="Forecast year used to define the default Fisher file and parameter ordering.",
    )

    parser.add_argument(
        "--fisher",
        type=str,
        default=None,
        help="Optional Fisher matrix path override. If omitted, the default file for the selected year is used.",
    )

    parser.add_argument(
        "--remove-prior",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Parameter name(s) whose Gaussian prior should be removed. "
            "Example: --remove-prior m_nu   or   --remove-prior m_nu w_0"
        ),
    )

    parser.add_argument(
        "--cosmo-yaml",
        type=str,
        default=str(DEFAULT_COSMO_YAML),
        help="Path to the cosmology YAML file containing prior widths.",
    )

    parser.add_argument(
        "--ia-yaml",
        type=str,
        default=str(DEFAULT_IA_YAML),
        help="Path to the intrinsic-alignment YAML file containing prior widths.",
    )

    parser.add_argument(
        "--gbias-yaml",
        type=str,
        default=str(DEFAULT_GALAXY_BIAS_YAML),
        help="Path to the galaxy-bias YAML file containing prior widths.",
    )

    parser.add_argument(
        "--no-ia",
        action="store_true",
        help="Exclude intrinsic-alignment parameters from the assumed parameter ordering.",
    )

    parser.add_argument(
        "--no-gbias",
        action="store_true",
        help="Exclude galaxy-bias parameters from the assumed parameter ordering.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. If omitted, a filename is generated automatically next to the input Fisher file.",
    )

    return parser.parse_args()


def main() -> None:
    """Load inputs, remove selected priors, and save the updated Fisher matrix."""
    args = parse_args()

    year = args.year

    if args.fisher is None:
        fisher_path = choose_default_fisher(year)
    else:
        fisher_path = Path(args.fisher)

    if not fisher_path.exists():
        raise FileNotFoundError(f"Fisher file not found: {fisher_path}")

    cosmo_yaml = Path(args.cosmo_yaml)
    ia_yaml = Path(args.ia_yaml)
    gbias_yaml = Path(args.gbias_yaml)

    include_cosmo = True
    include_gbias = not args.no_gbias
    include_ia = not args.no_ia

    param_names = build_parameter_names(
        year=year,
        include_cosmo=include_cosmo,
        include_gbias=include_gbias,
        include_ia=include_ia,
    )

    fisher = np.load(fisher_path)
    fisher = np.asarray(fisher, dtype=float)

    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"Fisher matrix must be square, got shape {fisher.shape}")

    if fisher.shape[0] != len(param_names):
        raise ValueError(
            f"Fisher shape is {fisher.shape}, but parameter list has length {len(param_names)}.\n"
            f"Parameter order assumed:\n{param_names}"
        )

    prior_sigmas = extract_prior_sigmas(
        cosmo_yaml=cosmo_yaml,
        ia_yaml=ia_yaml,
        gbias_yaml=gbias_yaml,
        year=year,
        include_cosmo=include_cosmo,
        include_gbias=include_gbias,
        include_ia=include_ia,
    )

    fisher_new, removed_info = remove_selected_priors_from_fisher(
        fisher=fisher,
        param_names=param_names,
        prior_sigmas=prior_sigmas,
        remove_params=args.remove_prior,
    )

    if args.output is None:
        output_path = make_output_name(fisher_path, args.remove_prior)
    else:
        output_path = Path(args.output)

    np.save(output_path, fisher_new)

    print("\n============================================================")
    print("Removed prior contribution(s) from Fisher matrix")
    print("============================================================")
    print(f"Input Fisher : {fisher_path}")
    print(f"Output Fisher: {output_path}")
    print(f"Year         : {year}")

    print("\nParameter order:")
    for i, name in enumerate(param_names):
        print(f"  [{i:02d}] {name}")

    print("\nRemoved prior Fisher diagonal terms:")
    for name in args.remove_prior:
        sigma = prior_sigmas[name]
        delta = removed_info[name]
        index = param_names.index(name)
        print(
            f"  {name:12s} index={index:2d}  sigma_prior={sigma:.6e}  "
            f"subtracted={delta:.6e}"
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
