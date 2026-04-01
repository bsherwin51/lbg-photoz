"""Plot Fisher-forecast constraints as GetDist triangle plots for a full 3x2pt parameter set.

This script reads a Fisher matrix, constructs the corresponding marginalized covariance,
builds the full parameter definition from cosmology / intrinsic-alignment / galaxy-bias
YAML files, and produces:

1. a full triangle plot for the entire parameter vector, and
2. optionally, a subset triangle plot for one parameter sector ("cosmo", "gbias", or "ia"),

while always deriving subset constraints from the full marginalized covariance.

Typical use
-----------
Auto-select the default Fisher matrix for Year 1:
    python plot_fisher.py --year y1

Auto-select the default Fisher matrix for Year 10:
    python plot_fisher.py --year y10

Plot only the cosmology subset while still marginalizing over all parameters:
    python plot_fisher.py --year y1 --subset cosmo

Use a specific Fisher matrix file:
    python plot_fisher.py --year y1 --fisher path/to/fisher.npy

Load a no-priors Fisher file inferred from the default filename pattern
(note that this script runs with no priors on m_nu by default):
    python plot_fisher.py --year y1 --no-priors m_nu

Load a no-priors Fisher file removing multiple priors in the filename:
    python plot_fisher.py --year y1 --no-priors m_nu w0 wa

Exclude IA or galaxy-bias parameters from the parameter definition:
    python plot_fisher.py --year y1 --no-ia
    python plot_fisher.py --year y1 --no-gbias

Choose a custom output directory:
    python plot_fisher.py --year y1 --output-dir outputs/plots

Notes
-----
- The subset plot is always extracted from the full covariance, so cosmology-only
  uncertainties still include marginalization over nuisance parameters present in the run.
- If --fisher is given, that file is used directly.
- If --fisher is omitted, the script auto-selects the default Fisher file from --year.
- If --no-priors is provided without --fisher, the script modifies the default Fisher
  filename by appending "_noprior_<param1>_<param2>...".
"""

import argparse
from pathlib import Path

import numpy as np
from getdist import plots as getdist_plots
from getdist.gaussian_mixtures import GaussianND

from fishers_srd_mu.helpers import (
    DEFAULT_COSMO_YAML,
    DEFAULT_GALAXY_BIAS_YAML,
    DEFAULT_IA_YAML,
    build_parameter_definition,
    choose_default_fisher,
    fisher_to_cov,
    get_sector_names,
    normalize_year,
    script_dir,
    validate_fisher_shape,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot Fisher constraints for a full 3x2pt parameter set and "
            "optionally for a marginalized parameter subset."
        )
    )

    parser.add_argument(
        "--year",
        type=str,
        required=True,
        choices=["y1", "y10"],
        help="Forecast year used to define defaults and parameter content.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["cosmo", "gbias", "ia"],
        help="Optional parameter sector to extract from the full covariance.",
    )

    parser.add_argument(
        "--fisher",
        type=str,
        default=None,
        help=(
            "Optional Fisher matrix path. If omitted, the script auto-selects "
            "the default Fisher file for the chosen year."
        ),
    )

    parser.add_argument(
        "--cosmo-yaml",
        type=str,
        default=None,
        help=(
            "Optional cosmology YAML override. Defaults to the standard "
            "cosmology YAML in this directory."
        ),
    )

    parser.add_argument(
        "--ia-yaml",
        type=str,
        default=str(DEFAULT_IA_YAML),
        help="Path to the intrinsic-alignment parameter YAML file.",
    )

    parser.add_argument(
        "--galaxy-bias-yaml",
        type=str,
        default=str(DEFAULT_GALAXY_BIAS_YAML),
        help="Path to the galaxy-bias parameter YAML file.",
    )

    parser.add_argument(
        "--no-ia",
        action="store_true",
        help="Exclude intrinsic-alignment parameters from the parameter vector.",
    )

    parser.add_argument(
        "--no-gbias",
        action="store_true",
        help="Exclude galaxy-bias parameters from the parameter vector.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(script_dir) / "plots"),
        help="Directory where output plots will be written.",
    )

    parser.add_argument(
        "--width-all",
        type=float,
        default=11.0,
        help="Figure width in inches for the full triangle plot.",
    )

    parser.add_argument(
        "--width-subset",
        type=float,
        default=7.0,
        help="Figure width in inches for the subset triangle plot.",
    )

    parser.add_argument(
        "--params",
        nargs="+",
        default=None,
        help=(
            "Optional explicit list of parameters to plot from the full "
            "marginalized covariance, e.g. --params omega_m sigma_8 w_0 w_a m_nu."
        ),
    )

    parser.add_argument(
        "--no-priors",
        nargs="+",
        default=["m_nu"],
        help=(
            "Infer a no-priors Fisher filename from the default Fisher file by "
            "appending '_noprior_<param1>_<param2>...'. Ignored if --fisher is given."
        ),
    )

    return parser.parse_args()


def build_getdist_gaussian_from_cov(
    theta0: np.ndarray,
    cov: np.ndarray,
    names: list[str],
    labels: list[str],
    label: str = "Fisher (Gaussian)",
) -> GaussianND:
    """Build a GetDist GaussianND object from a mean and covariance."""
    return GaussianND(
        mean=theta0,
        cov=cov,
        names=names,
        labels=labels,
        label=label,
    )


def print_diagnostics(fisher: np.ndarray, names: list[str]) -> np.ndarray:
    """Print marginalized and conditional parameter uncertainties."""
    cov = fisher_to_cov(fisher)

    sigma_marg = np.sqrt(np.diag(cov))
    sigma_cond = np.sqrt(1.0 / np.diag(fisher))

    print("\n============================================================")
    print("Parameter diagnostics")
    print("============================================================")
    print(f"{'parameter':15s} {'sigma_marg':>15s} {'sigma_cond':>15s}")
    print("------------------------------------------------------------")

    for name, sm, sc in zip(names, sigma_marg, sigma_cond):
        print(f"{name:15s} {sm:15.6e} {sc:15.6e}")

    print("------------------------------------------------------------")
    print("sigma_marg : marginalized uncertainty from full covariance")
    print("sigma_cond : conditional uncertainty from Fisher diagonal")
    print("============================================================\n")

    return sigma_marg


def subset_from_full_covariance(
    full_fisher: np.ndarray,
    full_theta0: np.ndarray,
    full_names: list[str],
    full_labels: list[str],
    want_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract a parameter subset from the full marginalized covariance."""
    full_cov = fisher_to_cov(full_fisher)
    idx = [full_names.index(name) for name in want_names]

    sub_cov = full_cov[np.ix_(idx, idx)]
    sub_theta0 = full_theta0[idx]
    sub_labels = [full_labels[i] for i in idx]

    return sub_cov, sub_theta0, sub_labels


def make_triangle_plot(
    gaussian: GaussianND,
    params: list[str],
    output_path: Path,
    width_inch: float = 9.0,
    contour_color: str = "#3b9ab2",
    line_width: float = 1.5,
) -> None:
    """Create and save a GetDist triangle plot."""
    plotter = getdist_plots.get_subplot_plotter(width_inch=width_inch)
    plotter.settings.linewidth = line_width
    plotter.settings.linewidth_contour = line_width
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 11
    plotter.settings.alpha_filled_add = 0.0

    param_limits = {}
    if "m_nu" in params:
        param_limits["m_nu"] = (0.0, None)

    plotter.triangle_plot(
        [gaussian],
        params=params,
        filled=[False],
        contour_colors=[contour_color],
        contour_lws=[line_width],
        contour_ls=["-"],
        param_limits=param_limits if param_limits else None,
    )

    plotter.export(str(output_path))
    print(f"Saved triangle plot to: {output_path}")


def resolve_fisher_path(args: argparse.Namespace) -> Path:
    """Resolve the Fisher matrix path from CLI options."""
    if args.fisher is not None:
        return Path(args.fisher)

    base_fisher = choose_default_fisher(args.year)

    if args.no_priors:
        suffix = "_noprior_" + "_".join(args.no_priors)
        return base_fisher.with_name(base_fisher.stem + suffix + base_fisher.suffix)

    return base_fisher


def main() -> None:
    """Run the Fisher plotting workflow."""
    args = parse_args()

    forecast_year = args.year
    year_clean = normalize_year(forecast_year)
    subset_sector = args.subset

    fisher_path = resolve_fisher_path(args)

    cosmo_yaml = Path(args.cosmo_yaml) if args.cosmo_yaml is not None else DEFAULT_COSMO_YAML
    ia_yaml = Path(args.ia_yaml)
    galaxy_bias_yaml = Path(args.galaxy_bias_yaml)

    include_cosmo = True
    include_ia = not args.no_ia
    include_galaxy_bias = not args.no_gbias

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_all = output_dir / f"getdist_triangle_all_y{year_clean}.pdf"

    if args.params is not None:
        subset_tag = "_".join(args.params)
        output_file_subset = output_dir / f"getdist_triangle_{subset_tag}_y{year_clean}.pdf"
    elif subset_sector is not None:
        output_file_subset = output_dir / f"getdist_triangle_{subset_sector}_y{year_clean}.pdf"
    else:
        output_file_subset = None

    print(f"\nUsing year           : y{year_clean}")
    print(f"Using Fisher file    : {fisher_path}")
    print(f"Using cosmology YAML : {cosmo_yaml}")
    print(f"Using gbias YAML     : {galaxy_bias_yaml}")
    print(f"Using IA YAML        : {ia_yaml}")
    print(f"Include cosmology    : {include_cosmo}")
    print(f"Include galaxy bias  : {include_galaxy_bias}")
    print(f"Include IA           : {include_ia}")

    param_names, param_labels, theta0 = build_parameter_definition(
        cosmo_path=cosmo_yaml,
        ia_path=ia_yaml if include_ia else None,
        galaxy_bias_path=galaxy_bias_yaml if include_galaxy_bias else None,
        forecast_year=forecast_year,
        include_cosmo=include_cosmo,
        include_ia=include_ia,
        include_galaxy_bias=include_galaxy_bias,
    )

    print("\nParameter order:")
    for i, name in enumerate(param_names):
        print(f"  [{i:02d}] {name}")

    fisher = np.load(fisher_path)
    fisher = np.asarray(fisher, dtype=float)

    validate_fisher_shape(fisher, param_names)
    print_diagnostics(fisher, param_names)

    full_cov = fisher_to_cov(fisher)

    gnd_all = build_getdist_gaussian_from_cov(
        theta0=theta0,
        cov=full_cov,
        names=param_names,
        labels=param_labels,
        label=f"3x2pt Y{year_clean} Fisher",
    )

    make_triangle_plot(
        gaussian=gnd_all,
        params=param_names,
        output_path=output_file_all,
        width_inch=args.width_all,
    )

    if args.params is not None:
        subset_names = args.params

        missing = [name for name in subset_names if name not in param_names]
        if missing:
            raise ValueError(
                f"Requested parameters are not in the full parameter list: {missing}"
            )

    elif subset_sector is not None:
        subset_names = get_sector_names(param_names, subset_sector)

        if not subset_names:
            raise ValueError(
                f"Subset '{subset_sector}' is empty for the current parameter set."
            )

    else:
        subset_names = None

    if subset_names is not None:
        sub_cov, sub_theta0, sub_labels = subset_from_full_covariance(
            full_fisher=fisher,
            full_theta0=theta0,
            full_names=param_names,
            full_labels=param_labels,
            want_names=subset_names,
        )

        if args.params is not None:
            print("\nExplicit parameter subset:")
        else:
            print(f"\nSubset sector: {subset_sector}")

        print("Subset parameters:")
        for i, name in enumerate(subset_names):
            print(f"  [{i:02d}] {name}")

        sub_sigma = np.sqrt(np.diag(sub_cov))
        print("\nMarginalized sigmas in subset (from FULL covariance):")
        for name, sig in zip(subset_names, sub_sigma):
            print(f"  {name:15s} {sig:.6e}")

        subset_label = (
            f"3x2pt Y{year_clean} Fisher (explicit subset)"
            if args.params is not None
            else f"3x2pt Y{year_clean} Fisher ({subset_sector} subset)"
        )

        gnd_subset = build_getdist_gaussian_from_cov(
            theta0=sub_theta0,
            cov=sub_cov,
            names=subset_names,
            labels=sub_labels,
            label=subset_label,
        )

        make_triangle_plot(
            gaussian=gnd_subset,
            params=subset_names,
            output_path=output_file_subset,
            width_inch=args.width_subset,
        )


if __name__ == "__main__":
    main()
