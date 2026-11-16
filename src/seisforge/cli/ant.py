"""Ambient-noise command line interfaces."""

from __future__ import annotations

import argparse


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "ant",
        help="Ambient-noise workflows.",
        description="Ambient-noise workflows.",
    )
    parser.set_defaults(func=_print_help, parser=parser)

    ant_subparsers = parser.add_subparsers(dest="ant_command", metavar="<command>")
    _register_rotate_ccf(ant_subparsers)
    _register_aftan(ant_subparsers)
    _register_eikonal(ant_subparsers)
    return parser


def _register_rotate_ccf(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "rotate-ccf",
        help="Rotate the CCFs of one station pair from ZNE to ZRT.",
        description=(
            "Rotate the ZNE SAC cross-correlations of one station pair to ZRT. "
            "Options override the config; loop in a shell script for many pairs."
        ),
    )
    parser.add_argument("-c", "--config-file", required=True, help="rotate-ccf YAML config.")
    parser.add_argument(
        "--input-template", default=None,
        help="Input SAC path with {component}, e.g. 'CCF/A_B_{component}.SAC'.",
    )
    parser.add_argument(
        "--output-template", default=None,
        help="Output SAC path with {component}, e.g. 'ZRT/A_B_{component}.SAC'.",
    )
    parser.add_argument(
        "--components", default=None, help="Comma-separated ZRT components, e.g. ZZ,ZR,RR."
    )
    parser.add_argument("--azimuth", type=float, default=None, help="Azimuth A->B in degrees.")
    parser.add_argument(
        "--back-azimuth", type=float, default=None, help="Back azimuth B->A in degrees."
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=None, help="Replace existing output files."
    )
    parser.set_defaults(func=_run_rotate_ccf)


def _register_aftan(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "aftan",
        help="Measure dispersion with pure-Python FTAN.",
        description="Measure dispersion curves from SAC CCFs with pure-Python FTAN.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="Path to one station-level AFTAN YAML configuration file.",
    )
    parser.add_argument(
        "--write-energy-map",
        action="store_true",
        default=None,
        help="Write FTAN energy maps as xarray/netCDF files.",
    )
    parser.add_argument(
        "--plot-energy-map",
        action="store_true",
        default=None,
        help="Write FTAN period-velocity energy map PNGs.",
    )
    parser.set_defaults(func=_run_aftan)


def _register_eikonal(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "eikonal",
        help=(
            "Image surface-wave phase velocity with Eikonal/Helmholtz fields."
        ),
        description=(
            "Build single-source Eikonal/Helmholtz fields and an "
            "azimuth-balanced isotropic phase-velocity stack."
        ),
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="Path to an Eikonal/Helmholtz YAML configuration file.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "fields", "stack"),
        default="all",
        help="Run the full workflow, event fields only, or stacking only.",
    )
    parser.set_defaults(func=_run_eikonal)


def _run_rotate_ccf(args: argparse.Namespace) -> int:
    from seisforge.ant.ccf_rotate import run_rotate_ccf_config

    result = run_rotate_ccf_config(
        args.config_file,
        input_template=args.input_template,
        output_template=args.output_template,
        components=args.components,
        azimuth=args.azimuth,
        back_azimuth=args.back_azimuth,
        overwrite=args.overwrite,
    )
    print(f"wrote {len(result.written)} rotated CCF files; log: {result.log_file}")
    return 0


def _run_aftan(args: argparse.Namespace) -> int:
    from seisforge.ant.aftan import run_aftan_config

    results = run_aftan_config(
        args.config_file,
        write_energy_map=args.write_energy_map,
        plot_energy_map=args.plot_energy_map,
    )
    print(f"measured {len(results)} AFTAN traces")
    return 0


def _run_eikonal(args: argparse.Namespace) -> int:
    from seisforge.ant.eikonal import run_eikonal_config

    outputs = run_eikonal_config(args.config_file, stage=args.stage)
    for output in outputs:
        print(f"wrote {output}")
    return 0


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0
