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
    return parser


def _register_rotate_ccf(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "rotate-ccf",
        help="Rotate CCFs from ZNE to ZRT.",
        description="Rotate SAC cross-correlation functions from ZNE to ZRT.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="Path to the rotate-ccf YAML configuration file.",
    )
    parser.add_argument("--input-dir", default=None, help="Override input directory.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--input-template",
        default=None,
        help="Override input template, e.g. '{component}.SAC'.",
    )
    parser.add_argument(
        "--output-template",
        default=None,
        help="Override output template, e.g. '{source_stem}_rot.SAC'.",
    )
    parser.add_argument(
        "--obj-components",
        "--components",
        dest="obj_components",
        default=None,
        help="Comma-separated output ZRT components, e.g. ZR,ZT,RR.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=None,
        help="Override interstation azimuth.",
    )
    parser.add_argument(
        "--back-azimuth",
        type=float,
        default=None,
        help="Override interstation back azimuth.",
    )
    parser.add_argument(
        "--radians",
        dest="degrees",
        action="store_false",
        default=None,
        help="Interpret azimuths as radians instead of degrees.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Allow replacing existing output SAC files.",
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


def _run_rotate_ccf(args: argparse.Namespace) -> int:
    from seisforge.ant.ccf import run_rotate_ccf_config

    results = run_rotate_ccf_config(
        args.config_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_template=args.input_template,
        output_template=args.output_template,
        obj_components=args.obj_components,
        azimuth=args.azimuth,
        back_azimuth=args.back_azimuth,
        degrees=args.degrees,
        overwrite=args.overwrite,
    )
    for result in results:
        label = result.job.name or str(result.job.input_dir)
        print(f"{label}: wrote {len(result.written)} rotated CCF files")
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


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0
