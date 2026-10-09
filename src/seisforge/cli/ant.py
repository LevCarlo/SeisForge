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
    _register_aftan_cycle_set(ant_subparsers)
    _register_eikonal(ant_subparsers)
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
    parser.add_argument(
        "--root-datadir",
        default=None,
        help="Override ZNE CCF root data directory.",
    )
    parser.add_argument(
        "--output-root-datadir",
        default=None,
        help="Override ZRT CCF output root data directory.",
    )
    parser.add_argument(
        "--source-station",
        default=None,
        help="Override source station code.",
    )
    parser.add_argument(
        "--receiver-station",
        default=None,
        help="Override receiver station code.",
    )
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


def _register_aftan_cycle_set(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "aftan-cycle-set",
        help="Preview or manually set one saved AFTAN phase-cycle branch.",
        description=(
            "Recompute phase velocity from the uncorrected curve stored in one "
            "AFTAN .vph.nc file. By default this is a read-only preview; pass "
            "--in-place to update the phase map and companion .dat file."
        ),
    )
    parser.add_argument(
        "phase_map",
        help="Path to a basic or PMF AFTAN phase map ending in .vph.nc.",
    )
    parser.add_argument(
        "--shift",
        type=int,
        required=True,
        help="Absolute global integer cycle shift relative to the uncorrected curve.",
    )
    parser.add_argument(
        "--dat-file",
        default=None,
        help="Companion dispersion .dat file; inferred from the phase-map name.",
    )
    parser.add_argument(
        "--phase-map-only",
        action="store_true",
        help="Update only the phase-map NetCDF and do not require a .dat file.",
    )
    parser.add_argument(
        "--reason",
        default=None,
        help="Short reason stored with an in-place manual override.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Apply the override; without this flag the command only previews it.",
    )
    parser.set_defaults(func=_run_aftan_cycle_set)


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

    results = run_rotate_ccf_config(
        args.config_file,
        root_datadir=args.root_datadir,
        output_root_datadir=args.output_root_datadir,
        source_station=args.source_station,
        receiver_station=args.receiver_station,
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


def _run_aftan_cycle_set(args: argparse.Namespace) -> int:
    from seisforge.ant.aftan import override_phase_cycle

    result = override_phase_cycle(
        args.phase_map,
        args.shift,
        dat_file=args.dat_file,
        phase_map_only=args.phase_map_only,
        write=args.in_place,
        reason=args.reason,
    )
    action = "updated" if result.written else "preview"
    print(
        f"{action}: shift {result.previous_shift:+d} -> "
        f"{result.requested_shift:+d}; "
        f"reference misfit={result.reference_misfit:.4f}; "
        f"vph={result.phase_velocity_min_km_s:.4f}-"
        f"{result.phase_velocity_max_km_s:.4f} km/s"
    )
    print(f"phase_map: {result.phase_map}")
    if result.dat_file is not None:
        print(f"dat_file: {result.dat_file}")
    if not result.written:
        print("No files were changed. Add --in-place to apply this override.")
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
