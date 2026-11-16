"""Ambient-noise command line interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    _register_collect_dispersion(ant_subparsers)
    _register_extract_station_curve(ant_subparsers)
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
        "--radial-convention",
        choices=("lin2008", "legacy"),
        default=None,
        help=(
            "R/T axis directions: lin2008 (default; R along the propagation "
            "at both stations, Lin et al. 2008 Fig. 2) or legacy (pre-change "
            "SeisForge; ZR, RZ, ZT, TZ have the opposite sign)."
        ),
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


def _register_collect_dispersion(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "collect-dispersion",
        help="Gather AFTAN .dat files into the Eikonal input dataset.",
        description=(
            "Gather AFTAN dispersion .dat files into the (path, period) NetCDF "
            "dataset that 'seisforge ant eikonal' reads. Negative-branch files "
            "become the reversed station pair; phase velocity and SNR are "
            "linearly resampled onto integer periods without extrapolation."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="AFTAN .dat files or quoted glob patterns, e.g. 'FTAN/**/*.dat'.",
    )
    parser.add_argument("-o", "--output", required=True, help="Output NetCDF file.")
    parser.add_argument(
        "--periods",
        required=True,
        help="Integer periods in s: '8,10,12' or 'start:stop:step', e.g. '8:40:2'.",
    )
    coordinates = parser.add_mutually_exclusive_group(required=True)
    coordinates.add_argument(
        "--stations",
        help="Station table: 'station longitude_deg latitude_deg' per line.",
    )
    coordinates.add_argument(
        "--sac-dir",
        help="Directory searched for each .dat's input SAC file (evlo/evla, stlo/stla).",
    )
    parser.add_argument(
        "--name-pattern",
        default="{source_station}_{receiver_station}_{component}*",
        help=(
            "Pattern for the SAC file stem (default: %(default)s); fields "
            "{source_station}, {receiver_station}, {component}, and '*'."
        ),
    )
    parser.add_argument(
        "--component",
        default=None,
        help="Stored CCF component (e.g. ZZ) when the file name has none.",
    )
    parser.add_argument(
        "--aftan-stage",
        choices=("basic", "pmf"),
        default="basic",
        help="Which AFTAN measurement to collect (default: %(default)s).",
    )
    parser.add_argument(
        "--period-axis",
        choices=("instant", "target"),
        default="instant",
        help=(
            "Period column the phase velocity is resampled on (default: "
            "%(default)s, the instantaneous period the velocity belongs to)."
        ),
    )
    parser.add_argument(
        "--longitude-convention",
        choices=("-180_180", "0_360"),
        default="-180_180",
        help="Longitude convention of the output; must match the Eikonal grid.",
    )
    parser.add_argument(
        "--mirror-stack",
        action="store_true",
        help="Also use each symmetric 'stack' branch as the reversed path.",
    )
    parser.add_argument(
        "--amplitude",
        action="store_true",
        help=(
            "Also write the AFTAN amplitude (needed for Helmholtz; meaningful "
            "only if CCF amplitudes share one normalization)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    parser.set_defaults(func=_run_collect_dispersion)


def _register_extract_station_curve(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "extract-station-curve",
        help="Write one station's phase-velocity curve from an Eikonal stack.",
        description=(
            "Sample an Eikonal stack at a station or point and write the "
            "three-column 'period velocity sigma' file read by 'seisforge inv'."
        ),
    )
    parser.add_argument("stack", help="Eikonal stack NetCDF (eikonal_isotropic_stack).")
    parser.add_argument("-o", "--output", required=True, help="Output .dat file.")
    parser.add_argument("--station", default=None, help="Station name in --stations.")
    parser.add_argument("--stations", default=None, help="Station table (see collect-dispersion).")
    parser.add_argument("--longitude", type=float, default=None, help="Longitude in degrees.")
    parser.add_argument("--latitude", type=float, default=None, help="Latitude in degrees.")
    parser.add_argument(
        "--field",
        choices=("eikonal", "helmholtz"),
        default="eikonal",
        help="Stacked field to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--sigma",
        dest="sigma_source",
        choices=("sem", "std"),
        default="sem",
        help=(
            "sem: standard error of the stacked mean (default); std: spread "
            "of single-source estimates."
        ),
    )
    parser.add_argument(
        "--sigma-floor-km-s",
        type=float,
        default=None,
        help=(
            "Lower bound on sigma in km/s (assumed value; default "
            "seisforge.ant.glue.DEFAULT_SIGMA_FLOOR_KM_S = 0.01)."
        ),
    )
    parser.add_argument(
        "--interpolation",
        choices=("bilinear", "nearest"),
        default="bilinear",
        help="How to sample the grid (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    parser.set_defaults(func=_run_extract_station_curve)


def _parse_periods(text: str) -> tuple[int, ...]:
    text = text.strip()
    if ":" in text:
        parts = [int(item) for item in text.split(":")]
        if len(parts) != 3 or parts[2] <= 0:
            raise ValueError("--periods range must be start:stop:step with step > 0.")
        start, stop, step = parts
        return tuple(range(start, stop + 1, step))
    return tuple(int(item) for item in text.split(",") if item.strip())


def _run_collect_dispersion(args: argparse.Namespace) -> int:
    from seisforge.ant.glue import CollectDispersionConfig, run_collect_dispersion
    from seisforge.ant.glue.io import expand_inputs

    config = CollectDispersionConfig(
        periods_s=_parse_periods(args.periods),
        station_table=Path(args.stations) if args.stations else None,
        sac_dir=Path(args.sac_dir) if args.sac_dir else None,
        name_pattern=args.name_pattern,
        component=args.component,
        aftan_stage=args.aftan_stage,
        period_axis=args.period_axis,
        longitude_convention=args.longitude_convention,
        mirror_stack=args.mirror_stack,
        include_amplitude=args.amplitude,
    )
    files = expand_inputs(args.inputs)
    if not files:
        raise FileNotFoundError(f"no files match {args.inputs}")
    result = run_collect_dispersion(files, config, args.output, overwrite=args.overwrite)
    for path, reason in result.skipped:
        print(f"skipped {path}: {reason}")
    print(
        f"wrote {args.output}: {result.dataset.sizes['path']} paths from "
        f"{len(result.used_files)} files, {len(config.periods_s)} periods"
    )
    return 0


def _run_extract_station_curve(args: argparse.Namespace) -> int:
    from seisforge.ant.glue import StationCurveConfig, run_extract_station_curve

    floor = {} if args.sigma_floor_km_s is None else {"sigma_floor_km_s": args.sigma_floor_km_s}
    config = StationCurveConfig(
        field=args.field,
        sigma_source=args.sigma_source,
        interpolation=args.interpolation,
        **floor,
    )
    curve = run_extract_station_curve(
        args.stack,
        args.output,
        config,
        station=args.station,
        station_table=args.stations,
        longitude_deg=args.longitude,
        latitude_deg=args.latitude,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output}: {curve.period_s.size} periods")
    if curve.dropped_period_s:
        print(f"dropped periods (masked or no sigma): {list(curve.dropped_period_s)}")
    return 0


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
        radial_convention=args.radial_convention,
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
