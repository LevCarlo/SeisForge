"""Inversion command line interfaces."""

from __future__ import annotations

import argparse


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "inv",
        help="Single-station inversion workflows.",
        description="Single-station inversion workflows.",
    )
    parser.set_defaults(func=_print_help, parser=parser)

    inv_subparsers = parser.add_subparsers(dest="inv_command", metavar="<command>")
    _register_disp(inv_subparsers)
    _register_hv(inv_subparsers)
    _register_disp_hv(inv_subparsers)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inv",
        required=True,
        help="Inversion YAML: model, priors, likelihood, and sampler settings.",
    )
    parser.add_argument(
        "--obs",
        default=None,
        help="Observation YAML: data files, data metadata, and forward settings.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for this run.",
    )
    parser.add_argument(
        "--prior-only",
        action="store_true",
        help="Run only the hard-prior sampler.",
    )
    parser.add_argument(
        "--with-prior",
        action="store_true",
        help="Run hard-prior sampling before posterior sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Show lightweight sampling progress.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable sampling progress output.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        help="Report progress every N MCMC steps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="File log level.",
    )


def _register_disp(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "disp",
        help="Invert dispersion observations.",
        description="Invert dispersion observations.",
    )
    _add_common_arguments(parser)
    parser.set_defaults(func=_run_disp)


def _register_hv(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "hv",
        help="Invert Rayleigh H/V observations.",
        description="Invert Rayleigh H/V observations.",
    )
    _add_common_arguments(parser)
    parser.set_defaults(func=_run_hv)


def _register_disp_hv(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "disp-hv",
        help="Jointly invert dispersion and Rayleigh H/V observations.",
        description="Jointly invert dispersion and Rayleigh H/V observations.",
    )
    _add_common_arguments(parser)
    parser.set_defaults(func=_run_disp_hv)


def _run_disp(args: argparse.Namespace) -> int:
    from seisforge.inv.driver import run_inversion

    run_inversion(
        kind="disp",
        inv_file=args.inv,
        obs_file=args.obs,
        output=args.output,
        prior_only=args.prior_only,
        with_prior=args.with_prior,
        overwrite=args.overwrite,
        progress=args.progress,
        progress_every=args.progress_every,
        log_level=args.log_level,
    )
    return 0


def _run_hv(args: argparse.Namespace) -> int:
    from seisforge.inv.driver import run_inversion

    run_inversion(
        kind="hv",
        inv_file=args.inv,
        obs_file=args.obs,
        output=args.output,
        prior_only=args.prior_only,
        with_prior=args.with_prior,
        overwrite=args.overwrite,
        progress=args.progress,
        progress_every=args.progress_every,
        log_level=args.log_level,
    )
    return 0


def _run_disp_hv(args: argparse.Namespace) -> int:
    from seisforge.inv.driver import run_inversion

    run_inversion(
        kind="disp_hv",
        inv_file=args.inv,
        obs_file=args.obs,
        output=args.output,
        prior_only=args.prior_only,
        with_prior=args.with_prior,
        overwrite=args.overwrite,
        progress=args.progress,
        progress_every=args.progress_every,
        log_level=args.log_level,
    )
    return 0


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0
