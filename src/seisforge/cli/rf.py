"""Receiver-function command line interfaces."""

from __future__ import annotations

import argparse


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "rf",
        help="Receiver-function workflows.",
        description="Receiver-function workflows.",
    )
    parser.set_defaults(func=_print_help, parser=parser)

    rf_subparsers = parser.add_subparsers(dest="rf_command", metavar="<command>")
    _register_qc(rf_subparsers)
    _register_hkseq(rf_subparsers)
    return parser


def _register_qc(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "qc",
        help="Run receiver-function quality control.",
        description="Run receiver-function quality control.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="Path to the RF QC YAML configuration file.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Save RF QC figures.",
    )
    parser.set_defaults(func=_run_qc)


def _register_hkseq(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "hkseq",
        help="Run sequential H-k analysis.",
        description="Run sequential H-k analysis.",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="Path to the H-k YAML configuration file.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="both",
        choices=["both", "high", "low"],
        help="H-k analysis mode.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Save H-k QC figures.",
    )
    parser.set_defaults(func=_run_hkseq)


def _run_qc(args: argparse.Namespace) -> int:
    from seisforge.rf.qc import run_rf_qc_config

    run_rf_qc_config(args.config_file, plot=args.plot)
    return 0


def _run_hkseq(args: argparse.Namespace) -> int:
    from seisforge.rf.hk import hkSeq

    params = _read_yaml(args.config_file)
    hkSeq(params, mode=args.mode, plot=args.plot)
    return 0


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        import yaml

        return yaml.safe_load(file)


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0
