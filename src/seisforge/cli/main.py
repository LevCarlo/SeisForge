"""Top-level command line interface for SeisForge."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from seisforge.cli import inv, rf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seisforge",
        description="SeisForge command line tools for seismic workflows.",
    )
    parser.set_defaults(func=_print_help, parser=parser)

    subparsers = parser.add_subparsers(dest="domain", metavar="<domain>")
    rf.register(subparsers)
    _register_placeholder(subparsers, "ant", "Ambient-noise tools.")
    inv.register(subparsers)

    return parser


def _register_placeholder(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> None:
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(func=_print_help, parser=parser)


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
