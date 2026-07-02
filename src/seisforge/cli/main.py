"""Top-level command line interface for SeisForge."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from seisforge.cli import ant, inv, rf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seisforge",
        description="SeisForge command line tools for seismic workflows.",
    )
    parser.set_defaults(func=_print_help, parser=parser)

    subparsers = parser.add_subparsers(dest="domain", metavar="<domain>")
    rf.register(subparsers)
    ant.register(subparsers)
    inv.register(subparsers)

    return parser


def _print_help(args: argparse.Namespace) -> int:
    args.parser.print_help()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
