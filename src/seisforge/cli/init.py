"""`seisforge init <workflow> [directory]`: copy an example's config template."""

from __future__ import annotations

import argparse

from seisforge.templates import WORKFLOWS, copy_template


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "init",
        help="Copy the config template of an example into a directory.",
        description=(
            "Copy the config file(s) of one example (examples/<NN_name>/) into DIRECTORY. "
            "Edit the lines marked '# change this' for your data. Existing files are "
            "kept unless --overwrite."
        ),
    )
    parser.add_argument("workflow", choices=list(WORKFLOWS), help="Which example's template.")
    parser.add_argument("directory", nargs="?", default=".", help="Target directory (default: .).")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files.")
    parser.set_defaults(func=_run_init)
    return parser


def _run_init(args: argparse.Namespace) -> int:
    try:
        written = copy_template(args.workflow, args.directory, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}")
        return 1
    example = WORKFLOWS[args.workflow][0]
    for path in written:
        print(f"wrote {path}")
    print(f"See examples/{example}/README.md, section 'Using your own data'.")
    return 0
