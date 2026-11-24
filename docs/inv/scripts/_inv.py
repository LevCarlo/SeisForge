"""Shared setup for docs/inv/scripts/*.py (not a figure itself).

Makes the ANT figure style (`docs/ant/scripts/_style.py`) importable, and
points `save` at `docs/inv/img/`. Each figure script calls
``_inv.save(draw, __file__)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ant" / "scripts"))

import _style  # noqa: E402

IMG_DIR = HERE.parent / "img"
REPO = HERE.parents[2]


def save(draw, script_file: str):
    """Write ``<stem>.svg`` and ``<stem>-dark.svg`` to ``docs/inv/img``."""
    return _style.save_themed(draw, script_file, out_dir=IMG_DIR)
