"""Shared test case for docs/hvsr/scripts/*.py (not a figure itself).

The model is the one of `examples/05_hvsr_forward`: one soft layer over a
stiffer half-space, whose SH resonance is f0 = Vs1 / (4 H) = 1 Hz. The
figure style comes from `docs/ant/scripts/_style.py`; figures are written to
`docs/hvsr/img/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ant" / "scripts"))
import _style  # noqa: E402

from seisforge.hvsr import ElasticLayerModel  # noqa: E402

IMG_DIR = Path(__file__).resolve().parents[1] / "img"

THICKNESS_KM = 0.05
VP_KM_S = (0.6, 2.0)
VS_KM_S = (0.2, 1.0)
DENSITY_G_CM3 = (1.8, 2.2)
MODEL = ElasticLayerModel([THICKNESS_KM], VP_KM_S, VS_KM_S, DENSITY_G_CM3)
F0_HZ = VS_KM_S[0] / (4.0 * THICKNESS_KM)  # 1 Hz


def save(draw, script_file: str) -> None:
    _style.save_themed(draw, script_file, out_dir=IMG_DIR)


__all__ = ["DENSITY_G_CM3", "F0_HZ", "MODEL", "THICKNESS_KM", "VP_KM_S", "VS_KM_S", "_style", "np", "save"]
