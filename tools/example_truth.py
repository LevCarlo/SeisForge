"""Truth model shared by the synthetic ambient-noise examples (02_aftan, 03_eikonal, full_chain).

The model is the truth of `examples/06_inversion` (`TRUTH` in
`make_example_06_inversion.py`: a 0-1 km sediment gradient over a B-spline
crust to 15 km), forwarded in the same uniform 1/32 km layers. Its
fundamental-mode Rayleigh phase velocity is the "known dispersion curve" of
the ambient-noise examples, and its ellipticity is the synthetic H/V of
`full_chain`. Everything is computed by the package (`seisforge.inv`); this
module only fixes the inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from make_example_06_inversion import EXAMPLE as INVERSION_EXAMPLE
from make_example_06_inversion import truth_layered_model
from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)

ROOT = Path(__file__).resolve().parents[1]

# Fine period grid on which the known curve is tabulated (synthetic CCFs
# interpolate it linearly in period). Assumed; wide enough for the 2-8 s band
# of the examples plus the one-octave ramps of `seisforge.ant.synthetic`.
CURVE_PERIODS_S = np.round(np.arange(0.8, 16.0 + 1e-9, 0.1), 2)


def truth_model():
    """Layered truth model (`seisforge.inv.model.LayeredModel`)."""
    return truth_layered_model(INVERSION_EXAMPLE / "inv.yaml", INVERSION_EXAMPLE / "obs.yaml")


def truth_phase_velocity(periods_s=CURVE_PERIODS_S):
    """Fundamental-mode Rayleigh phase velocity (km/s) of the truth model."""
    prediction = predict_dispersion(truth_model(), DispersionRequest(periods=periods_s))
    if not np.array_equal(prediction.periods, np.asarray(periods_s, dtype=float)):
        raise RuntimeError("The forward solver dropped periods of the truth curve.")
    return prediction.velocity


def truth_hv(periods_s):
    """Fundamental-mode Rayleigh ellipticity (H/V) of the truth model."""
    prediction = predict_rayleigh_hv(truth_model(), RayleighHVRequest(periods=periods_s))
    if not np.array_equal(prediction.periods, np.asarray(periods_s, dtype=float)):
        raise RuntimeError("The forward solver dropped periods of the truth H/V.")
    return prediction.hv
