"""Shared synthetic station pair for the dispersion figures (not a figure itself).

A ZZ CCF of a diffuse Rayleigh field (`seisforge.ant.synthetic.synthetic_rayleigh_ccf_zne`,
rotated with the package rotation) for a known phase-velocity curve, cut to
its positive lag and wrapped as an ObsPy trace with ``b = 0`` and ``dist``,
which is what the AFTAN driver passes to `AFTANMeasurement`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from obspy import Trace

from seisforge.ant.aftan.core import AFTANMeasurement
from seisforge.ant.aftan.models import AFTANAlphaConfig, AFTANBasicConfig, AFTANConfig, AFTANPMFConfig
from seisforge.ant.rotation import rotate_zne_ccf_to_zrt
from seisforge.ant.filters import gaussian_filter_trace
from seisforge.ant.synthetic import (
    group_velocity_from_phase,
    symmetric_ccf_time_axis,
    synthetic_ccf,
    synthetic_rayleigh_ccf_zne,
)

DISTANCE_KM = 400.0
AZIMUTH_DEG = 50.0
DELTA_S = 0.5
MIN_PERIOD_S, MAX_PERIOD_S = 6.0, 40.0
# Assumed smooth curve (illustration only): slow at short periods, faster at long.
PERIOD_S = np.linspace(1.0, 100.0, 991)
PHASE_VELOCITY_KM_S = 3.0 + 0.9 * np.tanh((PERIOD_S - 14.0) / 14.0)
GROUP_VELOCITY_KM_S = group_velocity_from_phase(PERIOD_S, PHASE_VELOCITY_KM_S)


def true_phase_velocity(period_s):
    return np.interp(period_s, PERIOD_S, PHASE_VELOCITY_KM_S)


def true_group_velocity(period_s):
    return np.interp(period_s, PERIOD_S, GROUP_VELOCITY_KM_S)


def zz_ccf(max_lag_s: float = 400.0) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided ZZ CCF (lag axis, values), band MIN_PERIOD_S/1.25 .. MAX_PERIOD_S*1.25."""
    time = symmetric_ccf_time_axis(max_lag_s, DELTA_S)
    zne = synthetic_rayleigh_ccf_zne(
        time, DISTANCE_KM, AZIMUTH_DEG, PERIOD_S, PHASE_VELOCITY_KM_S,
        min_period_s=MIN_PERIOD_S / 1.25, max_period_s=MAX_PERIOD_S * 1.25,
    )
    zz = rotate_zne_ccf_to_zrt(zne, AZIMUTH_DEG, AZIMUTH_DEG + 180.0, components=["ZZ"])["ZZ"]
    return time, zz


def positive_trace(time: np.ndarray, data: np.ndarray) -> Trace:
    keep = time >= 0
    trace = Trace(data=np.asarray(data[keep], dtype=float))
    trace.stats.delta = DELTA_S
    trace.stats.sac = {"b": 0.0, "dist": DISTANCE_KM}
    return trace


def noisy_trace(time: np.ndarray, zz: np.ndarray) -> tuple[Trace, Trace]:
    """(noise-free, noisy) positive-lag traces for the PMF figures.

    The noisy one adds (i) a weaker, faster interfering wave train (a second
    dispersive arrival with c = 4.2-4.6 km/s at 60 % of the main amplitude; an
    assumed stand-in for a precursor or higher mode) and (ii) band-limited
    random noise (12 s, alpha 2, 25 % of the main amplitude).
    """
    clean = positive_trace(time, zz)
    scale = np.abs(clean.data).max()
    fast = synthetic_ccf(time, DISTANCE_KM, PERIOD_S, 4.2 + 0.4 * PERIOD_S / PERIOD_S.max(),
                         min_period_s=5.0, max_period_s=20.0)
    fast = positive_trace(time, fast).data
    rng = np.random.default_rng(1)
    noise = gaussian_filter_trace(rng.normal(size=clean.data.size), DELTA_S, 12.0, 2.0)
    noisy = clean.copy()
    noisy.data = clean.data + 0.6 * scale * fast / np.abs(fast).max() + 0.25 * scale * noise / np.abs(noise).max()
    return clean, noisy


def prediction_file() -> Path:
    """Write the true curve to a temporary file (used only to fix the 2 pi cycle)."""
    path = Path(tempfile.mkdtemp()) / "prediction.dat"
    np.savetxt(path, np.column_stack([PERIOD_S, PHASE_VELOCITY_KM_S]))
    return path


def aftan_config(*, prediction: Path | None, pmf: bool = False, alpha: float = 20.0) -> AFTANConfig:
    """AFTAN settings for these figures; everything not set is the package default."""
    return AFTANConfig(
        min_period=MIN_PERIOD_S,
        max_period=MAX_PERIOD_S,
        velocity_min=2.0,
        velocity_max=4.5,
        pi_over_4=-1.0,  # ZZ correlation (docs/ant/dispersion.md, Appendix A)
        prediction_file=prediction,
        basic=AFTANBasicConfig(alpha=AFTANAlphaConfig(mode="constant", value=alpha)),
        pmf=AFTANPMFConfig(enabled=pmf, alpha=AFTANAlphaConfig(mode="constant", value=alpha)),
    )


def measurement(trace: Trace, config: AFTANConfig) -> AFTANMeasurement:
    return AFTANMeasurement(trace, Path("synthetic_ZZ.SAC"), config)
