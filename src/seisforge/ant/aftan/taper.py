"""Time-domain tapers applied to the AFTAN input waveform.

AFTAN measures one-sided cross-correlation branches that start at zero lag.
A taper whose length scales with the total trace length (the former
``Trace.taper(0.05)``) therefore down-weights short inter-station arrivals.
The default here follows Fortran AFTAN (``aftanpg.f``): keep the
group-velocity window ``[distance / vmax, distance / vmin]`` and add cosine
ramps of ``ramp_s`` seconds (the maximum analysed period) on both sides.
"""

from __future__ import annotations

import numpy as np
from obspy import Trace


TIME_TAPER_MODES = ("velocity_window", "none", "legacy_edge_fraction")


def group_velocity_window_bounds_s(
    distance_km: float,
    min_velocity_km_s: float,
    max_velocity_km_s: float,
) -> tuple[float, float]:
    """Return the arrival-time window spanned by a group-velocity range."""
    return distance_km / max_velocity_km_s, distance_km / min_velocity_km_s


def cosine_window_weights(
    time_s: np.ndarray,
    start_s: float,
    end_s: float,
    ramp_s: float,
) -> np.ndarray:
    """Unit weights on ``[start_s, end_s]`` with half-cosine ramps outside it.

    The rising ramp spans ``[start_s - ramp_s, start_s)`` and the falling ramp
    spans ``(end_s, end_s + ramp_s]``; weights are zero beyond the ramps.
    """
    time_s = np.asarray(time_s, dtype=float)
    weights = np.zeros(time_s.shape)
    weights[(time_s >= start_s) & (time_s <= end_s)] = 1.0
    if ramp_s > 0:
        rising = (time_s >= start_s - ramp_s) & (time_s < start_s)
        weights[rising] = 0.5 * (
            1.0 - np.cos(np.pi * (time_s[rising] - (start_s - ramp_s)) / ramp_s)
        )
        falling = (time_s > end_s) & (time_s <= end_s + ramp_s)
        weights[falling] = 0.5 * (
            1.0 + np.cos(np.pi * (time_s[falling] - end_s) / ramp_s)
        )
    return weights


def legacy_edge_fraction_weights(trace: Trace, fraction: float) -> np.ndarray:
    """Weights of ObsPy's ``Trace.taper(fraction)`` (pre-0.3 SeisForge AFTAN).

    Compatibility only: the ramp length scales with the trace length, not with
    the distance, so short-distance arrivals near zero lag are suppressed.
    """
    ones = Trace(data=np.ones(trace.stats.npts))
    ones.stats.delta = trace.stats.delta
    ones.taper(fraction)
    return np.asarray(ones.data, dtype=float)
