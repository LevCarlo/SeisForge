"""Synthetic cross-correlation functions (CCFs) from a known dispersion curve.

Used to build example and test data with a known answer; no file IO here.

Model
-----
A single surface-wave mode travelling a distance ``r`` with phase velocity
``c(f)``. The causal (positive-lag) branch is the far-field waveform

    g(t) = sum_f  W(f) cos(2 pi f (t - r / c(f)) + phi0),

written through its spectrum ``G(f) = W(f) exp(-2 pi i f r / c(f) + i phi0)``
and the inverse real FFT of numpy (kernel ``exp(+2 pi i f t)``). The phase
constant is ``phi0 = -pi_over_4 * pi / 4``: the far-field phase offset that
AFTAN removes with the same ``pi_over_4`` coefficient
(`seisforge.ant.aftan.phase`; -1 for ZZ, +1 for ZR). A synthetic CCF built
with a component's coefficient is therefore measured back at ``c(f)`` by
AFTAN run with ``pi_over_4: auto`` on that component; this is a convention
shared with AFTAN, not an independent check of it. The group velocity
``U = c / (1 - (f / c) dc/df)`` is independent of ``phi0`` and is the
independent check (`tests/test_ant_synthetic.py`).

The two-sided CCF is ``C(t) = g(|t|)`` (symmetric, as for a stacked CCF of a
diffuse field; the ``g(-t)`` tail near zero lag is left out) sampled on ``time_s = b + k * delta``, ``b = -(n - 1) / 2 * delta``.

``W(f)`` is 1 between ``1/max_period_s`` and ``1/min_period_s`` with cosine
ramps one octave wide outside the band (assumed shape; only its smoothness
matters, so that the waveform is compact in time). ``c`` is interpolated
linearly in period inside the given curve and held constant beyond its ends
(only reached inside the ramps). Amplitudes have no physical meaning: the
geometric spreading ``r**-0.5`` is applied so that amplitude decays with
distance, nothing else.

Units: km, km/s, s.
"""

from __future__ import annotations

import numpy as np

__all__ = ["symmetric_ccf_time_axis", "synthetic_ccf", "group_velocity_from_phase"]


def symmetric_ccf_time_axis(max_lag_s: float, delta_s: float) -> np.ndarray:
    """Lag times ``-max_lag_s ... +max_lag_s`` in steps of ``delta_s`` (odd length, 0 in the middle)."""
    if delta_s <= 0 or max_lag_s <= 0:
        raise ValueError("max_lag_s and delta_s must be positive.")
    n_half = int(round(max_lag_s / delta_s))
    return np.arange(-n_half, n_half + 1) * delta_s


def _band_weights(frequency_hz, min_period_s, max_period_s):
    f_low, f_high = 1.0 / max_period_s, 1.0 / min_period_s
    weights = np.zeros_like(frequency_hz)
    inside = (frequency_hz >= f_low) & (frequency_hz <= f_high)
    weights[inside] = 1.0
    low_ramp = (frequency_hz >= f_low / 2) & (frequency_hz < f_low)
    weights[low_ramp] = 0.5 - 0.5 * np.cos(np.pi * (frequency_hz[low_ramp] - f_low / 2) / (f_low / 2))
    high_ramp = (frequency_hz > f_high) & (frequency_hz <= 2 * f_high)
    weights[high_ramp] = 0.5 + 0.5 * np.cos(np.pi * (frequency_hz[high_ramp] - f_high) / f_high)
    return weights


def synthetic_ccf(
    time_s: np.ndarray,
    distance_km: float,
    period_s: np.ndarray,
    phase_velocity_km_s: np.ndarray,
    *,
    min_period_s: float,
    max_period_s: float,
    pi_over_4: float = -1.0,
) -> np.ndarray:
    """Symmetric two-sided CCF of one mode with phase velocity ``c(period)``.

    ``time_s`` must come from `symmetric_ccf_time_axis` (uniform, symmetric
    about 0). ``period_s`` must be increasing. The signal band
    ``[min_period_s, max_period_s]`` must lie inside the given curve. Returns
    an array like ``time_s``; see the module docstring for the formula.
    """
    time_s = np.asarray(time_s, dtype=float)
    period_s = np.asarray(period_s, dtype=float)
    velocity = np.asarray(phase_velocity_km_s, dtype=float)
    delta = time_s[1] - time_s[0]
    if time_s.size % 2 != 1 or not np.allclose(np.diff(time_s), delta) or not np.isclose(time_s[time_s.size // 2], 0.0):
        raise ValueError("time_s must be uniform, odd-length and symmetric about 0.")
    if period_s.shape != velocity.shape or period_s.size < 2 or np.any(np.diff(period_s) <= 0):
        raise ValueError("period_s must be increasing and match phase_velocity_km_s.")
    if not (period_s[0] <= min_period_s < max_period_s <= period_s[-1]):
        raise ValueError("The band [min_period_s, max_period_s] must lie inside the dispersion curve.")
    if np.any(velocity <= 0) or distance_km <= 0:
        raise ValueError("Velocities and distance must be positive.")
    if 2 * delta >= min_period_s:
        raise ValueError("Sampling too coarse for min_period_s (need 2 * delta < min_period_s).")

    n_half = time_s.size // 2
    n_fft = 4 * (n_half + 1)  # zero padding: the causal branch must not wrap around
    frequency = np.fft.rfftfreq(n_fft, delta)
    weights = _band_weights(frequency, min_period_s, max_period_s)
    used = weights > 0
    c = np.interp(1.0 / frequency[used], period_s, velocity)
    phi0 = -pi_over_4 * np.pi / 4
    spectrum = np.zeros(frequency.size, dtype=complex)
    spectrum[used] = weights[used] * np.exp(-2j * np.pi * frequency[used] * distance_km / c + 1j * phi0)
    causal = np.fft.irfft(spectrum, n_fft)[: n_half + 1] / np.sqrt(distance_km)
    return np.concatenate([causal[:0:-1], causal])


def group_velocity_from_phase(period_s: np.ndarray, phase_velocity_km_s: np.ndarray) -> np.ndarray:
    """Group velocity ``U = c / (1 + (T / c) dc/dT)`` by central differences in period.

    Equivalent to ``U = d omega / d k``; one-sided differences at the ends.
    """
    period_s = np.asarray(period_s, dtype=float)
    c = np.asarray(phase_velocity_km_s, dtype=float)
    dc_dt = np.gradient(c, period_s)
    return c / (1.0 + period_s / c * dc_dt)
