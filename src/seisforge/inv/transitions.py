"""Smooth-step shapes for gradational interfaces in 1-D velocity models.

An interface at depth ``z_i`` with 10-90 % width ``w`` blends the segment
above into the segment below with a weight ``S(z)`` that rises from 0 to 1:

    v(z) = (1 - S(z)) * v_upper(z) + S(z) * v_lower(z)

``S(z_i) = 0.5``, ``S`` reaches 10 % and 90 % at ``z_i -/+ w/2``, and ``S`` is
exactly 0 above and exactly 1 below a finite support ``z_i -/+ h``. Compact
support keeps one interface from leaking into the free surface, a thin layer,
or a neighbouring interface. ``w = 0`` is a hard step.

Shapes
------
``sigmoid``
    Logistic ``1 / (1 + exp(-k (z - z_i)))`` with steepness ``k = 2 ln 9 / w``,
    truncated at ``h = 2 w`` (where the logistic tail is 1.5e-4) and rescaled to
    reach exactly 0 and 1. Maximum gradient ``~ 1.1 dv / w``.
``smootherstep``
    Quintic ``6x^5 - 15x^4 + 10x^3`` on the support; C2-continuous with
    ``h = 0.987 w``. Maximum gradient ``1.875 dv / (2h) = 0.950 dv / w``.
"""

from __future__ import annotations

import numpy as np


TRANSITION_SHAPES = ("sigmoid", "smootherstep")

_LOGISTIC_STEEPNESS_TIMES_WIDTH = 2.0 * np.log(9.0)
_SIGMOID_HALF_SUPPORT_PER_WIDTH = 2.0
# 10-90 % width of smootherstep is 0.506727 of its full support.
_SMOOTHERSTEP_HALF_SUPPORT_PER_WIDTH = 0.5 / 0.5067270934230668


def transition_half_support_km(width_km: float, shape: str) -> float:
    """Half-thickness ``h`` outside which the transition weight is exactly 0 or 1."""
    _check_shape(shape)
    if width_km < 0:
        raise ValueError("transition width must be non-negative.")
    if shape == "sigmoid":
        return _SIGMOID_HALF_SUPPORT_PER_WIDTH * width_km
    return _SMOOTHERSTEP_HALF_SUPPORT_PER_WIDTH * width_km


def sigmoid_steepness_per_km(width_km: float) -> float:
    """Logistic steepness ``k`` matching a 10-90 % width ``w``: ``k = 2 ln 9 / w``."""
    if width_km <= 0:
        return np.inf
    return _LOGISTIC_STEEPNESS_TIMES_WIDTH / width_km


def transition_weight(
    z_km: np.ndarray,
    depth_km: float,
    width_km: float,
    shape: str = "sigmoid",
) -> np.ndarray:
    """Weight of the lower segment, 0 above and 1 below the interface.

    ``w = 0`` gives the hard step ``z >= depth_km``, which matches the segment
    assignment of a model without transitions.
    """
    _check_shape(shape)
    z = np.asarray(z_km, dtype=float)
    if width_km <= 0:
        return (z >= depth_km).astype(float)
    half = transition_half_support_km(width_km, shape)
    if shape == "sigmoid":
        k = sigmoid_steepness_per_km(width_km)
        u = np.clip(z - depth_km, -half, half)
        low = _logistic(-k * half)
        high = _logistic(k * half)
        return (_logistic(k * u) - low) / (high - low)
    x = np.clip((z - depth_km + half) / (2.0 * half), 0.0, 1.0)
    return x**3 * (x * (6.0 * x - 15.0) + 10.0)


def transition_weight_derivative_per_km(
    z_km: np.ndarray,
    depth_km: float,
    width_km: float,
    shape: str = "sigmoid",
) -> np.ndarray:
    """``dS/dz`` (1/km) of `transition_weight`; 0 outside the support.

    ``sigmoid``: ``k l (1 - l) / (l(kh) - l(-kh))`` with ``l = l(k (z - z_i))``.
    ``smootherstep``: ``30 x^2 (1 - x)^2 / (2h)``. Both peak at ``z_i``. A hard
    step (``w = 0``) has no finite derivative; it is reported as 0 everywhere,
    since a jump is not a gradient.
    """
    _check_shape(shape)
    z = np.asarray(z_km, dtype=float)
    if width_km <= 0:
        return np.zeros_like(z)
    half = transition_half_support_km(width_km, shape)
    inside = np.abs(z - depth_km) < half
    if shape == "sigmoid":
        k = sigmoid_steepness_per_km(width_km)
        logistic = _logistic(k * (z - depth_km))
        scale = _logistic(k * half) - _logistic(-k * half)
        slope = k * logistic * (1.0 - logistic) / scale
    else:
        x = (z - depth_km + half) / (2.0 * half)
        slope = 30.0 * x**2 * (1.0 - x) ** 2 / (2.0 * half)
    return np.where(inside, slope, 0.0)


def _logistic(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def _check_shape(shape: str) -> None:
    if shape not in TRANSITION_SHAPES:
        raise ValueError(f"transition shape must be one of {TRANSITION_SHAPES}.")
