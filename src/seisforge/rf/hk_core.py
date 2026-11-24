"""Pure H-k stacking functions for receiver functions.

The H-k grid uses ``k = Vp / Vs`` on axis 0 and crustal thickness ``H`` on
axis 1, so every delay or amplitude array has shape ``(n_k, n_H)``.

Phase delays relative to direct P for a layer of thickness ``H`` follow
Zhu & Kanamori (2000)::

    eta_s = sqrt(1 / Vs**2 - p**2),   eta_p = sqrt(1 / Vp**2 - p**2)
    t_Ps   = H * (eta_s - eta_p)
    t_PpPs = H * (eta_s + eta_p)
    t_PsPs = H * 2 * eta_s

With a known upper (sediment) layer of thickness ``h0`` the delays of the two
layers add: the target layer spans ``H - h0`` and the upper layer ``h0``.

Amplitudes at the delays are linearly interpolated between RF samples. Grid
points whose Ps delay lies within the direct-P pulse (shorter than
``min_ps_delay_s``) are excluded from the pick; when the stack there exceeds
the pick, direct P dominates and the pick is not a resolved conversion. Station estimates are the
median and the scaled MAD (Rousseeuw & Croux 1993) of the single-RF picks.

References: Zhu, L. & Kanamori, H. (2000), JGR 105(B2), 2969-2980;
Rousseeuw, P. J. & Croux, C. (1993), JASA 88(424), 1273-1283.
"""

from __future__ import annotations

import numpy as np


def vertical_slowness_s_per_km(
    velocity_km_s: np.ndarray,
    ray_parameter_s_per_km: float,
) -> np.ndarray:
    """Vertical slowness ``sqrt(1 / v**2 - p**2)``; NaN for post-critical rays."""
    velocity = np.asarray(velocity_km_s, dtype=float)
    return np.sqrt(1 / velocity**2 - ray_parameter_s_per_km**2)


def velocities_from_vpvs_ratio(
    vpvs_ratio: np.ndarray,
    *,
    vp_km_s: float | None = None,
    vs_km_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Vp, Vs)`` along the k axis with one velocity held fixed.

    A fixed ``vs_km_s`` takes precedence; otherwise ``vp_km_s`` is fixed and
    ``Vs = Vp / k``.
    """
    k = np.asarray(vpvs_ratio, dtype=float)
    if vs_km_s is not None:
        return vs_km_s * k, np.full_like(k, vs_km_s)
    if vp_km_s is None:
        raise ValueError("Either vp_km_s or vs_km_s must be provided.")
    return np.full_like(k, vp_km_s), vp_km_s / k


def hk_phase_delays_s(
    thickness_km: np.ndarray,
    vp_km_s: np.ndarray,
    vs_km_s: np.ndarray,
    ray_parameter_s_per_km: float,
    *,
    upper_thickness_km: float | None = None,
    upper_vp_km_s: np.ndarray | None = None,
    upper_vs_km_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ps, PpPs, and PsPs delays (s) on the ``(n_k, n_H)`` grid.

    ``vp_km_s``/``vs_km_s`` vary along k. When ``upper_thickness_km`` is
    given, ``thickness_km`` is the total depth to the target interface and the
    upper layer velocities (also along k) apply to the top ``upper_thickness_km``.
    """
    p = ray_parameter_s_per_km
    thickness = np.asarray(thickness_km, dtype=float)
    eta_s = vertical_slowness_s_per_km(vs_km_s, p)
    eta_p = vertical_slowness_s_per_km(vp_km_s, p)
    if upper_thickness_km is None:
        return (
            np.outer(eta_s - eta_p, thickness),
            np.outer(eta_s + eta_p, thickness),
            np.outer(2 * eta_s, thickness),
        )

    h0 = upper_thickness_km
    eta_s0 = vertical_slowness_s_per_km(upper_vs_km_s, p)
    eta_p0 = vertical_slowness_s_per_km(upper_vp_km_s, p)
    layer = thickness - h0
    return (
        np.outer(eta_s - eta_p, layer) + np.outer(eta_s0 - eta_p0, h0),
        np.outer(eta_s + eta_p, layer) + np.outer(eta_s0 + eta_p0, h0),
        np.outer(2 * eta_s, layer) + np.outer(2 * eta_s0, h0),
    )


def hk_grid(min_value: float, max_value: float, step: float) -> np.ndarray:
    """Search grid ``min_value + step * i`` with a step of exactly ``step``.

    The grid starts at ``min_value`` and ends at the last point not beyond
    ``max_value``. ``max_value`` itself is included when it lies on the grid;
    a remainder below ``1e-9 * step`` (floating-point rounding of a range
    that is a whole number of steps) counts as on the grid.
    """
    if not np.isfinite([min_value, max_value, step]).all():
        raise ValueError("Grid min, max and step must be finite.")
    if step <= 0:
        raise ValueError(f"Grid step must be positive, got {step}.")
    if max_value < min_value:
        raise ValueError(f"Grid max {max_value} is below min {min_value}.")
    n_steps = int(np.floor((max_value - min_value) / step + 1e-9))
    return min_value + step * np.arange(n_steps + 1)


def interpolate_at_times(
    data: np.ndarray,
    time_s: np.ndarray,
    begin_s: float,
    delta_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """RF amplitude at ``time_s`` by linear interpolation between samples.

    The trace is ``data[n]`` at ``begin_s + delta_s * n``. Times outside
    ``[begin_s, begin_s + delta_s * (npts - 1)]`` and non-finite times (for
    example post-critical delays) are invalid and contribute zero. Returns
    ``(amplitude, valid)`` with the shape of ``time_s``.

    Linear interpolation makes the amplitude a continuous function of the
    delay. Nearest-sample lookup, used before 0.2.0, made the stack constant
    over runs of grid points that map to the same sample, and the grid
    maximum then went to the first point of such a run.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("data must be a 1-D trace with at least two samples.")
    time_s = np.asarray(time_s, dtype=float)
    position = (time_s - begin_s) / delta_s
    valid = np.isfinite(position) & (position >= 0) & (position <= data.size - 1)
    position = np.where(valid, position, 0.0)
    left = np.minimum(np.floor(position).astype(np.int64), data.size - 2)
    fraction = position - left
    amplitude = (1 - fraction) * data[left] + fraction * data[left + 1]
    return np.where(valid, amplitude, 0.0), valid


def direct_p_mask(t_ps_s: np.ndarray, min_ps_delay_s: float) -> np.ndarray:
    """Grid points whose Ps delay is shorter than ``min_ps_delay_s``.

    Ps is the earliest of the three phases. Where its delay lies within the
    direct-P pulse, the stack samples direct P, not a conversion: as
    ``H -> 0`` it tends to ``(w1 + w2 - w3) A(0)``, a ridge carried by direct
    P. Post-critical (NaN) delays are not in the mask; they already
    contribute zero.
    """
    if min_ps_delay_s < 0:
        raise ValueError(f"min_ps_delay_s must be >= 0, got {min_ps_delay_s}.")
    t_ps_s = np.asarray(t_ps_s, dtype=float)
    return np.isfinite(t_ps_s) & (t_ps_s < min_ps_delay_s)


def hk_pick(stack: np.ndarray, direct_p: np.ndarray) -> tuple[int, int, bool]:
    """``(i_k, j_H, direct_p_dominates)``: the conversion pick of one stack.

    The pick is the stack maximum over the grid points outside ``direct_p``
    (``direct_p_mask``); ties go to the first point in C order (lowest k,
    then lowest H). ``direct_p_dominates`` is true when the stack inside
    ``direct_p`` exceeds that maximum: no conversion in the grid stacks
    above the direct-P ridge, and the pick is not a resolved layer. Raises
    ``ValueError`` when every point is in ``direct_p``.
    """
    stack = np.asarray(stack, dtype=float)
    direct_p = np.asarray(direct_p, dtype=bool)
    if np.all(direct_p):
        raise ValueError(
            "Every H-k grid point has a Ps delay shorter than min_ps_delay_s; "
            "extend the H range or lower min_ps_delay_s."
        )
    i_k, j_h = (int(i) for i in np.unravel_index(
        np.argmax(np.where(direct_p, -np.inf, stack)), stack.shape))
    dominates = bool(np.any(direct_p) and np.max(stack[direct_p]) > stack[i_k, j_h])
    return i_k, j_h, dominates


MAD_TO_STD = 1.4826
"""``1 / Phi^-1(3/4)``: scaled MAD equals the standard deviation of Gaussian data."""


def median_and_robust_std(values: np.ndarray) -> tuple[float, float]:
    """Median and ``1.4826 * median(|x - median|)`` of ``values``.

    The scaled MAD estimates the standard deviation for Gaussian scatter
    (Rousseeuw & Croux 1993) and is insensitive to a minority of outliers.
    One value gives a spread of 0; an empty input gives ``(nan, nan)``.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    center = float(np.median(values))
    return center, float(MAD_TO_STD * np.median(np.abs(values - center)))


def weighted_mahalanobis_distance(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row of ``points`` from the weighted mean.

    Mean ``mu = sum w x / sum w`` and covariance
    ``Sigma = sum w (x - mu)(x - mu)^T / sum w`` (pseudo-inverse if singular).
    Non-positive total weight falls back to equal weights.
    """
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1, 1)
    if np.sum(weights) <= 0:
        weights = np.ones_like(weights)
    total = np.sum(weights)
    mu = np.sum(points * weights, axis=0) / total
    centered = points - mu
    cov = (weights * centered).T @ centered / total
    cov_inv = np.linalg.pinv(cov)
    return np.sqrt(np.sum((centered @ cov_inv) * centered, axis=1))


def seispy_amplitude_correction(ray_parameter_s_per_km: float) -> float:
    """Empirical slowness-dependent RF amplitude scaling used by Seispy H-k.

    ``151.5478 p**2 + 3.2896 p + 0.2618`` with ``p`` in s/km equalises RF
    amplitudes across epicentral distances before stacking.
    """
    p = ray_parameter_s_per_km
    return 151.5478 * p**2 + 3.2896 * p + 0.2618


def hk_stack_amplitude(
    amp_ps: np.ndarray,
    amp_ppps: np.ndarray,
    amp_psps: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    """Weighted H-k stack ``w1 A(t_Ps) + w2 A(t_PpPs) - w3 A(t_PsPs)``.

    ``amp_psps`` is the RF amplitude at the PpSs + PsPs delay, which is
    negative for a velocity increase; the minus sign makes it add.
    """
    return weights[0] * amp_ps + weights[1] * amp_ppps - weights[2] * amp_psps
