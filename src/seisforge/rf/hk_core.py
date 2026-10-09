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


def nearest_sample_index(
    time_s: np.ndarray,
    begin_s: float,
    delta_s: float,
    npts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest trace sample for each time and a mask of usable times.

    Matches ``argmin(|t_axis - t|)`` over ``t_axis = begin_s + delta_s *
    arange(npts)`` (ties go to the earlier sample) in O(n) memory. Times that
    are non-finite or fall more than half a sample outside the trace are
    marked invalid instead of being clamped to the first/last sample.
    """
    time_s = np.asarray(time_s, dtype=float)
    t_axis = begin_s + delta_s * np.arange(npts)
    valid = (
        np.isfinite(time_s)
        & (time_s >= t_axis[0] - 0.5 * delta_s)
        & (time_s <= t_axis[-1] + 0.5 * delta_s)
    )
    estimate = np.floor((np.where(valid, time_s, begin_s) - begin_s) / delta_s)
    candidates = np.clip(
        estimate.astype(np.int64)[..., None] + np.arange(-1, 3),
        0,
        npts - 1,
    )
    distance = np.abs(t_axis[candidates] - np.where(valid, time_s, begin_s)[..., None])
    index = np.take_along_axis(
        candidates, np.argmin(distance, axis=-1)[..., None], axis=-1
    )[..., 0]
    return index, valid


def sample_at_times(
    data: np.ndarray,
    time_s: np.ndarray,
    begin_s: float,
    delta_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-sample amplitudes; invalid times contribute zero."""
    data = np.asarray(data)
    index, valid = nearest_sample_index(time_s, begin_s, delta_s, data.size)
    return np.where(valid, data[index], 0.0), valid


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
    """Weighted H-k stack ``w1 Ps + w2 PpPs - w3 PsPs`` (PsPs has negative polarity)."""
    return weights[0] * amp_ps + weights[1] * amp_ppps - weights[2] * amp_psps
