"""Surface-wave root finding for the native DFA backend."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .models import ElasticLayerModel
from .propagator import psv_surface_compliance, sh_surface_compliance


def _disba_model(model: ElasticLayerModel) -> tuple[np.ndarray, ...]:
    thickness = np.concatenate((model.thickness_km, [0.0]))
    return thickness, model.vp_km_s, model.vs_km_s, model.density_g_cm3


def approximate_phase_velocity_km_s(
    model: ElasticLayerModel,
    frequency_hz: float,
    mode: int,
    wave: str,
    *,
    dc_km_s: float = 0.001,
) -> float | None:
    """Obtain a robust CPS/disba phase-velocity seed for one mode."""

    try:
        from disba import PhaseDispersion

        solver = PhaseDispersion(*_disba_model(model), algorithm="dunkin", dc=dc_km_s)
        result = solver(np.array([1.0 / frequency_hz]), mode=mode, wave=wave)
    except Exception:
        return None
    if result.velocity.size != 1 or not np.isfinite(result.velocity[0]):
        return None
    return float(result.velocity[0])


def phase_velocity_seeds_km_s(
    model: ElasticLayerModel,
    frequencies_hz,
    mode: int,
    wave: str,
    *,
    dc_km_s: float = 0.001,
) -> np.ndarray:
    """Track one dispersion branch over a complete frequency axis.

    ``surf96`` uses the previously found root as the starting point for the
    next period. Calling it once per frequency discards that continuation and
    makes thin-layer/high-mode models unnecessarily prone to root failures.
    Missing parts of a higher-mode branch are returned as ``NaN``.
    """

    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")

    periods = 1.0 / frequencies
    order = np.argsort(periods, kind="stable")
    sorted_periods = periods[order]
    try:
        from disba import PhaseDispersion

        solver = PhaseDispersion(*_disba_model(model), algorithm="dunkin", dc=dc_km_s)
        result = solver(sorted_periods, mode=mode, wave=wave)
    except Exception:
        return np.full(frequencies.size, np.nan)

    sorted_seeds = np.full(frequencies.size, np.nan)
    # disba filters unavailable higher-mode periods from its result. Returned
    # periods are exact members of ``sorted_periods``, so searchsorted restores
    # their positions without interpolating across a modal cutoff.
    positions = np.searchsorted(sorted_periods, result.period)
    valid = positions < sorted_periods.size
    positions = positions[valid]
    velocities = np.asarray(result.velocity, dtype=np.float64)[valid]
    exact = np.isclose(
        sorted_periods[positions],
        np.asarray(result.period, dtype=np.float64)[valid],
        rtol=1.0e-12,
        atol=0.0,
    )
    sorted_seeds[positions[exact]] = velocities[exact]

    seeds = np.empty_like(sorted_seeds)
    seeds[order] = sorted_seeds
    return seeds


def _frequency_function(
    model: ElasticLayerModel,
    frequency_hz: float,
    horizontal_wavenumber_per_km: float,
    wave: str,
) -> float:
    if wave == "love":
        compliance = sh_surface_compliance(
            model, frequency_hz, horizontal_wavenumber_per_km
        )
        return float(np.real(1.0 / compliance))
    if wave == "rayleigh":
        compliance = psv_surface_compliance(
            model, frequency_hz, horizontal_wavenumber_per_km
        )
        # det(inv(C)) == 1/det(C), but the latter avoids forming an inverse of
        # the pole-valued compliance near a surface-wave root.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            determinant = np.linalg.det(compliance)
            return float(np.real(1.0 / determinant))
    raise ValueError("wave must be 'rayleigh' or 'love'")


def refine_surface_wavenumber_per_km(
    model: ElasticLayerModel,
    frequency_hz: float,
    phase_velocity_km_s: float,
    wave: str,
    *,
    velocity_half_width_km_s: float = 0.003,
) -> float:
    """Refine a disba seed against the SeisForge surface impedance."""

    omega = 2.0 * np.pi * frequency_hz
    low_velocity = max(phase_velocity_km_s - velocity_half_width_km_s, 1.0e-6)
    high_velocity = phase_velocity_km_s + velocity_half_width_km_s
    k_low = omega / high_velocity
    k_high = omega / low_velocity
    seed = omega / phase_velocity_km_s
    # Include the CPS seed itself. At closely spaced high modes the impedance
    # zero can be much narrower than an otherwise adequate uniform bracket.
    samples = np.unique(np.concatenate((np.linspace(k_low, k_high, 33), [seed])))
    values = np.array(
        [_frequency_function(model, frequency_hz, k, wave) for k in samples]
    )
    candidates = []
    for left, right, f_left, f_right in zip(
        samples[:-1], samples[1:], values[:-1], values[1:], strict=True
    ):
        if np.isfinite(f_left) and np.isfinite(f_right) and f_left * f_right <= 0.0:
            root = brentq(
                lambda k: _frequency_function(model, frequency_hz, k, wave),
                left,
                right,
                xtol=1.0e-12,
                rtol=1.0e-12,
            )
            candidates.append(root)
    seed_value = _frequency_function(model, frequency_hz, seed, wave)
    for relative_step in np.geomspace(1.0e-8, 1.0e-2, 25):
        for neighbor in (seed * (1.0 - relative_step), seed * (1.0 + relative_step)):
            if neighbor <= k_low or neighbor >= k_high:
                continue
            neighbor_value = _frequency_function(
                model, frequency_hz, neighbor, wave
            )
            if (
                np.isfinite(seed_value)
                and np.isfinite(neighbor_value)
                and seed_value * neighbor_value <= 0.0
            ):
                left, right = sorted((seed, neighbor))
                candidates.append(
                    brentq(
                        lambda k: _frequency_function(
                            model, frequency_hz, k, wave
                        ),
                        left,
                        right,
                        xtol=1.0e-12,
                        rtol=1.0e-12,
                    )
                )
                break
        if candidates:
            break
    if not candidates:
        def objective(k: float) -> float:
            return abs(_frequency_function(model, frequency_hz, float(k), wave))

        minimized = minimize_scalar(
            objective,
            bounds=(k_low, k_high),
            method="bounded",
            options={"xatol": 1.0e-12, "maxiter": 200},
        )
        seed_value = objective(seed)
        if (
            not minimized.success
            or not np.isfinite(minimized.fun)
            or minimized.fun > max(seed_value * 1.0e-2, 1.0e-9)
        ):
            raise RuntimeError(f"could not refine {wave} surface-wave root")
        return float(minimized.x)
    return min(candidates, key=lambda value: abs(value - seed))
