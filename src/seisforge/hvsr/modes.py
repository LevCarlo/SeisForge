"""Surface-wave pole residues contributing to diffuse-field Green functions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dispersion import (
    phase_velocity_seeds_km_s,
    refine_surface_wavenumber_per_km,
)
from .models import ElasticLayerModel, GreenFunctionContributions
from .propagator import psv_surface_compliance, sh_surface_compliance


@dataclass(frozen=True)
class SurfaceWaveSettings:
    max_rayleigh_modes: int = 20
    max_love_modes: int = 20
    dispersion_step_km_s: float = 0.001
    residue_step_fraction: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.max_rayleigh_modes < 0 or self.max_love_modes < 0:
            raise ValueError("surface-wave mode limits cannot be negative")
        if self.max_rayleigh_modes == 0 and self.max_love_modes == 0:
            raise ValueError("at least one surface-wave mode must be requested")
        if self.dispersion_step_km_s <= 0.0:
            raise ValueError("dispersion_step_km_s must be positive")
        if not 1.0e-8 <= self.residue_step_fraction <= 1.0e-2:
            raise ValueError("residue_step_fraction must lie in [1e-8, 1e-2]")


def _matrix_residue(function, root: float, step_fraction: float) -> np.ndarray:
    step = max(abs(root) * step_fraction, 1.0e-10)
    return 0.5 * step * (function(root + step) - function(root - step))


def _positive_residue(value: complex, *, label: str) -> float:
    real = float(np.real(value))
    tolerance = 1.0e-8 * max(abs(value), 1.0)
    if abs(np.imag(value)) > tolerance:
        raise RuntimeError(f"{label} pole residue has an unexpected imaginary part")
    if real < -tolerance:
        raise RuntimeError(f"{label} pole residue has negative spectral energy")
    return max(real, 0.0)


def surface_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: SurfaceWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Calculate Rayleigh/Love pole residues at the free surface."""

    settings = settings or SurfaceWaveSettings()
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    rayleigh_h = np.zeros(frequencies.size)
    rayleigh_v = np.zeros(frequencies.size)
    love_h = np.zeros(frequencies.size)

    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")

    for wave, max_modes in (
        ("rayleigh", settings.max_rayleigh_modes),
        ("love", settings.max_love_modes),
    ):
        for mode in range(max_modes):
            velocities = phase_velocity_seeds_km_s(
                model,
                frequencies,
                mode,
                wave,
                dc_km_s=settings.dispersion_step_km_s,
            )
            if not np.any(np.isfinite(velocities)):
                break
            for index, (frequency, velocity) in enumerate(
                zip(frequencies, velocities, strict=True)
            ):
                if not np.isfinite(velocity):
                    continue
                try:
                    root = refine_surface_wavenumber_per_km(
                        model,
                        frequency,
                        velocity,
                        wave,
                        velocity_half_width_km_s=(
                            3.0 * settings.dispersion_step_km_s
                        ),
                    )
                except RuntimeError:
                    # Closely spaced high modes can be returned by CPS without
                    # a matching, numerically resolvable impedance zero. Their
                    # residue cannot be evaluated reliably, so omit that mode
                    # instead of invalidating all frequencies in the request.
                    continue
                if wave == "rayleigh":
                    residue = _matrix_residue(
                        lambda k: psv_surface_compliance(model, frequency, k),
                        root,
                        settings.residue_step_fraction,
                    )
                    try:
                        horizontal = _positive_residue(
                            -np.pi * root * residue[0, 0] / 2.0,
                            label="Rayleigh horizontal",
                        )
                        vertical = _positive_residue(
                            -np.pi * root * residue[1, 1],
                            label="Rayleigh vertical",
                        )
                    except RuntimeError:
                        continue
                    rayleigh_h[index] += horizontal
                    rayleigh_v[index] += vertical
                else:
                    residue = _matrix_residue(
                        lambda k: sh_surface_compliance(model, frequency, k),
                        root,
                        settings.residue_step_fraction,
                    )
                    try:
                        horizontal = _positive_residue(
                            -np.pi * root * residue / 2.0,
                            label="Love horizontal",
                        )
                    except RuntimeError:
                        continue
                    love_h[index] += horizontal

    zeros = np.zeros_like(frequencies)
    return GreenFunctionContributions(
        rayleigh_horizontal=rayleigh_h,
        rayleigh_vertical=rayleigh_v,
        love_horizontal=love_h,
        psv_horizontal=zeros,
        psv_vertical=zeros,
        sh_horizontal=zeros,
    )
