"""Pure numerical formulas for Helmholtz amplitude correction."""

from __future__ import annotations

import numpy as np


def angular_frequency_rad_s(period_s: float) -> float:
    """Return angular frequency for a positive period."""
    if period_s <= 0:
        raise ValueError("period_s must be positive.")
    return 2.0 * np.pi / period_s


def amplitude_laplacian_correction_s2_per_km2(
    amplitude: np.ndarray,
    amplitude_laplacian_per_km2: np.ndarray,
    *,
    period_s: float,
) -> np.ndarray:
    """Return the Helmholtz term ``Laplacian(A) / (A * omega**2)``."""
    amplitude_array, laplacian = np.broadcast_arrays(
        np.asarray(amplitude, dtype=float),
        np.asarray(amplitude_laplacian_per_km2, dtype=float),
    )
    correction = np.full(amplitude_array.shape, np.nan)
    valid = (
        np.isfinite(amplitude_array)
        & np.isfinite(laplacian)
        & (amplitude_array != 0)
    )
    omega = angular_frequency_rad_s(period_s)
    correction[valid] = laplacian[valid] / (
        amplitude_array[valid] * omega**2
    )
    return correction


def corrected_slowness_squared_s2_per_km2(
    eikonal_slowness_s_per_km: np.ndarray,
    amplitude_correction_s2_per_km2: np.ndarray,
) -> np.ndarray:
    """Apply the Helmholtz correction to squared Eikonal slowness."""
    slowness, correction = np.broadcast_arrays(
        np.asarray(eikonal_slowness_s_per_km, dtype=float),
        np.asarray(amplitude_correction_s2_per_km2, dtype=float),
    )
    return slowness**2 - correction


def helmholtz_slowness_and_velocity(
    corrected_slowness_squared: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert positive corrected slowness squared to slowness and velocity."""
    squared = np.asarray(corrected_slowness_squared, dtype=float)
    slowness = np.full(squared.shape, np.nan)
    velocity = np.full(squared.shape, np.nan)
    valid = np.isfinite(squared) & (squared > 0)
    slowness[valid] = np.sqrt(squared[valid])
    velocity[valid] = 1.0 / slowness[valid]
    return slowness, velocity
