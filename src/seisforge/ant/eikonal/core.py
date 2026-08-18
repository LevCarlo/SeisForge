"""Pure numerical formulas for surface-wave Eikonal tomography."""

from __future__ import annotations

import numpy as np

from .grid import wrap_angle_180
from .models import GeographicGrid


def phase_travel_time_s(
    distance_km: np.ndarray, phase_velocity_km_s: np.ndarray
) -> np.ndarray:
    """Convert path-average phase velocity to phase travel time."""
    distance, velocity = np.broadcast_arrays(
        np.asarray(distance_km, dtype=float),
        np.asarray(phase_velocity_km_s, dtype=float),
    )
    output = np.full(distance.shape, np.nan)
    valid = np.isfinite(distance) & np.isfinite(velocity) & (velocity > 0)
    output[valid] = distance[valid] / velocity[valid]
    return output


def travel_time_gradient_s_per_km(
    travel_time_s: np.ndarray,
    grid: GeographicGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute north and east travel-time derivatives on a geographic grid."""
    return scalar_field_gradient_per_km(travel_time_s, grid)


def scalar_field_gradient_per_km(
    field_values: np.ndarray,
    grid: GeographicGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute north and east derivatives of a scalar field."""
    field = np.asarray(field_values, dtype=float)
    if field.shape != grid.shape:
        raise ValueError("scalar field shape does not match the grid.")
    if min(field.shape) < 3:
        raise ValueError("at least three grid points per axis are required.")
    gradient_north = np.gradient(field, grid.y_km, axis=0, edge_order=2)
    gradient_east = np.empty_like(field)
    for row in range(field.shape[0]):
        gradient_east[row] = np.gradient(
            field[row], grid.x_km_by_latitude[row], edge_order=2
        )
    return gradient_north, gradient_east


def travel_time_laplacian_s_per_km2(
    travel_time_s: np.ndarray,
    grid: GeographicGrid,
) -> np.ndarray:
    """Compute the horizontal travel-time Laplacian."""
    return scalar_field_laplacian_per_km2(travel_time_s, grid)


def scalar_field_laplacian_per_km2(
    field_values: np.ndarray,
    grid: GeographicGrid,
) -> np.ndarray:
    """Compute a scalar field's horizontal Laplacian on the metric grid."""
    gradient_north, gradient_east = scalar_field_gradient_per_km(
        field_values, grid
    )
    second_north = np.gradient(
        gradient_north, grid.y_km, axis=0, edge_order=2
    )
    second_east = np.empty_like(gradient_east)
    for row in range(gradient_east.shape[0]):
        second_east[row] = np.gradient(
            gradient_east[row],
            grid.x_km_by_latitude[row],
            edge_order=2,
        )
    return second_north + second_east


def apparent_slowness_s_per_km(
    gradient_north_s_per_km: np.ndarray,
    gradient_east_s_per_km: np.ndarray,
) -> np.ndarray:
    """Return Eikonal slowness, the magnitude of the travel-time gradient."""
    return np.hypot(gradient_north_s_per_km, gradient_east_s_per_km)


def apparent_velocity_km_s(slowness_s_per_km: np.ndarray) -> np.ndarray:
    """Convert positive local slowness to apparent phase velocity."""
    slowness = np.asarray(slowness_s_per_km, dtype=float)
    velocity = np.full(slowness.shape, np.nan, dtype=float)
    valid = np.isfinite(slowness) & (slowness > 0)
    velocity[valid] = 1.0 / slowness[valid]
    return velocity


def propagation_angle_deg(
    gradient_north_s_per_km: np.ndarray,
    gradient_east_s_per_km: np.ndarray,
) -> np.ndarray:
    """Return clockwise-from-north propagation azimuth in degrees."""
    return wrap_angle_180(
        np.rad2deg(
            np.arctan2(gradient_east_s_per_km, gradient_north_s_per_km)
        )
    )
