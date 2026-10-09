"""Pure numerical formulas for surface-wave Eikonal tomography."""

from __future__ import annotations

import numpy as np

from .grid import prime_vertical_radius_km, wrap_angle_180
from .models import GeographicGrid


LAPLACIAN_METRICS = ("ellipsoidal", "locally_flat")


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
    *,
    metric: str = "ellipsoidal",
) -> np.ndarray:
    """Compute the horizontal travel-time Laplacian."""
    return scalar_field_laplacian_per_km2(travel_time_s, grid, metric=metric)


def parallel_convergence_per_km(latitude_deg: np.ndarray) -> np.ndarray:
    """``(1/r) dr/dy = -tan(lat) / N`` for the parallel radius ``r = N cos(lat)``.

    ``y`` is meridian arc length, so ``dr/dy = (d(N cos lat)/d lat) / M =
    -sin(lat)`` exactly on the ellipsoid, with ``M`` and ``N`` the meridional
    and prime-vertical radii of curvature.
    """
    latitude_rad = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    return -np.tan(latitude_rad) / prime_vertical_radius_km(latitude_deg)


def scalar_field_laplacian_per_km2(
    field_values: np.ndarray,
    grid: GeographicGrid,
    *,
    metric: str = "ellipsoidal",
) -> np.ndarray:
    """Horizontal Laplacian in grid arc-length coordinates.

    With ``x`` the arc length along each parallel and ``y`` along the
    meridian, the surface Laplacian is

        f_xx + f_yy + (1/r) (dr/dy) f_y,   (1/r) dr/dy = -tan(lat) / N.

    ``metric="locally_flat"`` drops the last term; it reproduces SeisForge
    results before the metric term was added and is kept for comparison only.
    """
    if metric not in LAPLACIAN_METRICS:
        raise ValueError(f"metric must be one of {LAPLACIAN_METRICS}.")
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
    laplacian = second_north + second_east
    if metric == "ellipsoidal":
        laplacian = laplacian + (
            parallel_convergence_per_km(grid.latitude_deg)[:, None] * gradient_north
        )
    return laplacian


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
