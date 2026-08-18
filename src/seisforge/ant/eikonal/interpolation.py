"""Scattered-field interpolation backends for Eikonal tomography."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator

from .grid import local_xy_km
from .models import GeographicGrid, InterpolationConfig


def interpolate_travel_time_field(
    receiver_longitude_deg: np.ndarray,
    receiver_latitude_deg: np.ndarray,
    travel_time_s: np.ndarray,
    grid: GeographicGrid,
    config: InterpolationConfig,
    *,
    comparison: bool = False,
) -> np.ndarray:
    """Interpolate scattered travel times onto a geographic grid."""
    return interpolate_scalar_field(
        receiver_longitude_deg,
        receiver_latitude_deg,
        travel_time_s,
        grid,
        config,
        comparison=comparison,
    )


def interpolate_scalar_field(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    field_values: np.ndarray,
    grid: GeographicGrid,
    config: InterpolationConfig,
    *,
    comparison: bool = False,
) -> np.ndarray:
    """Interpolate finite scattered scalar values onto a geographic grid."""
    values = np.asarray(field_values, dtype=float)
    longitude = np.asarray(longitude_deg, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    finite = np.isfinite(values) & np.isfinite(longitude) & np.isfinite(latitude)
    longitude = longitude[finite]
    latitude = latitude[finite]
    values = values[finite]
    longitude, latitude, values = _deduplicate_measurements(
        longitude, latitude, values
    )
    if values.size < 3:
        raise ValueError("At least three unique field points are required.")

    if config.method == "scipy_rbf":
        smoothing = (
            config.stability_check.comparison_smoothing
            if comparison
            else config.smoothing
        )
        return _interpolate_scipy_rbf(
            longitude,
            latitude,
            values,
            grid,
            config,
            smoothing=smoothing,
        )
    if config.method == "gmt_surface":
        tension = (
            config.stability_check.comparison_tension
            if comparison
            else config.tension
        )
        return _interpolate_gmt_surface(
            longitude,
            latitude,
            values,
            grid,
            config,
            tension=tension,
        )
    raise ValueError(f"Unsupported interpolation method: {config.method}")


def _interpolate_scipy_rbf(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    values: np.ndarray,
    grid: GeographicGrid,
    config: InterpolationConfig,
    *,
    smoothing: float,
) -> np.ndarray:
    """Interpolate in a local metric projection with SciPy RBF."""

    center_longitude = float(grid.longitude_deg.mean())
    center_latitude = float(grid.latitude_deg.mean())
    input_x, input_y = local_xy_km(
        longitude_deg,
        latitude_deg,
        center_longitude_deg=center_longitude,
        center_latitude_deg=center_latitude,
    )
    grid_x, grid_y = local_xy_km(
        grid.longitude_2d_deg,
        grid.latitude_2d_deg,
        center_longitude_deg=center_longitude,
        center_latitude_deg=center_latitude,
    )
    points = np.column_stack((input_x, input_y))
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    neighbor_count = config.neighbors
    if neighbor_count is not None:
        neighbor_count = min(neighbor_count, values.size)
    interpolator = RBFInterpolator(
        points,
        values,
        kernel=config.kernel,
        smoothing=smoothing,
        neighbors=neighbor_count,
    )
    return interpolator(query).reshape(grid.shape)


def _interpolate_gmt_surface(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    values: np.ndarray,
    grid: GeographicGrid,
    config: InterpolationConfig,
    *,
    tension: float,
) -> np.ndarray:
    """Interpolate with PyGMT surface, matching surfpy's primary backend."""
    try:
        import pygmt
    except ImportError as exc:
        raise RuntimeError(
            "gmt_surface requires PyGMT and the GMT shared library. Install "
            "the 'gmt' SeisForge extra and GMT before selecting this backend."
        ) from exc

    region = [
        float(grid.longitude_deg[0]),
        float(grid.longitude_deg[-1]),
        float(grid.latitude_deg[0]),
        float(grid.latitude_deg[-1]),
    ]
    longitude_step = float(np.diff(grid.longitude_deg).mean())
    latitude_step = float(np.diff(grid.latitude_deg).mean())
    spacing = f"{longitude_step:.12g}/{latitude_step:.12g}"
    table = pd.DataFrame(
        {"longitude": longitude_deg, "latitude": latitude_deg, "value": values}
    )
    if config.block_reduce == "median":
        table = pygmt.blockmedian(data=table, region=region, spacing=spacing)
    surface = pygmt.surface(
        data=table,
        region=region,
        spacing=spacing,
        tension=tension,
    )
    return _aligned_pygmt_grid(surface, grid)


def _aligned_pygmt_grid(surface, grid: GeographicGrid) -> np.ndarray:
    """Validate and align PyGMT's latitude/longitude grid orientation."""
    if surface.ndim != 2:
        raise ValueError("PyGMT surface returned a non-2D grid.")
    latitude_dim, longitude_dim = surface.dims
    latitude = np.asarray(surface.coords[latitude_dim].values, dtype=float)
    longitude = np.asarray(surface.coords[longitude_dim].values, dtype=float)
    values = np.asarray(surface.values, dtype=float)
    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        values = values[::-1, :]
    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1]
        values = values[:, ::-1]
    if values.shape != grid.shape:
        raise ValueError(
            f"PyGMT grid shape {values.shape} does not match {grid.shape}."
        )
    if not np.allclose(latitude, grid.latitude_deg, atol=1e-8, rtol=0.0):
        raise ValueError("PyGMT latitude coordinates do not match the target grid.")
    if not np.allclose(longitude, grid.longitude_deg, atol=1e-8, rtol=0.0):
        raise ValueError("PyGMT longitude coordinates do not match the target grid.")
    return values


def _deduplicate_measurements(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounded = np.column_stack(
        (np.round(longitude_deg, 8), np.round(latitude_deg, 8))
    )
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    if unique.shape[0] == values.size:
        return longitude_deg, latitude_deg, values
    medians = np.empty(unique.shape[0], dtype=float)
    for index in range(unique.shape[0]):
        medians[index] = np.median(values[inverse == index])
    return unique[:, 0], unique[:, 1], medians
