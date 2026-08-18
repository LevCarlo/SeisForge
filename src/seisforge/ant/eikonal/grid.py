"""Geographic grid and distance helpers for Eikonal tomography."""

from __future__ import annotations

import numpy as np
from pyproj import Geod

from .models import GeographicGrid, GridConfig


WGS84 = Geod(ellps="WGS84")
EARTH_RADIUS_KM = 6371.0088


def build_geographic_grid(config: GridConfig) -> GeographicGrid:
    """Build an inclusive regular grid with metric coordinates for derivatives."""
    longitude_deg = _inclusive_axis(
        config.min_longitude_deg,
        config.max_longitude_deg,
        config.longitude_step_deg,
        name="longitude",
    )
    latitude_deg = _inclusive_axis(
        config.min_latitude_deg,
        config.max_latitude_deg,
        config.latitude_step_deg,
        name="latitude",
    )
    longitude_2d_deg, latitude_2d_deg = np.meshgrid(
        longitude_deg, latitude_deg
    )

    x_km_by_latitude = np.empty((latitude_deg.size, longitude_deg.size))
    for index, latitude in enumerate(latitude_deg):
        _, _, distance_m = WGS84.inv(
            longitude_deg[:-1],
            np.full(longitude_deg.size - 1, latitude),
            longitude_deg[1:],
            np.full(longitude_deg.size - 1, latitude),
        )
        x_km_by_latitude[index, 0] = 0.0
        x_km_by_latitude[index, 1:] = np.cumsum(distance_m) / 1000.0

    center_longitude = float(longitude_deg.mean())
    _, _, distance_m = WGS84.inv(
        np.full(latitude_deg.size - 1, center_longitude),
        latitude_deg[:-1],
        np.full(latitude_deg.size - 1, center_longitude),
        latitude_deg[1:],
    )
    y_km = np.concatenate(([0.0], np.cumsum(distance_m) / 1000.0))
    return GeographicGrid(
        longitude_deg=longitude_deg,
        latitude_deg=latitude_deg,
        longitude_2d_deg=longitude_2d_deg,
        latitude_2d_deg=latitude_2d_deg,
        x_km_by_latitude=x_km_by_latitude,
        y_km=y_km,
    )


def local_xy_km(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    *,
    center_longitude_deg: float,
    center_latitude_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project nearby geographic coordinates to local equirectangular km."""
    longitude = np.asarray(longitude_deg, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    x_km = (
        EARTH_RADIUS_KM
        * np.cos(np.deg2rad(center_latitude_deg))
        * np.deg2rad(longitude - center_longitude_deg)
    )
    y_km = EARTH_RADIUS_KM * np.deg2rad(latitude - center_latitude_deg)
    return x_km, y_km


def geodesic_distance_and_azimuth(
    source_longitude_deg: float,
    source_latitude_deg: float,
    receiver_longitude_deg: np.ndarray,
    receiver_latitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return WGS84 source-to-receiver distance (km) and azimuth (degrees)."""
    receiver_longitude = np.asarray(receiver_longitude_deg, dtype=float)
    receiver_latitude = np.asarray(receiver_latitude_deg, dtype=float)
    source_longitude = np.full(receiver_longitude.shape, source_longitude_deg)
    source_latitude = np.full(receiver_latitude.shape, source_latitude_deg)
    azimuth_deg, _, distance_m = WGS84.inv(
        source_longitude,
        source_latitude,
        receiver_longitude,
        receiver_latitude,
    )
    return distance_m / 1000.0, wrap_angle_180(azimuth_deg)


def wrap_angle_180(angle_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def _inclusive_axis(
    minimum: float, maximum: float, step: float, *, name: str
) -> np.ndarray:
    count_float = (maximum - minimum) / step
    count = int(round(count_float))
    if not np.isclose(count_float, count, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"{name} range must be an integer multiple of its grid step."
        )
    if count < 2:
        raise ValueError(f"{name} grid must contain at least three points.")
    return minimum + np.arange(count + 1, dtype=float) * step
