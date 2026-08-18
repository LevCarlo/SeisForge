"""Quality-control masks for gridded Eikonal measurements."""

from __future__ import annotations

import numpy as np

from .grid import geodesic_distance_and_azimuth, local_xy_km
from .models import GeographicGrid, RejectionReason, StationCoverageConfig


def assign_rejection_reason(
    rejection_reason: np.ndarray,
    mask: np.ndarray,
    reason: RejectionReason,
) -> None:
    """Assign a reason only to cells not rejected by an earlier QC stage."""
    select = np.asarray(mask, dtype=bool) & (
        rejection_reason == RejectionReason.ACCEPTED
    )
    rejection_reason[select] = int(reason)


def station_coverage_mask(
    grid: GeographicGrid,
    receiver_longitude_deg: np.ndarray,
    receiver_latitude_deg: np.ndarray,
    config: StationCoverageConfig,
) -> np.ndarray:
    """Return cells without the requested nearby/quadrant station coverage."""
    receiver_longitude = np.asarray(receiver_longitude_deg, dtype=float)
    receiver_latitude = np.asarray(receiver_latitude_deg, dtype=float)
    center_longitude = float(grid.longitude_deg.mean())
    center_latitude = float(grid.latitude_deg.mean())
    receiver_x, receiver_y = local_xy_km(
        receiver_longitude,
        receiver_latitude,
        center_longitude_deg=center_longitude,
        center_latitude_deg=center_latitude,
    )
    grid_x, grid_y = local_xy_km(
        grid.longitude_2d_deg,
        grid.latitude_2d_deg,
        center_longitude_deg=center_longitude,
        center_latitude_deg=center_latitude,
    )
    covered = np.zeros(grid.shape, dtype=bool)
    for row, column in np.ndindex(grid.shape):
        delta_x = receiver_x - grid_x[row, column]
        delta_y = receiver_y - grid_y[row, column]
        distance = np.hypot(delta_x, delta_y)
        nearby = bool(np.any(distance <= config.nearby_radius_km))
        in_quadrant_radius = distance <= config.quadrant_radius_km
        quadrant_count = _quadrant_count(
            delta_x[in_quadrant_radius], delta_y[in_quadrant_radius]
        )
        quadrants = quadrant_count >= config.min_quadrants
        if config.mode == "nearby":
            covered[row, column] = nearby
        elif config.mode == "quadrants":
            covered[row, column] = quadrants
        else:
            covered[row, column] = nearby or quadrants
    return ~covered


def source_distance_mask(
    grid: GeographicGrid,
    source_longitude_deg: float,
    source_latitude_deg: float,
    *,
    minimum_distance_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the near-source mask and source-to-grid great-circle azimuth."""
    distance_km, azimuth_deg = geodesic_distance_and_azimuth(
        source_longitude_deg,
        source_latitude_deg,
        grid.longitude_2d_deg,
        grid.latitude_2d_deg,
    )
    return distance_km < minimum_distance_km, azimuth_deg


def adjacent_to_nonfinite_mask(field: np.ndarray) -> np.ndarray:
    """Return finite cells sharing an edge with a non-finite cell."""
    invalid = ~np.isfinite(field)
    adjacent = np.zeros(invalid.shape, dtype=bool)
    adjacent[1:] |= invalid[:-1]
    adjacent[:-1] |= invalid[1:]
    adjacent[:, 1:] |= invalid[:, :-1]
    adjacent[:, :-1] |= invalid[:, 1:]
    return adjacent & ~invalid


def boundary_mask(shape: tuple[int, int], cells: int) -> np.ndarray:
    """Return a mask covering a fixed number of outer grid cells."""
    mask = np.zeros(shape, dtype=bool)
    if cells == 0:
        return mask
    cells = min(cells, (min(shape) + 1) // 2)
    mask[:cells] = True
    mask[-cells:] = True
    mask[:, :cells] = True
    mask[:, -cells:] = True
    return mask


def _quadrant_count(delta_x_km: np.ndarray, delta_y_km: np.ndarray) -> int:
    if delta_x_km.size == 0:
        return 0
    quadrant = (delta_x_km >= 0).astype(np.int8) * 2
    quadrant += (delta_y_km >= 0).astype(np.int8)
    return int(np.unique(quadrant).size)

