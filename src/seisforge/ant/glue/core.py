"""Pure array functions that connect AFTAN, Eikonal and the inversion.

Conventions
-----------
- Periods are in s, phase velocities in km/s, slowness in s/km, longitudes
  and latitudes in degrees.
- Resampling is piecewise-linear interpolation of the value in period
  between neighbouring measured samples. It never extrapolates: target
  periods outside the measured period range are NaN. Interpolation bridges
  any gap between neighbouring samples; AFTAN period grids are dense
  (fractions of a second), so a gap only appears where QC dropped samples.
- Grid sampling ("bilinear") interpolates linearly in longitude and latitude
  between the four surrounding grid nodes and returns NaN unless all four are
  valid; "nearest" takes the nearest node in degrees.
- Station-curve uncertainty: see :func:`station_sigma_km_s`.
"""

from __future__ import annotations

import numpy as np


def resample_to_periods(
    period_s: np.ndarray,
    value: np.ndarray,
    target_period_s: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ``value(period_s)`` onto ``target_period_s``.

    Non-finite samples are ignored. The remaining ``period_s`` must be
    strictly increasing; otherwise ``ValueError`` is raised (a non-monotonic
    AFTAN instantaneous-period curve is a bad pick, not something to sort).
    Targets outside ``[min(period_s), max(period_s)]`` are NaN.
    """
    period_s = np.asarray(period_s, dtype=float)
    value = np.asarray(value, dtype=float)
    target_period_s = np.asarray(target_period_s, dtype=float)
    if period_s.shape != value.shape or period_s.ndim != 1:
        raise ValueError("period_s and value must be 1-D arrays of equal length.")
    keep = np.isfinite(period_s) & np.isfinite(value)
    period_s = period_s[keep]
    value = value[keep]
    output = np.full(target_period_s.shape, np.nan)
    if period_s.size == 0:
        return output
    if np.any(np.diff(period_s) <= 0):
        raise ValueError("period_s must be strictly increasing.")
    inside = (target_period_s >= period_s[0]) & (target_period_s <= period_s[-1])
    output[inside] = np.interp(target_period_s[inside], period_s, value)
    return output


def wrap_longitude_deg(longitude_deg: np.ndarray, convention: str) -> np.ndarray:
    """Return longitudes in the ``"-180_180"`` or ``"0_360"`` convention."""
    longitude_deg = np.asarray(longitude_deg, dtype=float)
    if convention == "0_360":
        return np.mod(longitude_deg, 360.0)
    if convention == "-180_180":
        wrapped = np.mod(longitude_deg + 180.0, 360.0) - 180.0
        # Keep +180 as given instead of turning it into -180.
        return np.where(longitude_deg == 180.0, 180.0, wrapped)
    raise ValueError("convention must be '-180_180' or '0_360'.")


def sample_grid(
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    query_longitude_deg: float,
    query_latitude_deg: float,
    *,
    method: str = "bilinear",
) -> np.ndarray:
    """Sample ``values[..., latitude, longitude]`` at one point.

    ``longitude_deg`` and ``latitude_deg`` are the strictly increasing grid
    axes; ``values`` and ``valid`` have the grid in their last two axes
    (latitude, longitude) and any leading axes (e.g. period). The result has
    the leading shape; it is NaN where a needed node is not valid. A point
    outside the grid raises ``ValueError``.
    """
    longitude_deg = np.asarray(longitude_deg, dtype=float)
    latitude_deg = np.asarray(latitude_deg, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.shape[-2:] != (latitude_deg.size, longitude_deg.size):
        raise ValueError("values must end with (latitude, longitude) axes.")
    for axis in (longitude_deg, latitude_deg):
        if axis.size < 2 or np.any(np.diff(axis) <= 0):
            raise ValueError("grid axes must be strictly increasing with >= 2 nodes.")
    x, y = float(query_longitude_deg), float(query_latitude_deg)
    if not (longitude_deg[0] <= x <= longitude_deg[-1]) or not (
        latitude_deg[0] <= y <= latitude_deg[-1]
    ):
        raise ValueError(
            f"point ({x}, {y}) lies outside the grid "
            f"[{longitude_deg[0]}, {longitude_deg[-1]}] x "
            f"[{latitude_deg[0]}, {latitude_deg[-1]}]."
        )
    if method == "nearest":
        row = int(np.argmin(np.abs(latitude_deg - y)))
        column = int(np.argmin(np.abs(longitude_deg - x)))
        return np.where(valid[..., row, column], values[..., row, column], np.nan)
    if method != "bilinear":
        raise ValueError("method must be 'bilinear' or 'nearest'.")

    column = int(np.clip(np.searchsorted(longitude_deg, x) - 1, 0, longitude_deg.size - 2))
    row = int(np.clip(np.searchsorted(latitude_deg, y) - 1, 0, latitude_deg.size - 2))
    tx = (x - longitude_deg[column]) / (longitude_deg[column + 1] - longitude_deg[column])
    ty = (y - latitude_deg[row]) / (latitude_deg[row + 1] - latitude_deg[row])
    corners = [
        (row, column, (1 - tx) * (1 - ty)),
        (row, column + 1, tx * (1 - ty)),
        (row + 1, column, (1 - tx) * ty),
        (row + 1, column + 1, tx * ty),
    ]
    total = np.zeros(values.shape[:-2])
    usable = np.ones(values.shape[:-2], dtype=bool)
    for r, c, weight in corners:
        usable &= valid[..., r, c]
        total = total + weight * np.where(valid[..., r, c], values[..., r, c], 0.0)
    return np.where(usable, total, np.nan)


def velocity_std_from_slowness_std(
    slowness_s_per_km: np.ndarray,
    slowness_std_s_per_km: np.ndarray,
) -> np.ndarray:
    """First-order velocity spread from a slowness spread.

    ``v = 1/s`` so ``dv = -ds / s**2`` and ``std(v) ~= std(s) / s**2``
    (delta method; accurate while ``std(s) << s``).
    """
    slowness = np.asarray(slowness_s_per_km, dtype=float)
    spread = np.asarray(slowness_std_s_per_km, dtype=float)
    return spread / slowness**2


def station_sigma_km_s(
    sigma_km_s: np.ndarray,
    sigma_floor_km_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a lower bound to a one-sigma velocity uncertainty.

    Returns ``(max(sigma, floor), floored)`` where ``floored`` marks entries
    raised to the floor. NaN stays NaN (no uncertainty, no data point).

    The Eikonal stack's standard error of the mean assumes the single-source
    estimates are independent. They are not (shared stations, overlapping
    interpolation), so it is a lower bound on the error and can become
    implausibly small where many sources overlap; the floor keeps such points
    from dominating the inversion misfit.
    """
    if not np.isfinite(sigma_floor_km_s) or sigma_floor_km_s < 0:
        raise ValueError("sigma_floor_km_s must be finite and non-negative.")
    sigma = np.asarray(sigma_km_s, dtype=float)
    floored = np.isfinite(sigma) & (sigma < sigma_floor_km_s)
    return np.where(floored, sigma_floor_km_s, sigma), floored
