"""Integer-period phase travel-time correction."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .grid import local_xy_km
from .models import CycleSkipResult


def correct_cycle_skips(
    receiver_longitude_deg: np.ndarray,
    receiver_latitude_deg: np.ndarray,
    source_distance_km: np.ndarray,
    travel_time_s: np.ndarray,
    *,
    period_s: float,
    max_residual_s: float,
    trial_count: int,
    seed_station_stride: int,
) -> CycleSkipResult:
    """Unwrap phase travel times using locally propagated reference velocity."""
    longitude = np.asarray(receiver_longitude_deg, dtype=float)
    latitude = np.asarray(receiver_latitude_deg, dtype=float)
    distance = np.asarray(source_distance_km, dtype=float)
    travel_time = np.asarray(travel_time_s, dtype=float)
    finite = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & np.isfinite(distance)
        & np.isfinite(travel_time)
        & (distance > 0)
        & (travel_time > 0)
    )
    if finite.sum() < 2:
        return CycleSkipResult(
            travel_time_s=np.where(finite, travel_time, np.nan),
            valid=finite,
            cycle_count=np.zeros(travel_time.shape, dtype=np.int16),
            residual_s=np.where(finite, 0.0, np.nan),
        )

    center_longitude = float(np.mean(longitude[finite]))
    center_latitude = float(np.mean(latitude[finite]))
    x_km, y_km = local_xy_km(
        longitude,
        latitude,
        center_longitude_deg=center_longitude,
        center_latitude_deg=center_latitude,
    )
    pair_distance = cdist(np.column_stack((x_km, y_km)), np.column_stack((x_km, y_km)))
    finite_indices = np.flatnonzero(finite)
    center_distance = np.hypot(x_km, y_km)
    ordered_seeds = finite_indices[np.argsort(center_distance[finite_indices])]
    seeds = ordered_seeds[::seed_station_stride][:trial_count]
    if seeds.size == 0:
        seeds = ordered_seeds[:1]

    best: CycleSkipResult | None = None
    best_score: tuple[int, float] | None = None
    for seed in seeds:
        result = _unwrap_from_seed(
            seed,
            finite,
            pair_distance,
            distance,
            travel_time,
            period_s=period_s,
            max_residual_s=max_residual_s,
        )
        residual_sum = float(np.nansum(result.residual_s[result.valid]))
        score = (int(result.valid.sum()), -residual_sum)
        if best is None or score > best_score:
            best = result
            best_score = score
    assert best is not None
    return best


def _unwrap_from_seed(
    seed: int,
    finite: np.ndarray,
    pair_distance_km: np.ndarray,
    source_distance_km: np.ndarray,
    travel_time_s: np.ndarray,
    *,
    period_s: float,
    max_residual_s: float,
) -> CycleSkipResult:
    order = np.argsort(pair_distance_km[seed])
    corrected = np.full(travel_time_s.shape, np.nan, dtype=float)
    residual = np.full(travel_time_s.shape, np.nan, dtype=float)
    cycle_count = np.zeros(travel_time_s.shape, dtype=np.int16)
    valid = np.zeros(travel_time_s.shape, dtype=bool)
    corrected[seed] = travel_time_s[seed]
    residual[seed] = 0.0
    valid[seed] = True

    for index in order:
        if index == seed or not finite[index]:
            continue
        previous = np.flatnonzero(valid)
        if previous.size == 0:
            continue
        nearest = previous[np.argmin(pair_distance_km[index, previous])]
        reference_velocity = source_distance_km[nearest] / corrected[nearest]
        if not np.isfinite(reference_velocity) or reference_velocity <= 0:
            continue
        predicted = source_distance_km[index] / reference_velocity
        shift = int(np.rint((predicted - travel_time_s[index]) / period_s))
        candidate = travel_time_s[index] + shift * period_s
        candidate_residual = abs(candidate - predicted)
        residual[index] = candidate_residual
        cycle_count[index] = shift
        if candidate > 0 and candidate_residual <= max_residual_s:
            corrected[index] = candidate
            valid[index] = True
    return CycleSkipResult(
        travel_time_s=corrected,
        valid=valid,
        cycle_count=cycle_count,
        residual_s=residual,
    )

