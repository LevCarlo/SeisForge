"""Azimuth-balanced isotropic stacking for single-source Eikonal fields."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .models import StackingConfig


def stack_event_fields(
    event_fields: xr.Dataset,
    config: StackingConfig,
    *,
    include_helmholtz: bool | None = None,
) -> xr.Dataset:
    """Stack Eikonal and available Helmholtz single-source slowness fields."""
    output = _stack_slowness_fields(event_fields, config)
    if include_helmholtz is None:
        include_helmholtz = (
            "helmholtz_applied" in event_fields
            and bool(np.any(event_fields["helmholtz_applied"].values))
        )
    if (
        not include_helmholtz
        or "helmholtz_slowness_s_per_km" not in event_fields
        or "helmholtz_rejection_reason" not in event_fields
        or "helmholtz_source_accepted" not in event_fields
    ):
        return output

    helmholtz_fields = xr.Dataset(
        data_vars={
            "slowness_s_per_km": event_fields[
                "helmholtz_slowness_s_per_km"
            ],
            "great_circle_azimuth_deg": event_fields[
                "great_circle_azimuth_deg"
            ],
            "rejection_reason": event_fields["helmholtz_rejection_reason"],
            "source_accepted": event_fields["helmholtz_source_accepted"],
        },
        coords=event_fields.coords,
    )
    helmholtz_stack = _stack_slowness_fields(helmholtz_fields, config)
    helmholtz_stack = helmholtz_stack.rename(
        {name: f"helmholtz_{name}" for name in helmholtz_stack.data_vars}
    )
    return xr.merge((output, helmholtz_stack))


def _stack_slowness_fields(
    event_fields: xr.Dataset,
    config: StackingConfig,
) -> xr.Dataset:
    """Stack one family of single-source slowness fields."""
    slowness = np.asarray(event_fields["slowness_s_per_km"].values, dtype=float)
    azimuth = np.asarray(
        event_fields["great_circle_azimuth_deg"].values, dtype=float
    )
    reason = np.asarray(event_fields["rejection_reason"].values)
    source_accepted = np.asarray(
        event_fields["source_accepted"].values, dtype=bool
    )
    valid = (
        (reason == 0)
        & np.isfinite(slowness)
        & (slowness > 0)
        & source_accepted[:, None, None]
    )
    raw_count = valid.sum(axis=0).astype(np.int32)

    weights = _azimuth_weights(azimuth, valid, config)
    low_count = raw_count < config.min_raw_measurements_per_cell
    weights[:, low_count] = 0.0
    weights = _normalize_weights(weights)

    if config.outlier_rejection.enabled:
        for _ in range(config.outlier_rejection.iterations):
            mean, standard_deviation, count = _weighted_mean_and_std(
                slowness, weights
            )
            threshold = (
                config.outlier_rejection.max_standard_deviations
                * standard_deviation
            )
            outlier = (
                (weights > 0)
                & np.isfinite(threshold)[None, :, :]
                & (threshold[None, :, :] > 0)
                & (np.abs(slowness - mean[None, :, :]) > threshold[None, :, :])
            )
            weights[outlier] = 0.0
            weights = _normalize_weights(weights)
            if not np.any(outlier):
                break

    mean_slowness, slowness_std, qc_count = _weighted_mean_and_std(
        slowness, weights
    )
    phase_velocity = np.full(mean_slowness.shape, np.nan)
    positive = np.isfinite(mean_slowness) & (mean_slowness > 0)
    phase_velocity[positive] = 1.0 / mean_slowness[positive]

    event_velocity = np.full(slowness.shape, np.nan)
    event_velocity[valid] = 1.0 / slowness[valid]
    _, velocity_std, _ = _weighted_mean_and_std(
        event_velocity, weights
    )
    effective_count = np.zeros(mean_slowness.shape, dtype=float)
    sum_square_weight = np.sum(weights**2, axis=0)
    effective_count[sum_square_weight > 0] = (
        1.0 / sum_square_weight[sum_square_weight > 0]
    )
    velocity_sem = np.full(mean_slowness.shape, np.nan)
    sem_valid = (effective_count > 1) & np.isfinite(velocity_std)
    velocity_sem[sem_valid] = velocity_std[sem_valid] / np.sqrt(
        effective_count[sem_valid]
    )
    mask = (~positive) | low_count

    directional_coverage = _directional_coverage_deg(azimuth, weights > 0)
    output = xr.Dataset(
        data_vars={
            "phase_velocity_km_s": (
                ("latitude", "longitude"),
                phase_velocity,
            ),
            "slowness_s_per_km": (
                ("latitude", "longitude"),
                mean_slowness,
            ),
            "slowness_std_s_per_km": (
                ("latitude", "longitude"),
                slowness_std,
            ),
            "velocity_sem_km_s": (
                ("latitude", "longitude"),
                velocity_sem,
            ),
            "raw_measurement_count": (
                ("latitude", "longitude"),
                raw_count,
            ),
            "qc_measurement_count": (
                ("latitude", "longitude"),
                qc_count.astype(np.int32),
            ),
            "effective_measurement_count": (
                ("latitude", "longitude"),
                effective_count,
            ),
            "azimuthal_coverage_deg": (
                ("latitude", "longitude"),
                directional_coverage,
            ),
            "mask": (("latitude", "longitude"), mask),
        },
        coords={
            "latitude": event_fields["latitude"].values,
            "longitude": event_fields["longitude"].values,
        },
    )
    output["phase_velocity_km_s"].attrs["units"] = "km s-1"
    output["slowness_s_per_km"].attrs["units"] = "s km-1"
    output["slowness_std_s_per_km"].attrs["units"] = "s km-1"
    output["velocity_sem_km_s"].attrs["units"] = "km s-1"
    output["azimuthal_coverage_deg"].attrs["units"] = "degree"
    return output


def _azimuth_weights(
    azimuth_deg: np.ndarray,
    valid: np.ndarray,
    config: StackingConfig,
) -> np.ndarray:
    if not config.azimuth_weighting.enabled:
        return valid.astype(float)
    event_count = azimuth_deg.shape[0]
    neighbor_count = np.zeros(azimuth_deg.shape, dtype=np.int32)
    for event in range(event_count):
        difference = np.abs(azimuth_deg - azimuth_deg[event])
        difference = np.minimum(difference, 360.0 - difference)
        neighbor_count[event] = np.sum(
            valid
            & valid[event][None, :, :]
            & (difference <= config.azimuth_weighting.neighbor_width_deg),
            axis=0,
        )
    weights = np.zeros(azimuth_deg.shape, dtype=float)
    usable = valid & (neighbor_count > 0)
    if config.azimuth_weighting.require_peer:
        usable &= neighbor_count >= 2
    weights[usable] = 1.0 / neighbor_count[usable]
    _cap_large_weights(
        weights,
        config.azimuth_weighting.max_weight_standard_deviations,
    )
    return weights


def _cap_large_weights(weights: np.ndarray, standard_deviations: float) -> None:
    nonzero = weights > 0
    count = nonzero.sum(axis=0)
    total = weights.sum(axis=0)
    mean = np.zeros(total.shape)
    mean[count > 0] = total[count > 0] / count[count > 0]
    variance = np.sum(
        np.where(nonzero, (weights - mean[None, :, :]) ** 2, 0.0), axis=0
    )
    variance[count > 0] /= count[count > 0]
    cap = mean + standard_deviations * np.sqrt(variance)
    weights[:] = np.where(
        nonzero & (weights > cap[None, :, :]),
        cap[None, :, :],
        weights,
    )


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = weights.sum(axis=0)
    return np.divide(
        weights,
        total[None, :, :],
        out=np.zeros_like(weights),
        where=total[None, :, :] > 0,
    )


def _weighted_mean_and_std(
    values: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = (weights > 0) & np.isfinite(values)
    usable_weights = np.where(usable, weights, 0.0)
    total_weight = usable_weights.sum(axis=0)
    count = usable.sum(axis=0)
    mean = np.divide(
        np.sum(np.where(usable, usable_weights * values, 0.0), axis=0),
        total_weight,
        out=np.full(total_weight.shape, np.nan),
        where=total_weight > 0,
    )
    squared = np.sum(
        np.where(
            usable,
            usable_weights * (values - mean[None, :, :]) ** 2,
            0.0,
        ),
        axis=0,
    )
    correction = np.zeros(total_weight.shape)
    correction[count > 1] = count[count > 1] / (count[count > 1] - 1)
    variance = np.full(total_weight.shape, np.nan)
    variance[count > 1] = (
        squared[count > 1] / total_weight[count > 1] * correction[count > 1]
    )
    return mean, np.sqrt(variance), count


def _directional_coverage_deg(
    azimuth_deg: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    output = np.zeros(azimuth_deg.shape[1:], dtype=float)
    for row, column in np.ndindex(output.shape):
        angles = np.sort((azimuth_deg[:, row, column][valid[:, row, column]]) % 360)
        if angles.size < 2:
            continue
        gaps = np.diff(np.concatenate((angles, angles[:1] + 360.0)))
        output[row, column] = 360.0 - float(np.max(gaps))
    return output
