"""Single-source travel-time field and Eikonal measurement construction."""

from __future__ import annotations

import numpy as np

from .core import (
    apparent_slowness_s_per_km,
    apparent_velocity_km_s,
    phase_travel_time_s,
    propagation_angle_deg,
    scalar_field_laplacian_per_km2,
    travel_time_gradient_s_per_km,
    travel_time_laplacian_s_per_km2,
)
from .cycle_skip import correct_cycle_skips
from .grid import geodesic_distance_and_azimuth, wrap_angle_180
from .helmholtz import (
    amplitude_laplacian_correction_s2_per_km2,
    corrected_slowness_squared_s2_per_km2,
    helmholtz_slowness_and_velocity,
)
from .interpolation import interpolate_scalar_field, interpolate_travel_time_field
from .models import (
    CycleSkipConfig,
    EikonalMethodConfig,
    EikonalQCConfig,
    EventField,
    GeographicGrid,
    HelmholtzConfig,
    HelmholtzField,
    HelmholtzRejectionReason,
    InterpolationConfig,
    RejectionReason,
)
from .qc import (
    adjacent_to_nonfinite_mask,
    assign_rejection_reason,
    boundary_mask,
    source_distance_mask,
    station_coverage_mask,
)


def compute_event_field(
    *,
    source_station: str,
    source_longitude_deg: float,
    source_latitude_deg: float,
    receiver_longitude_deg: np.ndarray,
    receiver_latitude_deg: np.ndarray,
    phase_velocity_km_s: np.ndarray,
    amplitude: np.ndarray | None = None,
    period_s: int,
    grid: GeographicGrid,
    cycle_skip: CycleSkipConfig,
    interpolation: InterpolationConfig,
    method: EikonalMethodConfig,
    helmholtz: HelmholtzConfig = HelmholtzConfig(),
    qc: EikonalQCConfig,
) -> EventField:
    """Compute one virtual source's gridded local slowness measurement."""
    receiver_longitude = np.asarray(receiver_longitude_deg, dtype=float)
    receiver_latitude = np.asarray(receiver_latitude_deg, dtype=float)
    phase_velocity = np.asarray(phase_velocity_km_s, dtype=float)
    if amplitude is None:
        amplitude_values = np.full(phase_velocity.shape, np.nan)
    else:
        amplitude_values = np.asarray(amplitude, dtype=float)
        if amplitude_values.shape != phase_velocity.shape:
            raise ValueError("amplitude must have the phase-velocity array shape.")
    input_count = int(phase_velocity.size)
    finite = (
        np.isfinite(receiver_longitude)
        & np.isfinite(receiver_latitude)
        & np.isfinite(phase_velocity)
        & (phase_velocity > 0)
    )
    receiver_longitude = receiver_longitude[finite]
    receiver_latitude = receiver_latitude[finite]
    phase_velocity = phase_velocity[finite]
    amplitude_values = amplitude_values[finite]
    if phase_velocity.size < 3:
        return _empty_event_field(
            source_station=source_station,
            source_longitude_deg=source_longitude_deg,
            source_latitude_deg=source_latitude_deg,
            period_s=period_s,
            grid=grid,
            input_measurement_count=input_count,
        )

    distance_km, _ = geodesic_distance_and_azimuth(
        source_longitude_deg,
        source_latitude_deg,
        receiver_longitude,
        receiver_latitude,
    )
    travel_time = phase_travel_time_s(distance_km, phase_velocity)
    cycle_count = np.zeros(travel_time.shape, dtype=np.int16)
    cycle_rejected_count = 0
    if cycle_skip.enabled and period_s >= cycle_skip.min_period_s:
        cycle_result = correct_cycle_skips(
            receiver_longitude,
            receiver_latitude,
            distance_km,
            travel_time,
            period_s=float(period_s),
            max_residual_s=cycle_skip.max_residual_s,
            trial_count=cycle_skip.trial_count,
            seed_station_stride=cycle_skip.seed_station_stride,
        )
        keep = cycle_result.valid
        cycle_rejected_count = int((~keep).sum())
        receiver_longitude = receiver_longitude[keep]
        receiver_latitude = receiver_latitude[keep]
        travel_time = cycle_result.travel_time_s[keep]
        cycle_count = cycle_result.cycle_count[keep]
        amplitude_values = amplitude_values[keep]

    used_count = int(travel_time.size)
    if used_count < 3:
        return _empty_event_field(
            source_station=source_station,
            source_longitude_deg=source_longitude_deg,
            source_latitude_deg=source_latitude_deg,
            period_s=period_s,
            grid=grid,
            input_measurement_count=input_count,
            used_measurement_count=used_count,
            cycle_rejected_count=cycle_rejected_count,
        )

    interpolation_longitude = np.concatenate(
        ([source_longitude_deg], receiver_longitude)
    )
    interpolation_latitude = np.concatenate(
        ([source_latitude_deg], receiver_latitude)
    )
    interpolation_time = np.concatenate(([0.0], travel_time))
    try:
        gridded_time = interpolate_travel_time_field(
            interpolation_longitude,
            interpolation_latitude,
            interpolation_time,
            grid,
            interpolation,
        )
        comparison_time = None
        if interpolation.stability_check.enabled:
            comparison_time = interpolate_travel_time_field(
                interpolation_longitude,
                interpolation_latitude,
                interpolation_time,
                grid,
                interpolation,
                comparison=True,
            )
    except (ValueError, np.linalg.LinAlgError):
        return _empty_event_field(
            source_station=source_station,
            source_longitude_deg=source_longitude_deg,
            source_latitude_deg=source_latitude_deg,
            period_s=period_s,
            grid=grid,
            input_measurement_count=input_count,
            used_measurement_count=used_count,
            cycle_corrected_count=int(np.count_nonzero(cycle_count)),
            cycle_rejected_count=cycle_rejected_count,
        )

    gradient_north, gradient_east = travel_time_gradient_s_per_km(
        gridded_time, grid
    )
    slowness = apparent_slowness_s_per_km(gradient_north, gradient_east)
    velocity = apparent_velocity_km_s(slowness)
    propagation = propagation_angle_deg(gradient_north, gradient_east)
    laplacian = travel_time_laplacian_s_per_km2(gridded_time, grid)

    rejection = np.zeros(grid.shape, dtype=np.uint8)
    nonfinite = ~np.isfinite(gridded_time)
    assign_rejection_reason(
        rejection,
        nonfinite | adjacent_to_nonfinite_mask(gridded_time),
        RejectionReason.ADJACENT_TO_INVALID_FIELD,
    )
    if comparison_time is not None:
        unstable = (
            np.abs(gridded_time - comparison_time)
            > interpolation.stability_check.max_travel_time_difference_s
        )
        assign_rejection_reason(
            rejection, unstable, RejectionReason.INTERPOLATION_UNSTABLE
        )
    coverage_failure = station_coverage_mask(
        grid,
        receiver_longitude,
        receiver_latitude,
        qc.station_coverage,
    )
    assign_rejection_reason(
        rejection,
        coverage_failure,
        RejectionReason.INSUFFICIENT_STATION_COVERAGE,
    )
    velocity_failure = (
        ~np.isfinite(velocity)
        | (velocity < method.min_apparent_velocity_km_s)
        | (velocity > method.max_apparent_velocity_km_s)
    )
    assign_rejection_reason(
        rejection,
        velocity_failure,
        RejectionReason.APPARENT_VELOCITY_OUT_OF_RANGE,
    )
    minimum_distance_km = (
        qc.far_field.min_wavelengths
        * qc.far_field.reference_phase_velocity_km_s
        * period_s
    )
    near_source, great_circle_azimuth = source_distance_mask(
        grid,
        source_longitude_deg,
        source_latitude_deg,
        minimum_distance_km=minimum_distance_km,
    )
    assign_rejection_reason(
        rejection, near_source, RejectionReason.SOURCE_DISTANCE_TOO_SHORT
    )
    excessive_curvature = (
        ~np.isfinite(laplacian)
        | (
            np.abs(laplacian)
            > qc.max_abs_travel_time_laplacian_s_per_km2
        )
    )
    assign_rejection_reason(
        rejection,
        excessive_curvature,
        RejectionReason.TRAVEL_TIME_CURVATURE_TOO_LARGE,
    )
    assign_rejection_reason(
        rejection,
        boundary_mask(grid.shape, method.boundary_cells_to_mask),
        RejectionReason.GRID_BOUNDARY,
    )

    valid_count = int(np.count_nonzero(rejection == 0))
    valid_fraction = valid_count / rejection.size
    source_accepted = (
        valid_count > 0
        and valid_count >= qc.source_field.min_valid_grid_cells
        and valid_fraction >= qc.source_field.min_valid_fraction
    )
    helmholtz_field = _compute_helmholtz_field(
        receiver_longitude_deg=receiver_longitude,
        receiver_latitude_deg=receiver_latitude,
        amplitude=amplitude_values,
        period_s=period_s,
        grid=grid,
        interpolation=interpolation,
        config=helmholtz,
        eikonal_slowness_s_per_km=slowness,
        eikonal_rejection_reason=rejection,
        source_qc=qc,
    )
    return EventField(
        source_station=source_station,
        source_longitude_deg=source_longitude_deg,
        source_latitude_deg=source_latitude_deg,
        period_s=period_s,
        travel_time_s=gridded_time,
        travel_time_laplacian_s_per_km2=laplacian,
        slowness_s_per_km=slowness,
        apparent_velocity_km_s=velocity,
        propagation_angle_deg=propagation,
        great_circle_azimuth_deg=great_circle_azimuth,
        azimuth_deflection_deg=wrap_angle_180(
            propagation - great_circle_azimuth
        ),
        rejection_reason=rejection,
        amplitude=helmholtz_field.amplitude,
        amplitude_laplacian_per_km2=(
            helmholtz_field.amplitude_laplacian_per_km2
        ),
        amplitude_laplacian_correction_s2_per_km2=(
            helmholtz_field.amplitude_laplacian_correction_s2_per_km2
        ),
        helmholtz_slowness_s_per_km=helmholtz_field.slowness_s_per_km,
        helmholtz_phase_velocity_km_s=helmholtz_field.phase_velocity_km_s,
        helmholtz_rejection_reason=helmholtz_field.rejection_reason,
        input_measurement_count=input_count,
        used_measurement_count=used_count,
        amplitude_measurement_count=helmholtz_field.measurement_count,
        cycle_corrected_count=int(np.count_nonzero(cycle_count)),
        cycle_rejected_count=cycle_rejected_count,
        source_accepted=source_accepted,
        helmholtz_applied=helmholtz_field.applied,
        helmholtz_source_accepted=helmholtz_field.source_accepted,
    )


def _compute_helmholtz_field(
    *,
    receiver_longitude_deg: np.ndarray,
    receiver_latitude_deg: np.ndarray,
    amplitude: np.ndarray,
    period_s: int,
    grid: GeographicGrid,
    interpolation: InterpolationConfig,
    config: HelmholtzConfig,
    eikonal_slowness_s_per_km: np.ndarray,
    eikonal_rejection_reason: np.ndarray,
    source_qc: EikonalQCConfig,
) -> HelmholtzField:
    """Interpolate amplitude and apply the single-source Helmholtz correction."""
    enabled = config.enabled and period_s >= config.min_period_s
    valid_measurement = (
        np.isfinite(amplitude) & (amplitude > config.min_amplitude)
    )
    measurement_count = int(np.count_nonzero(valid_measurement))
    if not enabled or measurement_count < 3:
        return _empty_helmholtz_field(
            grid.shape, measurement_count=measurement_count
        )

    longitude = receiver_longitude_deg[valid_measurement]
    latitude = receiver_latitude_deg[valid_measurement]
    values = amplitude[valid_measurement]
    try:
        gridded_amplitude = interpolate_scalar_field(
            longitude, latitude, values, grid, interpolation
        )
        comparison_amplitude = None
        if interpolation.stability_check.enabled:
            comparison_amplitude = interpolate_scalar_field(
                longitude,
                latitude,
                values,
                grid,
                interpolation,
                comparison=True,
            )
    except (ValueError, np.linalg.LinAlgError):
        return _empty_helmholtz_field(
            grid.shape, measurement_count=measurement_count
        )

    laplacian = scalar_field_laplacian_per_km2(gridded_amplitude, grid)
    correction = amplitude_laplacian_correction_s2_per_km2(
        gridded_amplitude, laplacian, period_s=float(period_s)
    )
    corrected_squared = corrected_slowness_squared_s2_per_km2(
        eikonal_slowness_s_per_km, correction
    )
    corrected_slowness, corrected_velocity = helmholtz_slowness_and_velocity(
        corrected_squared
    )

    rejection = np.zeros(grid.shape, dtype=np.uint8)
    _assign_helmholtz_reason(
        rejection,
        eikonal_rejection_reason != RejectionReason.ACCEPTED,
        HelmholtzRejectionReason.EIKONAL_REJECTED,
    )
    invalid_amplitude = (
        ~np.isfinite(gridded_amplitude)
        | (gridded_amplitude <= config.min_amplitude)
        | adjacent_to_nonfinite_mask(gridded_amplitude)
    )
    _assign_helmholtz_reason(
        rejection,
        invalid_amplitude,
        HelmholtzRejectionReason.INVALID_AMPLITUDE_FIELD,
    )
    if comparison_amplitude is not None:
        unstable = np.abs(gridded_amplitude - comparison_amplitude) > (
            config.max_relative_interpolation_difference
            * np.abs(gridded_amplitude)
        )
        _assign_helmholtz_reason(
            rejection,
            unstable,
            HelmholtzRejectionReason.AMPLITUDE_INTERPOLATION_UNSTABLE,
        )
    max_abs_correction = 1.0 / config.curvature_reference_velocity_km_s**2
    excessive_curvature = (~np.isfinite(correction)) | (
        np.abs(correction) > max_abs_correction
    )
    _assign_helmholtz_reason(
        rejection,
        excessive_curvature,
        HelmholtzRejectionReason.AMPLITUDE_CURVATURE_TOO_LARGE,
    )
    _assign_helmholtz_reason(
        rejection,
        ~np.isfinite(corrected_squared) | (corrected_squared <= 0),
        HelmholtzRejectionReason.CORRECTED_SLOWNESS_SQUARED_NONPOSITIVE,
    )

    valid_count = int(np.count_nonzero(rejection == 0))
    valid_fraction = valid_count / rejection.size
    source_accepted = (
        valid_count > 0
        and valid_count >= source_qc.source_field.min_valid_grid_cells
        and valid_fraction >= source_qc.source_field.min_valid_fraction
    )
    return HelmholtzField(
        amplitude=gridded_amplitude,
        amplitude_laplacian_per_km2=laplacian,
        amplitude_laplacian_correction_s2_per_km2=correction,
        slowness_s_per_km=corrected_slowness,
        phase_velocity_km_s=corrected_velocity,
        rejection_reason=rejection,
        measurement_count=measurement_count,
        applied=True,
        source_accepted=source_accepted,
    )


def _empty_helmholtz_field(
    shape: tuple[int, int], *, measurement_count: int = 0
) -> HelmholtzField:
    empty = np.full(shape, np.nan)
    return HelmholtzField(
        amplitude=empty.copy(),
        amplitude_laplacian_per_km2=empty.copy(),
        amplitude_laplacian_correction_s2_per_km2=empty.copy(),
        slowness_s_per_km=empty.copy(),
        phase_velocity_km_s=empty.copy(),
        rejection_reason=np.full(
            shape,
            int(HelmholtzRejectionReason.NOT_COMPUTED),
            dtype=np.uint8,
        ),
        measurement_count=measurement_count,
        applied=False,
        source_accepted=False,
    )


def _assign_helmholtz_reason(
    rejection_reason: np.ndarray,
    mask: np.ndarray,
    reason: HelmholtzRejectionReason,
) -> None:
    select = np.asarray(mask, dtype=bool) & (
        rejection_reason == HelmholtzRejectionReason.ACCEPTED
    )
    rejection_reason[select] = int(reason)


def _empty_event_field(
    *,
    source_station: str,
    source_longitude_deg: float,
    source_latitude_deg: float,
    period_s: int,
    grid: GeographicGrid,
    input_measurement_count: int,
    used_measurement_count: int = 0,
    cycle_corrected_count: int = 0,
    cycle_rejected_count: int = 0,
) -> EventField:
    shape = grid.shape
    empty = np.full(shape, np.nan)
    _, azimuth = source_distance_mask(
        grid,
        source_longitude_deg,
        source_latitude_deg,
        minimum_distance_km=0.0,
    )
    helmholtz_field = _empty_helmholtz_field(shape)
    return EventField(
        source_station=source_station,
        source_longitude_deg=source_longitude_deg,
        source_latitude_deg=source_latitude_deg,
        period_s=period_s,
        travel_time_s=empty.copy(),
        travel_time_laplacian_s_per_km2=empty.copy(),
        slowness_s_per_km=empty.copy(),
        apparent_velocity_km_s=empty.copy(),
        propagation_angle_deg=empty.copy(),
        great_circle_azimuth_deg=azimuth,
        azimuth_deflection_deg=empty.copy(),
        rejection_reason=np.full(
            shape,
            int(RejectionReason.INSUFFICIENT_SOURCE_DATA),
            dtype=np.uint8,
        ),
        amplitude=helmholtz_field.amplitude,
        amplitude_laplacian_per_km2=(
            helmholtz_field.amplitude_laplacian_per_km2
        ),
        amplitude_laplacian_correction_s2_per_km2=(
            helmholtz_field.amplitude_laplacian_correction_s2_per_km2
        ),
        helmholtz_slowness_s_per_km=helmholtz_field.slowness_s_per_km,
        helmholtz_phase_velocity_km_s=helmholtz_field.phase_velocity_km_s,
        helmholtz_rejection_reason=helmholtz_field.rejection_reason,
        input_measurement_count=input_measurement_count,
        used_measurement_count=used_measurement_count,
        amplitude_measurement_count=0,
        cycle_corrected_count=cycle_corrected_count,
        cycle_rejected_count=cycle_rejected_count,
        source_accepted=False,
        helmholtz_applied=False,
        helmholtz_source_accepted=False,
    )
