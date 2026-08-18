"""Typed configuration and result models for Eikonal tomography."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np


class RejectionReason(IntEnum):
    """Reason that a gridded single-source measurement is unavailable."""

    ACCEPTED = 0
    INTERPOLATION_UNSTABLE = 1
    INSUFFICIENT_STATION_COVERAGE = 2
    APPARENT_VELOCITY_OUT_OF_RANGE = 3
    ADJACENT_TO_INVALID_FIELD = 4
    SOURCE_DISTANCE_TOO_SHORT = 5
    TRAVEL_TIME_CURVATURE_TOO_LARGE = 6
    GRID_BOUNDARY = 7
    INSUFFICIENT_SOURCE_DATA = 8


REJECTION_FLAG_MEANINGS = " ".join(
    reason.name.lower() for reason in RejectionReason
)


class HelmholtzRejectionReason(IntEnum):
    """Reason that a Helmholtz-corrected single-source value is unavailable."""

    ACCEPTED = 0
    NOT_COMPUTED = 1
    EIKONAL_REJECTED = 2
    AMPLITUDE_INTERPOLATION_UNSTABLE = 3
    INVALID_AMPLITUDE_FIELD = 4
    AMPLITUDE_CURVATURE_TOO_LARGE = 5
    CORRECTED_SLOWNESS_SQUARED_NONPOSITIVE = 6


HELMHOLTZ_REJECTION_FLAG_MEANINGS = " ".join(
    reason.name.lower() for reason in HelmholtzRejectionReason
)


@dataclass(frozen=True)
class EikonalInputConfig:
    dataset: Path
    periods_s: tuple[int, ...]
    min_snr: float | None = None


@dataclass(frozen=True)
class GridConfig:
    longitude_convention: str
    min_longitude_deg: float
    max_longitude_deg: float
    min_latitude_deg: float
    max_latitude_deg: float
    longitude_step_deg: float
    latitude_step_deg: float


@dataclass(frozen=True)
class CycleSkipConfig:
    enabled: bool = True
    min_period_s: float = 20.0
    max_residual_s: float = 10.0
    trial_count: int = 5
    seed_station_stride: int = 2


@dataclass(frozen=True)
class InterpolationStabilityConfig:
    enabled: bool = True
    comparison_smoothing: float = 0.2
    comparison_tension: float = 0.2
    max_travel_time_difference_s: float = 2.0


@dataclass(frozen=True)
class InterpolationConfig:
    method: str = "scipy_rbf"
    kernel: str = "thin_plate_spline"
    smoothing: float = 0.0
    neighbors: int | None = None
    block_reduce: str = "median"
    tension: float = 0.0
    stability_check: InterpolationStabilityConfig = field(
        default_factory=InterpolationStabilityConfig
    )


@dataclass(frozen=True)
class EikonalMethodConfig:
    gradient_scheme: str = "second_order"
    min_apparent_velocity_km_s: float = 2.0
    max_apparent_velocity_km_s: float = 5.0
    boundary_cells_to_mask: int = 1


@dataclass(frozen=True)
class HelmholtzConfig:
    enabled: bool = False
    min_period_s: float = 0.0
    min_amplitude: float = 0.0
    max_relative_interpolation_difference: float = 0.01
    curvature_reference_velocity_km_s: float = 4.0


@dataclass(frozen=True)
class StationCoverageConfig:
    mode: str = "nearby_or_quadrants"
    nearby_radius_km: float = 150.0
    quadrant_radius_km: float = 250.0
    min_quadrants: int = 4


@dataclass(frozen=True)
class FarFieldConfig:
    min_wavelengths: float = 3.0
    reference_phase_velocity_km_s: float = 3.5


@dataclass(frozen=True)
class SourceFieldConfig:
    min_valid_fraction: float = 0.05
    min_valid_grid_cells: int = 0


@dataclass(frozen=True)
class EikonalQCConfig:
    max_abs_travel_time_laplacian_s_per_km2: float = 0.005
    station_coverage: StationCoverageConfig = field(
        default_factory=StationCoverageConfig
    )
    far_field: FarFieldConfig = field(default_factory=FarFieldConfig)
    source_field: SourceFieldConfig = field(default_factory=SourceFieldConfig)


@dataclass(frozen=True)
class AzimuthWeightingConfig:
    enabled: bool = True
    neighbor_width_deg: float = 20.0
    require_peer: bool = True
    max_weight_standard_deviations: float = 3.0


@dataclass(frozen=True)
class OutlierRejectionConfig:
    enabled: bool = True
    max_standard_deviations: float = 2.0
    iterations: int = 1


@dataclass(frozen=True)
class StackingConfig:
    min_raw_measurements_per_cell: int = 50
    azimuth_weighting: AzimuthWeightingConfig = field(
        default_factory=AzimuthWeightingConfig
    )
    outlier_rejection: OutlierRejectionConfig = field(
        default_factory=OutlierRejectionConfig
    )


@dataclass(frozen=True)
class EikonalOutputConfig:
    directory: Path
    event_fields_directory: str = "event_fields"
    stack_file: str = "eikonal_stack.nc"
    overwrite: bool = False
    compression_level: int = 4

    @property
    def event_fields_path(self) -> Path:
        return self.directory / self.event_fields_directory

    @property
    def stack_path(self) -> Path:
        return self.directory / self.stack_file


@dataclass(frozen=True)
class ComputeConfig:
    workers: int = 1
    source_chunk_size: int = 50


@dataclass(frozen=True)
class EikonalRunConfig:
    project: str
    input: EikonalInputConfig
    grid: GridConfig
    cycle_skip: CycleSkipConfig = field(default_factory=CycleSkipConfig)
    interpolation: InterpolationConfig = field(
        default_factory=InterpolationConfig
    )
    eikonal: EikonalMethodConfig = field(default_factory=EikonalMethodConfig)
    helmholtz: HelmholtzConfig = field(default_factory=HelmholtzConfig)
    qc: EikonalQCConfig = field(default_factory=EikonalQCConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)
    output: EikonalOutputConfig | None = None
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    schema_version: int = 1


@dataclass(frozen=True)
class GeographicGrid:
    longitude_deg: np.ndarray
    latitude_deg: np.ndarray
    longitude_2d_deg: np.ndarray
    latitude_2d_deg: np.ndarray
    x_km_by_latitude: np.ndarray
    y_km: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.latitude_deg.size, self.longitude_deg.size


@dataclass(frozen=True)
class CycleSkipResult:
    travel_time_s: np.ndarray
    valid: np.ndarray
    cycle_count: np.ndarray
    residual_s: np.ndarray


@dataclass(frozen=True)
class HelmholtzField:
    amplitude: np.ndarray
    amplitude_laplacian_per_km2: np.ndarray
    amplitude_laplacian_correction_s2_per_km2: np.ndarray
    slowness_s_per_km: np.ndarray
    phase_velocity_km_s: np.ndarray
    rejection_reason: np.ndarray
    measurement_count: int
    applied: bool
    source_accepted: bool


@dataclass(frozen=True)
class EventField:
    source_station: str
    source_longitude_deg: float
    source_latitude_deg: float
    period_s: int
    travel_time_s: np.ndarray
    travel_time_laplacian_s_per_km2: np.ndarray
    slowness_s_per_km: np.ndarray
    apparent_velocity_km_s: np.ndarray
    propagation_angle_deg: np.ndarray
    great_circle_azimuth_deg: np.ndarray
    azimuth_deflection_deg: np.ndarray
    rejection_reason: np.ndarray
    amplitude: np.ndarray
    amplitude_laplacian_per_km2: np.ndarray
    amplitude_laplacian_correction_s2_per_km2: np.ndarray
    helmholtz_slowness_s_per_km: np.ndarray
    helmholtz_phase_velocity_km_s: np.ndarray
    helmholtz_rejection_reason: np.ndarray
    input_measurement_count: int
    used_measurement_count: int
    amplitude_measurement_count: int
    cycle_corrected_count: int
    cycle_rejected_count: int
    source_accepted: bool
    helmholtz_applied: bool
    helmholtz_source_accepted: bool
