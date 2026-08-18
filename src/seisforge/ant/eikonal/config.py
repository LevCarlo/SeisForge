"""YAML parsing and validation for Eikonal tomography."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .models import (
    AzimuthWeightingConfig,
    ComputeConfig,
    CycleSkipConfig,
    EikonalInputConfig,
    EikonalMethodConfig,
    EikonalOutputConfig,
    EikonalQCConfig,
    EikonalRunConfig,
    FarFieldConfig,
    GridConfig,
    HelmholtzConfig,
    InterpolationConfig,
    InterpolationStabilityConfig,
    OutlierRejectionConfig,
    SourceFieldConfig,
    StackingConfig,
    StationCoverageConfig,
)


def load_eikonal_config(path: str | Path) -> EikonalRunConfig:
    """Read an Eikonal YAML file and resolve paths relative to that file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    return build_eikonal_config(raw, base_dir=config_path.parent)


def build_eikonal_config(
    raw: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> EikonalRunConfig:
    """Build and validate a typed Eikonal configuration."""
    input_raw = _mapping(raw, "input")
    grid_raw = _mapping(raw, "grid")
    cycle_raw = _mapping(raw, "cycle_skip", required=False)
    interpolation_raw = _mapping(raw, "interpolation", required=False)
    stability_raw = _nested_mapping(
        interpolation_raw, "stability_check", required=False
    )
    method_raw = _mapping(raw, "eikonal", required=False)
    helmholtz_raw = _mapping(raw, "helmholtz", required=False)
    qc_raw = _mapping(raw, "qc", required=False)
    coverage_raw = _nested_mapping(qc_raw, "station_coverage", required=False)
    far_field_raw = _nested_mapping(qc_raw, "far_field", required=False)
    source_field_raw = _nested_mapping(qc_raw, "source_field", required=False)
    stacking_raw = _mapping(raw, "stacking", required=False)
    azimuth_raw = _nested_mapping(
        stacking_raw, "azimuth_weighting", required=False
    )
    outlier_raw = _nested_mapping(
        stacking_raw, "outlier_rejection", required=False
    )
    output_raw = _mapping(raw, "output")
    compute_raw = _mapping(raw, "compute", required=False)

    periods = _integer_periods(input_raw.get("periods_s", ()))
    input_config = EikonalInputConfig(
        dataset=_resolve_path(_required(input_raw, "dataset"), base_dir),
        periods_s=periods,
        min_snr=_optional_float(input_raw.get("min_snr")),
    )
    grid = GridConfig(
        longitude_convention=str(
            grid_raw.get("longitude_convention", "-180_180")
        ),
        min_longitude_deg=float(_required(grid_raw, "min_longitude_deg")),
        max_longitude_deg=float(_required(grid_raw, "max_longitude_deg")),
        min_latitude_deg=float(_required(grid_raw, "min_latitude_deg")),
        max_latitude_deg=float(_required(grid_raw, "max_latitude_deg")),
        longitude_step_deg=float(_required(grid_raw, "longitude_step_deg")),
        latitude_step_deg=float(_required(grid_raw, "latitude_step_deg")),
    )
    cycle_skip = CycleSkipConfig(
        enabled=bool(cycle_raw.get("enabled", True)),
        min_period_s=float(cycle_raw.get("min_period_s", 20.0)),
        max_residual_s=float(cycle_raw.get("max_residual_s", 10.0)),
        trial_count=int(cycle_raw.get("trial_count", 5)),
        seed_station_stride=int(cycle_raw.get("seed_station_stride", 2)),
    )
    interpolation = InterpolationConfig(
        method=str(interpolation_raw.get("method", "scipy_rbf")),
        kernel=str(interpolation_raw.get("kernel", "thin_plate_spline")),
        smoothing=float(interpolation_raw.get("smoothing", 0.0)),
        neighbors=_optional_int(interpolation_raw.get("neighbors")),
        block_reduce=str(interpolation_raw.get("block_reduce", "median")),
        tension=float(interpolation_raw.get("tension", 0.0)),
        stability_check=InterpolationStabilityConfig(
            enabled=bool(stability_raw.get("enabled", True)),
            comparison_smoothing=float(
                stability_raw.get("comparison_smoothing", 0.2)
            ),
            comparison_tension=float(
                stability_raw.get("comparison_tension", 0.2)
            ),
            max_travel_time_difference_s=float(
                stability_raw.get("max_travel_time_difference_s", 2.0)
            ),
        ),
    )
    method = EikonalMethodConfig(
        gradient_scheme=str(method_raw.get("gradient_scheme", "second_order")),
        min_apparent_velocity_km_s=float(
            method_raw.get("min_apparent_velocity_km_s", 2.0)
        ),
        max_apparent_velocity_km_s=float(
            method_raw.get("max_apparent_velocity_km_s", 5.0)
        ),
        boundary_cells_to_mask=int(
            method_raw.get("boundary_cells_to_mask", 1)
        ),
    )
    helmholtz = HelmholtzConfig(
        enabled=bool(helmholtz_raw.get("enabled", False)),
        min_period_s=float(helmholtz_raw.get("min_period_s", 0.0)),
        min_amplitude=float(helmholtz_raw.get("min_amplitude", 0.0)),
        max_relative_interpolation_difference=float(
            helmholtz_raw.get("max_relative_interpolation_difference", 0.01)
        ),
        curvature_reference_velocity_km_s=float(
            helmholtz_raw.get("curvature_reference_velocity_km_s", 4.0)
        ),
    )
    qc = EikonalQCConfig(
        max_abs_travel_time_laplacian_s_per_km2=float(
            qc_raw.get(
                "max_abs_travel_time_laplacian_s_per_km2", 0.005
            )
        ),
        station_coverage=StationCoverageConfig(
            mode=str(coverage_raw.get("mode", "nearby_or_quadrants")),
            nearby_radius_km=float(
                coverage_raw.get("nearby_radius_km", 150.0)
            ),
            quadrant_radius_km=float(
                coverage_raw.get("quadrant_radius_km", 250.0)
            ),
            min_quadrants=int(coverage_raw.get("min_quadrants", 4)),
        ),
        far_field=FarFieldConfig(
            min_wavelengths=float(far_field_raw.get("min_wavelengths", 3.0)),
            reference_phase_velocity_km_s=float(
                far_field_raw.get("reference_phase_velocity_km_s", 3.5)
            ),
        ),
        source_field=SourceFieldConfig(
            min_valid_fraction=float(
                source_field_raw.get("min_valid_fraction", 0.05)
            ),
            min_valid_grid_cells=int(
                source_field_raw.get("min_valid_grid_cells", 0)
            ),
        ),
    )
    stacking = StackingConfig(
        min_raw_measurements_per_cell=int(
            stacking_raw.get("min_raw_measurements_per_cell", 50)
        ),
        azimuth_weighting=AzimuthWeightingConfig(
            enabled=bool(azimuth_raw.get("enabled", True)),
            neighbor_width_deg=float(
                azimuth_raw.get("neighbor_width_deg", 20.0)
            ),
            require_peer=bool(azimuth_raw.get("require_peer", True)),
            max_weight_standard_deviations=float(
                azimuth_raw.get("max_weight_standard_deviations", 3.0)
            ),
        ),
        outlier_rejection=OutlierRejectionConfig(
            enabled=bool(outlier_raw.get("enabled", True)),
            max_standard_deviations=float(
                outlier_raw.get("max_standard_deviations", 2.0)
            ),
            iterations=int(outlier_raw.get("iterations", 1)),
        ),
    )
    output = EikonalOutputConfig(
        directory=_resolve_path(_required(output_raw, "directory"), base_dir),
        event_fields_directory=str(
            output_raw.get("event_fields_directory", "event_fields")
        ),
        stack_file=str(output_raw.get("stack_file", "eikonal_stack.nc")),
        overwrite=bool(output_raw.get("overwrite", False)),
        compression_level=int(output_raw.get("compression_level", 4)),
    )
    compute = ComputeConfig(
        workers=int(compute_raw.get("workers", 1)),
        source_chunk_size=int(compute_raw.get("source_chunk_size", 50)),
    )
    config = EikonalRunConfig(
        schema_version=int(raw.get("schema_version", 1)),
        project=str(raw.get("project", "")),
        input=input_config,
        grid=grid,
        cycle_skip=cycle_skip,
        interpolation=interpolation,
        eikonal=method,
        helmholtz=helmholtz,
        qc=qc,
        stacking=stacking,
        output=output,
        compute=compute,
    )
    _validate(config)
    return config


def _validate(config: EikonalRunConfig) -> None:
    if config.schema_version != 1:
        raise ValueError("schema_version must be 1.")
    if not config.input.periods_s:
        raise ValueError("input.periods_s must contain at least one integer period.")
    if any(period <= 0 for period in config.input.periods_s):
        raise ValueError("input.periods_s values must be positive.")
    if tuple(sorted(set(config.input.periods_s))) != config.input.periods_s:
        raise ValueError("input.periods_s must be unique and increasing.")
    if config.input.min_snr is not None and config.input.min_snr < 0:
        raise ValueError("input.min_snr must be non-negative or null.")
    if config.grid.longitude_convention not in {"-180_180", "0_360"}:
        raise ValueError(
            "grid.longitude_convention must be '-180_180' or '0_360'."
        )
    if config.grid.max_longitude_deg <= config.grid.min_longitude_deg:
        raise ValueError("grid longitude maximum must exceed its minimum.")
    if config.grid.max_latitude_deg <= config.grid.min_latitude_deg:
        raise ValueError("grid latitude maximum must exceed its minimum.")
    if not -90 <= config.grid.min_latitude_deg < config.grid.max_latitude_deg <= 90:
        raise ValueError("grid latitude bounds must lie within [-90, 90].")
    if config.grid.longitude_step_deg <= 0 or config.grid.latitude_step_deg <= 0:
        raise ValueError("grid longitude/latitude steps must be positive.")
    if config.cycle_skip.max_residual_s <= 0:
        raise ValueError("cycle_skip.max_residual_s must be positive.")
    if config.cycle_skip.trial_count < 1:
        raise ValueError("cycle_skip.trial_count must be at least 1.")
    if config.cycle_skip.seed_station_stride < 1:
        raise ValueError("cycle_skip.seed_station_stride must be at least 1.")
    if config.interpolation.method not in {"scipy_rbf", "gmt_surface"}:
        raise ValueError(
            "interpolation.method must be 'scipy_rbf' or 'gmt_surface'."
        )
    if config.interpolation.smoothing < 0:
        raise ValueError("interpolation.smoothing must be non-negative.")
    if config.interpolation.block_reduce not in {"median", "none"}:
        raise ValueError("interpolation.block_reduce must be 'median' or 'none'.")
    if not 0 <= config.interpolation.tension <= 1:
        raise ValueError("interpolation.tension must be between 0 and 1.")
    if (
        config.interpolation.stability_check.comparison_smoothing < 0
        or config.interpolation.stability_check.max_travel_time_difference_s <= 0
    ):
        raise ValueError("interpolation stability values must be non-negative/positive.")
    if not 0 <= config.interpolation.stability_check.comparison_tension <= 1:
        raise ValueError(
            "interpolation.stability_check.comparison_tension must be in [0, 1]."
        )
    if config.eikonal.gradient_scheme != "second_order":
        raise ValueError("eikonal.gradient_scheme currently supports second_order.")
    if config.eikonal.min_apparent_velocity_km_s <= 0:
        raise ValueError("minimum apparent velocity must be positive.")
    if (
        config.eikonal.max_apparent_velocity_km_s
        <= config.eikonal.min_apparent_velocity_km_s
    ):
        raise ValueError("maximum apparent velocity must exceed minimum.")
    if config.eikonal.boundary_cells_to_mask < 0:
        raise ValueError("eikonal.boundary_cells_to_mask must be non-negative.")
    if config.helmholtz.min_period_s < 0:
        raise ValueError("helmholtz.min_period_s must be non-negative.")
    if config.helmholtz.min_amplitude < 0:
        raise ValueError("helmholtz.min_amplitude must be non-negative.")
    if config.helmholtz.max_relative_interpolation_difference <= 0:
        raise ValueError(
            "helmholtz.max_relative_interpolation_difference must be positive."
        )
    if config.helmholtz.curvature_reference_velocity_km_s <= 0:
        raise ValueError(
            "helmholtz.curvature_reference_velocity_km_s must be positive."
        )
    if config.qc.station_coverage.mode not in {
        "nearby",
        "quadrants",
        "nearby_or_quadrants",
    }:
        raise ValueError("unsupported qc.station_coverage.mode.")
    if not 1 <= config.qc.station_coverage.min_quadrants <= 4:
        raise ValueError("qc.station_coverage.min_quadrants must be 1 through 4.")
    if not 0 <= config.qc.source_field.min_valid_fraction <= 1:
        raise ValueError("qc.source_field.min_valid_fraction must be in [0, 1].")
    if config.stacking.min_raw_measurements_per_cell < 1:
        raise ValueError("stacking minimum raw measurements must be positive.")
    if config.stacking.outlier_rejection.iterations < 0:
        raise ValueError("stacking outlier iterations must be non-negative.")
    if config.compute.workers == 0:
        raise ValueError("compute.workers cannot be zero.")
    if config.compute.source_chunk_size < 1:
        raise ValueError("compute.source_chunk_size must be positive.")
    if not 0 <= config.output.compression_level <= 9:
        raise ValueError("output.compression_level must be between 0 and 9.")


def _mapping(
    raw: dict[str, Any], key: str, *, required: bool = True
) -> dict[str, Any]:
    value = raw.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _nested_mapping(
    raw: dict[str, Any], key: str, *, required: bool = True
) -> dict[str, Any]:
    value = raw.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping.")
    return value


def _required(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"{key} is required.")
    return raw[key]


def _resolve_path(value: Any, base_dir: Path | None) -> Path:
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _integer_periods(values: Any) -> tuple[int, ...]:
    periods: list[int] = []
    for value in values:
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError("input.periods_s must contain integer periods.")
        periods.append(int(numeric))
    return tuple(periods)
