"""Xarray/NetCDF input validation and output encoding."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from pyproj import Geod
import xarray as xr
import yaml

from .models import (
    EikonalRunConfig,
    EventField,
    GeographicGrid,
    HELMHOLTZ_REJECTION_FLAG_MEANINGS,
    REJECTION_FLAG_MEANINGS,
    HelmholtzRejectionReason,
    RejectionReason,
)


REQUIRED_PATH_COORDINATES = (
    "source_station",
    "receiver_station",
    "source_longitude_deg",
    "source_latitude_deg",
    "receiver_longitude_deg",
    "receiver_latitude_deg",
)
WGS84 = Geod(ellps="WGS84")


def load_measurement_dataset(
    path: str | Path,
    *,
    requested_periods_s: tuple[int, ...],
    min_snr: float | None,
    longitude_convention: str | None = None,
    require_amplitude: bool = False,
) -> xr.Dataset:
    """Load the canonical directed-path measurement dataset into memory."""
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        dataset = opened.load()
    validate_measurement_dataset(
        dataset,
        requested_periods_s=requested_periods_s,
        min_snr=min_snr,
        longitude_convention=longitude_convention,
        require_amplitude=require_amplitude,
    )
    return dataset


def validate_measurement_dataset(
    dataset: xr.Dataset,
    *,
    requested_periods_s: tuple[int, ...],
    min_snr: float | None,
    longitude_convention: str | None = None,
    require_amplitude: bool = False,
) -> None:
    """Validate the public Eikonal input data contract."""
    if "path" not in dataset.dims or "period" not in dataset.dims:
        raise ValueError("measurement dataset must have path and period dimensions.")
    if int(dataset.attrs.get("schema_version", -1)) != 1:
        raise ValueError("measurement dataset schema_version must be 1.")
    if not str(dataset.attrs.get("component", "")).strip():
        raise ValueError("measurement dataset must define a component attribute.")
    dataset_longitude_convention = dataset.attrs.get("longitude_convention")
    if dataset_longitude_convention not in {"-180_180", "0_360"}:
        raise ValueError(
            "measurement dataset longitude_convention must be '-180_180' "
            "or '0_360'."
        )
    if (
        longitude_convention is not None
        and dataset_longitude_convention != longitude_convention
    ):
        raise ValueError(
            "measurement and grid longitude conventions must match."
        )
    if "period_s" not in dataset:
        raise ValueError("measurement dataset is missing period_s.")
    if dataset["period_s"].dims != ("period",):
        raise ValueError("period_s must have dimension ('period',).")
    periods = np.asarray(dataset["period_s"].values)
    if periods.dtype.kind not in "iu" or not np.all(np.isfinite(periods)):
        raise ValueError("period_s must contain finite integer periods.")
    integer_periods = periods.astype(int)
    if np.unique(integer_periods).size != integer_periods.size:
        raise ValueError("period_s values must be unique.")
    if np.any(np.diff(integer_periods) <= 0):
        raise ValueError("period_s values must be strictly increasing.")
    missing = sorted(set(requested_periods_s) - set(integer_periods.tolist()))
    if missing:
        raise ValueError(f"requested periods are missing from input: {missing}")
    for name in REQUIRED_PATH_COORDINATES:
        if name not in dataset:
            raise ValueError(f"measurement dataset is missing {name}.")
        if dataset[name].dims != ("path",):
            raise ValueError(f"{name} must have dimension ('path',).")
    if "phase_velocity_km_s" not in dataset:
        raise ValueError("measurement dataset is missing phase_velocity_km_s.")
    if dataset["phase_velocity_km_s"].dims != ("path", "period"):
        raise ValueError(
            "phase_velocity_km_s must have dimensions ('path', 'period')."
        )
    velocity = np.asarray(dataset["phase_velocity_km_s"].values, dtype=float)
    invalid_velocity = np.isfinite(velocity) & (velocity <= 0)
    if np.any(invalid_velocity):
        raise ValueError("finite phase_velocity_km_s values must be positive.")
    _validate_geographic_coordinates(dataset)
    _validate_directed_paths(dataset)
    _validate_station_coordinates(dataset)
    if min_snr is not None:
        if "snr" not in dataset:
            raise ValueError("input.min_snr requires an snr(path, period) variable.")
        if dataset["snr"].dims != ("path", "period"):
            raise ValueError("snr must have dimensions ('path', 'period').")
        if "snr_scale" not in dataset.attrs:
            raise ValueError("input.min_snr requires a dataset snr_scale attribute.")
        if dataset.attrs["snr_scale"] not in {"linear", "db"}:
            raise ValueError("snr_scale must be 'linear' or 'db'.")
    if "distance_km" in dataset:
        _validate_distances(dataset)
    if "measurement_valid" in dataset and dataset["measurement_valid"].dims != (
        "path",
        "period",
    ):
        raise ValueError(
            "measurement_valid must have dimensions ('path', 'period')."
        )
    if require_amplitude and "amplitude" not in dataset:
        raise ValueError("helmholtz.enabled requires amplitude(path, period).")
    if "amplitude" in dataset:
        if dataset["amplitude"].dims != ("path", "period"):
            raise ValueError("amplitude must have dimensions ('path', 'period').")
        amplitude = np.asarray(dataset["amplitude"].values, dtype=float)
        if np.any(np.isfinite(amplitude) & (amplitude <= 0)):
            raise ValueError("finite amplitude values must be positive.")


def event_fields_to_dataset(
    fields: list[EventField],
    grid: GeographicGrid,
    *,
    config: EikonalRunConfig,
) -> xr.Dataset:
    """Convert single-source result objects to one period-scoped Dataset."""
    if not fields:
        raise ValueError("cannot create an event-field dataset without sources.")
    period_s = fields[0].period_s
    if any(field.period_s != period_s for field in fields):
        raise ValueError("all event fields in one file must share a period.")
    source = np.asarray([field.source_station for field in fields], dtype=str)
    dimensions = ("source", "latitude", "longitude")

    def stack(name: str) -> np.ndarray:
        return np.stack([getattr(field, name) for field in fields], axis=0)

    dataset = xr.Dataset(
        data_vars={
            "travel_time_s": (dimensions, stack("travel_time_s")),
            "travel_time_laplacian_s_per_km2": (
                dimensions,
                stack("travel_time_laplacian_s_per_km2"),
            ),
            "slowness_s_per_km": (dimensions, stack("slowness_s_per_km")),
            "apparent_velocity_km_s": (
                dimensions,
                stack("apparent_velocity_km_s"),
            ),
            "propagation_angle_deg": (
                dimensions,
                stack("propagation_angle_deg"),
            ),
            "great_circle_azimuth_deg": (
                dimensions,
                stack("great_circle_azimuth_deg"),
            ),
            "azimuth_deflection_deg": (
                dimensions,
                stack("azimuth_deflection_deg"),
            ),
            "rejection_reason": (
                dimensions,
                stack("rejection_reason").astype(np.uint8),
            ),
            "amplitude": (dimensions, stack("amplitude")),
            "amplitude_laplacian_per_km2": (
                dimensions,
                stack("amplitude_laplacian_per_km2"),
            ),
            "amplitude_laplacian_correction_s2_per_km2": (
                dimensions,
                stack("amplitude_laplacian_correction_s2_per_km2"),
            ),
            "helmholtz_slowness_s_per_km": (
                dimensions,
                stack("helmholtz_slowness_s_per_km"),
            ),
            "helmholtz_phase_velocity_km_s": (
                dimensions,
                stack("helmholtz_phase_velocity_km_s"),
            ),
            "helmholtz_rejection_reason": (
                dimensions,
                stack("helmholtz_rejection_reason").astype(np.uint8),
            ),
            "source_longitude_deg": (
                "source",
                [field.source_longitude_deg for field in fields],
            ),
            "source_latitude_deg": (
                "source",
                [field.source_latitude_deg for field in fields],
            ),
            "input_measurement_count": (
                "source",
                [field.input_measurement_count for field in fields],
            ),
            "used_measurement_count": (
                "source",
                [field.used_measurement_count for field in fields],
            ),
            "amplitude_measurement_count": (
                "source",
                [field.amplitude_measurement_count for field in fields],
            ),
            "cycle_corrected_count": (
                "source",
                [field.cycle_corrected_count for field in fields],
            ),
            "cycle_rejected_count": (
                "source",
                [field.cycle_rejected_count for field in fields],
            ),
            "source_accepted": (
                "source",
                [field.source_accepted for field in fields],
            ),
            "helmholtz_applied": (
                "source",
                [field.helmholtz_applied for field in fields],
            ),
            "helmholtz_source_accepted": (
                "source",
                [field.helmholtz_source_accepted for field in fields],
            ),
        },
        coords={
            "source": source,
            "latitude": grid.latitude_deg,
            "longitude": grid.longitude_deg,
        },
        attrs={
            "schema_version": 1,
            "stage": "eikonal_event_fields",
            "project": config.project,
            "period_s": int(period_s),
            "input_dataset": str(config.input.dataset),
            "helmholtz_enabled": int(config.helmholtz.enabled),
            "helmholtz_formula": (
                "s_H^2=|grad(travel_time)|^2-laplacian(amplitude)/"
                "(amplitude*omega^2)"
            ),
            "config_yaml": config_yaml(config),
        },
    )
    _set_event_field_metadata(dataset)
    return dataset


def write_netcdf(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    overwrite: bool,
    compression_level: int,
) -> Path:
    """Atomically write an HDF5-backed NetCDF file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists and overwrite is false: {output}")
    encoding: dict[str, dict[str, Any]] = {}
    for name, variable in dataset.data_vars.items():
        if variable.dtype.kind in "iufb" and variable.ndim > 0:
            encoding[name] = {
                "compression": "gzip",
                "compression_opts": compression_level,
                "shuffle": True,
            }
        if variable.dtype.kind == "b":
            encoding.setdefault(name, {})["dtype"] = "int8"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataset.to_netcdf(temporary, engine="h5netcdf", encoding=encoding)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def config_yaml(config: EikonalRunConfig) -> str:
    """Serialize the resolved typed configuration for output provenance."""
    return yaml.safe_dump(_serializable(config), sort_keys=False)


def _set_event_field_metadata(dataset: xr.Dataset) -> None:
    units = {
        "travel_time_s": "s",
        "travel_time_laplacian_s_per_km2": "s km-2",
        "slowness_s_per_km": "s km-1",
        "apparent_velocity_km_s": "km s-1",
        "propagation_angle_deg": "degree",
        "great_circle_azimuth_deg": "degree",
        "azimuth_deflection_deg": "degree",
        "amplitude_laplacian_per_km2": "amplitude km-2",
        "amplitude_laplacian_correction_s2_per_km2": "s2 km-2",
        "helmholtz_slowness_s_per_km": "s km-1",
        "helmholtz_phase_velocity_km_s": "km s-1",
        "source_longitude_deg": "degree_east",
        "source_latitude_deg": "degree_north",
    }
    for name, unit in units.items():
        dataset[name].attrs["units"] = unit
    dataset["rejection_reason"].attrs.update(
        {
            "flag_values": np.asarray(
                [int(reason) for reason in RejectionReason], dtype=np.uint8
            ),
            "flag_meanings": REJECTION_FLAG_MEANINGS,
        }
    )
    dataset["helmholtz_rejection_reason"].attrs.update(
        {
            "flag_values": np.asarray(
                [int(reason) for reason in HelmholtzRejectionReason],
                dtype=np.uint8,
            ),
            "flag_meanings": HELMHOLTZ_REJECTION_FLAG_MEANINGS,
        }
    )


def _validate_geographic_coordinates(dataset: xr.Dataset) -> None:
    coordinates = {
        "source_longitude_deg": (-180.0, 360.0),
        "receiver_longitude_deg": (-180.0, 360.0),
        "source_latitude_deg": (-90.0, 90.0),
        "receiver_latitude_deg": (-90.0, 90.0),
    }
    for name, (minimum, maximum) in coordinates.items():
        values = np.asarray(dataset[name].values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if np.any((values < minimum) | (values > maximum)):
            raise ValueError(f"{name} contains values outside geographic bounds.")


def _validate_directed_paths(dataset: xr.Dataset) -> None:
    source = np.asarray(dataset["source_station"].values).astype(str)
    receiver = np.asarray(dataset["receiver_station"].values).astype(str)
    if np.any(source == receiver):
        raise ValueError("source_station and receiver_station must differ per path.")
    pairs = np.asarray([f"{left}\0{right}" for left, right in zip(source, receiver)])
    if np.unique(pairs).size != pairs.size:
        raise ValueError("duplicate directed source/receiver paths are not allowed.")


def _validate_station_coordinates(dataset: xr.Dataset) -> None:
    station = np.concatenate(
        (
            np.asarray(dataset["source_station"].values).astype(str),
            np.asarray(dataset["receiver_station"].values).astype(str),
        )
    )
    longitude = np.concatenate(
        (
            np.asarray(dataset["source_longitude_deg"].values, dtype=float),
            np.asarray(dataset["receiver_longitude_deg"].values, dtype=float),
        )
    )
    latitude = np.concatenate(
        (
            np.asarray(dataset["source_latitude_deg"].values, dtype=float),
            np.asarray(dataset["receiver_latitude_deg"].values, dtype=float),
        )
    )
    for identifier in np.unique(station):
        select = station == identifier
        if np.ptp(longitude[select]) > 1e-6 or np.ptp(latitude[select]) > 1e-6:
            raise ValueError(f"inconsistent coordinates for station {identifier}.")


def _validate_distances(dataset: xr.Dataset) -> None:
    if dataset["distance_km"].dims != ("path",):
        raise ValueError("distance_km must have dimension ('path',).")
    _, _, distance_m = WGS84.inv(
        dataset["source_longitude_deg"].values,
        dataset["source_latitude_deg"].values,
        dataset["receiver_longitude_deg"].values,
        dataset["receiver_latitude_deg"].values,
    )
    computed = distance_m / 1000.0
    stored = np.asarray(dataset["distance_km"].values, dtype=float)
    if not np.allclose(stored, computed, rtol=0.0, atol=0.5):
        raise ValueError("distance_km disagrees with WGS84 coordinates by >0.5 km.")


def _serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
