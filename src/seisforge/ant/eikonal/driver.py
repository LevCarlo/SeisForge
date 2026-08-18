"""Stage orchestration for Eikonal event fields and isotropic stacking."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import xarray as xr

from .config import load_eikonal_config
from .field import compute_event_field
from .grid import build_geographic_grid
from .io import (
    config_yaml,
    event_fields_to_dataset,
    load_measurement_dataset,
    write_netcdf,
)
from .models import EikonalRunConfig, EventField
from .stacking import stack_event_fields


def run_eikonal_config(
    config_path: str | Path,
    *,
    stage: str = "all",
) -> list[Path]:
    """Run event-field generation, stacking, or both from one YAML file."""
    config = load_eikonal_config(config_path)
    outputs: list[Path] = []
    if stage in {"all", "fields"}:
        outputs.extend(build_event_fields(config))
    if stage in {"all", "stack"}:
        outputs.append(stack_eikonal_fields(config))
    return outputs


def build_event_fields(config: EikonalRunConfig) -> list[Path]:
    """Compute and write one event-field Dataset per requested period."""
    measurements = load_measurement_dataset(
        config.input.dataset,
        requested_periods_s=config.input.periods_s,
        min_snr=config.input.min_snr,
        longitude_convention=config.grid.longitude_convention,
        require_amplitude=config.helmholtz.enabled,
    )
    grid = build_geographic_grid(config.grid)
    source_names = np.unique(measurements["source_station"].values.astype(str))
    output_paths: list[Path] = []
    _append_log(
        config,
        f"fields start: periods={len(config.input.periods_s)}, "
        f"sources={source_names.size}, paths={measurements.sizes['path']}",
    )
    for period_s in config.input.periods_s:
        fields = Parallel(
            n_jobs=config.compute.workers,
            batch_size=config.compute.source_chunk_size,
        )(
            delayed(_compute_source_period)(
                measurements, source, period_s, grid, config
            )
            for source in source_names
        )
        dataset = event_fields_to_dataset(fields, grid, config=config)
        output_path = _event_field_path(config, period_s)
        write_netcdf(
            dataset,
            output_path,
            overwrite=config.output.overwrite,
            compression_level=config.output.compression_level,
        )
        accepted = int(dataset["source_accepted"].values.sum())
        helmholtz_accepted = int(
            dataset["helmholtz_source_accepted"].values.sum()
        )
        _append_log(
            config,
            f"fields period={period_s}s: accepted_sources={accepted}/"
            f"{source_names.size}, helmholtz_accepted_sources="
            f"{helmholtz_accepted}/{source_names.size}, output={output_path}",
        )
        output_paths.append(output_path)
    return output_paths


def stack_eikonal_fields(config: EikonalRunConfig) -> Path:
    """Read period-scoped event fields and write one stacked Dataset."""
    period_datasets: list[xr.Dataset] = []
    for period_s in config.input.periods_s:
        path = _event_field_path(config, period_s)
        if not path.exists():
            raise FileNotFoundError(
                f"event fields for {period_s}s do not exist: {path}"
            )
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            event_fields = opened.load()
        if config.helmholtz.enabled:
            required_helmholtz = {
                "helmholtz_slowness_s_per_km",
                "helmholtz_rejection_reason",
                "helmholtz_source_accepted",
            }
            missing = sorted(required_helmholtz - set(event_fields.data_vars))
            if missing:
                raise ValueError(
                    f"event fields for {period_s}s predate Helmholtz output "
                    f"or are incomplete ({missing}); rerun the fields stage."
                )
        stacked = stack_event_fields(
            event_fields,
            config.stacking,
            include_helmholtz=config.helmholtz.enabled,
        )
        stacked = stacked.expand_dims(period=[period_s])
        period_datasets.append(stacked)
    output = xr.concat(period_datasets, dim="period")
    output = output.assign_coords(
        period_s=("period", np.asarray(config.input.periods_s, dtype=np.int32))
    )
    output.attrs.update(
        {
            "schema_version": 1,
            "stage": "eikonal_isotropic_stack",
            "project": config.project,
            "input_dataset": str(config.input.dataset),
            "helmholtz_enabled": int(config.helmholtz.enabled),
            "config_yaml": config_yaml(config),
        }
    )
    output_path = write_netcdf(
        output,
        config.output.stack_path,
        overwrite=config.output.overwrite,
        compression_level=config.output.compression_level,
    )
    _append_log(config, f"stack complete: output={output_path}")
    return output_path


def _compute_source_period(
    measurements: xr.Dataset,
    source_station: str,
    period_s: int,
    grid,
    config: EikonalRunConfig,
) -> EventField:
    source_by_path = measurements["source_station"].values.astype(str)
    path_select = source_by_path == source_station
    period_values = measurements["period_s"].values.astype(int)
    period_index = int(np.flatnonzero(period_values == period_s)[0])
    phase_velocity = np.asarray(
        measurements["phase_velocity_km_s"].values[path_select, period_index],
        dtype=float,
    )
    amplitude = None
    if "amplitude" in measurements:
        amplitude = np.asarray(
            measurements["amplitude"].values[path_select, period_index],
            dtype=float,
        )
    receiver_longitude = np.asarray(
        measurements["receiver_longitude_deg"].values[path_select], dtype=float
    )
    receiver_latitude = np.asarray(
        measurements["receiver_latitude_deg"].values[path_select], dtype=float
    )
    measurement_valid = np.isfinite(phase_velocity) & (phase_velocity > 0)
    if "measurement_valid" in measurements:
        measurement_valid &= np.asarray(
            measurements["measurement_valid"].values[path_select, period_index],
            dtype=bool,
        )
    if config.input.min_snr is not None:
        snr = np.asarray(
            measurements["snr"].values[path_select, period_index], dtype=float
        )
        measurement_valid &= np.isfinite(snr) & (snr >= config.input.min_snr)

    source_longitude_values = np.asarray(
        measurements["source_longitude_deg"].values[path_select], dtype=float
    )
    source_latitude_values = np.asarray(
        measurements["source_latitude_deg"].values[path_select], dtype=float
    )
    source_longitude = float(source_longitude_values[0])
    source_latitude = float(source_latitude_values[0])
    return compute_event_field(
        source_station=source_station,
        source_longitude_deg=source_longitude,
        source_latitude_deg=source_latitude,
        receiver_longitude_deg=receiver_longitude[measurement_valid],
        receiver_latitude_deg=receiver_latitude[measurement_valid],
        phase_velocity_km_s=phase_velocity[measurement_valid],
        amplitude=(
            amplitude[measurement_valid] if amplitude is not None else None
        ),
        period_s=period_s,
        grid=grid,
        cycle_skip=config.cycle_skip,
        interpolation=config.interpolation,
        method=config.eikonal,
        helmholtz=config.helmholtz,
        qc=config.qc,
    )


def _event_field_path(config: EikonalRunConfig, period_s: int) -> Path:
    return config.output.event_fields_path / f"{period_s}s.nc"


def _append_log(config: EikonalRunConfig, message: str) -> None:
    config.output.directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with (config.output.directory / "eikonal.log").open(
        "a", encoding="utf-8"
    ) as stream:
        stream.write(f"[{timestamp}] {message}\n")
