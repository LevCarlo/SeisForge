"""Orchestration for the AFTAN -> Eikonal -> inversion glue commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from seisforge.ant.aftan.phase import (
    physical_component_for_branch,
    physical_station_pair_for_branch,
    reciprocal_zrt_component,
)
from seisforge.ant.eikonal.io import validate_measurement_dataset

from .core import (
    resample_to_periods,
    sample_grid,
    station_sigma_km_s,
    velocity_std_from_slowness_std,
    wrap_longitude_deg,
)
from .io import (
    find_sac_file,
    measurement_dataset,
    parse_pair_name,
    read_aftan_dat,
    read_sac_pair_coordinates,
    read_stack,
    read_station_table,
    write_measurement_dataset,
    write_three_column_dat,
)
from .models import CollectDispersionConfig, StationCurveConfig


@dataclass(frozen=True)
class CollectResult:
    dataset: xr.Dataset
    used_files: tuple[Path, ...]
    skipped: tuple[tuple[Path, str], ...]


def collect_dispersion(
    files: list[Path] | tuple[Path, ...],
    config: CollectDispersionConfig,
) -> CollectResult:
    """Gather AFTAN ``.dat`` files into the Eikonal ``(path, period)`` dataset.

    Files of the other AFTAN stage are skipped. A file whose curve has no
    sample on the period grid, or whose period axis is not increasing, is
    skipped with a reason. Station or coordinate errors, mixed components or
    SNR scales, and duplicate directed paths raise.
    """
    stations = (
        read_station_table(config.station_table)
        if config.station_table is not None
        else None
    )
    target = np.asarray(config.periods_s, dtype=float)
    axis_column = (
        "instant_period_s" if config.period_axis == "instant" else "target_period_s"
    )
    paths: list[dict] = []
    used: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    components: set[str] = set()
    snr_scales: set[str] = set()
    for file in files:
        record = read_aftan_dat(file)
        meta = record["metadata"]
        if meta["stage"] != config.aftan_stage:
            continue
        columns = record["columns"]
        axis = columns[axis_column]
        try:
            velocity = resample_to_periods(axis, columns["phase_velocity_km_s"], target)
            snr = resample_to_periods(axis, record["snr"], target)
            amplitude = resample_to_periods(axis, columns["amplitude"], target)
        except ValueError as exc:
            skipped.append((Path(file), f"{axis_column}: {exc}"))
            continue
        if not np.any(np.isfinite(velocity)):
            skipped.append((Path(file), "no sample on the period grid"))
            continue

        sac_name = meta["input_file"]
        fields = parse_pair_name(Path(sac_name).stem, config.name_pattern)
        stored_component = config.component or fields["component"]
        branch = meta["selected_branch"]
        component = physical_component_for_branch(stored_component, branch)
        source, receiver = physical_station_pair_for_branch(
            fields["source_station"], fields["receiver_station"], branch
        )
        if stations is not None:
            coordinates = {}
            for name in (source, receiver):
                if name not in stations:
                    raise KeyError(f"station {name} ({file}) is not in the station table.")
                coordinates[name] = stations[name]
        else:
            sac_file = find_sac_file(config.sac_dir, sac_name)
            slon, slat, rlon, rlat = read_sac_pair_coordinates(sac_file)
            stored = {
                fields["source_station"]: (slon, slat),
                fields["receiver_station"]: (rlon, rlat),
            }
            coordinates = {source: stored[source], receiver: stored[receiver]}

        entry = {
            "phase_velocity_km_s": velocity,
            "snr": snr,
            "amplitude": amplitude,
            "aftan_file": str(file),
            "aftan_branch": branch,
        }
        directions = [(source, receiver, component)]
        if config.mirror_stack and branch == "stack":
            directions.append((receiver, source, reciprocal_zrt_component(component)))
        for src, rcv, comp in directions:
            components.add(comp)
            paths.append(
                {
                    **entry,
                    "source_station": src,
                    "receiver_station": rcv,
                    "source_longitude_deg": coordinates[src][0],
                    "source_latitude_deg": coordinates[src][1],
                    "receiver_longitude_deg": coordinates[rcv][0],
                    "receiver_latitude_deg": coordinates[rcv][1],
                }
            )
        snr_scales.add(record["snr_scale"])
        used.append(Path(file))

    if not paths:
        raise ValueError(
            f"no usable AFTAN '{config.aftan_stage}' measurements among {len(files)} files."
        )
    if len(components) != 1:
        raise ValueError(f"collected paths mix physical components {sorted(components)}.")
    if len(snr_scales) != 1:
        raise ValueError("collected files mix linear and dB SNR columns.")
    _raise_on_duplicate_paths(paths)
    for item in paths:
        for key in ("source_longitude_deg", "receiver_longitude_deg"):
            item[key] = float(wrap_longitude_deg(item[key], config.longitude_convention))

    attrs = {
        "schema_version": 1,
        "stage": "aftan_collected_dispersion",
        "component": components.pop(),
        "longitude_convention": config.longitude_convention,
        "snr_scale": snr_scales.pop(),
        "aftan_stage": config.aftan_stage,
        "period_axis": axis_column,
        "resampling": "linear in period between measured samples; no extrapolation",
        "coordinate_source": (
            f"station_table:{config.station_table}"
            if config.station_table is not None
            else f"sac_headers:{config.sac_dir} (evlo/evla source, stlo/stla receiver)"
        ),
        "name_pattern": config.name_pattern,
        "component_override": config.component or "",
        "mirror_stack": int(config.mirror_stack),
        "input_file_count": len(used),
        "skipped_file_count": len(skipped),
    }
    dataset = measurement_dataset(
        periods_s=config.periods_s,
        paths=paths,
        attrs=attrs,
        include_snr=True,
        include_amplitude=config.include_amplitude,
    )
    validate_measurement_dataset(
        dataset,
        requested_periods_s=config.periods_s,
        min_snr=None,
        longitude_convention=config.longitude_convention,
        require_amplitude=False,
    )
    return CollectResult(dataset, tuple(used), tuple(skipped))


def run_collect_dispersion(
    files: list[Path] | tuple[Path, ...],
    config: CollectDispersionConfig,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> CollectResult:
    """Collect AFTAN files and write the Eikonal input NetCDF."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    result = collect_dispersion(files, config)
    write_measurement_dataset(result.dataset, output, overwrite=overwrite)
    return result


def _raise_on_duplicate_paths(paths: list[dict]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for item in paths:
        key = (item["source_station"], item["receiver_station"])
        if key in seen:
            raise ValueError(
                f"directed path {key[0]} -> {key[1]} appears twice: "
                f"{seen[key]} and {item['aftan_file']}. Collect one branch per "
                "direction (e.g. remove a stack file or a duplicate pair)."
            )
        seen[key] = item["aftan_file"]


@dataclass(frozen=True)
class StationCurve:
    period_s: np.ndarray
    velocity_km_s: np.ndarray
    sigma_km_s: np.ndarray
    floored: np.ndarray
    dropped_period_s: tuple[int, ...]
    header: dict[str, object]


SIGMA_DESCRIPTIONS = {
    "sem": (
        "standard error of the azimuth-weighted mean single-source phase "
        "velocity: weighted sample std / sqrt(n_eff), n_eff = 1/sum(w^2) "
        "(stack variable {prefix}velocity_sem_km_s); assumes independent sources"
    ),
    "std": (
        "spread of single-source estimates: weighted sample std of slowness "
        "converted to velocity by std_s / s^2 (stack variable "
        "{prefix}slowness_std_s_per_km); does not shrink with source count"
    ),
}


def extract_station_curve(
    stack: xr.Dataset,
    longitude_deg: float,
    latitude_deg: float,
    config: StationCurveConfig,
    *,
    location_label: str = "",
) -> StationCurve:
    """Sample an Eikonal stack at one point as ``period, velocity, sigma``.

    Periods where the point (any needed node) is masked, or where the chosen
    uncertainty is undefined (fewer than two effective sources), are dropped
    and listed in the header.
    """
    prefix = "helmholtz_" if config.field == "helmholtz" else ""
    needed = [f"{prefix}phase_velocity_km_s", f"{prefix}slowness_s_per_km", f"{prefix}mask"]
    needed.append(
        f"{prefix}velocity_sem_km_s"
        if config.sigma_source == "sem"
        else f"{prefix}slowness_std_s_per_km"
    )
    missing = [name for name in needed if name not in stack]
    if missing:
        raise ValueError(f"stack is missing {missing} (field={config.field}).")
    grid_longitude = np.asarray(stack["longitude"].values, dtype=float)
    query_longitude = float(longitude_deg)
    if grid_longitude.max() > 180.0:
        query_longitude = float(wrap_longitude_deg(query_longitude, "0_360"))
    elif grid_longitude.min() < 0.0:
        query_longitude = float(wrap_longitude_deg(query_longitude, "-180_180"))

    def sample(name: str) -> np.ndarray:
        return sample_grid(
            grid_longitude,
            stack["latitude"].values,
            stack[name].transpose("period", "latitude", "longitude").values,
            valid,
            query_longitude,
            latitude_deg,
            method=config.interpolation,
        )

    valid = ~np.asarray(
        stack[f"{prefix}mask"].transpose("period", "latitude", "longitude").values,
        dtype=bool,
    )
    velocity = sample(f"{prefix}phase_velocity_km_s")
    if config.sigma_source == "sem":
        sigma = sample(f"{prefix}velocity_sem_km_s")
    else:
        sigma = velocity_std_from_slowness_std(
            sample(f"{prefix}slowness_s_per_km"),
            sample(f"{prefix}slowness_std_s_per_km"),
        )
    sigma, floored = station_sigma_km_s(sigma, config.sigma_floor_km_s)
    period_s = np.asarray(stack["period_s"].values, dtype=int)
    keep = np.isfinite(velocity) & np.isfinite(sigma) & (sigma > 0)
    dropped = tuple(int(value) for value in period_s[~keep])
    if not np.any(keep):
        raise ValueError(
            f"no valid period at ({longitude_deg}, {latitude_deg}) in the stack."
        )
    header = {
        "stage": "eikonal_station_curve",
        "stack_file": stack.encoding.get("source", ""),
        "location": location_label,
        "longitude_deg": float(longitude_deg),
        "latitude_deg": float(latitude_deg),
        "field": f"{prefix}phase_velocity_km_s",
        "interpolation": config.interpolation,
        "sigma_source": config.sigma_source,
        "sigma_definition": SIGMA_DESCRIPTIONS[config.sigma_source].format(prefix=prefix),
        "sigma_floor_km_s": f"{config.sigma_floor_km_s} (assumed value; sigma = max(source, floor))",
        "sigma_floored_periods_s": ",".join(str(p) for p in period_s[keep & floored]) or "none",
        "dropped_periods_s": ",".join(str(p) for p in dropped) or "none",
    }
    return StationCurve(
        period_s[keep],
        velocity[keep],
        sigma[keep],
        floored[keep],
        dropped,
        header,
    )


def run_extract_station_curve(
    stack_file: str | Path,
    output: str | Path,
    config: StationCurveConfig,
    *,
    station: str | None = None,
    station_table: str | Path | None = None,
    longitude_deg: float | None = None,
    latitude_deg: float | None = None,
    overwrite: bool = False,
) -> StationCurve:
    """Extract one station's curve and write the inversion's three-column file."""
    if station is not None:
        if station_table is None or longitude_deg is not None or latitude_deg is not None:
            raise ValueError("give station with station_table, or longitude/latitude.")
        stations = read_station_table(station_table)
        if station not in stations:
            raise KeyError(f"station {station} is not in {station_table}.")
        longitude_deg, latitude_deg = stations[station]
        label = f"station {station} from {station_table}"
    elif longitude_deg is None or latitude_deg is None:
        raise ValueError("give station with station_table, or longitude/latitude.")
    else:
        label = "point"
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    stack = read_stack(stack_file)
    stack.encoding["source"] = str(stack_file)
    curve = extract_station_curve(
        stack, longitude_deg, latitude_deg, config, location_label=label
    )
    write_three_column_dat(
        output,
        curve.period_s,
        curve.velocity_km_s,
        curve.sigma_km_s,
        header=curve.header,
        column_names="period_s phase_velocity_km_s sigma_km_s",
        overwrite=overwrite,
    )
    return curve
