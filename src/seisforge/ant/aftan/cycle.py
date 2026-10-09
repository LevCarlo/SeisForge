"""Post-pick phase-cycle inspection and explicit manual overrides."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from .core import apply_phase_cycle_shift
from .io import (
    _AFTAN_DAT_DEBUG_FORMATS,
    _AFTAN_DAT_DELIMITER,
    _AFTAN_DAT_MAIN_FORMATS,
)


@dataclass(frozen=True)
class PhaseCycleOverrideResult:
    """Summary of one requested phase-cycle override."""

    phase_map: Path
    dat_file: Path | None
    previous_shift: int
    requested_shift: int
    reference_score_cycles: float
    phase_velocity_min_km_s: float
    phase_velocity_max_km_s: float
    written: bool


def infer_dispersion_dat_path(phase_map: str | Path) -> Path:
    """Infer the companion ``.dat`` path from a ``.vph.nc`` path."""
    phase_map = Path(phase_map)
    suffix = ".vph.nc"
    if not phase_map.name.endswith(suffix):
        raise ValueError("phase map filename must end with '.vph.nc'.")
    return phase_map.with_name(phase_map.name[: -len(suffix)] + ".dat")


def override_phase_cycle(
    phase_map: str | Path,
    cycle_shift: int,
    *,
    dat_file: str | Path | None = None,
    phase_map_only: bool = False,
    write: bool = False,
    reason: str | None = None,
) -> PhaseCycleOverrideResult:
    """Preview or apply a manual global cycle shift to saved AFTAN outputs.

    The uncorrected phase velocity stored in the phase-map NetCDF is always the
    source of the new branch. This makes repeated overrides deterministic and
    prevents shifts from being accumulated accidentally.
    """
    phase_map_path = Path(phase_map)
    if not phase_map_path.exists():
        raise FileNotFoundError(phase_map_path)
    dat_path = None
    if not phase_map_only:
        dat_path = (
            Path(dat_file)
            if dat_file is not None
            else infer_dispersion_dat_path(phase_map_path)
        )
        if not dat_path.exists():
            raise FileNotFoundError(
                f"Companion dispersion file not found: {dat_path}. "
                "Use --phase-map-only to update only the NetCDF file."
            )

    with xr.open_dataset(phase_map_path) as opened:
        dataset = opened.load()
    updated, summary = _manual_cycle_dataset(dataset, int(cycle_shift), reason)

    if dat_path is not None:
        row_count = _dispersion_dat_row_count(dat_path)
        if row_count != updated.sizes["target_period"]:
            raise ValueError(
                f"{dat_path} has {row_count} data rows but {phase_map_path} has "
                f"{updated.sizes['target_period']} target periods."
            )

    if write:
        _atomic_write_netcdf(updated, phase_map_path)
        if dat_path is not None:
            _rewrite_dispersion_dat(
                dat_path,
                updated["picked_phase_velocity"].values,
                cycle_shift=int(cycle_shift),
                reason=reason,
            )

    return PhaseCycleOverrideResult(
        phase_map=phase_map_path,
        dat_file=dat_path,
        previous_shift=summary["previous_shift"],
        requested_shift=int(cycle_shift),
        reference_score_cycles=summary["reference_score_cycles"],
        phase_velocity_min_km_s=summary["phase_velocity_min_km_s"],
        phase_velocity_max_km_s=summary["phase_velocity_max_km_s"],
        written=bool(write),
    )


def _manual_cycle_dataset(
    dataset: xr.Dataset,
    cycle_shift: int,
    reason: str | None,
) -> tuple[xr.Dataset, dict[str, float | int]]:
    required = {
        "picked_phase_velocity",
        "picked_instant_period",
        "picked_phase_velocity_uncorrected",
        "reference_phase_velocity",
        "phase_cycle_shift",
    }
    missing = sorted(required.difference(dataset.variables))
    if missing:
        raise ValueError(f"phase map is missing required variables: {missing}")
    if "distance_km" not in dataset.attrs:
        raise ValueError("phase map is missing the distance_km attribute.")

    updated = dataset.copy(deep=True)
    period = np.asarray(updated["picked_instant_period"].values, dtype=float)
    uncorrected = np.asarray(
        updated["picked_phase_velocity_uncorrected"].values,
        dtype=float,
    )
    reference = np.asarray(updated["reference_phase_velocity"].values, dtype=float)
    distance_km = float(updated.attrs["distance_km"])
    corrected = apply_phase_cycle_shift(
        period,
        uncorrected,
        distance_km,
        cycle_shift,
    )

    previous_shift = int(updated["phase_cycle_shift"].item())
    _preserve_automatic_diagnostics(updated)
    valid = (
        np.isfinite(period)
        & np.isfinite(corrected)
        & np.isfinite(reference)
        & (period > 0)
        & (corrected > 0)
        & (reference > 0)
    )
    reference_score = (
        float(
            np.median(
                np.abs(
                    (
                        distance_km / corrected[valid]
                        - distance_km / reference[valid]
                    )
                    / period[valid]
                )
            )
        )
        if np.any(valid)
        else np.nan
    )

    updated["picked_phase_velocity"] = (
        ("target_period",),
        corrected,
        updated["picked_phase_velocity"].attrs,
    )
    updated["phase_cycle_shift"] = np.asarray(cycle_shift, dtype=np.int32)
    updated["phase_cycle_reference_score_cycles"] = np.asarray(
        reference_score, dtype=float
    )
    updated["phase_cycle_manual_override"] = np.asarray(1, dtype=np.int8)
    if "phase_cycle_reference_score_gap_cycles" in updated:
        updated["phase_cycle_reference_score_gap_cycles"] = np.asarray(
            np.nan, dtype=float
        )
    if "phase_cycle_search_limit_reached" in updated:
        updated["phase_cycle_search_limit_reached"] = np.asarray(0, dtype=np.int8)

    updated.attrs["phase_cycle_selection_source"] = "manual_override"
    updated.attrs["phase_cycle_manual_previous_shift"] = previous_shift
    updated.attrs["phase_cycle_manual_shift"] = int(cycle_shift)
    updated.attrs["phase_cycle_manual_override_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    if reason:
        updated.attrs["phase_cycle_manual_reason"] = str(reason)

    finite = corrected[np.isfinite(corrected)]
    return updated, {
        "previous_shift": previous_shift,
        "reference_score_cycles": reference_score,
        "phase_velocity_min_km_s": (
            float(np.min(finite)) if finite.size else np.nan
        ),
        "phase_velocity_max_km_s": (
            float(np.max(finite)) if finite.size else np.nan
        ),
    }


def _preserve_automatic_diagnostics(dataset: xr.Dataset) -> None:
    names = (
        "picked_phase_velocity",
        "phase_cycle_shift",
        "phase_cycle_reference_score_cycles",
        "phase_cycle_reference_score_gap_cycles",
        "phase_cycle_search_limit_reached",
    )
    for name in names:
        automatic_name = f"{name}_automatic"
        if name in dataset and automatic_name not in dataset:
            dataset[automatic_name] = dataset[name].copy(deep=True)


def _dispersion_dat_row_count(path: Path) -> int:
    data = np.loadtxt(path, comments="#", ndmin=2)
    return int(data.shape[0])


def _rewrite_dispersion_dat(
    path: Path,
    phase_velocity: np.ndarray,
    *,
    cycle_shift: int,
    reason: str | None,
) -> None:
    lines = path.read_text().splitlines()
    header_lines = [line for line in lines if line.lstrip().startswith("#")]
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] != phase_velocity.size:
        raise ValueError("dispersion row count does not match phase-map periods.")
    if data.shape[1] not in (len(_AFTAN_DAT_MAIN_FORMATS), len(_AFTAN_DAT_DEBUG_FORMATS)):
        raise ValueError(f"unsupported AFTAN .dat column count: {data.shape[1]}")
    data[:, 3] = phase_velocity
    formats = (
        _AFTAN_DAT_MAIN_FORMATS
        if data.shape[1] == len(_AFTAN_DAT_MAIN_FORMATS)
        else _AFTAN_DAT_DEBUG_FORMATS
    )
    metadata_prefixes = (
        "# phase_cycle_selection_source:",
        "# phase_cycle_manual_shift:",
        "# phase_cycle_manual_override_utc:",
        "# phase_cycle_manual_reason:",
    )
    header_lines = [
        line
        for line in header_lines
        if not line.startswith(metadata_prefixes)
    ]
    new_metadata = [
        "# phase_cycle_selection_source: manual_override",
        f"# phase_cycle_manual_shift: {int(cycle_shift)}",
        "# phase_cycle_manual_override_utc: "
        + datetime.now(timezone.utc).isoformat(),
    ]
    if reason:
        new_metadata.append(f"# phase_cycle_manual_reason: {reason}")

    temporary = path.with_name(f".{path.name}.cycle.tmp")
    with temporary.open("w") as stream:
        for line in new_metadata + header_lines:
            stream.write(line + "\n")
        np.savetxt(
            stream,
            data,
            fmt=formats,
            delimiter=_AFTAN_DAT_DELIMITER,
        )
    temporary.replace(path)


def _atomic_write_netcdf(dataset: xr.Dataset, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.cycle.tmp")
    try:
        dataset.to_netcdf(temporary, engine="scipy")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
