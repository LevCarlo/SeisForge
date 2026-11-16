"""Preview or apply a manual global 2 pi cycle shift to saved AFTAN outputs.

A stand-alone tool, not part of the SeisForge package or CLI. Run from
anywhere with SeisForge installed:

    python tools/aftan_cycle_set.py PAIR.vph.nc --shift N            # preview only
    python tools/aftan_cycle_set.py PAIR.vph.nc --shift N --in-place --reason "..."

Without ``--in-place`` no file is changed. With it, the phase map (written by
``aftan.energy_map.phase_velocity``) and its companion ``.dat`` are rewritten:
the new branch is always rebuilt from ``picked_phase_velocity_uncorrected``
(so repeated overrides do not accumulate), the automatic curve and
diagnostics are kept as ``*_automatic`` variables, and the shift, UTC time and
reason are recorded. The formulas are the package's
`seisforge.ant.aftan.core.apply_phase_cycle_shift` and
`phase_cycle_reference_misfit`; the ``.dat`` column formats are those of
`seisforge.ant.aftan.io`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from seisforge.ant.aftan.core import apply_phase_cycle_shift, phase_cycle_reference_misfit
from seisforge.ant.aftan.io import (
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
    reference_misfit: float
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
        reference_misfit=summary["reference_misfit"],
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
    reference_misfit = phase_cycle_reference_misfit(
        period, uncorrected, reference, distance_km, cycle_shift
    )

    updated["picked_phase_velocity"] = (
        ("target_period",),
        corrected,
        updated["picked_phase_velocity"].attrs,
    )
    updated["phase_cycle_shift"] = np.asarray(cycle_shift, dtype=np.int32)
    updated["phase_cycle_reference_misfit"] = np.asarray(
        reference_misfit, dtype=float
    )
    updated["phase_cycle_manual_override"] = np.asarray(1, dtype=np.int8)
    if "phase_cycle_reference_misfit_gap" in updated:
        updated["phase_cycle_reference_misfit_gap"] = np.asarray(
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
        "reference_misfit": reference_misfit,
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
        "phase_cycle_reference_misfit",
        "phase_cycle_reference_misfit_gap",
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
        "# phase_cycle_shift:",
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
        f"# phase_cycle_shift: {int(cycle_shift)}",
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute phase velocity from the uncorrected curve stored in one "
            "AFTAN .vph.nc file. By default this is a read-only preview; pass "
            "--in-place to update the phase map and companion .dat file."
        )
    )
    parser.add_argument("phase_map", help="AFTAN phase map ending in .vph.nc (basic or PMF).")
    parser.add_argument(
        "--shift", type=int, required=True,
        help="Absolute global integer cycle shift relative to the uncorrected curve.",
    )
    parser.add_argument("--dat-file", default=None, help="Companion .dat file; inferred from the phase-map name.")
    parser.add_argument(
        "--phase-map-only", action="store_true",
        help="Update only the phase-map NetCDF and do not require a .dat file.",
    )
    parser.add_argument("--reason", default=None, help="Short reason stored with an in-place override.")
    parser.add_argument(
        "--in-place", action="store_true",
        help="Apply the override; without this flag nothing is written.",
    )
    args = parser.parse_args(argv)
    result = override_phase_cycle(
        args.phase_map,
        args.shift,
        dat_file=args.dat_file,
        phase_map_only=args.phase_map_only,
        write=args.in_place,
        reason=args.reason,
    )
    action = "updated" if result.written else "preview"
    print(
        f"{action}: shift {result.previous_shift:+d} -> {result.requested_shift:+d}; "
        f"reference misfit={result.reference_misfit:.4f}; "
        f"vph={result.phase_velocity_min_km_s:.4f}-{result.phase_velocity_max_km_s:.4f} km/s"
    )
    print(f"phase_map: {result.phase_map}")
    if result.dat_file is not None:
        print(f"dat_file: {result.dat_file}")
    if not result.written:
        print("No files were changed. Add --in-place to apply this override.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
