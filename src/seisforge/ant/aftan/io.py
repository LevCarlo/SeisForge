"""SAC, path, YAML, and tabular output helpers for AFTAN."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
from obspy import Trace
import yaml

from .models import AFTANSNRConfig


_AFTAN_DAT_DELIMITER = "    "
_AFTAN_DAT_MAIN_FORMATS = (
    "%.6f",  # target_period_s
    "%.6f",  # instant_period_s
    "%.4f",  # group_velocity_km_s
    "%.4f",  # phase_velocity_km_s
    "%.6e",  # amplitude
    "%.3f",  # snr
)
_AFTAN_DAT_DEBUG_FORMATS = (
    "%.6f",  # target_period_s
    "%.6f",  # instant_period_s
    "%.4f",  # group_velocity_km_s
    "%.4f",  # phase_velocity_km_s
    "%.6e",  # amplitude
    "%.6f",  # phase_derivative_rad_s
    "%.6f",  # hilbert_phase_derivative_rad_s
    "%.4f",  # hilbert_instant_period_s
    "%.6f",  # instant_period_delta_s
    "%.3f",  # snr
)


def _write_dispersion_dat(
    path: Path,
    measured: dict[str, np.ndarray],
    *,
    debug: bool,
    snr_label: str = "snr",
    metadata: dict[str, object] | None = None,
) -> None:
    if debug:
        table = np.column_stack(
            [
                measured["target_period"],
                measured["period"],
                measured["group_velocity"],
                measured["phase_velocity"],
                measured["amplitude"],
                measured["phase_derivative"],
                measured["hilbert_phase_derivative"],
                measured["hilbert_period"],
                measured["hilbert_period"] - measured["period"],
                measured["snr"],
            ]
        )
        dat_fmt = _AFTAN_DAT_DEBUG_FORMATS
        dat_header = (
            "target_period_s instant_period_s group_velocity_km_s "
            "phase_velocity_km_s amplitude phase_derivative_rad_s "
            "hilbert_phase_derivative_rad_s hilbert_instant_period_s "
            f"instant_period_delta_s {snr_label}"
        )
    else:
        table = np.column_stack(
            [
                measured["target_period"],
                measured["period"],
                measured["group_velocity"],
                measured["phase_velocity"],
                measured["amplitude"],
                measured["snr"],
            ]
        )
        dat_fmt = _AFTAN_DAT_MAIN_FORMATS
        dat_header = (
            "target_period_s instant_period_s group_velocity_km_s "
            f"phase_velocity_km_s amplitude {snr_label}"
        )
    header_lines = _dat_metadata_lines(metadata) + [dat_header]
    np.savetxt(
        path,
        table,
        fmt=dat_fmt,
        delimiter=_AFTAN_DAT_DELIMITER,
        header="\n".join(header_lines),
    )


def _dat_metadata_lines(metadata: dict[str, object] | None) -> list[str]:
    if not metadata:
        return []
    lines = []
    for key, value in metadata.items():
        if value is None:
            continue
        lines.append(f"{key}: {value}")
    return lines


def _snr_column_name(config: AFTANSNRConfig) -> str:
    return "snr_db" if config.output_db else "snr"


def _find_sac_files(input_dir: Path, pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(str(input_dir / pattern)))]


def _trace_distance_km(trace: Trace) -> float:
    from obspy.geodetics.base import gps2dist_azimuth

    sac = getattr(trace.stats, "sac", None)
    if sac is not None:
        dist = _sac_float(sac, "dist")
        if dist is not None:
            return dist
        evla = _sac_float(sac, "evla")
        evlo = _sac_float(sac, "evlo")
        stla = _sac_float(sac, "stla")
        stlo = _sac_float(sac, "stlo")
        if None not in (evla, evlo, stla, stlo):
            distance_m, azimuth, back_azimuth = gps2dist_azimuth(evla, evlo, stla, stlo)
            sac.dist = distance_m / 1000.0
            sac.az = azimuth
            sac.baz = back_azimuth
            return sac.dist
    raise ValueError("SAC header dist or evla/evlo/stla/stlo is required.")


def _trace_times(trace: Trace) -> np.ndarray:
    b = _sac_float(getattr(trace.stats, "sac", None), "b")
    if b is None:
        b = 0.0
    return b + np.arange(trace.stats.npts) * trace.stats.delta


def _sac_float(sac, key: str) -> float | None:
    if sac is None:
        return None
    try:
        value = sac[key]
    except (KeyError, TypeError):
        value = getattr(sac, key, None)
    if value is None or float(value) == -12345.0:
        return None
    return float(value)


def _resolve_path(path, base_dir: Path | None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = base_dir / resolved
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
