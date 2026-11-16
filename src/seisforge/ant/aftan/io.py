"""SAC, path, YAML, and tabular output helpers for AFTAN."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
from obspy import Trace
import yaml

from seisforge.provenance import software_provenance

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
    header_lines = _dat_metadata_lines({**(metadata or {}), **software_provenance()})
    header_lines.append(dat_header)
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


_ZRT_CODES = frozenset(
    first + second for first in "ZRT" for second in "ZRT"
)


def _component_from_sac(path: Path, trace: Trace) -> tuple[str | None, str]:
    """ZRT component of a CCF file, from the SAC header or else the file name.

    Order: SAC ``kcmpnm`` when it is a two-letter ZRT code (as written by
    ``seisforge ant rotate-ccf``); otherwise the one distinct ZRT code among
    the tokens of the file stem split on ``_``, ``.`` and ``-`` (the
    ``*_{component}.SAC`` naming of the rotation output). Returns
    ``(component, how)``; ``component`` is None when neither source gives a
    single code, and ``how`` then says why. A header and a file name that
    name different codes raise ValueError.
    """
    sac = getattr(trace.stats, "sac", None)
    header = None
    if sac is not None and "kcmpnm" in sac:
        value = str(sac["kcmpnm"]).strip().upper()
        if value in _ZRT_CODES:
            header = value
    tokens = [token.upper() for token in _split_stem(path)]
    named = sorted({token for token in tokens if token in _ZRT_CODES})
    if header is not None:
        if named and header not in named:
            raise ValueError(
                f"{path.name}: SAC kcmpnm {header} disagrees with the component "
                f"{'/'.join(named)} in the file name."
            )
        return header, "sac_kcmpnm"
    if len(named) == 1:
        return named[0], "file_name"
    if named:
        return None, f"file name has several ZRT codes ({', '.join(named)})"
    return None, "no ZRT code in SAC kcmpnm or the file name"


def _radial_convention_from_sac(trace: Trace) -> str | None:
    """R/T convention recorded by `seisforge ant rotate-ccf` (SAC ``kuser0``), or None.

    Any other ``kuser0`` content (left by other software) counts as not recorded.
    """
    from seisforge.ant.rotation import RADIAL_CONVENTION_SAC_HEADER, RADIAL_CONVENTIONS

    sac = getattr(trace.stats, "sac", None)
    if sac is None or RADIAL_CONVENTION_SAC_HEADER not in sac:
        return None
    value = str(sac[RADIAL_CONVENTION_SAC_HEADER]).strip().lower()
    return value if value in RADIAL_CONVENTIONS else None


def _split_stem(path: Path) -> list[str]:
    stem = path.name
    if stem.upper().endswith(".SAC"):
        stem = stem[:-4]
    return [token for token in stem.replace(".", "_").replace("-", "_").split("_") if token]
