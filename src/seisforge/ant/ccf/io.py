"""SDS waveform reader, station table loader, and atomic NetCDF4/HDF5 IO."""

from __future__ import annotations

from collections.abc import Sequence
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import warnings

import numpy as np
from scipy import signal as sp_signal
import xarray as xr

from .config import StationRecord
from .core import expand_mask_for_ram

__all__ = [
    "SDSDayData",
    "decimate_day_data",
    "load_station_table",
    "read_netcdf",
    "read_netcdf_pair_slice",
    "read_sds_day_zz",
    "sds_file_path",
    "station_table_hash",
    "write_netcdf",
]

_REQUIRED_STATION_COLUMNS = {
    "network",
    "station",
    "location",
    "channel",
    "latitude_deg",
    "longitude_deg",
}


@dataclass(frozen=True)
class SDSDayData:
    """One UTC day of ZZ waveforms mapped onto the absolute ``k / fs`` grid."""

    utc_date: str
    waveforms: np.ndarray
    valid_mask: np.ndarray
    sampling_rate_hz: float
    input_fingerprint: str
    missing_stations: tuple[str, ...]
    next_day_missing_stations: tuple[str, ...]


def load_station_table(
    path: str | Path,
    *,
    expected_response_state: str,
) -> tuple[StationRecord, ...]:
    """Load and validate a CSV station table for ZZ CCF processing.

    Required columns: ``network,station,location,channel,latitude_deg,longitude_deg``.
    Optional columns: ``elevation_m``, ``response_state``, ``instrument_model``.
    Refuses duplicate SEED IDs or any station whose ``response_state`` conflicts
    with ``expected_response_state``.
    """
    table_path = Path(path)
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Station table is empty: {table_path}")
        missing = sorted(_REQUIRED_STATION_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"Station table {table_path} is missing required column(s): {', '.join(missing)}."
            )
        records: list[StationRecord] = []
        seen_seed_ids: set[str] = set()
        for row_num, row in enumerate(reader, start=2):
            resp = (row.get("response_state") or "").strip() or None
            if resp is not None and resp != expected_response_state:
                raise ValueError(
                    f"Row {row_num} ({row.get('network')}.{row.get('station')}): "
                    f"response_state={resp!r} conflicts with config response_state="
                    f"{expected_response_state!r}; mixing response states introduces "
                    "phase bias."
                )
            rec = StationRecord(
                network=str(row["network"]).strip(),
                station=str(row["station"]).strip(),
                location=str(row["location"]).strip(),
                channel=str(row["channel"]).strip(),
                latitude_deg=float(row["latitude_deg"]),
                longitude_deg=float(row["longitude_deg"]),
                elevation_m=float(row["elevation_m"]) if (row.get("elevation_m") or "").strip() else 0.0,
                response_state=resp or expected_response_state,
                instrument_model=(row.get("instrument_model") or "").strip(),
            )
            if rec.seed_id in seen_seed_ids:
                raise ValueError(f"Duplicate station entry in {table_path}: {rec.seed_id}")
            seen_seed_ids.add(rec.seed_id)
            records.append(rec)

    if not records:
        raise ValueError(f"Station table contains no station rows: {table_path}")

    if expected_response_state == "gain_only":
        models = {r.instrument_model for r in records if r.instrument_model}
        if len(models) > 1:
            warnings.warn(
                "response_state='gain_only' with multiple instrument models "
                f"({sorted(models)}); consider full instrument-response removal.",
                UserWarning,
                stacklevel=2,
            )

    return tuple(records)


def station_table_hash(stations: Sequence[StationRecord]) -> str:
    """Deterministic 16-hex-char hash of the ordered station table."""
    payload = [
        [
            s.network,
            s.station,
            s.location,
            s.channel,
            round(s.latitude_deg, 8),
            round(s.longitude_deg, 8),
            round(s.elevation_m, 4),
            s.response_state,
            s.instrument_model,
        ]
        for s in stations
    ]
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def sds_file_path(
    sds_root: str | Path,
    station: StationRecord,
    utc_date: str | date,
) -> Path:
    """Return the standard SeisComP Data Structure (SDS) daily MiniSEED path."""
    d = date.fromisoformat(utc_date) if isinstance(utc_date, str) else utc_date
    year = f"{d.year:04d}"
    doy = f"{d.timetuple().tm_yday:03d}"
    filename = (
        f"{station.network}.{station.station}.{station.location}."
        f"{station.channel}.D.{year}.{doy}"
    )
    return Path(sds_root) / year / station.network / station.station / f"{station.channel}.D" / filename


def _fingerprint_files(paths: Sequence[Path], sds_root: Path) -> str:
    """Hash relative paths, byte sizes, and mtime_ns of existing input files."""
    root = Path(sds_root).resolve()
    entries = []
    for path in sorted(paths):
        st = path.stat()
        try:
            rel = str(path.resolve().relative_to(root))
        except ValueError:
            rel = str(path.resolve())
        entries.append([rel, st.st_size, st.st_mtime_ns])
    raw = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _map_stream_onto_grid(
    stream: Any,
    *,
    grid_start_epoch_s: float,
    n_samples: int,
    sampling_rate_hz: float,
    grid_tolerance_samples: float,
    file_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map ObsPy Trace segments onto the absolute ``k / fs`` sample grid."""
    dt_s = 1.0 / float(sampling_rate_hz)
    out = np.full(n_samples, np.nan, dtype=np.float64)
    valid = np.zeros(n_samples, dtype=bool)

    for tr in stream:
        tr_fs = float(tr.stats.sampling_rate)
        if not np.isclose(tr_fs, sampling_rate_hz, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"{file_label}: trace sampling_rate={tr_fs} Hz does not match "
                f"configured sampling_rate_hz={sampling_rate_hz} Hz."
            )

        offset_samples = (float(tr.stats.starttime.timestamp) - grid_start_epoch_s) * sampling_rate_hz
        nearest_idx = int(round(offset_samples))
        residual = abs(offset_samples - nearest_idx)
        if residual > grid_tolerance_samples:
            raise ValueError(
                f"{file_label}: trace starttime {tr.stats.starttime.isoformat()} "
                f"is off the absolute k/fs grid by {residual:.4g} samples "
                f"(tolerance {grid_tolerance_samples}, dt={dt_s} s)."
            )

        tr_data = np.asarray(tr.data, dtype=np.float64)
        src_start = max(0, -nearest_idx)
        dst_start = max(0, nearest_idx)
        copy_len = min(tr_data.size - src_start, n_samples - dst_start)
        if copy_len <= 0:
            continue

        segment = tr_data[src_start : src_start + copy_len]
        finite = np.isfinite(segment)
        dst_slice = slice(dst_start, dst_start + copy_len)
        out[dst_slice] = np.where(finite, segment, np.nan)
        valid[dst_slice] = finite

    return out, valid


def read_sds_day_zz(
    sds_root: str | Path,
    stations: Sequence[StationRecord],
    utc_date: str | date,
    *,
    sampling_rate_hz: float,
    extra_next_day_s: float = 0.0,
    grid_tolerance_samples: float = 1e-3,
    gap_guard_s: float = 0.0,
    allow_missing_next_day: bool = True,
) -> SDSDayData:
    """Read one UTC day (plus optional next-day head) of ZZ SDS MiniSEED files.

    Parameters
    ----------
    sds_root : path-like
        Root of a standard SDS archive.
    stations : sequence of StationRecord
        Ordered stations defining axis 0 of the returned arrays.
    utc_date : str or datetime.date
        UTC calendar date (``YYYY-MM-DD``).
    sampling_rate_hz : float
        Expected sampling rate in Hz; every trace must match.
    extra_next_day_s : float, optional
        Seconds to read from the start of the following UTC day so cross-midnight
        windows are complete.
    grid_tolerance_samples : float, optional
        Maximum allowed deviation of trace start times from ``k / fs`` in
        fractions of a sample (assumed default: 1e-3).
    gap_guard_s : float, optional
        Extra invalid-mask margin dilated around internal gaps to shield against
        filter transients from external preparation scripts.
    allow_missing_next_day : bool, optional
        If False, raise ``FileNotFoundError`` when a next-day file is missing
        while ``extra_next_day_s > 0``.
    """
    from obspy import UTCDateTime, read

    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive.")
    if extra_next_day_s < 0 or not np.isfinite(extra_next_day_s):
        raise ValueError("extra_next_day_s must be finite and non-negative.")

    day_date = date.fromisoformat(utc_date) if isinstance(utc_date, str) else utc_date
    next_date = day_date + timedelta(days=1)
    day_start = UTCDateTime(day_date.year, day_date.month, day_date.day, 0, 0, 0)
    day_samples = int(round(86400.0 * sampling_rate_hz))
    extra_samples = int(round(extra_next_day_s * sampling_rate_hz))
    total_samples = day_samples + extra_samples

    n_stations = len(stations)
    waveforms = np.full((n_stations, total_samples), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n_stations, total_samples), dtype=bool)
    existing_files: list[Path] = []
    missing_stations: list[str] = []
    next_day_missing: list[str] = []

    for idx, sta in enumerate(stations):
        day_path = sds_file_path(sds_root, sta, day_date)
        if not day_path.exists():
            missing_stations.append(sta.seed_id)
            continue
        existing_files.append(day_path)
        st = read(str(day_path))
        arr_day, mask_day = _map_stream_onto_grid(
            st,
            grid_start_epoch_s=float(day_start.timestamp),
            n_samples=day_samples,
            sampling_rate_hz=sampling_rate_hz,
            grid_tolerance_samples=grid_tolerance_samples,
            file_label=str(day_path),
        )
        waveforms[idx, :day_samples] = arr_day
        valid_mask[idx, :day_samples] = mask_day

        if extra_samples > 0:
            next_path = sds_file_path(sds_root, sta, next_date)
            if not next_path.exists():
                if not allow_missing_next_day:
                    raise FileNotFoundError(
                        f"Missing next-day SDS file required for cross-midnight windows: {next_path}"
                    )
                next_day_missing.append(sta.seed_id)
            else:
                existing_files.append(next_path)
                st_next = read(
                    str(next_path),
                    starttime=day_start + 86400.0,
                    endtime=day_start + 86400.0 + extra_next_day_s,
                )
                arr_next, mask_next = _map_stream_onto_grid(
                    st_next,
                    grid_start_epoch_s=float((day_start + 86400.0).timestamp),
                    n_samples=extra_samples,
                    sampling_rate_hz=sampling_rate_hz,
                    grid_tolerance_samples=grid_tolerance_samples,
                    file_label=str(next_path),
                )
                waveforms[idx, day_samples:] = arr_next
                valid_mask[idx, day_samples:] = mask_next

    if gap_guard_s > 0.0:
        guard_samples = int(round(gap_guard_s * sampling_rate_hz))
        valid_mask = expand_mask_for_ram(valid_mask, guard_samples)
        waveforms = np.where(valid_mask, waveforms, np.nan)

    fingerprint = _fingerprint_files(existing_files, Path(sds_root))
    return SDSDayData(
        utc_date=day_date.isoformat(),
        waveforms=waveforms,
        valid_mask=valid_mask,
        sampling_rate_hz=float(sampling_rate_hz),
        input_fingerprint=fingerprint,
        missing_stations=tuple(missing_stations),
        next_day_missing_stations=tuple(next_day_missing),
    )


def decimate_day_data(
    day_data: SDSDayData,
    target_sampling_rate_hz: float,
) -> SDSDayData:
    """Anti-alias downsample an ``SDSDayData`` onto the exact ``k / fs_target`` grid.

    Each contiguous valid segment is resampled independently with
    ``scipy.signal.resample_poly`` (Kaiser ``beta=8.0``, linear edge padding)
    after trimming leading samples to align on a multiple of the downsampling
    denominator. Gaps remain masked and inputs are never mutated.
    """
    fs_in = float(day_data.sampling_rate_hz)
    fs_out = float(target_sampling_rate_hz)
    if fs_out <= 0.0 or not np.isfinite(fs_out):
        raise ValueError("target_sampling_rate_hz must be finite and positive.")
    if np.isclose(fs_out, fs_in, rtol=1e-9, atol=0.0):
        return day_data
    if fs_out > fs_in + 1e-12:
        raise ValueError(
            f"target_sampling_rate_hz ({fs_out} Hz) cannot exceed "
            f"input sampling_rate_hz ({fs_in} Hz)."
        )

    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    if not np.isclose(fs_in * up / down, fs_out, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"target_sampling_rate_hz ({fs_out} Hz) is not a clean rational "
            f"factor of input sampling_rate_hz ({fs_in} Hz)."
        )

    n_stations, n_samples_in = day_data.waveforms.shape
    n_samples_out = int(round(n_samples_in * fs_out / fs_in))
    out_wave = np.full((n_stations, n_samples_out), np.nan, dtype=np.float64)
    out_valid = np.zeros((n_stations, n_samples_out), dtype=bool)

    for idx in range(n_stations):
        mask = day_data.valid_mask[idx]
        if not np.any(mask):
            continue
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)

        for s_idx, e_idx in zip(starts, ends):
            s = int(s_idx)
            e = int(e_idx)
            # Align segment start upward to a multiple of `down` so
            # (s_aligned / fs_in) == ((s_aligned // down) * up) / fs_out.
            lead = (-s) % down
            s_aligned = s + lead
            if s_aligned >= e:
                continue
            seg = np.asarray(day_data.waveforms[idx, s_aligned:e], dtype=np.float64)
            y = sp_signal.resample_poly(seg, up, down, window=("kaiser", 8.0), padtype="line")

            out_start = (s_aligned // down) * up
            max_valid_end = ((e - 1) // down) * up + 1
            out_end = min(n_samples_out, out_start + int(y.size), max_valid_end)
            copy_len = out_end - out_start
            if copy_len <= 0:
                continue
            out_wave[idx, out_start:out_end] = y[:copy_len]
            out_valid[idx, out_start:out_end] = True

    return SDSDayData(
        utc_date=day_data.utc_date,
        waveforms=out_wave,
        valid_mask=out_valid,
        sampling_rate_hz=fs_out,
        input_fingerprint=day_data.input_fingerprint,
        missing_stations=day_data.missing_stations,
        next_day_missing_stations=day_data.next_day_missing_stations,
    )


def write_netcdf(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    overwrite: bool = False,
    compression_level: int = 4,
) -> Path:
    """Atomically write an HDF5-backed NetCDF4 file via temporary file + rename."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists and overwrite is False: {output}")

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

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        dataset.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)
        os.replace(tmp_path, output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return output


def read_netcdf(path: str | Path) -> xr.Dataset:
    """Load a CCF NetCDF4/HDF5 dataset into memory."""
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        return opened.load()


def read_netcdf_pair_slice(path: str | Path, start: int, end: int) -> xr.Dataset:
    """Lazily read only the ``[start:end]`` pair slice of a CCF NetCDF4/HDF5 file."""
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        return opened.isel(pair=slice(start, end)).load()
