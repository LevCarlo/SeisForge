#!/usr/bin/env python3
"""Stage-A standalone preprocessing script for Guangxi SEIS-S20 raw MiniSEED data.

Converts a single SEIS-S20 station directory (containing rolling ~80 MiB MiniSEED
files under ``Mseed*`` and optional status logs under ``Log*``) into a standard
daily SeisComP Data Structure (SDS) archive that satisfies the strict input
contract of ``seisforge ant ccf`` (``read_sds_day_zz``).

Design & Parallelism
--------------------
This script intentionally processes **one station directory per invocation** with
no internal multi-station loop or multiprocessing pool. Run multiple stations in
parallel using GNU ``parallel`` or a cluster job array:

.. code-block:: bash

    find /path/to/RawData/batch1 -mindepth 1 -maxdepth 1 -type d \\
        | parallel -j 8 \\
            python scripts/prep_seis_s20_sds.py {} \\
            --out-sds /path/to/sds_50hz \\
            --pz /path/to/SEIS-S20.PZ \\
            --fs-out 50.0 \\
            --channels BHE BHN BHZ

Physical Conventions & Assumed Defaults
---------------------------------------
* **Instrument response (SAC ``to none``)**: The ``SEIS-S20.PZ`` file (5 zeros,
  5 poles, ``CONSTANT = 160.2386``) represents the **velocity** transfer function
  directly (flat amplitude ~170 across 0.05–20 Hz). Deconvolution divides the
  spectrum by ``H(s)`` directly without an extra ``1 / s`` derivative.
* **Pre-filter**: When ``--pre-filt`` is omitted, defaults to
  ``(0.05, 0.10, 0.40 * fs_out, 0.45 * fs_out)`` Hz (for ``fs_out = 20 Hz``:
  ``0.05 0.10 8.0 9.0`` Hz; SEIS-S20 usable low-frequency band is optimistic to
  0.10 Hz / 10 s and stable above 0.20 Hz / 5 s).
* **Sub-sample alignment & resampling**: Raw SEIS-S20 streams (200 Hz) start off
  the absolute UTC ``k / fs_in`` grid by ~1–2 ms. Each contiguous segment is
  padded across UTC midnight (default ``--pad-s 600``), shifted onto the exact
  ``k / fs_in`` grid via ``float64`` frequency-domain phase shift (combined with
  pre-filtering and response deconvolution), trimmed to align with the decimation
  phase, and anti-alias downsampled with ``scipy.signal.resample_poly`` so every
  output sample lies on the exact ``k / fs_out`` grid (residual << 1e-3 samples).
* **Output format**: Standard SDS layout
  ``YEAR/NET/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY`` written atomically as
  ``FLOAT32`` MiniSEED. Existing files are skipped unless ``--overwrite`` is set.

Exit Codes
----------
* ``0``: All requested station-days completed (or skipped as already existing).
* ``1``: Fatal configuration, path, or input-indexing error.
* ``2``: Partial failure (one or more station-days raised a processing error).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from fractions import Fraction
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read
from scipy import fft as sp_fft
from scipy import signal as sp_signal

DEFAULT_PZ_PATH: Path | None = None
DEFAULT_FS_IN = 200.0
DEFAULT_FS_OUT = 20.0
DEFAULT_PAD_S = 600.0
DEFAULT_EDGE_TAPER_S = 60.0
DEFAULT_MIN_SEGMENT_S = 60.0
EXCESSIVE_FRAGMENTATION_THRESHOLD = 50
DEFAULT_WATER_LEVEL_DB = 60.0
DEFAULT_CLIP_THRESHOLD = 8388000.0  # 24-bit signed ADC saturation margin (2^23 - 1 = 8388607)
DEFAULT_INSTRUMENT_MODEL = "SEIS-S20"

_SAMPLING_STARTED_RE = re.compile(
    r"@Sampling\s+Started\s+"
    r"(?P<ymd>\d{2}-\d{2}-\d{2})\s+(?P<hm>\d{2}:\d{2})\s+"
    r"GPS_Lock:(?P<lock>[TF]),\s*SPS=(?P<sps>\d+),\s*"
    r"Pos:(?P<lat>-?\d+),(?P<lon>-?\d+),(?P<elev>-?\d+)"
)
_STATUS_LINE_RE = re.compile(
    r"#Status:(?P<ymd>\d{2}-\d{2}-\d{2})\s+(?P<hour>\d{2})\s+"
    r"BAT=(?P<bat>[+-]?\d+(?:\.\d+)?)V\s+"
    r"CHG=(?P<chg>[+-]?\d+(?:\.\d+)?)V\s+"
    r"Pitch=(?P<pitch>[+-]?\d+(?:\.\d+)?)\s+"
    r"Roll=(?P<roll>[+-]?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class PoleZeroResponse:
    """SAC Pole-Zero representation for ``to none`` velocity transfer function."""

    zeros: tuple[complex, ...]
    poles: tuple[complex, ...]
    constant: float


@dataclass(frozen=True)
class MseedIndexEntry:
    """Lightweight header index entry for one trace inside a raw MiniSEED file."""

    path: Path
    network: str
    station_raw: str
    location: str
    channel: str
    starttime: UTCDateTime
    endtime: UTCDateTime
    sampling_rate: float
    npts: int
    encoding: str


@dataclass(frozen=True)
class SamplingStartRecord:
    """Parsed ``@Sampling Started`` metadata from SEIS-S20 ``Log*`` files."""

    timestamp_utc: UTCDateTime
    gps_lock: bool
    sps: int
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    source_file: str


@dataclass(frozen=True)
class DailyLogStatus:
    """Daily aggregated battery and tilt summary from ``#Status:`` log lines."""

    utc_date: str
    count: int
    bat_min_v: float
    bat_max_v: float
    pitch_max_abs_deg: float
    roll_max_abs_deg: float


@dataclass(frozen=True)
class StationLogSummary:
    """Parsed SEIS-S20 instrument log summary."""

    sampling_starts: tuple[SamplingStartRecord, ...]
    selected_start: SamplingStartRecord | None
    daily_status: dict[str, DailyLogStatus]


@dataclass
class DayChannelQC:
    """Per-day, per-channel QC record written to CSV / JSONL."""

    utc_date: str
    network: str
    station: str
    raw_station: str
    location: str
    channel: str
    instrument_model: str
    response_state: str
    status: str
    output_path: str
    n_source_files: int = 0
    fs_in_hz: float = float("nan")
    fs_out_hz: float = float("nan")
    fs_anomaly: bool = False
    raw_n_segments: int = 0
    dropped_short_segments: int = 0
    out_n_traces: int = 0
    day_coverage_pct: float = 0.0
    gap_count: int = 0
    gap_total_s: float = 0.0
    overlap_count: int = 0
    overlap_total_s: float = 0.0
    frac_sample_s: float = float("nan")
    frac_sample_offset: float = float("nan")
    frac_offset_jump: bool = False
    frac_offset_max_jump_s: float = 0.0
    out_grid_residual_samples: float = float("nan")
    non_finite_count: int = 0
    zero_fraction: float = float("nan")
    clip_count: int = 0
    clip_fraction: float = float("nan")
    raw_rms: float = float("nan")
    raw_ptp: float = float("nan")
    proc_rms: float = float("nan")
    proc_ptp: float = float("nan")
    gps_lock: str = ""
    latitude_deg: float = float("nan")
    longitude_deg: float = float("nan")
    elevation_m: float = float("nan")
    bat_min_v: float = float("nan")
    bat_max_v: float = float("nan")
    pitch_max_abs_deg: float = float("nan")
    roll_max_abs_deg: float = float("nan")
    log_status_count: int = 0
    anomalies: str = ""
    error_message: str = ""


# ---------------------------------------------------------------------------
# Parsing helpers: SAC PZ, station naming, MiniSEED index, SEIS-S20 logs
# ---------------------------------------------------------------------------


def parse_sac_pz(path: str | Path) -> PoleZeroResponse:
    """Parse a SAC Pole-Zero file (``ZEROS`` / ``POLES`` / ``CONSTANT``)."""
    pz_path = Path(path)
    if not pz_path.is_file():
        raise FileNotFoundError(f"SAC PZ file not found: {pz_path}")

    lines = [
        line.strip()
        for line in pz_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("*")
    ]

    zeros: list[complex] = []
    poles: list[complex] = []
    constant: float | None = None

    idx = 0
    while idx < len(lines):
        tokens = lines[idx].split()
        key = tokens[0].upper()
        if key == "ZEROS":
            n_zeros = int(tokens[1])
            idx += 1
            count = 0
            while idx < len(lines) and count < n_zeros:
                parts = lines[idx].split()
                if parts[0].upper() in {"ZEROS", "POLES", "CONSTANT"}:
                    break
                zeros.append(complex(float(parts[0]), float(parts[1])))
                idx += 1
                count += 1
            while len(zeros) < n_zeros:
                zeros.append(0.0 + 0.0j)
        elif key == "POLES":
            n_poles = int(tokens[1])
            idx += 1
            count = 0
            while idx < len(lines) and count < n_poles:
                parts = lines[idx].split()
                if parts[0].upper() in {"ZEROS", "POLES", "CONSTANT"}:
                    break
                poles.append(complex(float(parts[0]), float(parts[1])))
                idx += 1
                count += 1
            if len(poles) != n_poles:
                raise ValueError(
                    f"Expected {n_poles} poles in {pz_path}, found {len(poles)}."
                )
        elif key == "CONSTANT":
            constant = float(tokens[1])
            idx += 1
        else:
            idx += 1

    if constant is None or not np.isfinite(constant) or constant == 0.0:
        raise ValueError(f"Missing or invalid CONSTANT in SAC PZ file: {pz_path}")

    return PoleZeroResponse(
        zeros=tuple(zeros),
        poles=tuple(poles),
        constant=float(constant),
    )


def eval_pz_transfer_function(
    pz: PoleZeroResponse,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Evaluate ``H(s)`` at ``s = i * 2 * pi * f`` under SAC ``to none`` convention."""
    s = 1j * 2.0 * np.pi * np.asarray(freqs_hz, dtype=np.float64)
    resp = np.full(s.shape, pz.constant, dtype=np.complex128)
    for zero in pz.zeros:
        resp *= s - zero
    for pole in pz.poles:
        resp /= s - pole
    return resp


def infer_station_code(station_dir: Path) -> str:
    """Extract point-ID station name from directory name (e.g. ``2164-0519`` -> ``2164``)."""
    name = station_dir.resolve().name
    if "-" in name:
        prefix = name.split("-", 1)[0].strip()
        if prefix:
            return prefix
    return name


def discover_mseed_files(station_dir: Path) -> list[Path]:
    """Find candidate raw MiniSEED files in ``station_dir/Mseed*`` or ``station_dir``."""
    mseed_dirs = sorted(
        p
        for p in station_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith("mseed")
    )
    search_dirs = mseed_dirs if mseed_dirs else [station_dir]

    candidates: list[Path] = []
    for directory in search_dirs:
        for item in sorted(directory.iterdir()):
            if not item.is_file():
                continue
            if item.name.startswith(".") or item.name.startswith("._"):
                continue
            candidates.append(item)
    return candidates


def build_mseed_index(
    files: Sequence[Path],
    logger: logging.Logger,
) -> tuple[list[MseedIndexEntry], int]:
    """Scan MiniSEED headers only to build a lightweight time index."""
    entries: list[MseedIndexEntry] = []
    skipped_files = 0
    for path in files:
        try:
            st = read(str(path), headonly=True)
        except Exception as exc:
            skipped_files += 1
            logger.warning("Skipping non-MiniSEED or unreadable file %s: %s", path, exc)
            continue
        for tr in st:
            mseed_meta = getattr(tr.stats, "mseed", None)
            encoding = str(getattr(mseed_meta, "encoding", "")) if mseed_meta else ""
            entries.append(
                MseedIndexEntry(
                    path=path,
                    network=str(tr.stats.network).strip(),
                    station_raw=str(tr.stats.station).strip(),
                    location=str(tr.stats.location).strip(),
                    channel=str(tr.stats.channel).strip(),
                    starttime=UTCDateTime(tr.stats.starttime),
                    endtime=UTCDateTime(tr.stats.endtime),
                    sampling_rate=float(tr.stats.sampling_rate),
                    npts=int(tr.stats.npts),
                    encoding=encoding,
                )
            )
    entries.sort(key=lambda e: (e.channel, e.starttime.timestamp, str(e.path)))
    return entries, skipped_files


def _parse_yy_mm_dd(ymd: str) -> date:
    yy, mm, dd = (int(part) for part in ymd.split("-"))
    return date(2000 + yy, mm, dd)


def parse_station_logs(
    station_dir: Path,
    earliest_data_start: UTCDateTime | None,
    logger: logging.Logger,
) -> StationLogSummary:
    """Parse optional ``Log*`` directory for GPS lock, coordinates, battery, and tilt."""
    log_dirs = sorted(
        p
        for p in station_dir.iterdir()
        if p.is_dir() and p.name.lower().startswith("log")
    )
    if not log_dirs:
        return StationLogSummary(sampling_starts=(), selected_start=None, daily_status={})

    starts: list[SamplingStartRecord] = []
    daily_buckets: dict[str, dict[str, list[float]]] = {}

    for log_dir in log_dirs:
        for log_file in sorted(log_dir.iterdir()):
            if not log_file.is_file() or log_file.name.startswith("."):
                continue
            try:
                text = log_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Failed to read log file %s: %s", log_file, exc)
                continue

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                m_start = _SAMPLING_STARTED_RE.search(line)
                if m_start:
                    d_obj = _parse_yy_mm_dd(m_start.group("ymd"))
                    hh, minute = (int(x) for x in m_start.group("hm").split(":"))
                    ts = UTCDateTime(d_obj.year, d_obj.month, d_obj.day, hh, minute, 0)
                    starts.append(
                        SamplingStartRecord(
                            timestamp_utc=ts,
                            gps_lock=(m_start.group("lock") == "T"),
                            sps=int(m_start.group("sps")),
                            latitude_deg=int(m_start.group("lat")) / 1e7,
                            longitude_deg=int(m_start.group("lon")) / 1e7,
                            elevation_m=int(m_start.group("elev")) / 1e3,
                            source_file=log_file.name,
                        )
                    )
                    continue

                m_stat = _STATUS_LINE_RE.search(line)
                if m_stat:
                    d_str = _parse_yy_mm_dd(m_stat.group("ymd")).isoformat()
                    bucket = daily_buckets.setdefault(
                        d_str, {"bat": [], "pitch": [], "roll": []}
                    )
                    bucket["bat"].append(float(m_stat.group("bat")))
                    bucket["pitch"].append(abs(float(m_stat.group("pitch"))))
                    bucket["roll"].append(abs(float(m_stat.group("roll"))))

    starts.sort(key=lambda r: r.timestamp_utc.timestamp)
    selected: SamplingStartRecord | None = None
    if starts:
        if earliest_data_start is not None:
            # Prefer the latest start record at or shortly before the earliest data.
            prior = [
                rec
                for rec in starts
                if rec.timestamp_utc.timestamp <= earliest_data_start.timestamp + 300.0
            ]
            if prior:
                selected = min(
                    prior,
                    key=lambda r: abs(r.timestamp_utc.timestamp - earliest_data_start.timestamp),
                )
            else:
                selected = min(
                    starts,
                    key=lambda r: abs(r.timestamp_utc.timestamp - earliest_data_start.timestamp),
                )
        else:
            selected = starts[-1]

    daily_status: dict[str, DailyLogStatus] = {}
    for d_str, vals in sorted(daily_buckets.items()):
        daily_status[d_str] = DailyLogStatus(
            utc_date=d_str,
            count=len(vals["bat"]),
            bat_min_v=min(vals["bat"]),
            bat_max_v=max(vals["bat"]),
            pitch_max_abs_deg=max(vals["pitch"]),
            roll_max_abs_deg=max(vals["roll"]),
        )

    return StationLogSummary(
        sampling_starts=tuple(starts),
        selected_start=selected,
        daily_status=daily_status,
    )


# ---------------------------------------------------------------------------
# Signal processing: fractional-sample shift, cosine pre-filter, PZ removal,
# and phase-aligned anti-aliasing resampling
# ---------------------------------------------------------------------------


def cosine_taper_spectrum(
    freqs_hz: np.ndarray,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
) -> np.ndarray:
    """Four-corner frequency-domain cosine taper (SAC / ObsPy ``pre_filt``)."""
    w = np.zeros_like(freqs_hz, dtype=np.float64)
    in_flat = (freqs_hz >= f2) & (freqs_hz <= f3)
    w[in_flat] = 1.0

    low_ramp = (freqs_hz > f1) & (freqs_hz < f2)
    if np.any(low_ramp):
        w[low_ramp] = 0.5 * (1.0 - np.cos(np.pi * (freqs_hz[low_ramp] - f1) / (f2 - f1)))

    high_ramp = (freqs_hz > f3) & (freqs_hz < f4)
    if np.any(high_ramp):
        w[high_ramp] = 0.5 * (1.0 + np.cos(np.pi * (freqs_hz[high_ramp] - f3) / (f4 - f3)))

    return w


def apply_edge_cosine_taper(data: np.ndarray, n_taper: int) -> np.ndarray:
    """Apply a symmetric Hann/cosine taper of ``n_taper`` samples to both ends."""
    out = np.array(data, dtype=np.float64, copy=True)
    n = out.size
    if n < 4 or n_taper <= 0:
        return out
    m = min(n_taper, n // 2)
    if m <= 1:
        return out
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(m, dtype=np.float64) / float(m)))
    out[:m] *= ramp
    out[-m:] *= ramp[::-1]
    return out


def signed_frac_sample_offset(starttime_epoch_s: float, fs_hz: float) -> tuple[float, float]:
    """Return ``(frac_samples, frac_seconds)`` in ``[-0.5, 0.5)`` relative to ``k / fs``."""
    scaled = starttime_epoch_s * fs_hz
    nearest = round(scaled)
    frac_samples = scaled - nearest
    return frac_samples, frac_samples / fs_hz


def process_contiguous_segment(
    data: np.ndarray,
    starttime: UTCDateTime,
    *,
    fs_in: float,
    fs_out: float,
    pz: PoleZeroResponse | None,
    pre_filt: tuple[float, float, float, float] | None,
    water_level_db: float | None,
    edge_taper_s: float,
) -> tuple[np.ndarray, int]:
    """Detrend, taper, fractionally align, deconvolve, and resample one segment.

    Returns
    -------
    resampled : np.ndarray
        ``float64`` waveform sampled at ``fs_out`` Hz.
    k_out_start : int
        Exact integer sample index on the absolute ``k / fs_out`` UTC grid
        (i.e. ``timestamp = k_out_start / fs_out``).
    """
    x = np.asarray(data, dtype=np.float64)
    n_orig = x.size
    if n_orig < 4:
        return np.empty(0, dtype=np.float64), 0

    # 1. Remove linear trend (and mean) to suppress DC leakage in FFT.
    x = sp_signal.detrend(x, type="linear")

    # 2. Cosine taper outer edges (capped at 5% of segment length).
    n_taper = min(int(round(edge_taper_s * fs_in)), max(0, int(0.05 * n_orig)))
    x = apply_edge_cosine_taper(x, n_taper)

    # 3. Fractional-sample alignment onto exact k / fs_in grid + optional PZ removal.
    t0 = float(starttime.timestamp)
    k0_in = int(round(t0 * fs_in))
    t_grid_in = k0_in / fs_in
    shift_s = t_grid_in - t0  # evaluate x(t + shift_s) -> multiply by exp(+i 2 pi f shift_s)

    need_fft = (abs(shift_s) > 1e-12) or (pz is not None) or (pre_filt is not None)
    if need_fft:
        # Pad by up to 60 s (or segment length) to prevent circular wrap-around.
        pad_fft = min(n_orig, int(round(60.0 * fs_in)))
        nfft = int(sp_fft.next_fast_len(n_orig + pad_fft, real=True))
        spec = sp_fft.rfft(x, n=nfft)
        freqs = sp_fft.rfftfreq(nfft, d=1.0 / fs_in)

        if abs(shift_s) > 1e-12:
            phase_shift = np.exp(1j * 2.0 * np.pi * freqs * shift_s)
            if nfft % 2 == 0:
                phase_shift[-1] = np.cos(np.pi * fs_in * shift_s)
            spec *= phase_shift

        if pz is not None:
            h_resp = eval_pz_transfer_function(pz, freqs)
            h_abs = np.abs(h_resp)
            if water_level_db is not None and water_level_db > 0.0:
                max_abs = float(np.max(h_abs))
                floor = max_abs * (10.0 ** (-water_level_db / 20.0))
            else:
                floor = 0.0
            denom = np.maximum(h_abs, floor)
            inv_h = np.zeros_like(h_resp)
            nz = h_abs > 0.0
            # Phase-preserving amplitude floor (same convention as ObsPy invert_spectrum).
            inv_h[nz] = np.conj(h_resp[nz] / h_abs[nz]) / denom[nz]
            inv_h[0] = 0.0
            spec *= inv_h

        if pre_filt is not None:
            f1, f2, f3, f4 = pre_filt
            spec *= cosine_taper_spectrum(freqs, f1, f2, f3, f4)
            spec[0] = 0.0

        x = sp_fft.irfft(spec, n=nfft)[:n_orig]

    # 4. Anti-aliasing rational resampling onto exact k / fs_out grid.
    if np.isclose(fs_in, fs_out, rtol=1e-9, atol=0.0):
        return x, k0_in

    ratio = Fraction(fs_out / fs_in).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    if not np.isclose(fs_in * up / down, fs_out, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"Unsupported non-rational resampling ratio: fs_in={fs_in} -> fs_out={fs_out}."
        )

    # Trim leading input samples so k0_aligned is divisible by `down`.
    # With gcd(up, down) == 1, k0_aligned % down == 0 guarantees that
    # t_start_out = k0_aligned / fs_in = ((k0_aligned // down) * up) / fs_out
    # lies exactly on the absolute k / fs_out grid.
    lead = (-k0_in) % down
    if lead >= x.size:
        return np.empty(0, dtype=np.float64), 0
    if lead > 0:
        x = x[lead:]
        k0_in += lead

    k_out_start = (k0_in // down) * up
    # Use padtype='line' and Kaiser beta=8.0 (~80 dB stopband, <1e-4 passband ripple).
    y = sp_signal.resample_poly(x, up, down, window=("kaiser", 8.0), padtype="line")
    return np.asarray(y, dtype=np.float64), k_out_start


def split_finite_segments(tr: Trace) -> list[Trace]:
    """Split a Trace around any NaN / Inf samples so FFTs never see non-finite values."""
    arr = np.asarray(tr.data, dtype=np.float64)
    finite = np.isfinite(arr)
    if np.all(finite):
        return [tr]
    if not np.any(finite):
        return []

    segments: list[Trace] = []
    padded = np.concatenate(([False], finite, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    dt = 1.0 / float(tr.stats.sampling_rate)
    for s_idx, e_idx in zip(starts, ends):
        sub = tr.copy()
        sub.data = arr[s_idx:e_idx].copy()
        sub.stats.starttime = tr.stats.starttime + s_idx * dt
        segments.append(sub)
    return segments


# ---------------------------------------------------------------------------
# SDS path & atomic I/O
# ---------------------------------------------------------------------------


def sds_daily_path(
    sds_root: Path,
    network: str,
    station: str,
    location: str,
    channel: str,
    utc_date: date,
) -> Path:
    """Return standard SDS daily MiniSEED path matching ``seisforge.ant.ccf.io``."""
    year = f"{utc_date.year:04d}"
    doy = f"{utc_date.timetuple().tm_yday:03d}"
    filename = f"{network}.{station}.{location}.{channel}.D.{year}.{doy}"
    return sds_root / year / network / station / f"{channel}.D" / filename


def write_mseed_atomic(stream: Stream, out_path: Path) -> None:
    """Atomically write a MiniSEED stream encoded as FLOAT32."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=out_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        stream.write(str(tmp_path), format="MSEED", encoding="FLOAT32", reclen=4096)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_text_atomic(content: str, out_path: Path) -> None:
    """Atomically write UTF-8 text via temp file + ``os.replace``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=out_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Per-day / per-channel orchestration & QC
# ---------------------------------------------------------------------------


def _analyze_raw_stream_for_day(
    raw_traces: list[Trace],
    *,
    day_start: UTCDateTime,
    day_end: UTCDateTime,
    expected_fs_in: float,
    clip_threshold: float | None,
) -> dict[str, Any]:
    """Compute gap/overlap, sub-sample timing, and raw signal quality metrics."""
    traces = sorted(raw_traces, key=lambda t: t.stats.starttime.timestamp)
    fs_values = [float(t.stats.sampling_rate) for t in traces]
    fs_in = fs_values[0]
    fs_anomaly = any(
        not np.isclose(fs_val, expected_fs_in, rtol=1e-6, atol=0.0)
        for fs_val in fs_values
    )

    # Sub-sample offset analysis across all contributing traces.
    frac_samples_list: list[float] = []
    frac_s_list: list[float] = []
    for tr in traces:
        fs_s, dt_s = signed_frac_sample_offset(
            float(tr.stats.starttime.timestamp), float(tr.stats.sampling_rate)
        )
        frac_samples_list.append(fs_s)
        frac_s_list.append(dt_s)

    max_jump_s = 0.0
    if len(frac_s_list) > 1:
        # Handle wrap-around at +/- 0.5 samples cleanly.
        diffs = []
        for a, b in zip(frac_samples_list[:-1], frac_samples_list[1:]):
            d_samp = abs(b - a)
            if d_samp > 0.5:
                d_samp = 1.0 - d_samp
            diffs.append(d_samp / fs_in)
        max_jump_s = max(diffs) if diffs else 0.0
    frac_offset_jump = (max_jump_s * fs_in) > 1e-3

    # Clip traces to [day_start, day_end) for day-scoped gap/overlap and stats.
    day_traces: list[Trace] = []
    for tr in traces:
        if tr.stats.endtime < day_start or tr.stats.starttime >= day_end:
            continue
        clipped = tr.slice(
            starttime=day_start,
            endtime=day_end - 1.0 / float(tr.stats.sampling_rate),
            nearest_sample=True,
        )
        if clipped.stats.npts > 0:
            day_traces.append(clipped)

    gap_count = 0
    gap_total_s = 0.0
    overlap_count = 0
    overlap_total_s = 0.0

    if day_traces:
        dt_in = 1.0 / fs_in
        tol_dt = 0.55 * dt_in
        # Leading gap from 00:00:00 UTC
        lead_gap = float(day_traces[0].stats.starttime.timestamp - day_start.timestamp)
        if lead_gap > tol_dt:
            gap_count += 1
            gap_total_s += lead_gap

        for prev_tr, cur_tr in zip(day_traces[:-1], day_traces[1:]):
            expected_next = float(prev_tr.stats.starttime.timestamp) + prev_tr.stats.npts * dt_in
            delta_t = float(cur_tr.stats.starttime.timestamp) - expected_next
            if delta_t > tol_dt:
                gap_count += 1
                gap_total_s += delta_t
            elif delta_t < -tol_dt:
                overlap_count += 1
                overlap_total_s += -delta_t

        # Trailing gap to 24:00:00 UTC
        last_end_next = float(day_traces[-1].stats.starttime.timestamp) + day_traces[-1].stats.npts * dt_in
        trail_gap = float(day_end.timestamp) - last_end_next
        if trail_gap > tol_dt:
            gap_count += 1
            gap_total_s += trail_gap

    # Raw amplitude / quality metrics within [day_start, day_end)
    total_samples = 0
    non_finite_count = 0
    zero_count = 0
    clip_count = 0
    sum_sq = 0.0
    finite_count = 0
    raw_min = np.inf
    raw_max = -np.inf

    for tr in day_traces:
        arr = np.asarray(tr.data, dtype=np.float64)
        total_samples += arr.size
        finite_mask = np.isfinite(arr)
        nf = int(arr.size - np.count_nonzero(finite_mask))
        non_finite_count += nf
        valid = arr[finite_mask]
        if valid.size == 0:
            continue
        finite_count += valid.size
        zero_count += int(np.count_nonzero(valid == 0.0))
        if clip_threshold is not None:
            clip_count += int(np.count_nonzero(np.abs(valid) >= clip_threshold))
        sum_sq += float(np.dot(valid, valid))
        raw_min = min(raw_min, float(np.min(valid)))
        raw_max = max(raw_max, float(np.max(valid)))

    raw_rms = float(np.sqrt(sum_sq / finite_count)) if finite_count > 0 else float("nan")
    raw_ptp = float(raw_max - raw_min) if finite_count > 0 else float("nan")
    zero_fraction = (zero_count / finite_count) if finite_count > 0 else float("nan")
    clip_fraction = (clip_count / finite_count) if finite_count > 0 else float("nan")

    return {
        "fs_in_hz": fs_in,
        "fs_anomaly": fs_anomaly,
        "raw_n_segments": len(day_traces),
        "gap_count": gap_count,
        "gap_total_s": gap_total_s,
        "overlap_count": overlap_count,
        "overlap_total_s": overlap_total_s,
        "frac_sample_s": frac_s_list[0],
        "frac_sample_offset": frac_samples_list[0],
        "frac_offset_jump": frac_offset_jump,
        "frac_offset_max_jump_s": max_jump_s,
        "non_finite_count": non_finite_count,
        "zero_fraction": zero_fraction,
        "clip_count": clip_count,
        "clip_fraction": clip_fraction,
        "raw_rms": raw_rms,
        "raw_ptp": raw_ptp,
    }


def process_day_channel(
    *,
    utc_date: date,
    channel: str,
    index_entries: Sequence[MseedIndexEntry],
    sds_root: Path,
    station_out: str,
    network_override: str | None,
    location_override: str | None,
    fs_out: float,
    expected_fs_in: float,
    pz: PoleZeroResponse | None,
    pre_filt: tuple[float, float, float, float] | None,
    water_level_db: float | None,
    pad_s: float,
    edge_taper_s: float,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
    clip_threshold: float | None,
    overwrite: bool,
    log_summary: StationLogSummary,
    instrument_model: str,
    logger: logging.Logger,
) -> DayChannelQC:
    """Process one UTC day and one channel into an atomic FLOAT32 SDS MiniSEED file."""
    day_start = UTCDateTime(utc_date.year, utc_date.month, utc_date.day, 0, 0, 0)
    day_end = day_start + 86400.0
    win_start = day_start - pad_s
    win_end = day_end + pad_s

    response_state = "velocity" if pz is not None else "gain_only"
    matching_entries = [
        e
        for e in index_entries
        if e.channel == channel
        and e.endtime >= win_start
        and e.starttime <= win_end
    ]

    # Determine header metadata from matching entries (or any entry for this channel).
    ref_entry = matching_entries[0] if matching_entries else next(
        (e for e in index_entries if e.channel == channel), None
    )
    net = network_override if network_override is not None else (ref_entry.network if ref_entry else "XX")
    loc = location_override if location_override is not None else (ref_entry.location if ref_entry else "00")
    raw_station = ref_entry.station_raw if ref_entry else ""
    out_path = sds_daily_path(sds_root, net, station_out, loc, channel, utc_date)

    qc = DayChannelQC(
        utc_date=utc_date.isoformat(),
        network=net,
        station=station_out,
        raw_station=raw_station,
        location=loc,
        channel=channel,
        instrument_model=instrument_model,
        response_state=response_state,
        status="pending",
        output_path=str(out_path),
        fs_out_hz=fs_out,
    )

    # Attach log metadata regardless of skip/no_data
    if log_summary.selected_start is not None:
        sel = log_summary.selected_start
        qc.gps_lock = "T" if sel.gps_lock else "F"
        qc.latitude_deg = sel.latitude_deg
        qc.longitude_deg = sel.longitude_deg
        qc.elevation_m = sel.elevation_m
    day_log = log_summary.daily_status.get(utc_date.isoformat())
    if day_log is not None:
        qc.bat_min_v = day_log.bat_min_v
        qc.bat_max_v = day_log.bat_max_v
        qc.pitch_max_abs_deg = day_log.pitch_max_abs_deg
        qc.roll_max_abs_deg = day_log.roll_max_abs_deg
        qc.log_status_count = day_log.count

    if out_path.exists() and not overwrite:
        qc.status = "skipped_exists"
        logger.info(
            "Skipping %s %s.%s.%s.%s (output already exists: %s)",
            utc_date.isoformat(), net, station_out, loc, channel, out_path,
        )
        return qc

    source_files = sorted({e.path for e in matching_entries})
    qc.n_source_files = len(source_files)
    if not source_files:
        qc.status = "no_data"
        qc.anomalies = "NO_DATA"
        logger.warning(
            "No raw files overlap %s [%s, %s] for channel %s",
            utc_date.isoformat(), win_start.isoformat(), win_end.isoformat(), channel,
        )
        return qc

    try:
        raw_stream = Stream()
        for src in source_files:
            part = read(str(src), starttime=win_start, endtime=win_end)
            for tr in part:
                if str(tr.stats.channel).strip() == channel and tr.stats.npts > 0:
                    raw_stream.append(tr)

        if len(raw_stream) == 0:
            qc.status = "no_data"
            qc.anomalies = "NO_DATA"
            return qc

        # Analyze raw traces before merging.
        diag = _analyze_raw_stream_for_day(
            list(raw_stream),
            day_start=day_start,
            day_end=day_end,
            expected_fs_in=expected_fs_in,
            clip_threshold=clip_threshold,
        )
        for key, val in diag.items():
            setattr(qc, key, val)

        if qc.raw_n_segments == 0:
            qc.status = "no_data"
            qc.anomalies = "NO_DATA"
            return qc

        # Verify all traces share a single sampling rate before merging.
        unique_fs = {round(float(tr.stats.sampling_rate), 6) for tr in raw_stream}
        if len(unique_fs) > 1:
            raise ValueError(
                f"Mixed input sampling rates in {utc_date.isoformat()} {channel}: {sorted(unique_fs)} Hz"
            )
        fs_in = float(raw_stream[0].stats.sampling_rate)

        # Merge seamless/overlapping segments (keeping gaps unmasked via split).
        merged = raw_stream.copy()
        merged.merge(method=1, interpolation_samples=0)
        contiguous_traces = merged.split()

        day_samples = int(round(86400.0 * fs_out))
        k_day_start = int(round(float(day_start.timestamp) * fs_out))
        k_day_end = k_day_start + day_samples

        out_traces: list[Trace] = []
        dropped_short = 0
        valid_samples_out = 0
        max_grid_residual = 0.0
        proc_sum_sq = 0.0
        proc_min = np.inf
        proc_max = -np.inf

        min_seg_samples = int(round(min_segment_s * fs_in)) if min_segment_s > 0.0 else 0

        for seg in contiguous_traces:
            for finite_seg in split_finite_segments(seg):
                if min_seg_samples > 0 and finite_seg.stats.npts < min_seg_samples:
                    dropped_short += 1
                    continue
                y, k_out_start = process_contiguous_segment(
                    finite_seg.data,
                    UTCDateTime(finite_seg.stats.starttime),
                    fs_in=fs_in,
                    fs_out=fs_out,
                    pz=pz,
                    pre_filt=pre_filt,
                    water_level_db=water_level_db,
                    edge_taper_s=edge_taper_s,
                )
                if y.size == 0:
                    continue

                k_out_end = k_out_start + y.size
                clip_start = max(k_out_start, k_day_start)
                clip_end = min(k_out_end, k_day_end)
                if clip_end <= clip_start:
                    continue

                seg_day = y[clip_start - k_out_start : clip_end - k_out_start]
                if not np.all(np.isfinite(seg_day)):
                    raise ValueError(
                        f"Non-finite values produced after processing {utc_date.isoformat()} {channel}."
                    )

                idx_in_day = clip_start - k_day_start
                tr_starttime = day_start + (idx_in_day / fs_out)

                # Verify grid alignment residual (same formula as read_sds_day_zz)
                offset_samples = (
                    float(tr_starttime.timestamp) - float(day_start.timestamp)
                ) * fs_out
                residual = abs(offset_samples - round(offset_samples))
                max_grid_residual = max(max_grid_residual, residual)

                seg_f32 = np.asarray(seg_day, dtype=np.float32)
                tr_out = Trace(
                    data=seg_f32,
                    header={
                        "network": net,
                        "station": station_out,
                        "location": loc,
                        "channel": channel,
                        "sampling_rate": fs_out,
                        "starttime": tr_starttime,
                    },
                )
                out_traces.append(tr_out)

                valid_samples_out += seg_f32.size
                seg_f64 = seg_f32.astype(np.float64)
                proc_sum_sq += float(np.dot(seg_f64, seg_f64))
                proc_min = min(proc_min, float(np.min(seg_f64)))
                proc_max = max(proc_max, float(np.max(seg_f64)))

        qc.dropped_short_segments = dropped_short
        qc.out_n_traces = len(out_traces)

        if not out_traces:
            qc.status = "no_data"
            skip_anoms: list[str] = []
            if dropped_short > 0:
                skip_anoms.append("SHORT_SEGMENTS_ONLY")
                if qc.raw_n_segments > EXCESSIVE_FRAGMENTATION_THRESHOLD:
                    skip_anoms.append("EXCESSIVE_FRAGMENTATION")
                logger.warning(
                    "SKIP_SHORT %s %s.%s.%s.%s | dropped=%d segment(s) shorter than %.1fs (raw_segs=%d)",
                    utc_date.isoformat(),
                    net,
                    station_out,
                    loc,
                    channel,
                    dropped_short,
                    min_segment_s,
                    qc.raw_n_segments,
                )
            else:
                skip_anoms.append("NO_DATA")
            qc.anomalies = ";".join(skip_anoms)
            return qc

        out_stream = Stream(traces=out_traces)
        out_stream.sort(keys=["starttime"])
        write_mseed_atomic(out_stream, out_path)

        qc.day_coverage_pct = 100.0 * valid_samples_out / float(day_samples)
        qc.out_grid_residual_samples = max_grid_residual
        qc.proc_rms = float(np.sqrt(proc_sum_sq / valid_samples_out))
        qc.proc_ptp = float(proc_max - proc_min)
        qc.status = "ok"

        anomalies: list[str] = []
        if qc.fs_anomaly:
            anomalies.append("FS_ANOMALY")
        if qc.gap_count > 0:
            anomalies.append("GAP")
        if qc.overlap_count > 0:
            anomalies.append("OVERLAP")
        if qc.frac_offset_jump:
            anomalies.append("FRAC_OFFSET_JUMP")
        if qc.non_finite_count > 0:
            anomalies.append("NON_FINITE")
        if qc.clip_count > 0:
            anomalies.append("CLIPPING")
        if np.isfinite(qc.zero_fraction) and qc.zero_fraction > 0.05:
            anomalies.append("HIGH_ZERO_FRACTION")
        if qc.day_coverage_pct < 99.0:
            anomalies.append("LOW_COVERAGE")
        if qc.gps_lock == "F":
            anomalies.append("GPS_UNLOCK")
        if qc.raw_n_segments > EXCESSIVE_FRAGMENTATION_THRESHOLD:
            anomalies.append("EXCESSIVE_FRAGMENTATION")
        qc.anomalies = ";".join(anomalies)

        log_msg = (
            "OK %s %s.%s.%s.%s | files=%d cov=%.2f%% gaps=%d(%.2fs) "
            "overlaps=%d(%.2fs) frac=%.3fms res=%.2e rms=%.3e->%.3e anomalies=[%s]"
        )
        log_args = (
            utc_date.isoformat(),
            net,
            station_out,
            loc,
            channel,
            qc.n_source_files,
            qc.day_coverage_pct,
            qc.gap_count,
            qc.gap_total_s,
            qc.overlap_count,
            qc.overlap_total_s,
            qc.frac_sample_s * 1e3,
            qc.out_grid_residual_samples,
            qc.raw_rms,
            qc.proc_rms,
            qc.anomalies or "NONE",
        )
        if anomalies:
            logger.warning(log_msg, *log_args)
        else:
            logger.info(log_msg, *log_args)

        return qc

    except Exception as exc:
        qc.status = "error"
        qc.error_message = str(exc)
        qc.anomalies = "ERROR"
        logger.exception(
            "FAILED %s %s.%s.%s.%s: %s",
            utc_date.isoformat(),
            net,
            station_out,
            loc,
            channel,
            exc,
        )
        return qc


# ---------------------------------------------------------------------------
# QC & Station-table writers
# ---------------------------------------------------------------------------


def write_qc_csv(records: Sequence[DayChannelQC], path: Path) -> None:
    """Atomically write QC records as CSV."""
    fieldnames = list(DayChannelQC.__dataclass_fields__.keys())
    lines: list[str] = []
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        row = asdict(rec)
        for k, v in row.items():
            if isinstance(v, float) and not np.isfinite(v):
                row[k] = ""
        writer.writerow(row)
    lines.append(buf.getvalue())
    write_text_atomic("".join(lines), path)


def write_qc_jsonl(records: Sequence[DayChannelQC], path: Path) -> None:
    """Atomically write QC records as JSON Lines."""
    out_lines: list[str] = []
    for rec in records:
        row = asdict(rec)
        for k, v in row.items():
            if isinstance(v, float) and not np.isfinite(v):
                row[k] = None
        out_lines.append(json.dumps(row, ensure_ascii=False))
    write_text_atomic("\n".join(out_lines) + ("\n" if out_lines else ""), path)


def write_station_table_snippet(
    *,
    path: Path,
    network: str,
    station: str,
    location: str,
    channels: Sequence[str],
    latitude_deg: float,
    longitude_deg: float,
    elevation_m: float,
    response_state: str,
    instrument_model: str,
) -> None:
    """Write a station-table CSV snippet directly readable by ``load_station_table``."""
    import io

    fieldnames = [
        "network",
        "station",
        "location",
        "channel",
        "latitude_deg",
        "longitude_deg",
        "elevation_m",
        "response_state",
        "instrument_model",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for cha in channels:
        writer.writerow(
            {
                "network": network,
                "station": station,
                "location": location,
                "channel": cha,
                "latitude_deg": f"{latitude_deg:.7f}",
                "longitude_deg": f"{longitude_deg:.7f}",
                "elevation_m": f"{elevation_m:.3f}",
                "response_state": response_state,
                "instrument_model": instrument_model,
            }
        )
    write_text_atomic(buf.getvalue(), path)


# ---------------------------------------------------------------------------
# CLI & main entry point
# ---------------------------------------------------------------------------


def _resolve_dates(
    *,
    explicit_dates: Sequence[str] | None,
    start_date: str | None,
    end_date: str | None,
    entries: Sequence[MseedIndexEntry],
) -> list[date]:
    if explicit_dates and (start_date or end_date):
        raise ValueError("Specify either --date or --start-date/--end-date, not both.")
    if explicit_dates:
        parsed = sorted({date.fromisoformat(d) for d in explicit_dates})
        return parsed
    if start_date or end_date:
        min_d = min(e.starttime.date for e in entries)
        max_d = max(e.endtime.date for e in entries)
        d0 = date.fromisoformat(start_date) if start_date else min_d
        d1 = date.fromisoformat(end_date) if end_date else max_d
        if d1 < d0:
            raise ValueError(f"--end-date ({d1}) precedes --start-date ({d0}).")
        days: list[date] = []
        cur = d0
        while cur <= d1:
            days.append(cur)
            cur += timedelta(days=1)
        return days

    min_d = min(e.starttime.date for e in entries)
    max_d = max(e.endtime.date for e in entries)
    days = []
    cur = min_d
    while cur <= max_d:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _setup_logger(log_file: Path | None, level_name: str) -> logging.Logger:
    logger = logging.getLogger(f"prep_seis_s20_sds.{os.getpid()}")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess one SEIS-S20 station directory into standard daily FLOAT32 "
            "SDS MiniSEED files compatible with seisforge ant ccf."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "station_dir",
        type=Path,
        help="Path to one raw SEIS-S20 station directory (e.g. .../2164-0519).",
    )
    parser.add_argument(
        "--out-sds",
        type=Path,
        required=True,
        help="Output SDS root directory.",
    )
    parser.add_argument(
        "--pz",
        type=Path,
        default=DEFAULT_PZ_PATH,
        help="Path to SEIS-S20 SAC Pole-Zero file (SAC 'to none' velocity convention).",
    )
    parser.add_argument(
        "--no-remove-response",
        action="store_true",
        help="Skip instrument-response removal (keeps gain_only state for comparison).",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=None,
        help="Output station code (default: prefix before '-' of station_dir name).",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Override output network code (default: keep raw MiniSEED network).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Override output location code (default: keep raw MiniSEED location).",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Channel code(s) to process (e.g. BHZ or BHE BHN BHZ; default: all found).",
    )
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        default=None,
        help="Target UTC date YYYY-MM-DD (repeatable).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive start UTC date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive end UTC date YYYY-MM-DD.",
    )
    parser.add_argument(
        "--fs-out",
        type=float,
        default=DEFAULT_FS_OUT,
        help="Target output sampling rate in Hz.",
    )
    parser.add_argument(
        "--expected-fs-in",
        type=float,
        default=DEFAULT_FS_IN,
        help="Expected raw input sampling rate in Hz for anomaly detection.",
    )
    parser.add_argument(
        "--pre-filt",
        nargs=4,
        type=float,
        metavar=("F1", "F2", "F3", "F4"),
        default=None,
        help=(
            "Four-corner cosine pre-filter frequencies in Hz "
            "(default: 0.05 0.10 0.40*fs_out 0.45*fs_out)."
        ),
    )
    parser.add_argument(
        "--water-level",
        type=float,
        default=DEFAULT_WATER_LEVEL_DB,
        help="Water level in dB for response deconvolution (<=0 disables).",
    )
    parser.add_argument(
        "--pad-s",
        type=float,
        default=DEFAULT_PAD_S,
        help="Cross-midnight context padding in seconds on both sides of each UTC day.",
    )
    parser.add_argument(
        "--edge-taper-s",
        type=float,
        default=DEFAULT_EDGE_TAPER_S,
        help="Outer cosine taper duration in seconds (capped at 5%% of segment length).",
    )
    parser.add_argument(
        "--min-segment-s",
        type=float,
        default=DEFAULT_MIN_SEGMENT_S,
        help=(
            "Drop contiguous raw segments shorter than this duration in seconds before "
            "FFT/resampling to avoid wasting time on heavily fragmented bad channels "
            "(default: 60.0; <=0 disables)."
        ),
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=DEFAULT_CLIP_THRESHOLD,
        help="Raw counts threshold for clipping detection (<=0 disables).",
    )
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=None,
        help="Output QC CSV path (default: <out-sds>/_qc/<station>_qc.csv).",
    )
    parser.add_argument(
        "--qc-jsonl",
        type=Path,
        default=None,
        help="Optional output QC JSONL path.",
    )
    parser.add_argument(
        "--station-table-snippet",
        type=Path,
        default=None,
        help=(
            "Output station-table CSV snippet path "
            "(default: <out-sds>/_qc/<station>_station.csv when coordinates are available)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: <out-sds>/_logs/<station>.log).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing daily SDS MiniSEED files.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    station_dir: Path = args.station_dir.resolve()
    if not station_dir.is_dir():
        print(f"ERROR: station_dir does not exist or is not a directory: {station_dir}", file=sys.stderr)
        return 1

    station_out = (args.station or infer_station_code(station_dir)).strip()
    if not station_out:
        print("ERROR: Could not determine non-empty station code.", file=sys.stderr)
        return 1

    out_sds: Path = args.out_sds.resolve()
    log_file = args.log_file if args.log_file is not None else (out_sds / "_logs" / f"{station_out}.log")
    logger = _setup_logger(log_file, args.log_level)

    if len(station_out) > 5:
        logger.warning(
            "Output station code %r exceeds 5 ASCII characters (MiniSEED header limit).",
            station_out,
        )

    if args.fs_out <= 0 or not np.isfinite(args.fs_out):
        logger.error("--fs-out must be finite and positive, got %s", args.fs_out)
        return 1
    if args.pad_s < 0 or not np.isfinite(args.pad_s):
        logger.error("--pad-s must be finite and non-negative, got %s", args.pad_s)
        return 1

    # Load PZ response unless --no-remove-response is set.
    pz: PoleZeroResponse | None = None
    pre_filt: tuple[float, float, float, float] | None = None
    if not args.no_remove_response:
        if args.pz is None:
            logger.error("Provide --pz <SAC_PZ_FILE> or pass --no-remove-response.")
            return 1
        try:
            pz = parse_sac_pz(args.pz)
        except Exception as exc:
            logger.error("Failed to load SAC PZ file %s: %s", args.pz, exc)
            return 1
        if args.pre_filt is not None:
            pre_filt = tuple(float(v) for v in args.pre_filt)  # type: ignore[assignment]
        else:
            pre_filt = (0.05, 0.10, 0.40 * args.fs_out, 0.45 * args.fs_out)
        f1, f2, f3, f4 = pre_filt
        if not (0.0 < f1 < f2 < f3 < f4):
            logger.error("Invalid --pre-filt frequencies (must satisfy 0 < f1 < f2 < f3 < f4): %s", pre_filt)
            return 1
        if f4 > 0.5 * args.fs_out + 1e-12:
            logger.warning(
                "pre_filt f4=%.4f Hz exceeds output Nyquist %.4f Hz; resample_poly will low-pass above Nyquist.",
                f4,
                0.5 * args.fs_out,
            )
    elif args.pre_filt is not None:
        pre_filt = tuple(float(v) for v in args.pre_filt)  # type: ignore[assignment]
        f1, f2, f3, f4 = pre_filt
        if not (0.0 < f1 < f2 < f3 < f4):
            logger.error("Invalid --pre-filt frequencies: %s", pre_filt)
            return 1

    logger.info(
        "Starting SEIS-S20 Stage-A prep | station_dir=%s | station_out=%s | "
        "fs_out=%.3f Hz | remove_response=%s | pre_filt=%s | pad_s=%.1f",
        station_dir,
        station_out,
        args.fs_out,
        pz is not None,
        pre_filt,
        args.pad_s,
    )

    # 1. Discover & index MiniSEED headers
    mseed_files = discover_mseed_files(station_dir)
    if not mseed_files:
        logger.error("No candidate MiniSEED files found in %s", station_dir)
        return 1

    entries, skipped_files = build_mseed_index(mseed_files, logger)
    if not entries:
        logger.error("No valid MiniSEED traces indexed from %d files in %s", len(mseed_files), station_dir)
        return 1

    raw_stations = sorted({e.station_raw for e in entries})
    discovered_channels = sorted({e.channel for e in entries})
    earliest_start = min(e.starttime for e in entries)
    latest_end = max(e.endtime for e in entries)
    logger.info(
        "Indexed %d trace headers from %d files (skipped %d) | raw_station=%s | "
        "channels=%s | span=%s .. %s",
        len(entries),
        len(mseed_files) - skipped_files,
        skipped_files,
        raw_stations,
        discovered_channels,
        earliest_start.isoformat(),
        latest_end.isoformat(),
    )

    # 2. Parse optional Log* files
    log_summary = parse_station_logs(station_dir, earliest_start, logger)
    if log_summary.selected_start is not None:
        sel = log_summary.selected_start
        logger.info(
            "Log metadata (%d start entries, selected %s from %s): "
            "GPS_Lock=%s SPS=%d lat=%.7f lon=%.7f elev=%.3fm | daily_status_days=%d",
            len(log_summary.sampling_starts),
            sel.timestamp_utc.isoformat(),
            sel.source_file,
            "T" if sel.gps_lock else "F",
            sel.sps,
            sel.latitude_deg,
            sel.longitude_deg,
            sel.elevation_m,
            len(log_summary.daily_status),
        )
        if not sel.gps_lock:
            logger.warning("GPS_Lock is FALSE in selected log entry for station %s!", station_out)
    else:
        logger.warning("No SEIS-S20 Log* metadata found under %s", station_dir)

    # 3. Resolve channels and dates
    if args.channels:
        requested_channels: list[str] = []
        for token in args.channels:
            for part in token.split(","):
                cleaned = part.strip()
                if cleaned:
                    requested_channels.append(cleaned)
        channels = [c for c in dict.fromkeys(requested_channels) if c in discovered_channels]
        missing_ch = sorted(set(requested_channels) - set(discovered_channels))
        if missing_ch:
            logger.warning("Requested channel(s) not found in raw archive: %s", missing_ch)
        if not channels:
            logger.error("None of the requested channels %s exist in %s", requested_channels, discovered_channels)
            return 1
    else:
        channels = discovered_channels

    try:
        target_dates = _resolve_dates(
            explicit_dates=args.dates,
            start_date=args.start_date,
            end_date=args.end_date,
            entries=entries,
        )
    except ValueError as exc:
        logger.error("Invalid date selection: %s", exc)
        return 1

    logger.info(
        "Processing %d UTC day(s) [%s .. %s] x %d channel(s) %s",
        len(target_dates),
        target_dates[0].isoformat(),
        target_dates[-1].isoformat(),
        len(channels),
        channels,
    )

    water_level_db = args.water_level if (args.water_level is not None and args.water_level > 0) else None
    clip_threshold = args.clip_threshold if (args.clip_threshold is not None and args.clip_threshold > 0) else None

    # 4. Run per-day, per-channel processing
    qc_records: list[DayChannelQC] = []
    for utc_day in target_dates:
        for cha in channels:
            rec = process_day_channel(
                utc_date=utc_day,
                channel=cha,
                index_entries=entries,
                sds_root=out_sds,
                station_out=station_out,
                network_override=args.network,
                location_override=args.location,
                fs_out=args.fs_out,
                expected_fs_in=args.expected_fs_in,
                pz=pz,
                pre_filt=pre_filt,
                water_level_db=water_level_db,
                pad_s=args.pad_s,
                edge_taper_s=args.edge_taper_s,
                min_segment_s=args.min_segment_s,
                clip_threshold=clip_threshold,
                overwrite=args.overwrite,
                log_summary=log_summary,
                instrument_model=DEFAULT_INSTRUMENT_MODEL,
                logger=logger,
            )
            qc_records.append(rec)

    # 5. Write QC table(s) & optional station-table snippet
    all_skipped = bool(qc_records) and all(r.status == "skipped_exists" for r in qc_records)
    qc_csv_path = args.qc_csv if args.qc_csv is not None else (out_sds / "_qc" / f"{station_out}_qc.csv")
    if all_skipped and qc_csv_path.exists() and not args.overwrite:
        logger.info(
            "All %d day/channel outputs already exist; preserving existing QC CSV: %s",
            len(qc_records),
            qc_csv_path,
        )
    else:
        write_qc_csv(qc_records, qc_csv_path)
        logger.info("Wrote QC CSV (%d rows): %s", len(qc_records), qc_csv_path)

    if args.qc_jsonl is not None:
        if all_skipped and args.qc_jsonl.exists() and not args.overwrite:
            logger.info(
                "All %d day/channel outputs already exist; preserving existing QC JSONL: %s",
                len(qc_records),
                args.qc_jsonl,
            )
        else:
            write_qc_jsonl(qc_records, args.qc_jsonl)
            logger.info("Wrote QC JSONL (%d rows): %s", len(qc_records), args.qc_jsonl)

    if log_summary.selected_start is not None:
        snippet_path = (
            args.station_table_snippet
            if args.station_table_snippet is not None
            else (out_sds / "_qc" / f"{station_out}_station.csv")
        )
        ref_net = args.network if args.network is not None else entries[0].network
        ref_loc = args.location if args.location is not None else entries[0].location
        write_station_table_snippet(
            path=snippet_path,
            network=ref_net,
            station=station_out,
            location=ref_loc,
            channels=channels,
            latitude_deg=log_summary.selected_start.latitude_deg,
            longitude_deg=log_summary.selected_start.longitude_deg,
            elevation_m=log_summary.selected_start.elevation_m,
            response_state="velocity" if pz is not None else "gain_only",
            instrument_model=DEFAULT_INSTRUMENT_MODEL,
        )
        logger.info("Wrote station table snippet: %s", snippet_path)
    elif args.station_table_snippet is not None:
        logger.warning(
            "Skipping --station-table-snippet (%s) because no GPS coordinates were found in Log*.",
            args.station_table_snippet,
        )

    n_ok = sum(1 for r in qc_records if r.status == "ok")
    n_skip = sum(1 for r in qc_records if r.status == "skipped_exists")
    n_nodata = sum(1 for r in qc_records if r.status == "no_data")
    n_err = sum(1 for r in qc_records if r.status == "error")
    n_anom = sum(1 for r in qc_records if r.anomalies)
    logger.info(
        "Completed station %s | ok=%d skipped=%d no_data=%d errors=%d with_anomalies=%d",
        station_out,
        n_ok,
        n_skip,
        n_nodata,
        n_err,
        n_anom,
    )

    return 2 if n_err > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
