"""Configuration schemas, strict input contracts, and YAML loaders for CCF."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
import difflib
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .core import minimum_nfft
from .models import CC_TYPES, STACK_METHODS, WHITEN_MODES, BandpassSpec, WhitenSpec
from .stacking import DAY_WEIGHTING_MODES

__all__ = [
    "BACKENDS",
    "RESPONSE_STATES",
    "SPECTRUM_DTYPES",
    "CCFProcessingConfig",
    "CCFRunConfig",
    "SDSInputConfig",
    "StackRunConfig",
    "StationRecord",
    "load_ccf_config",
    "load_stack_config",
    "parse_date_list",
]

RESPONSE_STATES = ("gain_only", "velocity", "displacement", "acceleration")
SPECTRUM_DTYPES = ("complex64", "complex128")
BACKENDS = ("numpy", "cupy")


def _check_unknown_keys(
    mapping: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if not unknown:
        return
    hints = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(allowed), n=1)
        if close:
            hints.append(f"{section}.{key} (did you mean {section}.{close[0]}?)")
        else:
            hints.append(f"{section}.{key}")
    raise ValueError(f"Unknown configuration key(s): {', '.join(hints)}.")


def parse_date_list(
    *,
    dates: list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str, ...]:
    """Parse explicit ISO dates or an inclusive ``[start_date, end_date]`` range."""
    if dates is not None and (start_date is not None or end_date is not None):
        raise ValueError("Provide either 'dates' or 'start_date'/'end_date', not both.")
    if dates is not None:
        parsed = tuple(date.fromisoformat(str(d)).isoformat() for d in dates)
        if not parsed:
            raise ValueError("dates must not be empty.")
        return tuple(dict.fromkeys(parsed))
    if start_date is None or end_date is None:
        raise ValueError("Provide either 'dates' or both 'start_date' and 'end_date'.")
    d0 = date.fromisoformat(str(start_date))
    d1 = date.fromisoformat(str(end_date))
    if d1 < d0:
        raise ValueError(f"end_date ({d1}) precedes start_date ({d0}).")
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return tuple(out)


@dataclass(frozen=True)
class StationRecord:
    """One station-channel entry from the deployment station table."""

    network: str
    station: str
    location: str
    channel: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0
    response_state: str | None = None
    instrument_model: str = ""

    def __post_init__(self) -> None:
        if not self.network.strip() or not self.station.strip() or not self.channel.strip():
            raise ValueError("network, station, and channel must be non-empty strings.")
        if not (-90.0 <= self.latitude_deg <= 90.0) or not np.isfinite(self.latitude_deg):
            raise ValueError(f"Invalid latitude_deg for {self.seed_id}: {self.latitude_deg}.")
        if not (-180.0 <= self.longitude_deg <= 360.0) or not np.isfinite(self.longitude_deg):
            raise ValueError(f"Invalid longitude_deg for {self.seed_id}: {self.longitude_deg}.")
        if not np.isfinite(self.elevation_m):
            raise ValueError(f"Invalid elevation_m for {self.seed_id}: {self.elevation_m}.")
        if self.response_state is not None and self.response_state not in RESPONSE_STATES:
            raise ValueError(
                f"response_state for {self.seed_id} must be one of {RESPONSE_STATES}, "
                f"got {self.response_state!r}."
            )

    @property
    def station_id(self) -> str:
        return f"{self.network}.{self.station}"

    @property
    def seed_id(self) -> str:
        return f"{self.network}.{self.station}.{self.location}.{self.channel}"


@dataclass(frozen=True)
class SDSInputConfig:
    """Explicit input contract for pre-prepared standard SDS daily MiniSEED data.

    Because MiniSEED headers cannot encode instrument-response state, effective
    passband, or resampling filter transients, callers declare these explicitly
    so the CCF stage can refuse mismatched inputs rather than silently mixing
    phases or whitening outside the valid band.
    """

    sds_root: Path
    station_table: Path
    sampling_rate_hz: float
    response_state: str
    valid_passband_hz: tuple[float, float] | None = None
    grid_tolerance_samples: float = 1e-3  # assumed default
    gap_guard_s: float = 0.0
    allow_missing_next_day: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "sds_root", Path(self.sds_root))
        object.__setattr__(self, "station_table", Path(self.station_table))
        if self.sampling_rate_hz <= 0 or not np.isfinite(self.sampling_rate_hz):
            raise ValueError("sampling_rate_hz must be finite and positive.")
        if self.response_state not in RESPONSE_STATES:
            raise ValueError(
                f"response_state must be one of {RESPONSE_STATES}, got {self.response_state!r}."
            )
        if self.valid_passband_hz is not None:
            if len(self.valid_passband_hz) != 2:
                raise ValueError("valid_passband_hz must be a 2-tuple (f_min_hz, f_max_hz).")
            f_lo, f_hi = float(self.valid_passband_hz[0]), float(self.valid_passband_hz[1])
            nyquist_hz = 0.5 * float(self.sampling_rate_hz)
            if not (0.0 <= f_lo < f_hi <= nyquist_hz + 1e-12):
                raise ValueError(
                    f"valid_passband_hz ({f_lo}, {f_hi}) must satisfy "
                    f"0 <= f_min < f_max <= Nyquist ({nyquist_hz} Hz)."
                )
            object.__setattr__(self, "valid_passband_hz", (f_lo, f_hi))
        if self.grid_tolerance_samples <= 0 or self.grid_tolerance_samples >= 0.5:
            raise ValueError("grid_tolerance_samples must lie in (0, 0.5).")
        if self.gap_guard_s < 0 or not np.isfinite(self.gap_guard_s):
            raise ValueError("gap_guard_s must be finite and non-negative.")


@dataclass(frozen=True)
class CCFProcessingConfig:
    """Validated scientific parameters for ZZ cross-correlation."""

    cc_type: str = "cc"
    target_sampling_rate_hz: float | None = None
    window_length_s: float = 500.0
    step_s: float | None = None
    max_lag_s: float = 60.0
    lag_guard_s: float = 20.0
    ram_half_width_s: float | None = None
    ram_bandpass: BandpassSpec | None = None
    pcc2_bandpass: BandpassSpec | None = None
    whiten: WhitenSpec | None = None
    whiten_mode: str = "vector_norm"
    taper_fraction: float = 0.05  # assumed default
    circular: bool = False
    nfft: int | None = None
    include_autocorr: bool = True
    max_distance_km: float | None = None
    block_size: int = 64  # assumed default
    spectrum_dtype: str = "complex64"

    def __post_init__(self) -> None:
        if self.cc_type not in CC_TYPES:
            raise ValueError(f"cc_type must be one of {CC_TYPES}, got {self.cc_type!r}.")
        if self.target_sampling_rate_hz is not None and (
            self.target_sampling_rate_hz <= 0 or not np.isfinite(self.target_sampling_rate_hz)
        ):
            raise ValueError("target_sampling_rate_hz must be finite and positive when set.")
        if self.window_length_s <= 0 or not np.isfinite(self.window_length_s):
            raise ValueError("window_length_s must be finite and positive.")
        if self.step_s is not None and (self.step_s <= 0 or not np.isfinite(self.step_s)):
            raise ValueError("step_s must be finite and positive when set.")
        if self.max_lag_s <= 0 or not np.isfinite(self.max_lag_s):
            raise ValueError("max_lag_s must be finite and positive.")
        if self.lag_guard_s < 0 or not np.isfinite(self.lag_guard_s):
            raise ValueError("lag_guard_s must be finite and non-negative.")
        if self.ram_half_width_s is not None and (
            self.ram_half_width_s < 0 or not np.isfinite(self.ram_half_width_s)
        ):
            raise ValueError("ram_half_width_s must be finite and non-negative.")
        if self.whiten_mode not in WHITEN_MODES:
            raise ValueError(
                f"whiten_mode must be one of {WHITEN_MODES}, got {self.whiten_mode!r}."
            )
        if not (0.0 <= self.taper_fraction <= 0.5):
            raise ValueError("taper_fraction must lie in [0.0, 0.5].")
        if self.max_distance_km is not None and (
            self.max_distance_km <= 0 or not np.isfinite(self.max_distance_km)
        ):
            raise ValueError("max_distance_km must be finite and positive when set.")
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1.")
        if self.spectrum_dtype not in SPECTRUM_DTYPES:
            raise ValueError(
                f"spectrum_dtype must be one of {SPECTRUM_DTYPES}, got {self.spectrum_dtype!r}."
            )

    def resolved_step_s(self) -> float:
        return self.window_length_s if self.step_s is None else float(self.step_s)


@dataclass(frozen=True)
class CCFRunConfig:
    """Complete validated configuration for a daily ZZ CCF run."""

    input: SDSInputConfig
    processing: CCFProcessingConfig
    output_dir: Path
    dates: tuple[str, ...] = field(default_factory=tuple)
    overwrite: bool = False
    compression_level: int = 4  # assumed default
    max_workers: int = 1
    backend: str = "numpy"

    def effective_sampling_rate_hz(self) -> float:
        """Return the working sampling rate used for correlation and output."""
        if self.processing.target_sampling_rate_hz is not None:
            return float(self.processing.target_sampling_rate_hz)
        return float(self.input.sampling_rate_hz)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.dates:
            raise ValueError("dates must contain at least one UTC date.")
        if not (0 <= self.compression_level <= 9):
            raise ValueError("compression_level must lie in [0, 9].")
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}.")

        input_fs = float(self.input.sampling_rate_hz)
        proc = self.processing
        if proc.target_sampling_rate_hz is not None:
            target_fs = float(proc.target_sampling_rate_hz)
            if target_fs > input_fs + 1e-12:
                raise ValueError(
                    f"processing.target_sampling_rate_hz ({target_fs} Hz) cannot exceed "
                    f"input.sampling_rate_hz ({input_fs} Hz); upsampling is not supported."
                )
            ratio = Fraction(target_fs / input_fs).limit_denominator(1000)
            if not np.isclose(input_fs * ratio.numerator / ratio.denominator, target_fs, rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"processing.target_sampling_rate_hz ({target_fs} Hz) is not a clean "
                    f"rational factor of input.sampling_rate_hz ({input_fs} Hz)."
                )

        fs = self.effective_sampling_rate_hz()
        nyquist_hz = 0.5 * fs
        win_samples = int(round(proc.window_length_s * fs))
        max_lag_samples = int(round(proc.max_lag_s * fs))
        guard_samples = int(round(proc.lag_guard_s * fs))

        if proc.nfft is not None:
            min_req = minimum_nfft(
                win_samples,
                max_lag_samples,
                guard_samples,
                circular=proc.circular,
                next_fast=False,
            )
            if proc.circular and proc.nfft != win_samples:
                raise ValueError("circular=True requires nfft == window_samples.")
            if not proc.circular and proc.nfft < min_req:
                raise ValueError(
                    f"nfft ({proc.nfft}) is smaller than minimum wrap-around-free length "
                    f"({min_req}) for window_length_s + max_lag_s + lag_guard_s."
                )

        if proc.whiten is not None:
            if proc.whiten.taper_max_hz > nyquist_hz + 1e-12:
                raise ValueError(
                    f"whiten.taper_max_hz ({proc.whiten.taper_max_hz}) exceeds Nyquist "
                    f"({nyquist_hz} Hz)."
                )
            if self.input.valid_passband_hz is not None:
                f_lo, f_hi = self.input.valid_passband_hz
                if proc.whiten.taper_min_hz < f_lo - 1e-12 or proc.whiten.taper_max_hz > f_hi + 1e-12:
                    raise ValueError(
                        f"Whitening band [{proc.whiten.taper_min_hz}, {proc.whiten.taper_max_hz}] Hz "
                        f"falls outside declared input valid_passband_hz [{f_lo}, {f_hi}] Hz."
                    )

        for label, bp in (
            ("ram_bandpass", proc.ram_bandpass),
            ("pcc2_bandpass", proc.pcc2_bandpass),
        ):
            if bp is not None and bp.freq_max_hz >= nyquist_hz:
                raise ValueError(
                    f"{label}.freq_max_hz ({bp.freq_max_hz}) must be below Nyquist ({nyquist_hz} Hz)."
                )

    def parameter_hash(self) -> str:
        """Deterministic 16-hex-char hash of scientific parameters for resume checks.

        Runtime-only knobs (``overwrite``, ``compression_level``, ``max_workers``,
        ``block_size``, ``backend``) are intentionally excluded so changing
        parallelism or block size does not invalidate cached daily products.
        """
        proc_dict = asdict(self.processing)
        proc_dict.pop("block_size", None)
        payload = {
            "sampling_rate_hz": self.input.sampling_rate_hz,
            "response_state": self.input.response_state,
            "valid_passband_hz": self.input.valid_passband_hz,
            "grid_tolerance_samples": self.input.grid_tolerance_samples,
            "gap_guard_s": self.input.gap_guard_s,
            "processing": proc_dict,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()[:16]


@dataclass(frozen=True)
class StackRunConfig:
    """Validated configuration for multi-day ZZ stacking."""

    daily_dir: Path
    output_file: Path
    dates: tuple[str, ...] | None = None
    method: str = "linear"
    day_weighting: str = "windows"
    pws_power: float = 2.0
    tf_pws_freq_min_hz: float | None = None
    tf_pws_freq_max_hz: float | None = None
    tf_pws_sigma: float = 1.0
    tf_pws_smooth_samples: int = 0
    tf_pws_normalize_weights: bool = False
    tf_pws_pair_chunk_size: int = 64  # assumed default
    tf_pws_dtype: str = "complex128"
    trim_guard: bool = False
    max_workers: int = 1
    backend: str = "numpy"
    overwrite: bool = False
    compression_level: int = 4  # assumed default

    def __post_init__(self) -> None:
        object.__setattr__(self, "daily_dir", Path(self.daily_dir))
        object.__setattr__(self, "output_file", Path(self.output_file))
        if self.method not in STACK_METHODS:
            raise ValueError(f"method must be one of {STACK_METHODS}, got {self.method!r}.")
        if self.day_weighting not in DAY_WEIGHTING_MODES:
            raise ValueError(
                f"day_weighting must be one of {DAY_WEIGHTING_MODES}, "
                f"got {self.day_weighting!r}."
            )
        if self.pws_power <= 0 or not np.isfinite(self.pws_power):
            raise ValueError("pws_power must be finite and positive.")
        if self.tf_pws_freq_min_hz is not None and (
            self.tf_pws_freq_min_hz < 0 or not np.isfinite(self.tf_pws_freq_min_hz)
        ):
            raise ValueError("tf_pws_freq_min_hz must be finite and non-negative.")
        if self.tf_pws_freq_max_hz is not None and (
            self.tf_pws_freq_max_hz <= 0 or not np.isfinite(self.tf_pws_freq_max_hz)
        ):
            raise ValueError("tf_pws_freq_max_hz must be finite and positive.")
        if (
            self.tf_pws_freq_min_hz is not None
            and self.tf_pws_freq_max_hz is not None
            and self.tf_pws_freq_max_hz <= self.tf_pws_freq_min_hz
        ):
            raise ValueError("tf_pws_freq_max_hz must be greater than tf_pws_freq_min_hz.")
        if self.tf_pws_sigma <= 0 or not np.isfinite(self.tf_pws_sigma):
            raise ValueError("tf_pws_sigma must be finite and positive.")
        if self.tf_pws_smooth_samples < 0:
            raise ValueError("tf_pws_smooth_samples must be non-negative.")
        if self.tf_pws_pair_chunk_size < 1:
            raise ValueError("tf_pws_pair_chunk_size must be at least 1.")
        if self.tf_pws_dtype not in SPECTRUM_DTYPES:
            raise ValueError(
                f"tf_pws_dtype must be one of {SPECTRUM_DTYPES}, got {self.tf_pws_dtype!r}."
            )
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}.")
        if not (0 <= self.compression_level <= 9):
            raise ValueError("compression_level must lie in [0, 9].")


def _parse_bandpass(raw: dict[str, Any] | None, section: str) -> BandpassSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be a mapping.")
    _check_unknown_keys(raw, {"freq_min_hz", "freq_max_hz", "order", "zerophase"}, section)
    return BandpassSpec(
        freq_min_hz=float(raw["freq_min_hz"]),
        freq_max_hz=float(raw["freq_max_hz"]),
        order=int(raw.get("order", 4)),
        zerophase=bool(raw.get("zerophase", True)),
    )


def _parse_whiten(raw: dict[str, Any] | None) -> WhitenSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("processing.whiten must be a mapping.")
    _check_unknown_keys(
        raw,
        {
            "freq_min_hz",
            "freq_max_hz",
            "taper_min_hz",
            "taper_max_hz",
            "smooth_half_width_hz",
            "water_level",
        },
        "processing.whiten",
    )
    return WhitenSpec(
        freq_min_hz=float(raw["freq_min_hz"]),
        freq_max_hz=float(raw["freq_max_hz"]),
        taper_min_hz=float(raw["taper_min_hz"]),
        taper_max_hz=float(raw["taper_max_hz"]),
        smooth_half_width_hz=float(raw.get("smooth_half_width_hz", 0.02)),
        water_level=float(raw.get("water_level", 1e-6)),
    )


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_ccf_config(config_path: str | Path) -> CCFRunConfig:
    """Load and validate a YAML configuration file for daily ZZ CCF computation."""
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("CCF config must be a YAML mapping.")
    _check_unknown_keys(raw, {"input", "processing", "output", "run"}, "root")

    inp = raw.get("input") or {}
    proc = raw.get("processing") or {}
    out = raw.get("output") or {}
    run = raw.get("run") or {}

    _check_unknown_keys(
        inp,
        {
            "sds_root",
            "station_table",
            "sampling_rate_hz",
            "response_state",
            "valid_passband_hz",
            "grid_tolerance_samples",
            "gap_guard_s",
            "allow_missing_next_day",
        },
        "input",
    )
    _check_unknown_keys(
        proc,
        {
            "cc_type",
            "target_sampling_rate_hz",
            "window_length_s",
            "step_s",
            "max_lag_s",
            "lag_guard_s",
            "ram_half_width_s",
            "ram_bandpass",
            "pcc2_bandpass",
            "whiten",
            "whiten_mode",
            "taper_fraction",
            "circular",
            "nfft",
            "include_autocorr",
            "max_distance_km",
            "block_size",
            "spectrum_dtype",
        },
        "processing",
    )
    _check_unknown_keys(out, {"dir", "overwrite", "compression_level"}, "output")
    _check_unknown_keys(
        run,
        {"dates", "start_date", "end_date", "max_workers", "backend"},
        "run",
    )

    base = path.parent
    valid_pb = inp.get("valid_passband_hz")
    sds_cfg = SDSInputConfig(
        sds_root=_resolve_path(inp["sds_root"], base),
        station_table=_resolve_path(inp["station_table"], base),
        sampling_rate_hz=float(inp["sampling_rate_hz"]),
        response_state=str(inp["response_state"]),
        valid_passband_hz=None if valid_pb is None else (float(valid_pb[0]), float(valid_pb[1])),
        grid_tolerance_samples=float(inp.get("grid_tolerance_samples", 1e-3)),
        gap_guard_s=float(inp.get("gap_guard_s", 0.0)),
        allow_missing_next_day=bool(inp.get("allow_missing_next_day", True)),
    )

    proc_cfg = CCFProcessingConfig(
        cc_type=str(proc.get("cc_type", "cc")),
        target_sampling_rate_hz=(
            None
            if proc.get("target_sampling_rate_hz") is None
            else float(proc["target_sampling_rate_hz"])
        ),
        window_length_s=float(proc.get("window_length_s", 500.0)),
        step_s=None if proc.get("step_s") is None else float(proc["step_s"]),
        max_lag_s=float(proc.get("max_lag_s", 60.0)),
        lag_guard_s=float(proc.get("lag_guard_s", 20.0)),
        ram_half_width_s=(
            None if proc.get("ram_half_width_s") is None else float(proc["ram_half_width_s"])
        ),
        ram_bandpass=_parse_bandpass(proc.get("ram_bandpass"), "processing.ram_bandpass"),
        pcc2_bandpass=_parse_bandpass(proc.get("pcc2_bandpass"), "processing.pcc2_bandpass"),
        whiten=_parse_whiten(proc.get("whiten")),
        whiten_mode=str(proc.get("whiten_mode", "vector_norm")),
        taper_fraction=float(proc.get("taper_fraction", 0.05)),
        circular=bool(proc.get("circular", False)),
        nfft=None if proc.get("nfft") is None else int(proc["nfft"]),
        include_autocorr=bool(proc.get("include_autocorr", True)),
        max_distance_km=(
            None if proc.get("max_distance_km") is None else float(proc["max_distance_km"])
        ),
        block_size=int(proc.get("block_size", 64)),
        spectrum_dtype=str(proc.get("spectrum_dtype", "complex64")),
    )

    dates = parse_date_list(
        dates=run.get("dates"),
        start_date=run.get("start_date"),
        end_date=run.get("end_date"),
    )

    return CCFRunConfig(
        input=sds_cfg,
        processing=proc_cfg,
        output_dir=_resolve_path(out["dir"], base),
        dates=dates,
        overwrite=bool(out.get("overwrite", False)),
        compression_level=int(out.get("compression_level", 4)),
        max_workers=int(run.get("max_workers", 1)),
        backend=str(run.get("backend", "numpy")),
    )


def load_stack_config(config_path: str | Path) -> StackRunConfig:
    """Load and validate a YAML configuration file for multi-day ZZ stacking."""
    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Stack config must be a YAML mapping.")
    _check_unknown_keys(raw, {"input", "stack", "output"}, "root")

    inp = raw.get("input") or {}
    stk = raw.get("stack") or {}
    out = raw.get("output") or {}

    _check_unknown_keys(inp, {"daily_dir", "dates", "start_date", "end_date"}, "input")
    _check_unknown_keys(
        stk,
        {
            "method",
            "day_weighting",
            "pws_power",
            "tf_pws_freq_min_hz",
            "tf_pws_freq_max_hz",
            "tf_pws_sigma",
            "tf_pws_smooth_samples",
            "tf_pws_normalize_weights",
            "tf_pws_pair_chunk_size",
            "tf_pws_dtype",
            "max_workers",
            "backend",
            "trim_guard",
        },
        "stack",
    )
    _check_unknown_keys(out, {"file", "overwrite", "compression_level"}, "output")

    dates: tuple[str, ...] | None = None
    if any(inp.get(k) is not None for k in ("dates", "start_date", "end_date")):
        dates = parse_date_list(
            dates=inp.get("dates"),
            start_date=inp.get("start_date"),
            end_date=inp.get("end_date"),
        )

    base = path.parent
    return StackRunConfig(
        daily_dir=_resolve_path(inp["daily_dir"], base),
        output_file=_resolve_path(out["file"], base),
        dates=dates,
        method=str(stk.get("method", "linear")),
        day_weighting=str(stk.get("day_weighting", "windows")),
        pws_power=float(stk.get("pws_power", 2.0)),
        tf_pws_freq_min_hz=(
            None if stk.get("tf_pws_freq_min_hz") is None else float(stk["tf_pws_freq_min_hz"])
        ),
        tf_pws_freq_max_hz=(
            None if stk.get("tf_pws_freq_max_hz") is None else float(stk["tf_pws_freq_max_hz"])
        ),
        tf_pws_sigma=float(stk.get("tf_pws_sigma", 1.0)),
        tf_pws_smooth_samples=int(stk.get("tf_pws_smooth_samples", 0)),
        tf_pws_normalize_weights=bool(stk.get("tf_pws_normalize_weights", False)),
        tf_pws_pair_chunk_size=int(stk.get("tf_pws_pair_chunk_size", 64)),
        tf_pws_dtype=str(stk.get("tf_pws_dtype", "complex128")),
        max_workers=int(stk.get("max_workers", 1)),
        backend=str(stk.get("backend", "numpy")),
        trim_guard=bool(stk.get("trim_guard", False)),
        overwrite=bool(out.get("overwrite", False)),
        compression_level=int(out.get("compression_level", 4)),
    )
