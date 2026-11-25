"""Orchestration drivers for daily ZZ CCF computation and multi-day stacking."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
import os
from pathlib import Path
import time

import numpy as np
from pyproj import Geod

from seisforge.provenance import software_provenance

from .config import (
    CCFRunConfig,
    StackRunConfig,
    StationRecord,
    load_ccf_config,
    load_stack_config,
)
from .core import correlate_zz_day, upper_triangle_pairs
from .io import (
    decimate_day_data,
    load_station_table,
    read_netcdf,
    read_netcdf_pair_slice,
    read_sds_day_zz,
    station_table_hash,
    write_netcdf,
)
from .models import StackedCCFResult
from .products import (
    daily_ccf_to_dataset,
    dataset_to_daily_ccf,
    stacked_ccf_to_dataset,
)
from .stacking import (
    LinearStackAccumulator,
    PWSStackAccumulator,
    TFPWSStackAccumulator,
    _resolve_stockwell_grid,
)

__all__ = [
    "CCFDayOutcome",
    "CCFRunSummary",
    "StackRunSummary",
    "daily_ccf_path",
    "run_ccf_config",
    "run_ccf_day",
    "run_stack_config",
]

_WGS84 = Geod(ellps="WGS84")


@dataclass(frozen=True)
class CCFDayOutcome:
    """Result summary for one UTC day in a CCF run."""

    utc_date: str
    output_file: Path
    status: str  # "computed" or "resumed"
    elapsed_s: float
    n_pairs: int
    n_active_pairs: int
    missing_stations: tuple[str, ...]
    next_day_missing_stations: tuple[str, ...]


@dataclass(frozen=True)
class CCFRunSummary:
    """Summary returned by :func:`run_ccf_config`."""

    config: CCFRunConfig
    outcomes: tuple[CCFDayOutcome, ...]
    log_file: Path


@dataclass(frozen=True)
class StackRunSummary:
    """Summary returned by :func:`run_stack_config`."""

    config: StackRunConfig
    output_file: Path
    dates_used: tuple[str, ...]
    n_pairs: int
    log_file: Path


def daily_ccf_path(output_dir: str | Path, utc_date: str) -> Path:
    """Canonical daily ZZ CCF NetCDF path for ``utc_date``."""
    return Path(output_dir) / f"ccf_ZZ_{utc_date}.nc"


def _filter_pair_indices(
    stations: Sequence[StationRecord],
    *,
    include_autocorr: bool,
    max_distance_km: float | None,
) -> np.ndarray:
    pairs = upper_triangle_pairs(len(stations), include_autocorr=include_autocorr)
    if max_distance_km is None or pairs.shape[0] == 0:
        return pairs

    src_lat = np.asarray([stations[i].latitude_deg for i in pairs[:, 0]], dtype=np.float64)
    src_lon = np.asarray([stations[i].longitude_deg for i in pairs[:, 0]], dtype=np.float64)
    rcv_lat = np.asarray([stations[j].latitude_deg for j in pairs[:, 1]], dtype=np.float64)
    rcv_lon = np.asarray([stations[j].longitude_deg for j in pairs[:, 1]], dtype=np.float64)
    _, _, dist_m = _WGS84.inv(src_lon, src_lat, rcv_lon, rcv_lat)
    dist_km = np.asarray(dist_m, dtype=np.float64) / 1000.0
    same = pairs[:, 0] == pairs[:, 1]
    dist_km = np.where(same, 0.0, dist_km)
    return pairs[dist_km <= float(max_distance_km)]


def _matches_resume_hashes(
    path: Path,
    *,
    param_hash: str,
    station_hash: str,
    input_fingerprint: str,
) -> bool:
    if not path.exists():
        return False
    try:
        ds = read_netcdf(path)
    except Exception:
        return False
    return (
        str(ds.attrs.get("stage", "")) == "ccf_daily"
        and str(ds.attrs.get("param_hash", "")) == param_hash
        and str(ds.attrs.get("station_list_hash", "")) == station_hash
        and str(ds.attrs.get("input_fingerprint", "")) == input_fingerprint
    )


def _init_single_thread_blas() -> None:
    """Avoid BLAS/OpenMP thread oversubscription inside ProcessPool workers."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def run_ccf_day(
    config: CCFRunConfig,
    utc_date: str,
    *,
    stations: tuple[StationRecord, ...] | None = None,
    param_hash: str | None = None,
    station_hash: str | None = None,
    pair_indices: np.ndarray | None = None,
) -> CCFDayOutcome:
    """Compute (or resume) one UTC day of ZZ cross-correlations."""
    t0 = time.perf_counter()
    if stations is None:
        stations = load_station_table(
            config.input.station_table,
            expected_response_state=config.input.response_state,
        )
    if param_hash is None:
        param_hash = config.parameter_hash()
    if station_hash is None:
        station_hash = station_table_hash(stations)
    if pair_indices is None:
        pair_indices = _filter_pair_indices(
            stations,
            include_autocorr=config.processing.include_autocorr,
            max_distance_km=config.processing.max_distance_km,
        )

    out_path = daily_ccf_path(config.output_dir, utc_date)
    proc = config.processing
    step_s = proc.resolved_step_s()
    extra_next_day_s = max(0.0, proc.window_length_s - step_s) if step_s < proc.window_length_s else 0.0
    # Cross-midnight windows starting late in day D may need up to window_length_s
    # past 00:00 of day D+1 when step_s aligns a window before 24:00.
    if extra_next_day_s == 0.0 and (86400.0 % proc.window_length_s) > 1e-6:
        extra_next_day_s = proc.window_length_s

    day_data = read_sds_day_zz(
        config.input.sds_root,
        stations,
        utc_date,
        sampling_rate_hz=config.input.sampling_rate_hz,
        extra_next_day_s=extra_next_day_s,
        grid_tolerance_samples=config.input.grid_tolerance_samples,
        gap_guard_s=config.input.gap_guard_s,
        allow_missing_next_day=config.input.allow_missing_next_day,
    )

    if not config.overwrite and _matches_resume_hashes(
        out_path,
        param_hash=param_hash,
        station_hash=station_hash,
        input_fingerprint=day_data.input_fingerprint,
    ):
        existing_ds = read_netcdf(out_path)
        n_win = np.asarray(existing_ds["n_windows"].values, dtype=np.int64)
        return CCFDayOutcome(
            utc_date=utc_date,
            output_file=out_path,
            status="resumed",
            elapsed_s=time.perf_counter() - t0,
            n_pairs=int(n_win.size),
            n_active_pairs=int(np.sum(n_win > 0)),
            missing_stations=day_data.missing_stations,
            next_day_missing_stations=day_data.next_day_missing_stations,
        )

    if out_path.exists() and not config.overwrite and _matches_resume_hashes(
        out_path, param_hash=param_hash, station_hash=station_hash, input_fingerprint=day_data.input_fingerprint
    ) is False:
        # Existing file has stale hashes/inputs -> recompute and replace atomically
        # unless overwrite=False with identical inputs? Wait: per resume policy,
        # stale hash triggers recomputation cleanly via overwrite=True on replace.
        pass

    if proc.target_sampling_rate_hz is not None:
        day_data = decimate_day_data(day_data, proc.target_sampling_rate_hz)

    spec_dtype = np.complex64 if proc.spectrum_dtype == "complex64" else np.complex128
    daily_res = correlate_zz_day(
        day_data.waveforms,
        day_data.valid_mask,
        sampling_rate_hz=day_data.sampling_rate_hz,
        window_length_s=proc.window_length_s,
        max_lag_s=proc.max_lag_s,
        lag_guard_s=proc.lag_guard_s,
        step_s=proc.step_s,
        cc_type=proc.cc_type,
        ram_half_width_s=proc.ram_half_width_s,
        ram_bandpass=proc.ram_bandpass,
        pcc2_bandpass=proc.pcc2_bandpass,
        whiten=proc.whiten,
        whiten_mode=proc.whiten_mode,
        taper_fraction=proc.taper_fraction,
        circular=proc.circular,
        nfft=proc.nfft,
        block_size=proc.block_size,
        pair_indices=pair_indices,
        include_autocorr=proc.include_autocorr,
        spectrum_dtype=spec_dtype,
        backend=config.backend,
    )

    ds = daily_ccf_to_dataset(
        daily_res,
        stations,
        utc_date=utc_date,
        param_hash=param_hash,
        station_list_hash=station_hash,
        input_fingerprint=day_data.input_fingerprint,
        extra_attrs={
            "response_state": config.input.response_state,
            "input_sampling_rate_hz": float(config.input.sampling_rate_hz),
            "whiten_mode": proc.whiten_mode,
            "window_length_s": float(proc.window_length_s),
            "step_s": float(step_s),
            "backend": config.backend,
        },
    )
    # Stale outputs with mismatched hashes are refreshed atomically.
    write_netcdf(ds, out_path, overwrite=True, compression_level=config.compression_level)
    return CCFDayOutcome(
        utc_date=utc_date,
        output_file=out_path,
        status="computed",
        elapsed_s=time.perf_counter() - t0,
        n_pairs=int(daily_res.n_windows.size),
        n_active_pairs=int(np.sum(daily_res.n_windows > 0)),
        missing_stations=day_data.missing_stations,
        next_day_missing_stations=day_data.next_day_missing_stations,
    )


def _run_day_worker(args: tuple[CCFRunConfig, str, tuple[StationRecord, ...], str, str, np.ndarray]) -> CCFDayOutcome:
    cfg, utc_date, stations, p_hash, s_hash, pairs = args
    return run_ccf_day(
        cfg,
        utc_date,
        stations=stations,
        param_hash=p_hash,
        station_hash=s_hash,
        pair_indices=pairs,
    )


def run_ccf_config(
    config_or_path: CCFRunConfig | str | Path,
    *,
    dates: Sequence[str] | None = None,
    overwrite: bool | None = None,
    max_workers: int | None = None,
    backend: str | None = None,
) -> CCFRunSummary:
    """Run daily ZZ cross-correlation from a config object or YAML file."""
    cfg = (
        config_or_path
        if isinstance(config_or_path, CCFRunConfig)
        else load_ccf_config(config_or_path)
    )
    if dates is not None:
        cfg = replace(cfg, dates=tuple(dates))
    if overwrite is not None:
        cfg = replace(cfg, overwrite=bool(overwrite))
    if max_workers is not None:
        cfg = replace(cfg, max_workers=int(max_workers))
    if backend is not None:
        cfg = replace(cfg, backend=str(backend))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = cfg.output_dir / "ccf.log"
    stations = load_station_table(
        cfg.input.station_table,
        expected_response_state=cfg.input.response_state,
    )
    p_hash = cfg.parameter_hash()
    s_hash = station_table_hash(stations)
    pairs = _filter_pair_indices(
        stations,
        include_autocorr=cfg.processing.include_autocorr,
        max_distance_km=cfg.processing.max_distance_km,
    )

    lines = [
        "",
        f"[{datetime.now().isoformat(timespec='seconds')}] ccf start",
        f"n_stations: {len(stations)}, n_pairs: {pairs.shape[0]}, n_dates: {len(cfg.dates)}",
        f"cc_type: {cfg.processing.cc_type}, backend: {cfg.backend}, max_workers: {cfg.max_workers}",
        f"param_hash: {p_hash}, station_list_hash: {s_hash}",
        *(f"{k}: {v}" for k, v in software_provenance().items()),
    ]
    try:
        if cfg.max_workers > 1 and len(cfg.dates) > 1:
            worker_args = [
                (cfg, d, stations, p_hash, s_hash, pairs) for d in cfg.dates
            ]
            with ProcessPoolExecutor(
                max_workers=min(cfg.max_workers, len(cfg.dates)),
                initializer=_init_single_thread_blas,
            ) as pool:
                outcomes = tuple(pool.map(_run_day_worker, worker_args))
        else:
            outcomes = tuple(
                run_ccf_day(
                    cfg,
                    d,
                    stations=stations,
                    param_hash=p_hash,
                    station_hash=s_hash,
                    pair_indices=pairs,
                )
                for d in cfg.dates
            )

        for out in outcomes:
            lines.append(
                f"date={out.utc_date}: status={out.status}, active_pairs={out.n_active_pairs}/{out.n_pairs}, "
                f"elapsed_s={out.elapsed_s:.2f}, missing={len(out.missing_stations)}, "
                f"next_day_missing={len(out.next_day_missing_stations)}"
            )
        lines.append("status: ok")
        return CCFRunSummary(config=cfg, outcomes=outcomes, log_file=log_file)
    except Exception as exc:
        lines.append(f"status: error, {type(exc).__name__}: {exc}")
        raise
    finally:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")


def _stations_from_daily_dataset(ds) -> tuple[StationRecord, ...]:
    """Reconstruct ordered StationRecord stubs from a daily CCF dataset."""
    pair_i = np.asarray(ds["pair_i"].values, dtype=np.int64)
    pair_j = np.asarray(ds["pair_j"].values, dtype=np.int64)
    n_stations = int(max(np.max(pair_i), np.max(pair_j))) + 1
    by_idx: dict[int, StationRecord] = {}

    src_sta = np.asarray(ds["source_station"].values).astype(str)
    rcv_sta = np.asarray(ds["receiver_station"].values).astype(str)
    src_lat = np.asarray(ds["source_latitude_deg"].values, dtype=np.float64)
    src_lon = np.asarray(ds["source_longitude_deg"].values, dtype=np.float64)
    rcv_lat = np.asarray(ds["receiver_latitude_deg"].values, dtype=np.float64)
    rcv_lon = np.asarray(ds["receiver_longitude_deg"].values, dtype=np.float64)

    for k in range(pair_i.size):
        for idx, name, lat, lon in (
            (int(pair_i[k]), str(src_sta[k]), float(src_lat[k]), float(src_lon[k])),
            (int(pair_j[k]), str(rcv_sta[k]), float(rcv_lat[k]), float(rcv_lon[k])),
        ):
            if idx not in by_idx:
                net, sta = name.split(".", 1) if "." in name else ("", name)
                by_idx[idx] = StationRecord(
                    network=net or "XX",
                    station=sta,
                    location="",
                    channel="ZZ",
                    latitude_deg=lat,
                    longitude_deg=lon,
                )
    return tuple(by_idx[i] for i in range(n_stations))


def _make_tf_pws_accumulator(cfg: StackRunConfig) -> TFPWSStackAccumulator:
    return TFPWSStackAccumulator(
        power=cfg.pws_power,
        day_weighting=cfg.day_weighting,
        freq_min_hz=cfg.tf_pws_freq_min_hz,
        freq_max_hz=cfg.tf_pws_freq_max_hz,
        sigma=cfg.tf_pws_sigma,
        smooth_samples=cfg.tf_pws_smooth_samples,
        normalize_weights=cfg.tf_pws_normalize_weights,
        spectrum_dtype=cfg.tf_pws_dtype,
        backend=cfg.backend,
    )


def _run_tf_pws_chunk_worker(
    args: tuple[StackRunConfig, tuple[Path, ...], int, int],
) -> StackedCCFResult:
    """Worker computing one ``[start:end]`` pair chunk of a tf-PWS stack."""
    cfg, daily_files, start, end = args
    acc = _make_tf_pws_accumulator(cfg)
    for path in daily_files:
        ds = read_netcdf_pair_slice(path, start, end)
        acc.add_day(dataset_to_daily_ccf(ds))
    return acc.finalize()


def _run_stack_tf_pws_chunked(
    cfg: StackRunConfig,
    daily_files: Sequence[Path],
) -> tuple[StackedCCFResult, tuple[StationRecord, ...], list[str], str, str]:
    """Stream daily CCFs into a tf-PWS stack with bounded memory and optional multiprocessing."""
    first_ds = read_netcdf(daily_files[0])
    stations = _stations_from_daily_dataset(first_ds)
    param_hash = str(first_ds.attrs.get("param_hash", ""))
    station_hash = str(first_ds.attrs.get("station_list_hash", ""))
    first_daily = dataset_to_daily_ccf(first_ds)
    n_pairs = int(first_daily.n_windows.size)

    # Cap pair chunk size by a 1 GiB working-set budget per worker so high-fs /
    # long-lag Stockwell stacks never thrash host RAM.
    resolved_nfft, lo, hi, _ = _resolve_stockwell_grid(
        int(first_daily.lag_s.size),
        float(first_daily.sampling_rate_hz),
        freq_min_hz=cfg.tf_pws_freq_min_hz,
        freq_max_hz=cfg.tf_pws_freq_max_hz,
    )
    n_freq = max(1, hi - lo + 1)
    bytes_per_elem = 8 if cfg.tf_pws_dtype == "complex64" else 16
    bytes_per_pair = max(1, 4 * n_freq * resolved_nfft * bytes_per_elem)
    auto_chunk = max(1, (1024 * 1024 * 1024) // bytes_per_pair)
    chunk_size = min(cfg.tf_pws_pair_chunk_size, auto_chunk)
    use_multiprocess = cfg.backend == "numpy" and cfg.max_workers > 1 and n_pairs > 1
    if use_multiprocess:
        balanced_chunk = max(1, (n_pairs + cfg.max_workers - 1) // cfg.max_workers)
        chunk_size = min(chunk_size, balanced_chunk)

    # Fast metadata validation pass across all daily files (reads only 1 pair header).
    dates_used: list[str] = [str(first_ds.attrs.get("utc_date", daily_files[0].stem))]
    for path in daily_files[1:]:
        hdr = read_netcdf_pair_slice(path, 0, min(1, n_pairs))
        if str(hdr.attrs.get("station_list_hash", "")) != station_hash:
            raise ValueError(
                f"station_list_hash mismatch in {path}; cannot stack incompatible station tables."
            )
        dates_used.append(str(hdr.attrs.get("utc_date", path.stem)))

    chunk_ranges = [(s, min(s + chunk_size, n_pairs)) for s in range(0, n_pairs, chunk_size)]

    if len(chunk_ranges) == 1:
        acc = _make_tf_pws_accumulator(cfg)
        acc.add_day(first_daily)
        for path in daily_files[1:]:
            acc.add_day(dataset_to_daily_ccf(read_netcdf(path)))
        return acc.finalize(), stations, dates_used, param_hash, station_hash

    daily_tuple = tuple(daily_files)
    worker_args = [(cfg, daily_tuple, start, end) for start, end in chunk_ranges]
    if use_multiprocess:
        effective_workers = min(cfg.max_workers, len(chunk_ranges))
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            initializer=_init_single_thread_blas,
        ) as pool:
            chunk_results = list(pool.map(_run_tf_pws_chunk_worker, worker_args))
    else:
        chunk_results = [_run_tf_pws_chunk_worker(arg) for arg in worker_args]

    template = chunk_results[0]
    stacked = StackedCCFResult(
        lag_s=template.lag_s,
        pair_indices=np.vstack([res.pair_indices for res in chunk_results]),
        ccf=np.vstack([res.ccf for res in chunk_results]),
        n_windows=np.concatenate([res.n_windows for res in chunk_results]),
        n_days=np.concatenate([res.n_days for res in chunk_results]),
        method="tf_pws",
        cc_type=template.cc_type,
        sampling_rate_hz=template.sampling_rate_hz,
        max_lag_s=template.max_lag_s,
        lag_guard_s=template.lag_guard_s,
        pws_power=template.pws_power,
    )
    return stacked, stations, dates_used, param_hash, station_hash


def run_stack_config(
    config_or_path: StackRunConfig | str | Path,
    *,
    overwrite: bool | None = None,
    max_workers: int | None = None,
    backend: str | None = None,
) -> StackRunSummary:
    """Stream daily ZZ NetCDF files into a multi-day linear, PWS, or tf-PWS stack."""
    cfg = (
        config_or_path
        if isinstance(config_or_path, StackRunConfig)
        else load_stack_config(config_or_path)
    )
    if overwrite is not None:
        cfg = replace(cfg, overwrite=bool(overwrite))
    if max_workers is not None:
        cfg = replace(cfg, max_workers=int(max_workers))
    if backend is not None:
        cfg = replace(cfg, backend=str(backend))

    if cfg.output_file.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Stack output exists and overwrite is False: {cfg.output_file}"
        )

    if cfg.dates is not None:
        daily_files = [daily_ccf_path(cfg.daily_dir, d) for d in cfg.dates]
        missing = [str(p) for p in daily_files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing daily CCF file(s): {', '.join(missing)}")
    else:
        daily_files = sorted(cfg.daily_dir.glob("ccf_ZZ_*.nc"))
        if not daily_files:
            raise FileNotFoundError(f"No ccf_ZZ_*.nc files found in {cfg.daily_dir}.")

    log_file = cfg.output_file.parent / "stack_ccf.log"
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"[{datetime.now().isoformat(timespec='seconds')}] stack-ccf start",
        f"daily_dir: {cfg.daily_dir}, n_files: {len(daily_files)}, method: {cfg.method}",
        *(f"{k}: {v}" for k, v in software_provenance().items()),
    ]
    try:
        extra_attrs: dict[str, object] = {"day_weighting": cfg.day_weighting}
        if cfg.method == "tf_pws":
            stacked, stations, dates_used, param_hash, station_hash = _run_stack_tf_pws_chunked(
                cfg, daily_files
            )
            extra_attrs["tf_pws_sigma"] = float(cfg.tf_pws_sigma)
            extra_attrs["tf_pws_smooth_samples"] = int(cfg.tf_pws_smooth_samples)
            extra_attrs["tf_pws_normalize_weights"] = int(cfg.tf_pws_normalize_weights)
            if cfg.tf_pws_freq_min_hz is not None:
                extra_attrs["tf_pws_freq_min_hz"] = float(cfg.tf_pws_freq_min_hz)
            if cfg.tf_pws_freq_max_hz is not None:
                extra_attrs["tf_pws_freq_max_hz"] = float(cfg.tf_pws_freq_max_hz)
        else:
            acc = (
                LinearStackAccumulator(day_weighting=cfg.day_weighting)
                if cfg.method == "linear"
                else PWSStackAccumulator(power=cfg.pws_power, day_weighting=cfg.day_weighting)
            )
            dates_used = []
            stations_opt: tuple[StationRecord, ...] | None = None
            param_hash = ""
            station_hash = ""

            for path in daily_files:
                ds = read_netcdf(path)
                if stations_opt is None:
                    stations_opt = _stations_from_daily_dataset(ds)
                    param_hash = str(ds.attrs.get("param_hash", ""))
                    station_hash = str(ds.attrs.get("station_list_hash", ""))
                else:
                    if str(ds.attrs.get("station_list_hash", "")) != station_hash:
                        raise ValueError(
                            f"station_list_hash mismatch in {path}; cannot stack incompatible station tables."
                        )
                daily_res = dataset_to_daily_ccf(ds)
                acc.add_day(daily_res)
                dates_used.append(str(ds.attrs.get("utc_date", path.stem)))

            assert stations_opt is not None
            stations = stations_opt
            stacked = (
                acc.finalize(trim_guard=cfg.trim_guard)
                if isinstance(acc, LinearStackAccumulator)
                else acc.finalize()
            )
        out_ds = stacked_ccf_to_dataset(
            stacked,
            stations,
            dates=dates_used,
            param_hash=param_hash,
            station_list_hash=station_hash,
            extra_attrs=extra_attrs,
        )
        write_netcdf(
            out_ds,
            cfg.output_file,
            overwrite=cfg.overwrite,
            compression_level=cfg.compression_level,
        )
        lines.append(f"wrote: {cfg.output_file}, pairs={stacked.n_windows.size}")
        lines.append("status: ok")
        return StackRunSummary(
            config=cfg,
            output_file=cfg.output_file,
            dates_used=tuple(dates_used),
            n_pairs=int(stacked.n_windows.size),
            log_file=log_file,
        )
    except Exception as exc:
        lines.append(f"status: error, {type(exc).__name__}: {exc}")
        raise
    finally:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
