"""SNR definitions used by SeisForge AFTAN."""

from __future__ import annotations

import numpy as np
from obspy import Trace
from scipy import signal

from .io import _trace_times
from .models import AFTANSNRConfig


def _window_snr(
    trace: Trace,
    distance_km: float,
    period: float,
    group_velocity: float,
    max_period: float,
    config: AFTANSNRConfig,
) -> float:
    if config.definition == "pyftan":
        return _format_snr(
            _pyftan_window_snr_ratio(trace, distance_km, max_period, config),
            config,
        )
    if period <= 0 or group_velocity <= 0:
        return config.fill
    times = _trace_times(trace)
    arrival_time = distance_km / group_velocity
    before_periods = (
        config.signal_before_periods
        if config.signal_before_periods is not None
        else config.signal_half_width_factor
    )
    after_periods = (
        config.signal_after_periods
        if config.signal_after_periods is not None
        else config.signal_half_width_factor
    )
    signal_start = max(times[0], arrival_time - before_periods * period)
    signal_end = min(times[-1], arrival_time + after_periods * period)
    signal_data = _slice_by_time(trace.data, times, signal_start, signal_end)

    if config.noise_mode == "tail":
        noise_start = signal_end + config.dsn
        noise_end = noise_start + config.nlen
        trace_end = times[-1]
        if noise_end > trace_end:
            shift = min(noise_end - trace_end, config.dsn)
            noise_start -= shift
            noise_end -= shift
        noise_data = _slice_by_time(trace.data, times, noise_start, noise_end)
    elif config.noise_mode == "complement":
        guard = config.noise_guard_factor * period
        noise_mask = (times < signal_start - guard) | (times > signal_end + guard)
        noise_data = trace.data[noise_mask]
    else:
        raise ValueError(f"Unsupported SNR noise mode: {config.noise_mode}")

    return _format_snr(_signal_noise_ratio(signal_data, noise_data, config.fill), config)


def _pyftan_window_snr_ratio(
    trace: Trace,
    distance_km: float,
    max_period: float,
    config: AFTANSNRConfig,
) -> float:
    times = _trace_times(trace)
    signal_start = max(0.0, distance_km / config.vmax - config.bfact * max_period)
    signal_end = distance_km / config.vmin + config.efact * max_period
    noise_start = signal_end + config.dsn
    noise_end = noise_start + config.nlen
    trace_end = times[-1]
    if signal_end > trace_end:
        signal_end = trace_end - config.nlen
        noise_start = signal_end
        noise_end = trace_end
    elif noise_end > trace_end:
        shift = min(noise_end - trace_end, config.dsn)
        noise_start -= shift
        noise_end -= shift
    signal_data = _slice_by_time(trace.data, times, signal_start, signal_end)
    noise_data = _slice_by_time(trace.data, times, noise_start, noise_end)
    return _signal_noise_ratio(signal_data, noise_data, config.fill)


def _signal_noise_ratio(
    signal_data: np.ndarray,
    noise_data: np.ndarray,
    fill: float,
) -> float:
    if signal_data.size == 0 or noise_data.size == 0:
        return fill
    noise_rms = np.sqrt(np.mean(noise_data**2))
    if noise_rms == 0:
        return fill
    return float(np.max(np.abs(signal_data)) / noise_rms)


def _aftan_diagram_snr(
    envelope: np.ndarray,
    peak_index: int,
    config: AFTANSNRConfig,
) -> float:
    if envelope.size == 0:
        return config.fill
    peak_index = int(np.clip(peak_index, 0, envelope.size - 1))
    peak_indices = signal.find_peaks(envelope)[0]
    if peak_indices.size == 0:
        peak_indices = np.array([peak_index])
    if peak_index not in peak_indices:
        peak_indices = np.sort(np.append(peak_indices, peak_index))
    current = int(np.searchsorted(peak_indices, peak_index))
    left_bound = int(peak_indices[current - 1]) if current > 0 else 0
    right_bound = (
        int(peak_indices[current + 1])
        if current < peak_indices.size - 1
        else envelope.size - 1
    )
    left_min = float(np.min(envelope[left_bound : peak_index + 1]))
    right_min = float(np.min(envelope[peak_index : right_bound + 1]))
    denom = np.sqrt(left_min * right_min)
    if denom <= 0:
        return config.fill
    return _format_snr(float(envelope[peak_index] / denom), config)


def _format_snr(ratio: float, config: AFTANSNRConfig) -> float:
    if not np.isfinite(ratio) or ratio <= 0:
        return config.fill
    if config.output_db:
        return float(20.0 * np.log10(ratio))
    return float(ratio)


def _snr(
    trace: Trace,
    distance_km: float,
    period: float,
    group_velocity: float,
    config: AFTANSNRConfig,
) -> float:
    """Backward-compatible local-window SNR helper for tests and notebooks."""
    return _window_snr(trace, distance_km, period, group_velocity, period, config)


def _slice_by_time(
    data: np.ndarray,
    times: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    mask = (times >= start) & (times <= end)
    return data[mask]
