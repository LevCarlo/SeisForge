"""Diagnostic AFTAN maps and their xarray products.

``AFTANMeasurement`` produces the FTAN amplitudes and picks; this module turns
them into self-describing datasets:

- the group-velocity energy map, ``amplitude(target_period, velocity)`` with
  ``velocity = distance / time``;
- the phase-velocity cycle-candidate map, Gaussian bumps at the phase
  velocities implied by ``2 pi N`` cycle ambiguities of the picked phase.
"""

from __future__ import annotations

from math import tau

import numpy as np
import xarray as xr


def amplitude_time_to_velocity(
    amplitude: np.ndarray,
    time_s: np.ndarray,
    distance_km: float,
    velocity_km_s: np.ndarray,
) -> np.ndarray:
    """Resample ``amplitude(period, time)`` onto ``velocity = distance / time``."""
    out = np.full((amplitude.shape[0], velocity_km_s.size), np.nan)
    query_time = distance_km / velocity_km_s
    for idx, amp in enumerate(amplitude):
        out[idx] = np.interp(query_time, time_s, amp, left=np.nan, right=np.nan)
    return out


def phase_velocity_candidate_map(
    period: np.ndarray,
    group_velocity: np.ndarray,
    phase: np.ndarray,
    picked_phase_velocity: np.ndarray,
    amplitude: np.ndarray,
    distance_km: float,
    velocity: np.ndarray,
    cycle_count: int,
    normalize: bool,
) -> np.ndarray:
    """Amplitude-weighted Gaussian bumps at cycle-shifted phase velocities.

    For each picked period, candidate velocities are
    ``omega * distance / (omega * t_group - phase + 2 pi N)`` for ``N`` within
    ``cycle_count`` cycles of the picked branch.
    """
    out = np.zeros((period.size, velocity.size), dtype=float)
    if velocity.size < 2:
        return out

    sigma = max(float(np.median(np.diff(velocity))) * 1.5, 1e-6)
    offsets = np.arange(-int(cycle_count), int(cycle_count) + 1)
    for idx, (per, gv, ph, pv, amp) in enumerate(
        zip(period, group_velocity, phase, picked_phase_velocity, amplitude)
    ):
        if not all(np.isfinite(value) for value in (per, gv, ph, pv, amp)):
            continue
        if per <= 0 or gv <= 0 or pv <= 0:
            continue
        omega = tau / per
        time = distance_km / gv
        base = omega * time - ph
        selected_denominator = omega * distance_km / pv
        selected_cycle = int(np.rint((selected_denominator - base) / tau))
        cycles = selected_cycle + offsets
        for cycle in cycles:
            denominator = base + cycle * tau
            if abs(denominator) < 1e-10:
                continue
            candidate_velocity = omega * distance_km / denominator
            if not np.isfinite(candidate_velocity):
                continue
            if candidate_velocity < velocity[0] or candidate_velocity > velocity[-1]:
                continue
            out[idx] += amp * np.exp(
                -0.5 * ((velocity - candidate_velocity) / sigma) ** 2
            )

    if normalize:
        out = _normalize_rows(out)
    return out


def _align_to_period_grid(
    target_period: np.ndarray,
    picked_target_period: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Align picked values to the full target-period grid."""
    aligned = np.full(target_period.size, np.nan, dtype=float)
    if picked_target_period.size == 0:
        return aligned

    indices = np.searchsorted(target_period, picked_target_period)
    for period, item, value in zip(picked_target_period, indices, values):
        candidates = []
        if item < target_period.size:
            candidates.append(item)
        if item > 0:
            candidates.append(item - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(target_period[idx] - period))
        tolerance = max(1e-8, abs(period) * 1e-8)
        if abs(target_period[best] - period) <= tolerance:
            aligned[best] = value
    return aligned


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    scale = np.nanmax(values, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return values / scale


def energy_map_dataset(
    measured: dict[str, np.ndarray],
    *,
    target_period: np.ndarray,
    velocity_km_s: np.ndarray,
    map_amplitude: np.ndarray,
    normalize: bool,
    debug: bool,
    attrs: dict[str, object],
) -> xr.Dataset:
    """Group-velocity energy map with picks aligned to ``target_period``."""
    picked_target_period = measured["target_period"]

    def aligned(values):
        return (
            ("target_period",),
            _align_to_period_grid(target_period, picked_target_period, values),
        )

    data_vars = {
        "amplitude": (("target_period", "velocity"), map_amplitude),
        "picked_group_velocity": aligned(measured["group_velocity"]),
        "picked_instant_period": aligned(measured["period"]),
        "picked_amplitude": aligned(measured["amplitude"]),
        "picked_snr": aligned(measured["snr"]),
    }
    if debug:
        data_vars.update(
            picked_hilbert_instant_period=aligned(measured["hilbert_period"]),
            picked_instant_period_delta=aligned(
                measured["hilbert_period"] - measured["period"]
            ),
        )
    if normalize:
        data_vars["normalized_amplitude"] = (
            ("target_period", "velocity"),
            _normalize_rows(map_amplitude),
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={"target_period": target_period, "velocity": velocity_km_s},
        attrs=attrs,
    )


def phase_velocity_map_dataset(
    measured: dict[str, np.ndarray],
    *,
    phase_velocity_km_s: np.ndarray,
    candidate_amplitude: np.ndarray,
    attrs: dict[str, object],
) -> xr.Dataset:
    """Phase-velocity cycle-candidate map plus global cycle-selection diagnostics."""
    return xr.Dataset(
        data_vars={
            "phase_candidate_amplitude": (
                ("target_period", "phase_velocity"),
                candidate_amplitude,
            ),
            "picked_phase_velocity": (
                ("target_period",),
                measured["phase_velocity"],
                {"long_name": "cycle-corrected phase velocity", "units": "km/s"},
            ),
            "picked_phase_velocity_uncorrected": (
                ("target_period",),
                measured["phase_velocity_uncorrected"],
                {
                    "long_name": "phase velocity before global cycle correction",
                    "units": "km/s",
                },
            ),
            "reference_phase_velocity": (
                ("target_period",),
                measured["phase_reference_velocity"],
                {
                    "long_name": (
                        "reference phase velocity evaluated at picked instant period"
                    ),
                    "units": "km/s",
                },
            ),
            "picked_group_velocity": (("target_period",), measured["group_velocity"]),
            "picked_instant_period": (("target_period",), measured["period"]),
            "picked_amplitude": (("target_period",), measured["amplitude"]),
            "phase_cycle_shift": (
                (),
                measured["phase_cycle_shift"],
                {"long_name": "global integer cycle shift applied to phase slowness"},
            ),
            "phase_cycle_reference_score_cycles": (
                (),
                measured["phase_cycle_reference_score_cycles"],
                {
                    "long_name": "median absolute reference residual after correction",
                    "units": "cycles",
                },
            ),
            "phase_cycle_reference_score_gap_cycles": (
                (),
                measured["phase_cycle_reference_score_gap_cycles"],
                {
                    "long_name": (
                        "reference-score gap between best and second-best shifts"
                    ),
                    "units": "cycles",
                },
            ),
            "phase_cycle_candidate_reference_score_cycles": (
                ("cycle_shift_candidate",),
                measured["phase_cycle_candidate_reference_score_cycles"],
                {"units": "cycles"},
            ),
            "phase_cycle_valid_count": (
                (),
                measured["phase_cycle_valid_count"],
                {"long_name": "period count used for global cycle selection"},
            ),
            "phase_cycle_search_limit_reached": (
                (),
                measured["phase_cycle_search_limit_reached"],
                {"long_name": "one when the selected shift reaches max_shift"},
            ),
        },
        coords={
            "target_period": measured["target_period"],
            "phase_velocity": phase_velocity_km_s,
            "cycle_shift_candidate": measured["phase_cycle_candidate_shift"],
        },
        attrs=attrs,
    )
