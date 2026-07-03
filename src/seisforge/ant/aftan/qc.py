"""Quality-control helpers for picked AFTAN branches."""

from __future__ import annotations

import numpy as np

from .models import AFTANQCConfig, AFTANResult


def _period_qc(measured: dict[str, np.ndarray]) -> dict[str, float | int]:
    target = np.asarray(measured["target_period"], dtype=float)
    instant = np.asarray(measured["period"], dtype=float)
    group_velocity = np.asarray(measured["group_velocity"], dtype=float)
    amplitude = np.asarray(measured["amplitude"], dtype=float)
    initial_count = int(np.asarray(measured.get("initial_target_period_count", target.size)))
    retained_count = int(target.size)
    retained_fraction = retained_count / initial_count if initial_count > 0 else 0.0
    valid = np.isfinite(target) & np.isfinite(instant) & (target > 0)
    if not np.any(valid):
        return {
            "qc_initial_period_count": initial_count,
            "qc_retained_period_count": retained_count,
            "qc_retained_fraction": float(retained_fraction),
            "qc_valid_points": int(0),
            "qc_period_abs_median": float("nan"),
            "qc_period_rel_median": float("nan"),
            "qc_period_rel_max": float("nan"),
            "qc_group_velocity_min": float("nan"),
            "qc_group_velocity_max": float("nan"),
            "qc_amplitude_max": float("nan"),
        }
    period_abs = np.abs(instant[valid] - target[valid])
    period_rel = period_abs / target[valid]
    finite_group = group_velocity[np.isfinite(group_velocity)]
    finite_amp = amplitude[np.isfinite(amplitude)]
    return {
        "qc_initial_period_count": initial_count,
        "qc_retained_period_count": retained_count,
        "qc_retained_fraction": float(retained_fraction),
        "qc_valid_points": int(np.count_nonzero(valid)),
        "qc_period_abs_median": float(np.nanmedian(period_abs)),
        "qc_period_rel_median": float(np.nanmedian(period_rel)),
        "qc_period_rel_max": float(np.nanmax(period_rel)),
        "qc_group_velocity_min": float(np.nanmin(finite_group))
        if finite_group.size
        else float("nan"),
        "qc_group_velocity_max": float(np.nanmax(finite_group))
        if finite_group.size
        else float("nan"),
        "qc_amplitude_max": float(np.nanmax(finite_amp)) if finite_amp.size else float("nan"),
    }


def _qc_warnings(qc: dict[str, float | int], config: AFTANQCConfig) -> tuple[str, ...]:
    warnings = []
    period_rel_max = float(qc["qc_period_rel_max"])
    if np.isfinite(period_rel_max) and period_rel_max > config.period_rel_warning:
        warnings.append(
            "instant_period differs strongly from target_period: "
            f"max_relative_mismatch={period_rel_max:.3g}, "
            f"threshold={config.period_rel_warning:.3g}; "
            "check alpha, energy map, and phase-derivative stability"
        )
    retained_fraction = float(qc["qc_retained_fraction"])
    if (
        config.min_valid_fraction > 0
        and np.isfinite(retained_fraction)
        and retained_fraction < config.min_valid_fraction
    ):
        warnings.append(
            "short final branch: "
            f"retained_fraction={retained_fraction:.3g}, "
            f"threshold={config.min_valid_fraction:.3g}; "
            "consider rejecting this trace before interpolation"
        )
    return tuple(warnings)


def _qc_log_lines(result: AFTANResult) -> list[str]:
    return _format_qc_log_lines(result.input_file.name, result.branch, result.qc)


def _format_qc_log_lines(
    input_name: str,
    branch: str,
    qc: dict[str, float | int],
    *,
    prefix: str = "qc",
) -> list[str]:
    return [
        (
            f"{prefix} {input_name} [{branch}]: "
            f"retained_periods={qc['qc_retained_period_count']}/"
            f"{qc['qc_initial_period_count']}, "
            f"retained_fraction={qc['qc_retained_fraction']:.6g}, "
            f"valid_points={qc['qc_valid_points']}, "
            f"period_abs_median={qc['qc_period_abs_median']:.6g} s, "
            f"period_rel_median={qc['qc_period_rel_median']:.6g}, "
            f"period_rel_max={qc['qc_period_rel_max']:.6g}"
        ),
        (
            f"{prefix} {input_name} [{branch}]: "
            f"group_velocity_range="
            f"{qc['qc_group_velocity_min']:.6g}-"
            f"{qc['qc_group_velocity_max']:.6g} km/s, "
            f"amplitude_max={qc['qc_amplitude_max']:.6g}"
        ),
    ]


def _raise_short_branch_if_requested(
    qc: dict[str, float | int],
    config: AFTANQCConfig,
    label: str,
) -> None:
    retained_fraction = float(qc["qc_retained_fraction"])
    if (
        config.fail_on_short_branch
        and config.min_valid_fraction > 0
        and np.isfinite(retained_fraction)
        and retained_fraction < config.min_valid_fraction
    ):
        raise ValueError(
            f"{label}: retained_fraction {retained_fraction:.3g} is below "
            f"aftan.qc.min_valid_fraction {config.min_valid_fraction:.3g}."
        )
