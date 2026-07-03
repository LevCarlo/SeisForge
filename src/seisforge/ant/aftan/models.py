"""Data models for station-level AFTAN measurements."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from obspy import Trace


@dataclass(frozen=True)
class AFTANPeriodSamplingConfig:
    mode: str
    count: int | None = None
    step: float | None = None
    dfreq: float | None = None
    min_count: int | None = None
    periods: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AFTANAlphaConfig:
    mode: str
    value: float | None = None
    factor: float = 1.0
    distance_nodes: tuple[float, ...] | None = None
    alpha_nodes: tuple[float, ...] | None = None


@dataclass(frozen=True)
class AFTANBasicConfig:
    alpha: AFTANAlphaConfig
    trig_threshold: float = 50.0
    jump_points: int = 3


@dataclass(frozen=True)
class AFTANPMFPeriodBoundsConfig:
    mode: str = "raw"
    step: float | None = None
    min_method: str = "floor"
    max_method: str = "ceil"


@dataclass(frozen=True)
class AFTANPMFConfig:
    enabled: bool = False
    alpha: AFTANAlphaConfig = field(
        default_factory=lambda: AFTANAlphaConfig(mode="constant", value=20.0)
    )
    trig_threshold: float = 20.0
    jump_points: int = 3
    nalpha: float = 2.0
    min_half_length: float = 5.0
    amplitude_ratio: float = 0.2
    window_factor: float = 1.0
    period_bounds: AFTANPMFPeriodBoundsConfig = field(
        default_factory=AFTANPMFPeriodBoundsConfig
    )


@dataclass(frozen=True)
class AFTANSNRConfig:
    definition: str = "local"
    output_db: bool = False
    noise_mode: str = "tail"
    signal_half_width_factor: float = 1.0
    signal_before_periods: float | None = None
    signal_after_periods: float | None = None
    noise_guard_factor: float = 1.0
    bfact: float = 1.0
    efact: float = 0.0
    dsn: float = 500.0
    nlen: float = 500.0
    vmax: float = 4.5
    vmin: float = 1.0
    fill: float = 3.0


@dataclass(frozen=True)
class AFTANEnergyMapConfig:
    enabled: bool = False
    plot: bool = False
    velocity_count: int = 240
    normalize: bool = True
    phase_velocity: bool = False
    phase_cycle_count: int = 5


@dataclass(frozen=True)
class AFTANQCConfig:
    period_rel_warning: float = 0.35
    min_valid_fraction: float = 0.0
    fail_on_short_branch: bool = False


@dataclass(frozen=True)
class AFTANConfig:
    debug: bool = False
    min_period: float = 0.5
    max_period: float = 10.0
    max_period_nwl: float = 0.5
    reference_velocity: float = 4.0
    period_sampling: AFTANPeriodSamplingConfig = field(
        default_factory=lambda: AFTANPeriodSamplingConfig(
            mode="uniform",
            count=60,
        )
    )
    velocity_min: float = 0.5
    velocity_max: float = 5.5
    pi_over_4: float = -1.0
    branch: str = "positive"
    prediction_file: Path | None = None
    basic: AFTANBasicConfig = field(
        default_factory=lambda: AFTANBasicConfig(
            alpha=AFTANAlphaConfig(mode="constant", value=20.0),
            trig_threshold=50.0,
            jump_points=3,
        )
    )
    pmf: AFTANPMFConfig = field(default_factory=AFTANPMFConfig)
    snr: AFTANSNRConfig = field(default_factory=AFTANSNRConfig)
    energy_map: AFTANEnergyMapConfig = field(default_factory=AFTANEnergyMapConfig)
    qc: AFTANQCConfig = field(default_factory=AFTANQCConfig)


@dataclass(frozen=True)
class StationAFTANConfig:
    station: str
    input_dir: Path
    output_dir: Path
    sac_pattern: str = "*.SAC"
    overwrite: bool = False
    aftan: AFTANConfig = field(default_factory=AFTANConfig)


@dataclass(frozen=True)
class AFTANResult:
    input_file: Path
    branch: str
    output_dat: Path
    output_energy_map: Path | None
    output_energy_plot: Path | None
    target_period: np.ndarray
    period: np.ndarray
    group_velocity: np.ndarray
    phase_velocity: np.ndarray
    amplitude: np.ndarray
    snr: np.ndarray
    distance_km: float
    alpha: float
    alpha_mode: str
    qc: dict[str, float | int]
    warnings: tuple[str, ...] = ()
    output_npz: Path | None = None
    pmf_alpha: float | None = None
    pmf_alpha_mode: str | None = None
    output_phase_map: Path | None = None
    output_phase_plot: Path | None = None
    output_pmf_npz: Path | None = None
    output_pmf_dat: Path | None = None
    output_pmf_energy_map: Path | None = None
    output_pmf_energy_plot: Path | None = None
    output_pmf_phase_map: Path | None = None
    output_pmf_phase_plot: Path | None = None
    pmf_qc: dict[str, float | int] | None = None
    pmf_warnings: tuple[str, ...] = ()
    pmf_period_min: float | None = None
    pmf_period_max: float | None = None
    pmf_raw_period_min: float | None = None
    pmf_raw_period_max: float | None = None


@dataclass(frozen=True)
class BranchTrace:
    name: str
    trace: Trace
    warnings: tuple[str, ...] = ()
