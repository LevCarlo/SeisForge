"""Pure-Python FTAN dispersion measurement for ambient-noise CCFs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import glob
from math import tau
from pathlib import Path
from typing import Any

import numpy as np
from obspy import Trace, read
from scipy import interpolate, signal
from scipy.fft import ifft
import yaml

from seisforge.ant.filters import (
    GaussianFTANFilter,
    PhaseMatchedFilter,
    gaussian_alpha_aftan_scaling,
    gaussian_alpha_from_distance,
    gaussian_alpha_pyftan_constant,
)


_BRANCHES = {"positive", "negative", "both", "stack"}
_ALPHA_MODES = {
    "constant",
    "empirical_distance_table",
    "aftan_distance_scaling",
    "pyftan_constant",
}
_PERIOD_SAMPLING_MODES = {"geomspace", "uniform", "list"}
_PMF_PERIOD_BOUND_MODES = {"raw", "step"}
_PMF_PERIOD_BOUND_METHODS = {"floor", "ceil", "nearest"}
_SNR_DEFINITIONS = {"local", "pyftan", "aftan"}
_SNR_NOISE_MODES = {"tail", "complement"}


@dataclass(frozen=True)
class AFTANPeriodSamplingConfig:
    mode: str
    count: int | None = None
    step: float | None = None
    points_per_octave: float | None = None
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


@dataclass(frozen=True)
class AFTANPMFPeriodBoundsConfig:
    mode: str = "raw"
    step: float | None = None
    min_method: str = "floor"
    max_method: str = "ceil"


@dataclass(frozen=True)
class AFTANPMFConfig:
    enabled: bool = False
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
    trig_threshold: float = 50.0
    jump_points: int = 3
    prediction_file: Path | None = None
    basic: AFTANBasicConfig = field(
        default_factory=lambda: AFTANBasicConfig(
            alpha=AFTANAlphaConfig(mode="constant", value=20.0)
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
    output_npz: Path
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


@dataclass(frozen=True)
class BranchTrace:
    name: str
    trace: Trace
    warnings: tuple[str, ...] = ()


_AFTAN_KEYS = (
    "period",
    "amplitude",
    "phase",
    "group_velocity",
    "phase_derivative",
    "hilbert_phase",
    "hilbert_phase_derivative",
    "hilbert_period",
)
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
    np.savetxt(
        path,
        table,
        fmt=dat_fmt,
        delimiter=_AFTAN_DAT_DELIMITER,
        header=dat_header,
    )


def _snr_column_name(config: AFTANSNRConfig) -> str:
    return "snr_db" if config.output_db else "snr"


def run_aftan_config(
    config_file,
    *,
    write_energy_map: bool | None = None,
    plot_energy_map: bool | None = None,
) -> list[AFTANResult]:
    """Run one station-level AFTAN config."""
    config_path = Path(config_file)
    config = build_station_aftan_config(_read_yaml(config_path), config_path.parent)
    config = _override_energy_map_config(config, write_energy_map, plot_energy_map)
    return run_station_aftan(config)


def build_station_aftan_config(
    config: dict[str, Any],
    base_dir: str | Path | None = None,
) -> StationAFTANConfig:
    """Build a station-level AFTAN config from a plain YAML dictionary."""
    base_path = Path(base_dir) if base_dir is not None else None
    io_config = config.get("io", {})
    aftan_config = config.get("aftan", {})

    station = str(config.get("station") or io_config.get("station") or "")
    if not station:
        raise ValueError("AFTAN config requires station.")
    for key in ("input_dir", "output_dir"):
        if key not in io_config:
            raise ValueError(f"AFTAN config io.{key} is required.")

    return StationAFTANConfig(
        station=station,
        input_dir=_resolve_path(io_config["input_dir"], base_path),
        output_dir=_resolve_path(io_config["output_dir"], base_path),
        sac_pattern=str(io_config.get("sac_pattern", "*.SAC")),
        overwrite=bool(io_config.get("overwrite", False)),
        aftan=_build_aftan_config(aftan_config, base_path),
    )


def run_station_aftan(config: StationAFTANConfig) -> list[AFTANResult]:
    """Measure dispersion for all SAC files in one station config."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.output_dir / "aftan.log"
    log_lines = _start_log(config)
    try:
        files = _find_sac_files(config.input_dir, config.sac_pattern)
        log_lines.append(f"matched_sac_files: {len(files)}")
        for path in files:
            log_lines.append(f"input: {path}")
        if not files:
            raise FileNotFoundError(
                f"No SAC files matched {config.input_dir / config.sac_pattern}"
            )

        results = []
        for path in files:
            path_results = run_aftan_file(path, config)
            for result in path_results:
                log_lines.append(
                    f"wrote {path.name} [{result.branch}]: "
                    f"{result.output_npz}, {result.output_dat}; "
                    f"n_period={result.period.size}, "
                    f"effective_period_range={np.nanmin(result.target_period):.6g}-"
                    f"{np.nanmax(result.target_period):.6g} s, "
                    f"distance_km={result.distance_km:.3f}, "
                    f"alpha_mode={result.alpha_mode}, alpha={result.alpha:.6g}"
                )
                for warning in result.warnings:
                    log_lines.append(f"warning {path.name} [{result.branch}]: {warning}")
                log_lines.extend(_qc_log_lines(result))
                if result.output_energy_map is not None:
                    log_lines.append(f"wrote energy_map: {result.output_energy_map}")
                if result.output_energy_plot is not None:
                    log_lines.append(f"wrote summary_plot: {result.output_energy_plot}")
                if result.output_phase_map is not None:
                    log_lines.append(f"wrote phase_velocity_map: {result.output_phase_map}")
                if result.output_pmf_dat is not None and result.output_pmf_npz is not None:
                    log_lines.append(
                        f"wrote PMF {path.name} [{result.branch}]: "
                        f"{result.output_pmf_npz}, {result.output_pmf_dat}"
                    )
                    for warning in result.pmf_warnings:
                        log_lines.append(
                            f"warning PMF {path.name} [{result.branch}]: {warning}"
                        )
                    if result.pmf_qc is not None:
                        log_lines.extend(
                            _format_qc_log_lines(
                                path.name,
                                result.branch,
                                result.pmf_qc,
                                prefix="qc PMF",
                            )
                        )
                if result.output_pmf_energy_map is not None:
                    log_lines.append(f"wrote pmf_energy_map: {result.output_pmf_energy_map}")
                if result.output_pmf_energy_plot is not None:
                    log_lines.append(f"wrote pmf_summary_plot: {result.output_pmf_energy_plot}")
                if result.output_pmf_phase_map is not None:
                    log_lines.append(
                        f"wrote pmf_phase_velocity_map: {result.output_pmf_phase_map}"
                    )
            results.extend(path_results)
        log_lines.append(f"status: ok, measured {len(results)} files")
        _append_log(log_file, log_lines)
        return results
    except Exception as exc:
        log_lines.append(f"status: error, {type(exc).__name__}: {exc}")
        _append_log(log_file, log_lines)
        raise


def run_aftan_file(path: str | Path, config: StationAFTANConfig) -> list[AFTANResult]:
    """Measure dispersion for one SAC file."""
    path = Path(path)
    trace = read(str(path))[0]
    branch_traces = _prepare_branch_traces(trace, config.aftan.branch)

    results = []
    for branch_trace in branch_traces:
        results.append(_run_aftan_branch(path, branch_trace, config))
    return results


def _run_aftan_branch(
    path: Path,
    branch_trace: BranchTrace,
    config: StationAFTANConfig,
) -> AFTANResult:
    output_npz = config.output_dir / f"{path.stem}.{branch_trace.name}.npz"
    output_dat = config.output_dir / f"{path.stem}.{branch_trace.name}.dat"
    output_energy_map = config.output_dir / f"{path.stem}.{branch_trace.name}.basic_ftan.nc"
    output_energy_plot = config.output_dir / f"{path.stem}.{branch_trace.name}.basic_ftan.png"
    output_phase_map = config.output_dir / f"{path.stem}.{branch_trace.name}.phase_velocity.nc"
    output_pmf_npz = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf.npz"
    output_pmf_dat = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf.dat"
    output_pmf_energy_map = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf_ftan.nc"
    output_pmf_energy_plot = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf_ftan.png"
    output_pmf_phase_map = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf_phase_velocity.nc"
    output_paths = [output_npz, output_dat]
    if config.aftan.pmf.enabled:
        output_paths.extend([output_pmf_npz, output_pmf_dat])
    if config.aftan.energy_map.enabled:
        output_paths.append(output_energy_map)
        if config.aftan.energy_map.phase_velocity:
            output_paths.append(output_phase_map)
        if config.aftan.pmf.enabled:
            output_paths.append(output_pmf_energy_map)
            if config.aftan.energy_map.phase_velocity:
                output_paths.append(output_pmf_phase_map)
    if config.aftan.energy_map.plot:
        output_paths.append(output_energy_plot)
        if config.aftan.pmf.enabled:
            output_paths.append(output_pmf_energy_plot)
    if any(item.exists() for item in output_paths) and not config.overwrite:
        raise FileExistsError(
            f"One or more output files for {path.name} already exist. Enable io.overwrite."
        )

    measurement = AFTANMeasurement(branch_trace.trace, path, config.aftan)
    measured = measurement.measure()
    qc = _period_qc(measured)
    warnings = branch_trace.warnings + _qc_warnings(qc, config.aftan.qc)
    _raise_short_branch_if_requested(qc, config.aftan.qc, f"{path.name} [{branch_trace.name}]")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "branch": branch_trace.name,
        "alpha": measurement.alpha,
        "alpha_mode": measurement.alpha_mode,
        "period_sampling_mode": config.aftan.period_sampling.mode,
        "debug": config.aftan.debug,
        "target_period": measured["target_period"],
        "period": measured["period"],
        "instant_period": measured["period"],
        "group_velocity": measured["group_velocity"],
        "phase_velocity": measured["phase_velocity"],
        "amplitude": measured["amplitude"],
        "snr": measured["snr"],
        "snr_definition": config.aftan.snr.definition,
        "snr_output_db": config.aftan.snr.output_db,
        "distance_km": measurement.distance_km,
        "input_file": str(path),
        **qc,
    }
    if config.aftan.debug:
        npz_payload.update(
            phase_derivative_rad_s=measured["phase_derivative"],
            hilbert_phase_derivative_rad_s=measured["hilbert_phase_derivative"],
            hilbert_instant_period=measured["hilbert_period"],
            instant_period_delta=measured["hilbert_period"] - measured["period"],
        )
    np.savez_compressed(output_npz, **npz_payload)
    _write_dispersion_dat(
        output_dat,
        measured,
        debug=config.aftan.debug,
        snr_label=_snr_column_name(config.aftan.snr),
    )

    written_energy_map = None
    written_energy_plot = None
    written_phase_map = None
    if config.aftan.energy_map.enabled or config.aftan.energy_map.plot:
        dataset = measurement.energy_map_dataset(
            measured,
            stage="basic",
            source_file=path,
            branch=branch_trace.name,
            velocity_count=config.aftan.energy_map.velocity_count,
            normalize=config.aftan.energy_map.normalize,
        )
        if config.aftan.energy_map.enabled:
            dataset.to_netcdf(output_energy_map, engine="scipy")
            written_energy_map = output_energy_map
        phase_dataset = None
        if config.aftan.energy_map.phase_velocity:
            phase_dataset = measurement.phase_velocity_map_dataset(
                measured,
                stage="basic",
                source_file=path,
                branch=branch_trace.name,
                velocity_count=config.aftan.energy_map.velocity_count,
                normalize=config.aftan.energy_map.normalize,
                cycle_count=config.aftan.energy_map.phase_cycle_count,
            )
            if config.aftan.energy_map.enabled:
                phase_dataset.to_netcdf(output_phase_map, engine="scipy")
                written_phase_map = output_phase_map
        if config.aftan.energy_map.plot:
            _write_aftan_summary_plot(
                dataset,
                output_energy_plot,
                phase_dataset=phase_dataset,
                snr_label=_snr_column_name(config.aftan.snr),
            )
            written_energy_plot = output_energy_plot

    written_pmf_npz = None
    written_pmf_dat = None
    written_pmf_energy_map = None
    written_pmf_energy_plot = None
    written_pmf_phase_map = None
    pmf_qc = None
    pmf_warnings: tuple[str, ...] = ()
    pmf_measured = None
    if config.aftan.pmf.enabled:
        pmf_measured = measurement.measure_pmf(measured)
        pmf_qc = _period_qc(pmf_measured)
        pmf_warnings = _qc_warnings(pmf_qc, config.aftan.qc)
        _raise_short_branch_if_requested(
            pmf_qc,
            config.aftan.qc,
            f"PMF {path.name} [{branch_trace.name}]",
        )
        np.savez_compressed(
            output_pmf_npz,
            stage="pmf",
            branch=branch_trace.name,
            alpha=measurement.alpha,
            alpha_mode=measurement.alpha_mode,
            period_sampling_mode=config.aftan.period_sampling.mode,
            pmf_period_min=pmf_measured["pmf_period_min"],
            pmf_period_max=pmf_measured["pmf_period_max"],
            pmf_raw_period_min=pmf_measured["pmf_raw_period_min"],
            pmf_raw_period_max=pmf_measured["pmf_raw_period_max"],
            pmf_period_bounds_mode=config.aftan.pmf.period_bounds.mode,
            pmf_period_bounds_step=(
                np.nan
                if config.aftan.pmf.period_bounds.step is None
                else config.aftan.pmf.period_bounds.step
            ),
            pmf_period_bounds_min_method=config.aftan.pmf.period_bounds.min_method,
            pmf_period_bounds_max_method=config.aftan.pmf.period_bounds.max_method,
            target_period=pmf_measured["target_period"],
            period=pmf_measured["period"],
            instant_period=pmf_measured["period"],
            group_velocity=pmf_measured["group_velocity"],
            phase_velocity=pmf_measured["phase_velocity"],
            amplitude=pmf_measured["amplitude"],
            snr=pmf_measured["snr"],
            snr_definition=config.aftan.snr.definition,
            snr_output_db=config.aftan.snr.output_db,
            distance_km=measurement.distance_km,
            input_file=str(path),
            **pmf_qc,
        )
        _write_dispersion_dat(
            output_pmf_dat,
            pmf_measured,
            debug=config.aftan.debug,
            snr_label=_snr_column_name(config.aftan.snr),
        )
        written_pmf_npz = output_pmf_npz
        written_pmf_dat = output_pmf_dat
        if config.aftan.energy_map.enabled or config.aftan.energy_map.plot:
            pmf_dataset = measurement.energy_map_dataset(
                pmf_measured,
                stage="pmf",
                source_file=path,
                branch=branch_trace.name,
                velocity_count=config.aftan.energy_map.velocity_count,
                normalize=config.aftan.energy_map.normalize,
                signal_data=pmf_measured["signal"],
            )
            if config.aftan.energy_map.enabled:
                pmf_dataset.to_netcdf(output_pmf_energy_map, engine="scipy")
                written_pmf_energy_map = output_pmf_energy_map
            pmf_phase_dataset = None
            if config.aftan.energy_map.phase_velocity:
                pmf_phase_dataset = measurement.phase_velocity_map_dataset(
                    pmf_measured,
                    stage="pmf",
                    source_file=path,
                    branch=branch_trace.name,
                    velocity_count=config.aftan.energy_map.velocity_count,
                    normalize=config.aftan.energy_map.normalize,
                    cycle_count=config.aftan.energy_map.phase_cycle_count,
                )
                if config.aftan.energy_map.enabled:
                    pmf_phase_dataset.to_netcdf(output_pmf_phase_map, engine="scipy")
                    written_pmf_phase_map = output_pmf_phase_map
            if config.aftan.energy_map.plot:
                _write_aftan_summary_plot(
                    pmf_dataset,
                    output_pmf_energy_plot,
                    phase_dataset=pmf_phase_dataset,
                    snr_label=_snr_column_name(config.aftan.snr),
                )
                written_pmf_energy_plot = output_pmf_energy_plot

    return AFTANResult(
        input_file=path,
        branch=branch_trace.name,
        output_npz=output_npz,
        output_dat=output_dat,
        output_energy_map=written_energy_map,
        output_energy_plot=written_energy_plot,
        target_period=measured["target_period"],
        period=measured["period"],
        group_velocity=measured["group_velocity"],
        phase_velocity=measured["phase_velocity"],
        amplitude=measured["amplitude"],
        snr=measured["snr"],
        distance_km=measurement.distance_km,
        alpha=measurement.alpha,
        alpha_mode=measurement.alpha_mode,
        qc=qc,
        warnings=warnings,
        output_phase_map=written_phase_map,
        output_phase_plot=None,
        output_pmf_npz=written_pmf_npz,
        output_pmf_dat=written_pmf_dat,
        output_pmf_energy_map=written_pmf_energy_map,
        output_pmf_energy_plot=written_pmf_energy_plot,
        output_pmf_phase_map=written_pmf_phase_map,
        output_pmf_phase_plot=None,
        pmf_qc=pmf_qc,
        pmf_warnings=pmf_warnings,
    )


class AFTANMeasurement:
    """A compact, config-local Python AFTAN implementation."""

    def __init__(self, trace: Trace, path: Path, config: AFTANConfig):
        self.trace = trace.copy()
        self.path = Path(path)
        self.config = config
        self.trace.taper(0.05)
        self.distance_km = _trace_distance_km(self.trace)
        self.alpha = _resolve_alpha(config.basic.alpha, self.distance_km)
        self.alpha_mode = config.basic.alpha.mode
        self.min_period = config.min_period
        self.max_period = min(
            config.max_period,
            self.distance_km / (config.reference_velocity * config.max_period_nwl),
        )
        if self.max_period <= self.min_period:
            raise ValueError(
                f"{self.path.name}: max_period {self.max_period:.3f} <= "
                f"min_period {self.min_period:.3f}"
            )
        self.target_periods = _build_period_grid(
            config.period_sampling,
            self.min_period,
            self.max_period,
        )
        self.nfft = int(2 ** np.ceil(np.log2(self.trace.stats.npts)))
        self.time = _trace_times(self.trace)
        self.pred_period, self.pred_velocity = _load_prediction(config)

    @property
    def nfreq(self) -> int:
        return self.target_periods.size

    @property
    def omega0(self) -> np.ndarray:
        return tau / self.target_periods

    @property
    def periods0(self) -> np.ndarray:
        return self.target_periods

    def measure(self) -> dict[str, np.ndarray]:
        return self._measure_signal(
            self.trace.data,
            self.target_periods,
            self.alpha,
            self.config.trig_threshold,
            self.config.jump_points,
        )

    def measure_pmf(self, basic: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        period_min, period_max, raw_period_min, raw_period_max = _pmf_period_bounds(
            basic["period"],
            self.config.min_period,
            self.config.max_period,
            self.config.pmf.period_bounds,
        )
        target_periods = _build_period_grid(
            self.config.period_sampling,
            period_min,
            period_max,
        )
        cleaned = PhaseMatchedFilter(
            dt=self.trace.stats.delta,
            distance_km=self.distance_km,
            periods=basic["period"],
            group_velocity=basic["group_velocity"],
            alpha=self.alpha,
            nalpha=self.config.pmf.nalpha,
            min_half_length=self.config.pmf.min_half_length,
            amplitude_ratio=self.config.pmf.amplitude_ratio,
            window_factor=self.config.pmf.window_factor,
            nfft=self.nfft,
        ).apply(self.trace.data)
        measured = self._measure_signal(
            cleaned,
            target_periods,
            self.alpha,
            self.config.trig_threshold,
            self.config.jump_points,
        )
        measured["signal"] = cleaned
        measured["pmf_period_min"] = np.asarray(period_min)
        measured["pmf_period_max"] = np.asarray(period_max)
        measured["pmf_raw_period_min"] = np.asarray(raw_period_min)
        measured["pmf_raw_period_max"] = np.asarray(raw_period_max)
        return measured

    def _measure_signal(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
        trig_threshold: float,
        jump_points: int,
    ) -> dict[str, np.ndarray]:
        par_all, raw = self._raw_ftan(signal_data, target_periods, alpha)
        corrected = self._jump_corrected(
            par_all,
            raw,
            trig_threshold,
            jump_points,
        )
        period = corrected["period"]
        group_velocity = corrected["group_velocity"]
        phase_velocity = self._phase_velocity(
            period,
            group_velocity,
            corrected["phase"],
        )
        snr = self._spectral_snr(
            signal_data,
            period,
            group_velocity,
            self.config.snr,
            alpha,
            target_periods,
        )
        return {
            "target_period": corrected["center_period"],
            "initial_target_period_count": np.asarray(target_periods.size),
            "period": period,
            "group_velocity": group_velocity,
            "phase_velocity": phase_velocity,
            "phase": corrected["phase"],
            "amplitude": corrected["amplitude"],
            "phase_derivative": corrected["phase_derivative"],
            "hilbert_phase": corrected["hilbert_phase"],
            "hilbert_phase_derivative": corrected["hilbert_phase_derivative"],
            "hilbert_period": corrected["hilbert_period"],
            "snr": snr,
        }

    def _raw_ftan(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
        ft = self._ftan_complex(signal_data, target_periods, alpha)
        amplitude = np.abs(ft)
        phase = np.angle(ft)
        hilbert_phase = self._hilbert_phase(ft)
        out = {key: np.zeros(target_periods.size) for key in _AFTAN_KEYS}
        par_all = []

        for idx, (omega, amp_col, phase_col, hilbert_phase_col) in enumerate(
            zip(tau / target_periods, amplitude.T, phase.T, hilbert_phase.T)
        ):
            peak_indices = signal.find_peaks(amp_col[: self.trace.stats.npts])[0]
            peak_indices = self._velocity_window_indices(peak_indices)
            if peak_indices.size == 0:
                peak_indices = np.array([self._fallback_peak_index(amp_col)])

            par = {key: np.zeros(peak_indices.size) for key in _AFTAN_KEYS}
            dphase, par["amplitude"], par["phase"], dt_offset, peak_indices = self._interp(
                amp_col,
                phase_col,
                peak_indices,
                omega,
            )
            hilbert_dphase, par["hilbert_phase"] = self._interp_phase_only(
                hilbert_phase_col,
                peak_indices,
                omega,
                dt_offset,
            )
            inst_period = tau / (dphase / self.trace.stats.delta)
            hilbert_period = tau / (hilbert_dphase / self.trace.stats.delta)
            t_max = self.time[0] + (peak_indices + dt_offset) * self.trace.stats.delta
            group_velocity = self.distance_km / t_max
            valid = np.isfinite(inst_period) & np.isfinite(group_velocity)
            valid &= group_velocity >= self.config.velocity_min
            valid &= group_velocity <= self.config.velocity_max
            if not np.any(valid):
                valid = np.ones_like(group_velocity, dtype=bool)
            par["period"] = inst_period
            par["group_velocity"] = group_velocity
            par["phase_derivative"] = dphase / self.trace.stats.delta
            par["hilbert_phase_derivative"] = hilbert_dphase / self.trace.stats.delta
            par["hilbert_period"] = hilbert_period
            par_all.append(par)

            score = np.where(valid, par["amplitude"], -np.inf)
            imax = int(np.argmax(score))
            for key in _AFTAN_KEYS:
                out[key][idx] = par[key][imax]

        out["center_period"] = target_periods
        return par_all, out

    def energy_map_dataset(
        self,
        measured: dict[str, np.ndarray],
        *,
        stage: str,
        source_file: Path,
        branch: str,
        velocity_count: int,
        normalize: bool,
        signal_data: np.ndarray | None = None,
    ):
        """Build an xarray dataset for the FTAN group-velocity energy map."""
        import xarray as xr

        if signal_data is None:
            signal_data = self.trace.data
        target_period = measured.get("energy_target_period", measured["target_period"])
        amplitude = np.abs(
            self._ftan_complex(
                signal_data,
                np.asarray(target_period, dtype=float),
                self.alpha,
            )
        )
        amplitude = amplitude[: self.trace.stats.npts, :].T
        velocity = np.linspace(
            self.config.velocity_min,
            self.config.velocity_max,
            int(velocity_count),
        )
        map_amplitude = self._amplitude_time_to_velocity(amplitude, velocity)
        picked_target_period = measured["target_period"]
        data_vars = {
            "amplitude": (("target_period", "velocity"), map_amplitude),
            "picked_group_velocity": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["group_velocity"],
                ),
            ),
            "picked_instant_period": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["period"],
                ),
            ),
            "picked_amplitude": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["amplitude"],
                ),
            ),
            "picked_snr": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["snr"],
                ),
            ),
        }
        if self.config.debug:
            data_vars.update(
                picked_hilbert_instant_period=(
                    ("target_period",),
                    _align_to_period_grid(
                        target_period,
                        picked_target_period,
                        measured["hilbert_period"],
                    ),
                ),
                picked_instant_period_delta=(
                    ("target_period",),
                    _align_to_period_grid(
                        target_period,
                        picked_target_period,
                        measured["hilbert_period"] - measured["period"],
                    ),
                ),
            )
        if normalize:
            scale = np.nanmax(map_amplitude, axis=1, keepdims=True)
            scale[~np.isfinite(scale) | (scale == 0)] = 1.0
            data_vars["normalized_amplitude"] = (
                ("target_period", "velocity"),
                map_amplitude / scale,
            )
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "target_period": target_period,
                "velocity": velocity,
            },
            attrs={
                "stage": stage,
                "branch": branch,
                "input_file": str(source_file),
                "distance_km": float(self.distance_km),
                "alpha": float(self.alpha),
                "alpha_mode": self.alpha_mode,
                "velocity_min": float(self.config.velocity_min),
                "velocity_max": float(self.config.velocity_max),
                "period_min": float(np.nanmin(target_period)),
                "period_max": float(np.nanmax(target_period)),
                "period_sampling_mode": self.config.period_sampling.mode,
                "signal_source": "raw_waveform" if stage == "basic" else "pmf_clean_waveform",
                "debug": bool(self.config.debug),
            },
        )
        return dataset

    def phase_velocity_map_dataset(
        self,
        measured: dict[str, np.ndarray],
        *,
        stage: str,
        source_file: Path,
        branch: str,
        velocity_count: int,
        normalize: bool,
        cycle_count: int,
    ):
        """Build an xarray dataset that shows phase-velocity cycle candidates."""
        import xarray as xr

        phase_velocity = np.linspace(
            self.config.velocity_min,
            self.config.velocity_max,
            int(velocity_count),
        )
        candidate_amplitude = _phase_velocity_candidate_map(
            measured["period"],
            measured["group_velocity"],
            measured["phase"],
            measured["phase_velocity"],
            measured["amplitude"],
            self.distance_km,
            phase_velocity,
            cycle_count,
            normalize,
        )
        return xr.Dataset(
            data_vars={
                "phase_candidate_amplitude": (
                    ("target_period", "phase_velocity"),
                    candidate_amplitude,
                ),
                "picked_phase_velocity": (
                    ("target_period",),
                    measured["phase_velocity"],
                ),
                "picked_group_velocity": (
                    ("target_period",),
                    measured["group_velocity"],
                ),
                "picked_instant_period": (
                    ("target_period",),
                    measured["period"],
                ),
                "picked_amplitude": (
                    ("target_period",),
                    measured["amplitude"],
                ),
            },
            coords={
                "target_period": measured["target_period"],
                "phase_velocity": phase_velocity,
            },
            attrs={
                "stage": stage,
                "branch": branch,
                "input_file": str(source_file),
                "distance_km": float(self.distance_km),
                "alpha": float(self.alpha),
                "alpha_mode": self.alpha_mode,
                "velocity_min": float(self.config.velocity_min),
                "velocity_max": float(self.config.velocity_max),
                "period_min": float(np.nanmin(measured["target_period"])),
                "period_max": float(np.nanmax(measured["target_period"])),
                "period_sampling_mode": self.config.period_sampling.mode,
                "phase_cycle_count": int(cycle_count),
                "description": "phase-velocity cycle-candidate diagnostic map",
            },
        )

    def _amplitude_time_to_velocity(
        self,
        amplitude: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        times = self.time[: self.trace.stats.npts]
        out = np.full((amplitude.shape[0], velocity.size), np.nan)
        for idx, amp in enumerate(amplitude):
            query_time = self.distance_km / velocity
            out[idx] = np.interp(query_time, times, amp, left=np.nan, right=np.nan)
        return out

    def _ftan_complex(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        spc_mat = self._multiple_filter(signal_data, target_periods, alpha)
        return ifft(spc_mat, axis=0)

    def _hilbert_phase(self, ft: np.ndarray) -> np.ndarray:
        real_filtered = ft.real[: self.trace.stats.npts, :]
        analytic = signal.hilbert(real_filtered, axis=0)
        phase = np.zeros_like(ft.real)
        phase[: self.trace.stats.npts, :] = np.angle(analytic)
        return phase

    def _multiple_filter(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        return GaussianFTANFilter(
            self.trace.stats.delta,
            target_periods,
            alpha,
            nfft=self.nfft,
        ).spectra(signal_data)

    def _velocity_window_indices(self, indices: np.ndarray) -> np.ndarray:
        time = self.time[0] + indices * self.trace.stats.delta
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = self.distance_km / time
        mask = np.isfinite(velocity)
        mask &= velocity >= self.config.velocity_min
        mask &= velocity <= self.config.velocity_max
        return indices[mask]

    def _fallback_peak_index(self, amplitude: np.ndarray) -> int:
        npts = self.trace.stats.npts
        time = self.time[:npts]
        tmin = self.distance_km / self.config.velocity_max
        tmax = self.distance_km / self.config.velocity_min
        mask = (time >= tmin) & (time <= tmax)
        if np.any(mask):
            candidates = np.where(mask)[0]
            return int(candidates[np.argmax(amplitude[candidates])])
        return int(np.argmax(amplitude[:npts]))

    def _interp(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        indices: np.ndarray,
        omega: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = indices[(indices > 0) & (indices < self.trace.stats.npts - 1)]
        if indices.size == 0:
            indices = np.array([min(max(1, self.trace.stats.npts // 2), self.trace.stats.npts - 2)])

        left = indices - 1
        right = indices + 1
        denom = amplitude[left] + amplitude[right] - 2 * amplitude[indices]
        denom[denom == 0] = np.nan
        dt_offset = (amplitude[left] - amplitude[right]) / denom / 2
        dt_offset[np.isnan(denom)] = 0

        amp_i = _parabola(
            dt_offset,
            amplitude[left],
            amplitude[indices],
            amplitude[right],
        )
        phase_left = phase[left]
        phase_mid = phase[indices].copy()
        phase_right = phase[right].copy()
        delta = self.trace.stats.delta
        phase_mid -= tau * np.around((phase_mid - phase_left - omega * delta) / tau)
        phase_right -= tau * np.around((phase_right - phase_mid - omega * delta) / tau)
        dphase = dt_offset * (phase_left + phase_right - 2 * phase_mid)
        dphase += (phase_right - phase_left) / 2
        phase_i = (
            _parabola(dt_offset, phase_left, phase_mid, phase_right)
            + self.config.pi_over_4 * np.pi / 4
        )
        return dphase, amp_i, phase_i, dt_offset, indices

    def _interp_phase_only(
        self,
        phase: np.ndarray,
        indices: np.ndarray,
        omega: float,
        dt_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        left = indices - 1
        right = indices + 1
        delta = self.trace.stats.delta
        phase_left = phase[left]
        phase_mid = phase[indices].copy()
        phase_right = phase[right].copy()
        phase_mid -= tau * np.around((phase_mid - phase_left - omega * delta) / tau)
        phase_right -= tau * np.around((phase_right - phase_mid - omega * delta) / tau)
        dphase = dt_offset * (phase_left + phase_right - 2 * phase_mid)
        dphase += (phase_right - phase_left) / 2
        phase_i = (
            _parabola(dt_offset, phase_left, phase_mid, phase_right)
            + self.config.pi_over_4 * np.pi / 4
        )
        return dphase, phase_i

    def _jump_corrected(
        self,
        par_all: list[dict[str, np.ndarray]],
        raw: dict[str, np.ndarray],
        trig_threshold: float,
        jump_points: int,
    ) -> dict[str, np.ndarray]:
        out = {key: np.asarray(value).copy() for key, value in raw.items()}
        _, trig, on = self._trigger(raw["center_period"], raw["group_velocity"], trig_threshold)
        if not on:
            return out

        jumps = np.where(np.abs(trig[1:] - trig[:-1]) == 2)[0] + 1
        if jumps.size >= 2:
            close = np.where((jumps[1:] - jumps[:-1]) <= jump_points)[0]
            for item in close:
                temp = {key: value.copy() for key, value in out.items()}
                for k in range(jumps[item], jumps[item + 1]):
                    par = par_all[k]
                    delta = np.abs(par["group_velocity"] - temp["group_velocity"][k - 1])
                    best = int(np.argmin(delta))
                    for key in _AFTAN_KEYS:
                        temp[key][k] = par[key][best]
                _, test_trig, test_on = self._trigger(
                    raw["center_period"],
                    temp["group_velocity"],
                    trig_threshold,
                )
                if not np.any(np.abs(test_trig[jumps[item] - 1 : jumps[item + 1] + 1]) == 1):
                    out = temp
                    trig = test_trig
                    on = test_on

        if on:
            start, stop = _longest_branch(trig)
            out = {key: value[start : stop + 1] for key, value in out.items()}
        return out

    @staticmethod
    def _trigger(
        period: np.ndarray,
        velocity: np.ndarray,
        trig_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        curvature = _curvature(period, velocity)
        trig = np.zeros(curvature.size)
        trig[curvature > trig_threshold] = 1
        trig[curvature < -trig_threshold] = -1
        return curvature, trig, bool(np.any(np.abs(trig) == 1))

    def _phase_velocity(
        self,
        period: np.ndarray,
        group_velocity: np.ndarray,
        phase: np.ndarray,
    ) -> np.ndarray:
        distance = self.distance_km
        omega = tau / period
        time = distance / group_velocity
        slowness = 1 / group_velocity
        phase_velocity = np.zeros(group_velocity.size)

        c_pred = self._predicted_phase_velocity(period[-1])
        phase_pred = omega[-1] * time[-1] - omega[-1] / c_pred * distance
        cycle = round((phase_pred - phase[-1]) / tau)
        phase_velocity[-1] = omega[-1] * distance / (
            omega[-1] * time[-1] - (phase[-1] + cycle * tau)
        )

        domega = omega[:-1] - omega[1:]
        for idx in range(omega.size - 2, -1, -1):
            wave_number = omega[idx + 1] / phase_velocity[idx + 1]
            wave_number += domega[idx] * (slowness[idx] + slowness[idx + 1]) / 2
            phase_pred = omega[idx] * time[idx] - wave_number * distance
            cycle = round((phase[idx] - phase_pred) / tau)
            phase_velocity[idx] = omega[idx] * distance / (
                omega[idx] * time[idx] - phase[idx] + cycle * tau
            )
        return phase_velocity

    def _predicted_phase_velocity(self, period: float) -> float:
        return float(
            interpolate.interp1d(
                self.pred_period,
                self.pred_velocity,
                fill_value="extrapolate",
                assume_sorted=True,
            )(period)
        )

    def _spectral_snr(
        self,
        signal_data: np.ndarray,
        period: np.ndarray,
        group_velocity: np.ndarray,
        config: AFTANSNRConfig,
        alpha: float,
        target_periods: np.ndarray,
    ) -> np.ndarray:
        ft = self._ftan_complex(signal_data, target_periods, alpha)
        envelope = np.abs(ft[: self.trace.stats.npts, :])
        snr = np.zeros(period.size)
        work_trace = Trace(header=self.trace.stats)
        for idx in range(period.size):
            nearest = int(np.argmin(np.abs(target_periods - period[idx])))
            if config.definition == "aftan":
                peak_time = self.distance_km / group_velocity[idx]
                peak_index = int(
                    np.argmin(np.abs(self.time[: self.trace.stats.npts] - peak_time))
                )
                snr[idx] = _aftan_diagram_snr(envelope[:, nearest], peak_index, config)
            else:
                work_trace.data = envelope[:, nearest]
                snr[idx] = _window_snr(
                    work_trace,
                    self.distance_km,
                    period[idx],
                    group_velocity[idx],
                    self.max_period,
                    config,
                )
            if snr[idx] <= 0:
                break
        return snr


def _build_aftan_config(config: dict[str, Any], base_path: Path | None) -> AFTANConfig:
    basic = config.get("basic", {})
    pmf = config.get("pmf", {})
    snr = config.get("snr", {})
    energy_map = config.get("energy_map", {})
    qc_config = config.get("qc", {})
    velocity_window = config.get("velocity_window", {})
    prediction_file = config.get("prediction_file")
    branch = str(config.get("branch", "")).lower()
    if branch not in _BRANCHES:
        raise ValueError(f"aftan.branch must be one of {sorted(_BRANCHES)}.")
    aftan = AFTANConfig(
        debug=bool(config.get("debug", False)),
        min_period=float(config.get("min_period", 0.5)),
        max_period=float(config.get("max_period", 10.0)),
        max_period_nwl=float(config.get("max_period_nwl", 0.5)),
        reference_velocity=float(config.get("reference_velocity", 4.0)),
        period_sampling=_build_period_sampling_config(
            config.get("period_sampling", {})
        ),
        velocity_min=float(velocity_window.get("min", config.get("velocity_min", 0.5))),
        velocity_max=float(velocity_window.get("max", config.get("velocity_max", 5.5))),
        pi_over_4=float(config.get("pi_over_4", -1.0)),
        branch=branch,
        trig_threshold=float(
            config.get("trig_threshold", basic.get("trig_threshold", 50.0))
        ),
        jump_points=int(config.get("jump_points", basic.get("jump_points", 3))),
        prediction_file=(
            _resolve_path(prediction_file, base_path) if prediction_file else None
        ),
        basic=AFTANBasicConfig(
            alpha=_build_alpha_config(basic.get("alpha", {})),
        ),
        pmf=AFTANPMFConfig(
            enabled=bool(pmf.get("enabled", False)),
            nalpha=float(pmf.get("nalpha", 2.0)),
            min_half_length=float(pmf.get("min_half_length", 5.0)),
            amplitude_ratio=float(pmf.get("amplitude_ratio", 0.2)),
            window_factor=float(pmf.get("window_factor", 1.0)),
            period_bounds=_build_pmf_period_bounds_config(
                pmf.get("period_bounds", {})
            ),
        ),
        snr=_build_snr_config(snr),
        energy_map=AFTANEnergyMapConfig(
            enabled=bool(energy_map.get("enabled", False)),
            plot=bool(energy_map.get("plot", False)),
            velocity_count=int(energy_map.get("velocity_count", 240)),
            normalize=bool(energy_map.get("normalize", True)),
            phase_velocity=bool(energy_map.get("phase_velocity", False)),
            phase_cycle_count=int(energy_map.get("phase_cycle_count", 5)),
        ),
        qc=AFTANQCConfig(
            period_rel_warning=float(qc_config.get("period_rel_warning", 0.35)),
            min_valid_fraction=float(qc_config.get("min_valid_fraction", 0.0)),
            fail_on_short_branch=bool(qc_config.get("fail_on_short_branch", False)),
        ),
    )
    _validate_aftan_config(aftan)
    return aftan


def _validate_aftan_config(config: AFTANConfig) -> None:
    if config.min_period <= 0:
        raise ValueError("aftan.min_period must be positive.")
    if config.max_period <= config.min_period:
        raise ValueError("aftan.max_period must be greater than aftan.min_period.")
    if config.max_period_nwl <= 0:
        raise ValueError("aftan.max_period_nwl must be positive.")
    if config.reference_velocity <= 0:
        raise ValueError("aftan.reference_velocity must be positive.")
    if config.velocity_min <= 0 or config.velocity_max <= 0:
        raise ValueError("aftan.velocity_window min/max must be positive.")
    if config.velocity_max <= config.velocity_min:
        raise ValueError("aftan.velocity_window.max must be greater than min.")
    if config.trig_threshold <= 0:
        raise ValueError("aftan.trig_threshold must be positive.")
    if config.jump_points < 0:
        raise ValueError("aftan.jump_points must be non-negative.")
    if config.pmf.nalpha <= 0:
        raise ValueError("aftan.pmf.nalpha must be positive.")
    if config.pmf.min_half_length < 0:
        raise ValueError("aftan.pmf.min_half_length must be non-negative.")
    if config.pmf.amplitude_ratio < 0:
        raise ValueError("aftan.pmf.amplitude_ratio must be non-negative.")
    if config.pmf.window_factor <= 0:
        raise ValueError("aftan.pmf.window_factor must be positive.")
    if config.energy_map.velocity_count < 2:
        raise ValueError("aftan.energy_map.velocity_count must be at least 2.")
    if config.energy_map.phase_cycle_count < 0:
        raise ValueError("aftan.energy_map.phase_cycle_count must be non-negative.")
    if config.qc.period_rel_warning < 0:
        raise ValueError("aftan.qc.period_rel_warning must be non-negative.")
    if not 0 <= config.qc.min_valid_fraction <= 1:
        raise ValueError("aftan.qc.min_valid_fraction must be between 0 and 1.")


def _build_snr_config(config: dict[str, Any]) -> AFTANSNRConfig:
    if not isinstance(config, dict):
        raise ValueError("aftan.snr must be a mapping.")
    definition = str(config.get("definition", "local")).lower()
    if definition not in _SNR_DEFINITIONS:
        raise ValueError(f"aftan.snr.definition must be one of {sorted(_SNR_DEFINITIONS)}.")
    noise_mode = str(config.get("noise_mode", "tail")).lower()
    if noise_mode not in _SNR_NOISE_MODES:
        raise ValueError(f"aftan.snr.noise_mode must be one of {sorted(_SNR_NOISE_MODES)}.")
    signal_half_width_factor = float(
        config.get(
            "signal_half_width_factor",
            config.get("signal_half_width_periods", 1.0),
        )
    )
    signal_before_periods = (
        float(config["signal_before_periods"])
        if "signal_before_periods" in config
        else None
    )
    signal_after_periods = (
        float(config["signal_after_periods"])
        if "signal_after_periods" in config
        else None
    )
    noise_guard_factor = float(
        config.get("noise_guard_factor", config.get("noise_guard_periods", 1.0))
    )
    if signal_half_width_factor < 0:
        raise ValueError("aftan.snr.signal_half_width_factor must be non-negative.")
    if signal_before_periods is not None and signal_before_periods < 0:
        raise ValueError("aftan.snr.signal_before_periods must be non-negative.")
    if signal_after_periods is not None and signal_after_periods < 0:
        raise ValueError("aftan.snr.signal_after_periods must be non-negative.")
    if noise_guard_factor < 0:
        raise ValueError("aftan.snr.noise_guard_factor must be non-negative.")
    dsn = float(config.get("dsn", 500.0))
    nlen = float(config.get("nlen", 500.0))
    vmax = float(config.get("vmax", 4.5))
    vmin = float(config.get("vmin", 1.0))
    fill = float(config.get("fill", 3.0))
    if dsn < 0:
        raise ValueError("aftan.snr.dsn must be non-negative.")
    if nlen <= 0:
        raise ValueError("aftan.snr.nlen must be positive.")
    if vmin <= 0 or vmax <= 0:
        raise ValueError("aftan.snr.vmin/vmax must be positive.")
    if vmax <= vmin:
        raise ValueError("aftan.snr.vmax must be greater than vmin.")
    if fill <= 0:
        raise ValueError("aftan.snr.fill must be positive.")
    return AFTANSNRConfig(
        definition=definition,
        output_db=bool(config.get("output_db", False)),
        noise_mode=noise_mode,
        signal_half_width_factor=signal_half_width_factor,
        signal_before_periods=signal_before_periods,
        signal_after_periods=signal_after_periods,
        noise_guard_factor=noise_guard_factor,
        bfact=float(config.get("bfact", 1.0)),
        efact=float(config.get("efact", 0.0)),
        dsn=dsn,
        nlen=nlen,
        vmax=vmax,
        vmin=vmin,
        fill=fill,
    )


def _override_energy_map_config(
    config: StationAFTANConfig,
    write_energy_map: bool | None,
    plot_energy_map: bool | None,
) -> StationAFTANConfig:
    if write_energy_map is None and plot_energy_map is None:
        return config
    energy_map = config.aftan.energy_map
    new_energy_map = AFTANEnergyMapConfig(
        enabled=energy_map.enabled if write_energy_map is None else write_energy_map,
        plot=energy_map.plot if plot_energy_map is None else plot_energy_map,
        velocity_count=energy_map.velocity_count,
        normalize=energy_map.normalize,
        phase_velocity=energy_map.phase_velocity,
        phase_cycle_count=energy_map.phase_cycle_count,
    )
    new_aftan = AFTANConfig(
        debug=config.aftan.debug,
        min_period=config.aftan.min_period,
        max_period=config.aftan.max_period,
        max_period_nwl=config.aftan.max_period_nwl,
        reference_velocity=config.aftan.reference_velocity,
        period_sampling=config.aftan.period_sampling,
        velocity_min=config.aftan.velocity_min,
        velocity_max=config.aftan.velocity_max,
        pi_over_4=config.aftan.pi_over_4,
        branch=config.aftan.branch,
        trig_threshold=config.aftan.trig_threshold,
        jump_points=config.aftan.jump_points,
        prediction_file=config.aftan.prediction_file,
        basic=config.aftan.basic,
        pmf=config.aftan.pmf,
        snr=config.aftan.snr,
        energy_map=new_energy_map,
        qc=config.aftan.qc,
    )
    return StationAFTANConfig(
        station=config.station,
        input_dir=config.input_dir,
        output_dir=config.output_dir,
        sac_pattern=config.sac_pattern,
        overwrite=config.overwrite,
        aftan=new_aftan,
    )


def _find_sac_files(input_dir: Path, pattern: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(str(input_dir / pattern)))]


def _build_period_sampling_config(config: dict[str, Any]) -> AFTANPeriodSamplingConfig:
    if not isinstance(config, dict):
        raise ValueError("aftan.period_sampling must be a mapping.")
    mode = str(config.get("mode", "")).lower()
    if mode not in _PERIOD_SAMPLING_MODES:
        raise ValueError(
            f"aftan.period_sampling.mode must be one of {sorted(_PERIOD_SAMPLING_MODES)}."
        )
    count = config.get("count")
    step = config.get("step")
    points_per_octave = config.get("points_per_octave")
    periods = config.get("periods")
    if mode == "geomspace" and count is None and points_per_octave is None:
        raise ValueError(
            "aftan.period_sampling.count or points_per_octave is required "
            "for geomspace."
        )
    if mode == "uniform" and count is None and step is None:
        raise ValueError("aftan.period_sampling.count or step is required for uniform mode.")
    if mode == "list" and not periods:
        raise ValueError("aftan.period_sampling.periods is required for list mode.")
    if count is not None and int(count) < 2:
        raise ValueError("aftan.period_sampling.count must be at least 2.")
    if step is not None and float(step) <= 0:
        raise ValueError("aftan.period_sampling.step must be positive.")
    if points_per_octave is not None and float(points_per_octave) <= 0:
        raise ValueError("aftan.period_sampling.points_per_octave must be positive.")
    return AFTANPeriodSamplingConfig(
        mode=mode,
        count=int(count) if count is not None else None,
        step=float(step) if step is not None else None,
        points_per_octave=float(points_per_octave)
        if points_per_octave is not None
        else None,
        periods=tuple(float(item) for item in periods) if periods is not None else None,
    )


def _build_pmf_period_bounds_config(config: dict[str, Any]) -> AFTANPMFPeriodBoundsConfig:
    if not isinstance(config, dict):
        raise ValueError("aftan.pmf.period_bounds must be a mapping.")
    mode = str(config.get("mode", "raw")).lower()
    if mode not in _PMF_PERIOD_BOUND_MODES:
        raise ValueError(
            f"aftan.pmf.period_bounds.mode must be one of "
            f"{sorted(_PMF_PERIOD_BOUND_MODES)}."
        )
    min_method = str(config.get("min_method", "floor")).lower()
    max_method = str(config.get("max_method", "ceil")).lower()
    if min_method not in _PMF_PERIOD_BOUND_METHODS:
        raise ValueError(
            f"aftan.pmf.period_bounds.min_method must be one of "
            f"{sorted(_PMF_PERIOD_BOUND_METHODS)}."
        )
    if max_method not in _PMF_PERIOD_BOUND_METHODS:
        raise ValueError(
            f"aftan.pmf.period_bounds.max_method must be one of "
            f"{sorted(_PMF_PERIOD_BOUND_METHODS)}."
        )
    step = config.get("step")
    if mode == "step" and step is None:
        step = 0.1
    step = float(step) if step is not None else None
    if step is not None and step <= 0:
        raise ValueError("aftan.pmf.period_bounds.step must be positive.")
    return AFTANPMFPeriodBoundsConfig(
        mode=mode,
        step=step,
        min_method=min_method,
        max_method=max_method,
    )


def _build_period_grid(
    config: AFTANPeriodSamplingConfig,
    min_period: float,
    max_period: float,
) -> np.ndarray:
    if min_period <= 0 or max_period <= min_period:
        raise ValueError("period range must satisfy 0 < min_period < max_period.")
    if config.mode == "list":
        periods = np.asarray(config.periods, dtype=float)
        mask = (periods >= min_period) & (periods <= max_period)
        periods = np.unique(periods[mask])
        if periods.size == 0:
            raise ValueError("list period_sampling has no periods inside the usable range.")
        return periods

    if config.mode == "geomspace":
        count = _period_count_from_sampling(config, min_period, max_period)
        if count < 2:
            raise ValueError("period_sampling must produce at least two periods.")
        return np.geomspace(min_period, max_period, count)
    if config.mode == "uniform":
        if config.step is not None:
            periods = np.arange(min_period, max_period + config.step * 0.5, config.step)
            periods = periods[periods <= max_period]
            if periods.size == 0 or periods[-1] < max_period:
                periods = np.append(periods, max_period)
            return periods
        count = _period_count_from_sampling(config, min_period, max_period)
        if count < 2:
            raise ValueError("period_sampling must produce at least two periods.")
        return np.linspace(min_period, max_period, count)
    raise ValueError(f"Unsupported period sampling mode: {config.mode}")


def _period_count_from_sampling(
    config: AFTANPeriodSamplingConfig,
    min_period: float,
    max_period: float,
) -> int:
    if config.count is not None:
        return int(config.count)
    octave_span = np.log2(max_period / min_period)
    return int(np.ceil(octave_span * float(config.points_per_octave))) + 1


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


def _pmf_period_bounds(
    apparent_period: np.ndarray,
    config_min_period: float,
    config_max_period: float,
    bounds_config: AFTANPMFPeriodBoundsConfig,
) -> tuple[float, float, float, float]:
    valid = np.asarray(apparent_period, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size < 2:
        raise ValueError("PMF requires at least two valid apparent periods.")
    raw_period_min = max(float(config_min_period), float(np.min(valid)))
    raw_period_max = min(float(config_max_period), float(np.max(valid)))
    period_min = raw_period_min
    period_max = raw_period_max

    if bounds_config.mode == "step":
        if bounds_config.step is None:
            raise ValueError("PMF step period bounds require a positive step.")
        period_min = _snap_period_bound(
            raw_period_min,
            bounds_config.step,
            bounds_config.min_method,
        )
        period_max = _snap_period_bound(
            raw_period_max,
            bounds_config.step,
            bounds_config.max_method,
        )
        period_min = max(float(config_min_period), period_min)
        period_max = min(float(config_max_period), period_max)

    if period_max <= period_min:
        raise ValueError(
            f"PMF period range is empty after first-pass bounds: "
            f"{period_min:.6g}-{period_max:.6g} s."
        )
    return period_min, period_max, raw_period_min, raw_period_max


def _snap_period_bound(value: float, step: float, method: str) -> float:
    scaled = value / step
    if method == "floor":
        snapped = np.floor(scaled)
    elif method == "ceil":
        snapped = np.ceil(scaled)
    elif method == "nearest":
        snapped = np.rint(scaled)
    else:
        raise ValueError(f"Unsupported PMF period bound method: {method}")
    return float(np.round(snapped * step, 12))


def _phase_velocity_candidate_map(
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
        scale = np.nanmax(out, axis=1, keepdims=True)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        out = out / scale
    return out


def _build_alpha_config(config: dict[str, Any]) -> AFTANAlphaConfig:
    if not isinstance(config, dict):
        raise ValueError("aftan.basic.alpha must be a mapping with a mode.")
    mode = str(config.get("mode", "")).lower()
    if mode not in _ALPHA_MODES:
        raise ValueError(f"aftan.basic.alpha.mode must be one of {sorted(_ALPHA_MODES)}.")
    value = config.get("value")
    factor = float(config.get("factor", 1.0))
    distance_nodes = config.get("distance_nodes")
    alpha_nodes = config.get("alpha_nodes")
    if mode == "constant" and value is None:
        raise ValueError("aftan.basic.alpha.value is required when mode is constant.")
    if value is not None and float(value) <= 0:
        raise ValueError("aftan.basic.alpha.value must be positive.")
    if factor <= 0:
        raise ValueError("aftan.basic.alpha.factor must be positive.")
    if (distance_nodes is None) != (alpha_nodes is None):
        raise ValueError("distance_nodes and alpha_nodes must be provided together.")
    if alpha_nodes is not None and any(float(item) <= 0 for item in alpha_nodes):
        raise ValueError("aftan.basic.alpha.alpha_nodes must be positive.")
    return AFTANAlphaConfig(
        mode=mode,
        value=float(value) if value is not None else None,
        factor=factor,
        distance_nodes=tuple(float(item) for item in distance_nodes)
        if distance_nodes is not None
        else None,
        alpha_nodes=tuple(float(item) for item in alpha_nodes)
        if alpha_nodes is not None
        else None,
    )


def _resolve_alpha(config: AFTANAlphaConfig, distance_km: float) -> float:
    if config.mode == "constant":
        return float(config.value)
    if config.mode == "empirical_distance_table":
        return gaussian_alpha_from_distance(
            distance_km,
            np.asarray(config.distance_nodes) if config.distance_nodes is not None else None,
            np.asarray(config.alpha_nodes) if config.alpha_nodes is not None else None,
        )
    if config.mode == "aftan_distance_scaling":
        return gaussian_alpha_aftan_scaling(distance_km, factor=config.factor)
    if config.mode == "pyftan_constant":
        return gaussian_alpha_pyftan_constant(factor=config.factor)
    raise ValueError(f"Unsupported alpha mode: {config.mode}")


def _prepare_branch_traces(trace: Trace, branch: str) -> list[BranchTrace]:
    if branch == "positive":
        return [BranchTrace("positive", _positive_branch_trace(trace))]
    if branch == "negative":
        if not _has_negative_lag(trace):
            return [
                BranchTrace(
                    "positive",
                    _positive_branch_trace(trace),
                    ("requested negative branch but trace is not symmetric; using positive branch",),
                )
            ]
        return [BranchTrace("negative", _negative_branch_trace(trace))]
    if branch == "both":
        if not _has_two_sided_lags(trace):
            return [
                BranchTrace(
                    "positive",
                    _positive_branch_trace(trace),
                    ("requested both branches but trace is not symmetric; using positive branch",),
                )
            ]
        return [
            BranchTrace("positive", _positive_branch_trace(trace)),
            BranchTrace("negative", _negative_branch_trace(trace)),
        ]
    if branch == "stack":
        if not _has_two_sided_lags(trace):
            return [
                BranchTrace(
                    "positive",
                    _positive_branch_trace(trace),
                    ("requested stack branch but trace is not symmetric; using positive branch",),
                )
            ]
        return [BranchTrace("stack", _stack_branch_trace(trace))]
    raise ValueError(f"Unsupported branch: {branch}")


def _positive_branch_trace(trace: Trace) -> Trace:
    times = _trace_times(trace)
    mask = times >= 0
    if not np.any(mask):
        raise ValueError("Trace does not contain positive lags.")
    return _branch_trace(trace, trace.data[mask], max(float(times[mask][0]), 0.0))


def _negative_branch_trace(trace: Trace) -> Trace:
    times = _trace_times(trace)
    mask = times <= 0
    if not np.any(mask):
        raise ValueError("Trace does not contain negative lags.")
    data = trace.data[mask][::-1]
    branch_times = np.abs(times[mask][::-1])
    order = np.argsort(branch_times)
    return _branch_trace(trace, data[order], max(float(branch_times[order][0]), 0.0))


def _stack_branch_trace(trace: Trace) -> Trace:
    positive = _positive_branch_trace(trace)
    negative = _negative_branch_trace(trace)
    npts = min(positive.stats.npts, negative.stats.npts)
    data = (positive.data[:npts] + negative.data[:npts]) / 2.0
    b = max(
        _sac_float(getattr(positive.stats, "sac", None), "b") or 0.0,
        _sac_float(getattr(negative.stats, "sac", None), "b") or 0.0,
    )
    return _branch_trace(trace, data, b)


def _branch_trace(trace: Trace, data: np.ndarray, b: float) -> Trace:
    branch = Trace(data=np.asarray(data, dtype=trace.data.dtype))
    branch.stats = trace.stats.copy()
    branch.stats.npts = branch.data.size
    if hasattr(branch.stats, "sac"):
        branch.stats.sac.b = float(b)
        branch.stats.sac.e = float(b + (branch.data.size - 1) * branch.stats.delta)
        branch.stats.sac.npts = branch.data.size
    return branch


def _has_negative_lag(trace: Trace) -> bool:
    return bool(np.any(_trace_times(trace) < 0))


def _has_two_sided_lags(trace: Trace) -> bool:
    times = _trace_times(trace)
    return bool(np.any(times < 0) and np.any(times >= 0))


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


def _load_prediction(config: AFTANConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.prediction_file is not None:
        period, velocity = np.loadtxt(config.prediction_file, unpack=True)
        return np.asarray(period), np.asarray(velocity)
    return (
        np.array([config.min_period, config.max_period], dtype=float),
        np.array([config.reference_velocity, config.reference_velocity], dtype=float),
    )


def _parabola(t, left, center, right):
    return ((left + right - 2 * center) / 2) * t**2 + ((right - left) / 2) * t + center


def _curvature(period: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    if period.size < 3:
        return np.zeros(period.size)
    x = np.log(period)
    y = velocity
    d10 = x[1:-1] - x[:-2]
    d21 = x[2:] - x[1:-1]
    d20 = x[2:] - x[:-2]
    curvature = ((y[2:] - y[1:-1]) / d21 - (y[1:-1] - y[:-2]) / d10) / d20
    return np.concatenate(([0.0], curvature, [0.0]))


def _longest_branch(trigger: np.ndarray) -> tuple[int, int]:
    indices = np.where(np.abs(trigger) == 1)[0]
    indices = np.concatenate(([0], indices, [trigger.size - 1]))
    branch = int(np.argmax(indices[1:] - indices[:-1]))
    return int(indices[branch]), int(indices[branch + 1])


def _split_or_symmetrize(trace: Trace, nlag_out: int = 1) -> list[Trace]:
    if nlag_out != 1:
        raise ValueError("Only symmetric one-lag output is supported for now.")
    sym = Trace()
    sym.data = _symmetrize_data(trace.data)
    sym.stats = trace.stats.copy()
    sym.stats.npts = sym.data.size
    sym.stats.starttime += _sac_float(trace.stats.sac, "e") or 0.0
    if hasattr(sym.stats, "sac"):
        sym.stats.sac.b = 0.0
        sym.stats.sac.npts = sym.data.size
    return [sym]


def _symmetrize_data(data: np.ndarray) -> np.ndarray:
    npts = data.size
    if npts % 2 == 0:
        left = npts // 2 - 1
        right = left + 1
        return data[left::-1] + data[right:]
    half = npts // 2
    return data[half::-1] + data[half:]


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
    signal_start = max(times[0], distance_km / config.vmax - config.bfact * max_period)
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


def _write_aftan_summary_plot(dataset, path: Path, *, phase_dataset=None, snr_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amplitude_name = (
        "normalized_amplitude"
        if "normalized_amplitude" in dataset.data_vars
        else "amplitude"
    )
    has_phase = phase_dataset is not None
    height_ratios = [0.4, 0.4, 0.2] if has_phase else [0.7, 0.3]
    nrows = 3 if has_phase else 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7.4, 7.0 if has_phase else 5.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    group_ax = axes[0]
    snr_ax = axes[-1]
    group_mesh = group_ax.pcolormesh(
        dataset["target_period"],
        dataset["velocity"],
        dataset[amplitude_name].transpose("velocity", "target_period"),
        shading="auto",
        cmap="viridis",
    )
    group_ax.plot(
        dataset["target_period"],
        dataset["picked_group_velocity"],
        color="white",
        linewidth=1.4,
    )
    group_ax.set_ylabel("Group velocity (km/s)")
    group_ax.set_title(f"{dataset.attrs.get('stage', 'ftan')} FTAN summary")
    fig.colorbar(group_mesh, ax=group_ax, label=amplitude_name.replace("_", " "))

    if has_phase:
        phase_ax = axes[1]
        phase_mesh = phase_ax.pcolormesh(
            phase_dataset["target_period"],
            phase_dataset["phase_velocity"],
            phase_dataset["phase_candidate_amplitude"].transpose(
                "phase_velocity",
                "target_period",
            ),
            shading="auto",
            cmap="magma",
        )
        phase_ax.plot(
            phase_dataset["target_period"],
            phase_dataset["picked_phase_velocity"],
            color="white",
            linewidth=1.4,
        )
        phase_ax.set_ylabel("Phase velocity (km/s)")
        fig.colorbar(phase_mesh, ax=phase_ax, label="candidate amplitude")

    snr_ax.plot(
        dataset["target_period"],
        dataset["picked_snr"],
        color="black",
        linewidth=1.2,
    )
    snr_ax.set_ylabel(snr_label)
    snr_ax.set_xlabel("Period (s)")
    snr_ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _start_log(config: StationAFTANConfig) -> list[str]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    return [
        "",
        f"[{timestamp}] aftan start",
        f"station: {config.station}",
        f"input_dir: {config.input_dir}",
        f"sac_pattern: {config.sac_pattern}",
        f"output_dir: {config.output_dir}",
        f"requested_branch: {config.aftan.branch}",
        f"debug: {config.aftan.debug}",
        f"period_range: {config.aftan.min_period}-{config.aftan.max_period} s",
        (
            "period_sampling: "
            f"mode={config.aftan.period_sampling.mode}, "
            f"count={config.aftan.period_sampling.count}, "
            f"step={config.aftan.period_sampling.step}, "
            f"points_per_octave={config.aftan.period_sampling.points_per_octave}, "
            f"n_list_periods={len(config.aftan.period_sampling.periods or ())}"
        ),
        (
            "alpha: "
            f"mode={config.aftan.basic.alpha.mode}, "
            f"value={config.aftan.basic.alpha.value}, "
            f"factor={config.aftan.basic.alpha.factor}"
        ),
        (
            "jump_correction: "
            f"trig_threshold={config.aftan.trig_threshold}, "
            f"jump_points={config.aftan.jump_points}"
        ),
        (
            "pmf: "
            f"enabled={config.aftan.pmf.enabled}, "
            "gaussian_alpha_source=basic.alpha, "
            f"period_bounds_mode={config.aftan.pmf.period_bounds.mode}, "
            f"period_bounds_step={config.aftan.pmf.period_bounds.step}, "
            f"period_bounds_min_method={config.aftan.pmf.period_bounds.min_method}, "
            f"period_bounds_max_method={config.aftan.pmf.period_bounds.max_method}"
        ),
        (
            "energy_map: "
            f"enabled={config.aftan.energy_map.enabled}, "
            f"plot={config.aftan.energy_map.plot}, "
            f"velocity_count={config.aftan.energy_map.velocity_count}, "
            f"phase_velocity={config.aftan.energy_map.phase_velocity}, "
            f"phase_cycle_count={config.aftan.energy_map.phase_cycle_count}"
        ),
        (
            "snr: "
            f"definition={config.aftan.snr.definition}, "
            f"output_db={config.aftan.snr.output_db}, "
            f"noise_mode={config.aftan.snr.noise_mode}, "
            f"signal_half_width_factor="
            f"{config.aftan.snr.signal_half_width_factor}, "
            f"signal_before_periods={config.aftan.snr.signal_before_periods}, "
            f"signal_after_periods={config.aftan.snr.signal_after_periods}, "
            f"noise_guard_factor={config.aftan.snr.noise_guard_factor}, "
            f"dsn={config.aftan.snr.dsn}, "
            f"nlen={config.aftan.snr.nlen}"
        ),
        (
            "velocity_window: "
            f"{config.aftan.velocity_min}-{config.aftan.velocity_max} km/s"
        ),
    ]


def _append_log(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")
