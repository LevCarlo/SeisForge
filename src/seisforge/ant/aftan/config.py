"""Configuration parsing and validation for AFTAN workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import _resolve_path
from .models import (
    AFTANAlphaConfig,
    AFTANBasicConfig,
    AFTANConfig,
    AFTANEnergyMapConfig,
    AFTANPMFConfig,
    AFTANPMFPeriodBoundsConfig,
    AFTANPeriodSamplingConfig,
    AFTANQCConfig,
    AFTANSNRConfig,
    StationAFTANConfig,
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
        prediction_file=(
            _resolve_path(prediction_file, base_path) if prediction_file else None
        ),
        basic=AFTANBasicConfig(
            alpha=_build_alpha_config(basic.get("alpha", {}), label="aftan.basic.alpha"),
            trig_threshold=float(
                basic.get("trig_threshold", config.get("trig_threshold", 50.0))
            ),
            jump_points=int(basic.get("jump_points", config.get("jump_points", 3))),
        ),
        pmf=AFTANPMFConfig(
            enabled=bool(pmf.get("enabled", False)),
            alpha=_build_alpha_config(
                pmf.get("alpha", basic.get("alpha", {})),
                label="aftan.pmf.alpha",
            ),
            trig_threshold=float(
                pmf.get("trig_threshold", config.get("trig_threshold", 20.0))
            ),
            jump_points=int(pmf.get("jump_points", config.get("jump_points", 3))),
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
    if config.basic.trig_threshold <= 0:
        raise ValueError("aftan.basic.trig_threshold must be positive.")
    if config.basic.jump_points < 0:
        raise ValueError("aftan.basic.jump_points must be non-negative.")
    if config.pmf.trig_threshold <= 0:
        raise ValueError("aftan.pmf.trig_threshold must be positive.")
    if config.pmf.jump_points < 0:
        raise ValueError("aftan.pmf.jump_points must be non-negative.")
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
    if definition == "local" and noise_mode not in _SNR_NOISE_MODES:
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
    if definition == "local" and signal_half_width_factor < 0:
        raise ValueError("aftan.snr.signal_half_width_factor must be non-negative.")
    if definition == "local" and signal_before_periods is not None and signal_before_periods < 0:
        raise ValueError("aftan.snr.signal_before_periods must be non-negative.")
    if definition == "local" and signal_after_periods is not None and signal_after_periods < 0:
        raise ValueError("aftan.snr.signal_after_periods must be non-negative.")
    if definition == "local" and noise_guard_factor < 0:
        raise ValueError("aftan.snr.noise_guard_factor must be non-negative.")
    bfact = float(config.get("bfact", 1.0))
    efact = float(config.get("efact", 0.0))
    dsn = float(config.get("dsn", 500.0))
    nlen = float(config.get("nlen", 500.0))
    vmax = float(config.get("vmax", 4.5))
    vmin = float(config.get("vmin", 1.0))
    fill = float(config.get("fill", 3.0))
    if fill <= 0:
        raise ValueError("aftan.snr.fill must be positive.")
    uses_tail_window = definition == "pyftan" or (
        definition == "local" and noise_mode == "tail"
    )
    if uses_tail_window and dsn < 0:
        raise ValueError("aftan.snr.dsn must be non-negative.")
    if uses_tail_window and nlen <= 0:
        raise ValueError("aftan.snr.nlen must be positive.")
    if definition == "pyftan" and (vmin <= 0 or vmax <= 0):
        raise ValueError("aftan.snr.vmin/vmax must be positive.")
    if definition == "pyftan" and vmax <= vmin:
        raise ValueError("aftan.snr.vmax must be greater than vmin.")
    return AFTANSNRConfig(
        definition=definition,
        output_db=bool(config.get("output_db", False)),
        noise_mode=noise_mode,
        signal_half_width_factor=signal_half_width_factor,
        signal_before_periods=signal_before_periods,
        signal_after_periods=signal_after_periods,
        noise_guard_factor=noise_guard_factor,
        bfact=bfact,
        efact=efact,
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
    dfreq = config.get("dfreq")
    min_count = config.get("min_count", config.get("min_nfreq"))
    periods = config.get("periods")
    if mode == "geomspace" and count is None and dfreq is None:
        raise ValueError(
            "aftan.period_sampling.count or dfreq is required "
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
    if dfreq is not None and float(dfreq) <= 0:
        raise ValueError("aftan.period_sampling.dfreq must be positive.")
    if min_count is not None and int(min_count) < 2:
        raise ValueError("aftan.period_sampling.min_count must be at least 2.")
    return AFTANPeriodSamplingConfig(
        mode=mode,
        count=int(count) if count is not None else None,
        step=float(step) if step is not None else None,
        dfreq=float(dfreq) if dfreq is not None else None,
        min_count=int(min_count) if min_count is not None else None,
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


def _build_alpha_config(
    config: dict[str, Any],
    *,
    label: str = "aftan.basic.alpha",
) -> AFTANAlphaConfig:
    if not isinstance(config, dict):
        raise ValueError(f"{label} must be a mapping with a mode.")
    mode = str(config.get("mode", "")).lower()
    if mode not in _ALPHA_MODES:
        raise ValueError(f"{label}.mode must be one of {sorted(_ALPHA_MODES)}.")
    value = config.get("value")
    factor = float(config.get("factor", 1.0))
    distance_nodes = config.get("distance_nodes")
    alpha_nodes = config.get("alpha_nodes")
    if mode == "constant" and value is None:
        raise ValueError(f"{label}.value is required when mode is constant.")
    if value is not None and float(value) <= 0:
        raise ValueError(f"{label}.value must be positive.")
    if factor <= 0:
        raise ValueError(f"{label}.factor must be positive.")
    if (distance_nodes is None) != (alpha_nodes is None):
        raise ValueError("distance_nodes and alpha_nodes must be provided together.")
    if alpha_nodes is not None and any(float(item) <= 0 for item in alpha_nodes):
        raise ValueError(f"{label}.alpha_nodes must be positive.")
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
