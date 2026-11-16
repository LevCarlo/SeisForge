"""Configuration parsing and validation for AFTAN workflows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from seisforge.ant.filters import PMF_CLEAN_WINDOW_MODES

from .io import _resolve_path
from .models import (
    AFTANAlphaConfig,
    AFTANBasicConfig,
    AFTANConfig,
    AFTANEnergyMapConfig,
    AFTANPhaseCycleConfig,
    AFTANPMFConfig,
    AFTANPeriodSamplingConfig,
    AFTANQCConfig,
    AFTANSNRConfig,
    AFTANShortDistanceGuardConfig,
    AFTANTimeTaperConfig,
    StationAFTANConfig,
)
from .phase import RADIAL_CONVENTION_MODES, normalize_component
from .taper import TIME_TAPER_MODES


_BRANCHES = {"positive", "negative", "both", "stack"}
_ALPHA_MODES = {
    "constant",
    "empirical_distance_table",
    "aftan_distance_scaling",
    "pyftan_constant",
}
_PERIOD_SAMPLING_MODES = {"geomspace", "uniform", "list"}
_SNR_DEFINITIONS = {"local", "pyftan", "aftan"}
_SNR_NOISE_MODES = {"tail", "complement"}
_PI_OVER_4_MODES = {"auto", "manual"}

# Key names removed in 0.2.0 (aliases of older SeisForge configs) -> the
# current key. A config that still uses one stops with this hint instead of
# being read silently. Paths are relative to the top of the config.
_SNR_LOCAL_KEYS = (
    "noise_mode",
    "signal_half_width_factor",
    "signal_before_periods",
    "signal_after_periods",
    "noise_guard_factor",
)
_SNR_PYFTAN_KEYS = ("bfact", "efact", "vmax", "vmin", "fill")
REMOVED_AFTAN_KEYS: dict[tuple[str, ...], str] = {
    ("component",): "components",
    ("io", "source_station"): "source_station (top level)",
    ("io", "receiver_station"): "receiver_station (top level)",
    ("io", "station"): "station (top level)",
    ("io", "component"): "components (top level)",
    ("io", "components"): "components (top level)",
    ("io", "input_root_dir"): "io.input_root_datadir",
    ("io", "output_root_dir"): "io.output_root_datadir",
    ("aftan", "velocity_min"): "aftan.velocity_window.min",
    ("aftan", "velocity_max"): "aftan.velocity_window.max",
    ("aftan", "trig_threshold"): "aftan.basic.trig_threshold and aftan.pmf.trig_threshold",
    ("aftan", "jump_points"): "aftan.basic.jump_points and aftan.pmf.jump_points",
    ("aftan", "reference_velocity"): "short_distance_period_guard.reference_velocity",
    ("aftan", "max_period_nwl"): "short_distance_period_guard.max_period_nwl",
    ("short_distance_period_guard", "reference_velocity_km_s"): (
        "short_distance_period_guard.reference_velocity"
    ),
    ("short_distance_period_guard", "min_wavelengths"): (
        "short_distance_period_guard.max_period_nwl"
    ),
    **{("aftan", "snr", key): f"aftan.snr.local.{key}" for key in _SNR_LOCAL_KEYS},
    ("aftan", "snr", "signal_half_width_periods"): "aftan.snr.local.signal_half_width_factor",
    ("aftan", "snr", "noise_guard_periods"): "aftan.snr.local.noise_guard_factor",
    ("aftan", "snr", "local", "signal_half_width_periods"): (
        "aftan.snr.local.signal_half_width_factor"
    ),
    ("aftan", "snr", "local", "noise_guard_periods"): "aftan.snr.local.noise_guard_factor",
    **{("aftan", "snr", key): f"aftan.snr.pyftan.{key}" for key in _SNR_PYFTAN_KEYS},
    **{
        ("aftan", "snr", key): f"aftan.snr.local.tail.{key} or aftan.snr.pyftan.{key}"
        for key in ("dsn", "nlen")
    },
    ("aftan", "time_taper", "ramp_s"): "aftan.time_taper.velocity_window.ramp_s",
    ("aftan", "time_taper", "fraction"): "aftan.time_taper.legacy_edge_fraction.fraction",
    **{
        ("aftan", "period_sampling", key): f"aftan.period_sampling.<mode>.{key}"
        for key in ("count", "step", "dfreq", "min_count", "periods")
    },
    **{
        ("aftan", "period_sampling", mode, "min_nfreq"): (
            f"aftan.period_sampling.{mode}.min_count"
        )
        for mode in ("geomspace", "uniform", "list")
    },
    **{
        ("aftan", stage, "alpha", key): f"aftan.{stage}.alpha.<mode>.{key}"
        for stage in ("basic", "pmf")
        for key in ("value", "factor", "distance_nodes", "alpha_nodes")
    },
}


def _reject_removed_keys(config: dict[str, Any]) -> None:
    """Stop on any key of `REMOVED_AFTAN_KEYS`, naming the current key."""
    found = []
    for path, current in REMOVED_AFTAN_KEYS.items():
        node: Any = config
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and path[-1] in node:
            found.append(f"{'.'.join(path)} (use {current})")
    if found:
        raise ValueError(
            "AFTAN config uses removed (unknown) keys: " + "; ".join(found) + "."
        )


def build_station_aftan_config(
    config: dict[str, Any],
    base_dir: str | Path | None = None,
) -> StationAFTANConfig:
    """Build a station-level AFTAN config from a plain YAML dictionary."""
    base_path = Path(base_dir) if base_dir is not None else None
    _reject_removed_keys(config)
    io_config = config.get("io", {})
    aftan_config = config.get("aftan", {})
    if not isinstance(aftan_config, dict):
        raise ValueError("aftan must be a mapping.")
    if "phase_cycle" in aftan_config:
        raise ValueError(
            "aftan.phase_cycle is not supported; use top-level "
            "phase_cycle_selection with only enabled and max_shift."
        )

    source_station = _optional_string(config.get("source_station"))
    receiver_station = _optional_string(config.get("receiver_station"))
    components = _build_components(config.get("components"))

    station = _optional_string(config.get("station"))
    if source_station is not None and station is None:
        station = source_station

    if components:
        if not source_station or not receiver_station:
            raise ValueError(
                "AFTAN config with component(s) requires source_station and "
                "receiver_station."
            )
        if "sac_pattern" in io_config:
            raise ValueError(
                "AFTAN pair config (with components) uses io.input_template, "
                "not io.sac_pattern."
            )
        input_root = io_config.get("input_root_datadir")
        output_root = io_config.get("output_root_datadir")
        if input_root is None:
            raise ValueError("AFTAN config io.input_root_datadir is required.")
        if output_root is None:
            raise ValueError("AFTAN config io.output_root_datadir is required.")
        pair_name = f"{source_station}_{receiver_station}"
        input_dir = _resolve_path(input_root, base_path) / source_station / pair_name
        output_dir = _resolve_path(output_root, base_path) / source_station / pair_name
        input_template = str(io_config.get("input_template", "*_{component}.SAC"))
        sac_pattern = input_template
    else:
        if not station:
            raise ValueError(
                "AFTAN config requires either station or source_station/receiver_station."
            )
        for key in ("input_dir", "output_dir"):
            if key not in io_config:
                raise ValueError(f"AFTAN config io.{key} is required.")
        input_dir = _resolve_path(io_config["input_dir"], base_path)
        output_dir = _resolve_path(io_config["output_dir"], base_path)
        input_template = None
        sac_pattern = str(io_config.get("sac_pattern", "*.SAC"))

    return StationAFTANConfig(
        station=station,
        input_dir=input_dir,
        output_dir=output_dir,
        sac_pattern=sac_pattern,
        overwrite=bool(io_config.get("overwrite", False)),
        aftan=_build_aftan_config(
            aftan_config,
            base_path,
            phase_cycle_selection=_build_phase_cycle_selection_config(
                config.get("phase_cycle_selection"),
            ),
            short_distance_guard=_build_short_distance_guard_config(
                config.get("short_distance_period_guard"),
            ),
        ),
        source_station=source_station,
        receiver_station=receiver_station,
        components=components,
        input_template=input_template,
    )


def _build_aftan_config(
    config: dict[str, Any],
    base_path: Path | None,
    *,
    phase_cycle_selection: AFTANPhaseCycleConfig | None = None,
    short_distance_guard: AFTANShortDistanceGuardConfig | None = None,
) -> AFTANConfig:
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
    pi_over_4, pi_over_4_mode = _build_pi_over_4_config(config)
    aftan = AFTANConfig(
        debug=bool(config.get("debug", False)),
        min_period=float(config.get("min_period", 0.5)),
        max_period=float(config.get("max_period", 10.0)),
        period_sampling=_build_period_sampling_config(
            config.get("period_sampling", {})
        ),
        velocity_min=float(velocity_window.get("min", 0.5)),
        velocity_max=float(velocity_window.get("max", 5.5)),
        pi_over_4=pi_over_4,
        pi_over_4_mode=pi_over_4_mode,
        radial_convention=str(config.get("radial_convention", "auto")).strip().lower(),
        branch=branch,
        prediction_file=(
            _resolve_path(prediction_file, base_path) if prediction_file else None
        ),
        basic=AFTANBasicConfig(
            alpha=_build_alpha_config(basic.get("alpha", {}), label="aftan.basic.alpha"),
            trig_threshold=float(basic.get("trig_threshold", 50.0)),
            jump_points=int(basic.get("jump_points", 3)),
        ),
        pmf=AFTANPMFConfig(
            enabled=bool(pmf.get("enabled", False)),
            alpha=_build_alpha_config(
                pmf.get("alpha", basic.get("alpha", {})),
                label="aftan.pmf.alpha",
            ),
            trig_threshold=float(pmf.get("trig_threshold", 20.0)),
            jump_points=int(pmf.get("jump_points", 3)),
            nalpha=float(pmf.get("nalpha", 2.0)),
            min_half_length=float(pmf.get("min_half_length", 5.0)),
            amplitude_ratio=float(pmf.get("amplitude_ratio", 0.2)),
            window_factor=float(pmf.get("window_factor", 1.0)),
            clean_window=str(pmf.get("clean_window", "symmetric")),
        ),
        snr=_build_snr_config(snr),
        time_taper=_build_time_taper_config(config.get("time_taper", {})),
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
        phase_cycle=(
            phase_cycle_selection
            if phase_cycle_selection is not None
            else AFTANPhaseCycleConfig()
        ),
        short_distance_guard=(
            short_distance_guard
            if short_distance_guard is not None
            else AFTANShortDistanceGuardConfig()
        ),
    )
    _validate_aftan_config(aftan)
    return aftan


def _validate_aftan_config(config: AFTANConfig) -> None:
    if config.min_period <= 0:
        raise ValueError("aftan.min_period must be positive.")
    if config.max_period <= config.min_period:
        raise ValueError("aftan.max_period must be greater than aftan.min_period.")
    if config.short_distance_guard.reference_velocity <= 0:
        raise ValueError(
            "short_distance_period_guard.reference_velocity must be positive."
        )
    if config.short_distance_guard.max_period_nwl <= 0:
        raise ValueError("short_distance_period_guard.max_period_nwl must be positive.")
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
    if config.pmf.clean_window not in PMF_CLEAN_WINDOW_MODES:
        raise ValueError(
            f"aftan.pmf.clean_window must be one of {PMF_CLEAN_WINDOW_MODES}."
        )
    if config.energy_map.velocity_count < 2:
        raise ValueError("aftan.energy_map.velocity_count must be at least 2.")
    if config.energy_map.phase_cycle_count < 0:
        raise ValueError("aftan.energy_map.phase_cycle_count must be non-negative.")
    if config.phase_cycle.max_shift < 0:
        raise ValueError("phase_cycle_selection.max_shift must be non-negative.")
    if config.qc.period_rel_warning < 0:
        raise ValueError("aftan.qc.period_rel_warning must be non-negative.")
    if not 0 <= config.qc.min_valid_fraction <= 1:
        raise ValueError("aftan.qc.min_valid_fraction must be between 0 and 1.")
    if config.pi_over_4_mode not in _PI_OVER_4_MODES:
        raise ValueError(f"aftan.pi_over_4.mode must be one of {sorted(_PI_OVER_4_MODES)}.")
    if config.radial_convention not in RADIAL_CONVENTION_MODES:
        raise ValueError(
            f"aftan.radial_convention must be one of {list(RADIAL_CONVENTION_MODES)}."
        )


def _build_short_distance_guard_config(config: Any) -> AFTANShortDistanceGuardConfig:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("short_distance_period_guard must be a mapping.")
    return AFTANShortDistanceGuardConfig(
        enabled=bool(config.get("enabled", True)),
        reference_velocity=float(config.get("reference_velocity", 4.0)),
        max_period_nwl=float(config.get("max_period_nwl", 0.5)),
    )


def _build_phase_cycle_selection_config(
    config: Any,
) -> AFTANPhaseCycleConfig:
    """Build the top-level SeisForge phase-cycle selection extension."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("phase_cycle_selection must be a mapping.")
    unexpected = sorted(set(config).difference({"enabled", "max_shift"}))
    if unexpected:
        raise ValueError(
            "phase_cycle_selection has unsupported fields: "
            + ", ".join(unexpected)
        )
    return AFTANPhaseCycleConfig(
        enabled=bool(config.get("enabled", True)),
        max_shift=int(config.get("max_shift", 2)),
    )


def _build_pi_over_4_config(config: dict[str, Any]) -> tuple[float, str]:
    # Default auto in both run modes: the constant is looked up per file from
    # its stored component (pair config, else SAC kcmpnm / file name).
    value = config.get("pi_over_4")
    if value is None:
        return -1.0, "auto"
    if isinstance(value, dict):
        mode = str(value.get("mode", "auto")).lower()
        if mode not in _PI_OVER_4_MODES:
            raise ValueError(
                f"aftan.pi_over_4.mode must be one of {sorted(_PI_OVER_4_MODES)}."
            )
        if mode == "manual":
            if "value" not in value:
                raise ValueError("aftan.pi_over_4.value is required for manual mode.")
            return float(value["value"]), mode
        return -1.0, mode
    return float(value), "manual"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _build_components(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    components = tuple(normalize_component(item) for item in items if item)
    if not components:
        raise ValueError("AFTAN component(s) cannot be empty.")
    if len(set(components)) != len(components):
        raise ValueError("AFTAN component(s) must not contain duplicates.")
    return components


def _nested_option_config(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config option {key} must be a mapping.")
    return value


def _build_snr_config(config: dict[str, Any]) -> AFTANSNRConfig:
    if not isinstance(config, dict):
        raise ValueError("aftan.snr must be a mapping.")
    definition = str(config.get("definition", "local")).lower()
    if definition not in _SNR_DEFINITIONS:
        raise ValueError(f"aftan.snr.definition must be one of {sorted(_SNR_DEFINITIONS)}.")
    local_config = _nested_option_config(config, "local")
    pyftan_config = _nested_option_config(config, "pyftan")
    local_tail_config = _nested_option_config(local_config, "tail")
    noise_mode = str(local_config.get("noise_mode", "tail")).lower()
    if definition == "local" and noise_mode not in _SNR_NOISE_MODES:
        raise ValueError(f"aftan.snr.noise_mode must be one of {sorted(_SNR_NOISE_MODES)}.")
    signal_half_width_factor = float(local_config.get("signal_half_width_factor", 1.0))
    signal_before_value = local_config.get("signal_before_periods")
    signal_after_value = local_config.get("signal_after_periods")
    signal_before_periods = (
        float(signal_before_value) if signal_before_value is not None else None
    )
    signal_after_periods = (
        float(signal_after_value) if signal_after_value is not None else None
    )
    noise_guard_factor = float(local_config.get("noise_guard_factor", 1.0))
    if definition == "local" and signal_half_width_factor < 0:
        raise ValueError("aftan.snr.signal_half_width_factor must be non-negative.")
    if definition == "local" and signal_before_periods is not None and signal_before_periods < 0:
        raise ValueError("aftan.snr.signal_before_periods must be non-negative.")
    if definition == "local" and signal_after_periods is not None and signal_after_periods < 0:
        raise ValueError("aftan.snr.signal_after_periods must be non-negative.")
    if definition == "local" and noise_guard_factor < 0:
        raise ValueError("aftan.snr.noise_guard_factor must be non-negative.")
    bfact = float(pyftan_config.get("bfact", 1.0))
    efact = float(pyftan_config.get("efact", 0.0))
    # The tail-noise window of the definition in use: local.tail or pyftan.
    tail_config = pyftan_config if definition == "pyftan" else local_tail_config
    dsn = float(tail_config.get("dsn", 500.0))
    nlen = float(tail_config.get("nlen", 500.0))
    vmax = float(pyftan_config.get("vmax", 4.5))
    vmin = float(pyftan_config.get("vmin", 1.0))
    fill = float(pyftan_config.get("fill", 3.0))
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
    new_aftan = replace(config.aftan, energy_map=new_energy_map)
    return replace(config, aftan=new_aftan)


def _build_time_taper_config(config: Any) -> AFTANTimeTaperConfig:
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("aftan.time_taper must be a mapping.")
    mode = str(config.get("mode", "velocity_window")).lower()
    if mode not in TIME_TAPER_MODES:
        raise ValueError(f"aftan.time_taper.mode must be one of {list(TIME_TAPER_MODES)}.")
    window_config = _nested_option_config(config, "velocity_window")
    legacy_config = _nested_option_config(config, "legacy_edge_fraction")
    ramp_s = window_config.get("ramp_s")
    fraction = float(legacy_config.get("fraction", 0.05))
    if ramp_s is not None and float(ramp_s) < 0:
        raise ValueError("aftan.time_taper.velocity_window.ramp_s must be non-negative.")
    if not 0 < fraction <= 0.5:
        raise ValueError("aftan.time_taper.legacy_edge_fraction.fraction must be in (0, 0.5].")
    return AFTANTimeTaperConfig(
        mode=mode,
        ramp_s=float(ramp_s) if ramp_s is not None else None,
        legacy_edge_fraction=fraction,
    )


def _build_period_sampling_config(config: dict[str, Any]) -> AFTANPeriodSamplingConfig:
    if not isinstance(config, dict):
        raise ValueError("aftan.period_sampling must be a mapping.")
    mode = str(config.get("mode", "")).lower()
    if mode not in _PERIOD_SAMPLING_MODES:
        raise ValueError(
            f"aftan.period_sampling.mode must be one of {sorted(_PERIOD_SAMPLING_MODES)}."
        )
    mode_config = _nested_option_config(config, mode)
    count = mode_config.get("count")
    step = mode_config.get("step")
    dfreq = mode_config.get("dfreq")
    min_count = mode_config.get("min_count")
    periods = mode_config.get("periods")
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
    mode_config = _nested_option_config(config, mode)
    value = mode_config.get("value")
    factor = float(mode_config.get("factor", 1.0))
    distance_nodes = mode_config.get("distance_nodes")
    alpha_nodes = mode_config.get("alpha_nodes")
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
