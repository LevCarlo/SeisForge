from __future__ import annotations

import numpy as np
from obspy import Trace
import pytest
import xarray as xr

from seisforge.ant.aftan import (
    AFTANSNRConfig,
    AFTANQCConfig,
    AFTANPMFPeriodBoundsConfig,
    _align_to_period_grid,
    _aftan_diagram_snr,
    _override_energy_map_config,
    _period_qc,
    _pmf_period_bounds,
    _qc_warnings,
    _snr,
    build_station_aftan_config,
    run_aftan_config,
)


def _write_wave_packet(path, distance_km=100.0, velocity_km_s=3.0):
    delta = 0.5
    time = np.arange(0.0, 400.0, delta)
    center = distance_km / velocity_km_s
    envelope = np.exp(-0.5 * ((time - center) / 10.0) ** 2)
    data = envelope * np.cos(2 * np.pi * time / 5.0)
    trace = Trace(data=data.astype(np.float32))
    trace.stats.delta = delta
    trace.stats.sac = {"b": 0.0, "dist": distance_km}
    trace.write(str(path), format="SAC")


def _write_symmetric_wave_packet(path, distance_km=120.0, velocity_km_s=3.0):
    delta = 0.5
    time = np.arange(-400.0, 400.0 + delta, delta)
    center = distance_km / velocity_km_s
    positive = np.exp(-0.5 * ((time - center) / 10.0) ** 2)
    negative = np.exp(-0.5 * ((time + center) / 10.0) ** 2)
    data = (positive + negative) * np.cos(2 * np.pi * time / 5.0)
    trace = Trace(data=data.astype(np.float32))
    trace.stats.delta = delta
    trace.stats.sac = {"b": time[0], "dist": distance_km}
    trace.write(str(path), format="SAC")


def _alpha_config_lines(mode="constant"):
    if mode == "constant":
        return [
            "    alpha:",
            "      mode: constant",
            "      value: 20.0",
        ]
    if mode == "empirical_distance_table":
        return [
            "    alpha:",
            "      mode: empirical_distance_table",
        ]
    raise ValueError(mode)


def test_build_station_aftan_config_is_station_scoped(tmp_path):
    config = {
        "station": "WT.2001",
        "io": {
            "input_dir": "ccf",
            "output_dir": "ftan",
            "sac_pattern": "*_ZR.SAC",
        },
        "aftan": {
            "branch": "positive",
            "min_period": 2.0,
            "max_period": 8.0,
            "period_sampling": {"mode": "uniform", "step": 0.5},
            "velocity_window": {"min": 2.0, "max": 4.5},
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
            "snr": {"noise_mode": "complement"},
        },
    }

    built = build_station_aftan_config(config, base_dir=tmp_path)

    assert built.station == "WT.2001"
    assert built.input_dir == tmp_path / "ccf"
    assert built.output_dir == tmp_path / "ftan"
    assert built.sac_pattern == "*_ZR.SAC"
    assert built.aftan.debug is False
    assert built.aftan.branch == "positive"
    assert built.aftan.min_period == 2.0
    assert built.aftan.velocity_min == 2.0
    assert built.aftan.period_sampling.mode == "uniform"
    assert built.aftan.period_sampling.step == 0.5
    assert built.aftan.basic.alpha.mode == "constant"
    assert built.aftan.basic.alpha.value == 18.0
    assert built.aftan.snr.noise_mode == "complement"


def test_energy_map_override_preserves_jump_correction_config(tmp_path):
    config = {
        "station": "WT.2001",
        "io": {
            "input_dir": "ccf",
            "output_dir": "ftan",
            "sac_pattern": "*_ZR.SAC",
        },
        "aftan": {
            "branch": "positive",
            "min_period": 2.0,
            "max_period": 8.0,
            "period_sampling": {"mode": "uniform", "step": 0.5},
            "velocity_window": {"min": 2.0, "max": 4.5},
            "trig_threshold": 123.0,
            "jump_points": 7,
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
        },
    }
    built = build_station_aftan_config(config, base_dir=tmp_path)

    overridden = _override_energy_map_config(
        built,
        write_energy_map=True,
        plot_energy_map=True,
    )

    assert overridden.aftan.trig_threshold == 123.0
    assert overridden.aftan.jump_points == 7


def test_build_station_aftan_config_rejects_invalid_numeric_parameters(tmp_path):
    config = {
        "station": "WT.2001",
        "io": {
            "input_dir": "ccf",
            "output_dir": "ftan",
        },
        "aftan": {
            "branch": "positive",
            "min_period": 2.0,
            "max_period": 8.0,
            "max_period_nwl": 0.0,
            "period_sampling": {"mode": "uniform", "step": 0.5},
            "velocity_window": {"min": 2.0, "max": 4.5},
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
        },
    }

    with pytest.raises(ValueError, match="max_period_nwl"):
        build_station_aftan_config(config, base_dir=tmp_path)


def test_short_branch_qc_warns_when_retained_fraction_is_small():
    measured = {
        "initial_target_period_count": np.asarray(10),
        "target_period": np.array([1.0, 2.0]),
        "period": np.array([1.0, 2.0]),
        "group_velocity": np.array([3.0, 3.1]),
        "amplitude": np.array([1.0, 0.8]),
    }

    qc = _period_qc(measured)
    warnings = _qc_warnings(
        qc,
        AFTANQCConfig(min_valid_fraction=0.5, fail_on_short_branch=False),
    )

    assert qc["qc_retained_period_count"] == 2
    assert qc["qc_initial_period_count"] == 10
    assert np.isclose(qc["qc_retained_fraction"], 0.2)
    assert any("short final branch" in warning for warning in warnings)


def test_snr_complement_uses_noise_outside_period_scaled_signal_window():
    delta = 1.0
    time = np.arange(0.0, 101.0, delta)
    data = np.full(time.size, 2.0, dtype=np.float32)
    data[(time >= 48.0) & (time <= 52.0)] = 10.0
    trace = Trace(data=data)
    trace.stats.delta = delta
    trace.stats.sac = {"b": 0.0}
    config = AFTANSNRConfig(
        noise_mode="complement",
        signal_half_width_factor=1.0,
        noise_guard_factor=1.0,
    )

    snr = _snr(trace, distance_km=150.0, period=2.0, group_velocity=3.0, config=config)

    assert np.isclose(snr, 5.0)


def test_snr_can_be_output_in_db():
    delta = 1.0
    time = np.arange(0.0, 101.0, delta)
    data = np.full(time.size, 2.0, dtype=np.float32)
    data[(time >= 48.0) & (time <= 52.0)] = 10.0
    trace = Trace(data=data)
    trace.stats.delta = delta
    trace.stats.sac = {"b": 0.0}
    config = AFTANSNRConfig(
        output_db=True,
        noise_mode="complement",
        signal_half_width_factor=1.0,
        noise_guard_factor=1.0,
    )

    snr = _snr(trace, distance_km=150.0, period=2.0, group_velocity=3.0, config=config)

    assert np.isclose(snr, 20.0 * np.log10(5.0))


def test_aftan_diagram_snr_uses_peak_over_neighboring_minima():
    envelope = np.array([4.0, 1.0, 10.0, 2.0, 6.0], dtype=float)
    config = AFTANSNRConfig(definition="aftan")

    snr = _aftan_diagram_snr(envelope, peak_index=2, config=config)

    assert np.isclose(snr, 10.0 / np.sqrt(1.0 * 2.0))


def test_run_aftan_config_writes_dispersion_outputs(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  branch: positive",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    results = run_aftan_config(config_file)

    assert len(results) == 1
    result = results[0]
    assert result.output_npz.exists()
    assert result.output_dat.exists()
    dat_lines = result.output_dat.read_text().splitlines()
    data_lines = [line for line in dat_lines if line and not line.startswith("#")]
    assert data_lines
    assert {len(line.split()) for line in data_lines} == {6}
    for line in data_lines:
        fields = line.split()
        assert "e" in fields[4].lower()
        assert not any("e" in field.lower() for field in fields[:4] + fields[5:])
    assert result.output_npz.exists()
    with np.load(result.output_npz) as data:
        assert "hilbert_instant_period" not in data
    log_file = tmp_path / "ftan" / "aftan.log"
    assert log_file.exists()
    log_text = log_file.read_text()
    assert "requested_branch: positive" in log_text
    assert "alpha_mode=constant" in log_text
    assert "qc WT.2001_WT.2100_ZR_pws.SAC [positive]" in log_text
    assert "period_sampling: mode=uniform" in log_text
    assert result.branch == "positive"
    assert result.alpha == 20.0
    assert result.alpha_mode == "constant"
    assert result.output_energy_map is None
    assert result.output_energy_plot is None
    assert result.target_period.size == result.period.size
    assert result.amplitude.size == result.period.size
    assert result.period.size > 0
    assert np.all(np.isfinite(result.group_velocity))
    assert np.all(np.isfinite(result.phase_velocity))
    assert np.nanmedian(result.group_velocity) > 2.0
    assert np.nanmedian(result.group_velocity) < 4.5


def test_run_aftan_config_optionally_writes_energy_map_and_plot(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  branch: positive",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  energy_map:",
                "    enabled: true",
                "    plot: true",
                "    velocity_count: 64",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines("empirical_distance_table"),
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    assert result.output_energy_map is not None
    assert result.output_energy_map.exists()
    assert result.output_energy_plot is not None
    assert result.output_energy_plot.exists()
    dataset = xr.open_dataset(result.output_energy_map, engine="scipy")
    try:
        assert dataset.attrs["stage"] == "basic"
        assert dataset.attrs["branch"] == "positive"
        assert dataset.attrs["alpha_mode"] == "empirical_distance_table"
        assert dataset.attrs["period_sampling_mode"] == "uniform"
        assert dataset.attrs["alpha"] == 8.0
        assert dataset.sizes["velocity"] == 64
        assert dataset.sizes["target_period"] > 0
        assert dataset["picked_group_velocity"].sizes["target_period"] == dataset.sizes[
            "target_period"
        ]
        assert "amplitude" in dataset
        assert "normalized_amplitude" in dataset
        assert "picked_group_velocity" in dataset
        assert dataset["picked_group_velocity"].dims == ("target_period",)
        assert "picked_hilbert_instant_period" not in dataset
    finally:
        dataset.close()


def test_align_to_period_grid_fills_missing_picked_periods():
    target_period = np.array([1.0, 2.0, 3.0, 4.0])
    picked_period = np.array([2.0, 4.0])
    values = np.array([3.1, 3.4])

    aligned = _align_to_period_grid(target_period, picked_period, values)

    assert np.isnan(aligned[0])
    assert aligned[1] == 3.1
    assert np.isnan(aligned[2])
    assert aligned[3] == 3.4


def test_run_aftan_config_writes_debug_phase_outputs_when_requested(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  debug: true",
                "  branch: positive",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  energy_map:",
                "    enabled: true",
                "    velocity_count: 64",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    data_lines = [
        line
        for line in result.output_dat.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert {len(line.split()) for line in data_lines} == {10}
    for line in data_lines:
        fields = line.split()
        assert "e" in fields[4].lower()
        assert not any("e" in field.lower() for field in fields[:4] + fields[5:])
    with np.load(result.output_npz) as data:
        assert bool(data["debug"])
        assert "hilbert_instant_period" in data
    dataset = xr.open_dataset(result.output_energy_map, engine="scipy")
    try:
        assert dataset.attrs["debug"]
        assert dataset["picked_hilbert_instant_period"].dims == ("target_period",)
        assert dataset["picked_instant_period_delta"].dims == ("target_period",)
    finally:
        dataset.close()


def test_run_aftan_config_optionally_writes_phase_velocity_map(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  branch: positive",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  energy_map:",
                "    enabled: true",
                "    plot: true",
                "    phase_velocity: true",
                "    phase_cycle_count: 3",
                "    velocity_count: 64",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    assert result.output_phase_map is not None
    assert result.output_phase_map.exists()
    assert result.output_phase_plot is None
    assert result.output_energy_plot is not None
    assert result.output_energy_plot.exists()
    energy_dataset = xr.open_dataset(result.output_energy_map, engine="scipy")
    try:
        assert energy_dataset["picked_snr"].dims == ("target_period",)
    finally:
        energy_dataset.close()
    dataset = xr.open_dataset(result.output_phase_map, engine="scipy")
    try:
        assert dataset.attrs["phase_cycle_count"] == 3
        assert dataset.sizes["phase_velocity"] == 64
        assert dataset["phase_candidate_amplitude"].dims == (
            "target_period",
            "phase_velocity",
        )
        assert dataset["picked_phase_velocity"].dims == ("target_period",)
    finally:
        dataset.close()


def test_run_aftan_config_optionally_writes_pmf_outputs(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  branch: positive",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
                "  pmf:",
                "    enabled: true",
                "    period_bounds:",
                "      mode: step",
                "      step: 0.1",
                "      min_method: floor",
                "      max_method: ceil",
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    assert result.output_pmf_npz is not None
    assert result.output_pmf_npz.exists()
    assert result.output_pmf_dat is not None
    assert result.output_pmf_dat.exists()
    pmf_lines = [
        line
        for line in result.output_pmf_dat.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert pmf_lines
    assert {len(line.split()) for line in pmf_lines} == {6}
    with np.load(result.output_pmf_npz) as data:
        assert data["stage"] == "pmf"
        assert data["pmf_raw_period_min"] >= result.period.min()
        assert data["pmf_raw_period_max"] <= result.period.max()
        assert data["pmf_period_min"] <= data["pmf_raw_period_min"]
        assert data["pmf_period_max"] >= data["pmf_raw_period_max"]
        assert data["pmf_period_bounds_mode"] == "step"
        assert data["pmf_period_bounds_step"] == 0.1
        assert data["target_period"].size > 0


def test_pmf_period_bounds_can_snap_to_decimal_step():
    bounds = AFTANPMFPeriodBoundsConfig(
        mode="step",
        step=0.1,
        min_method="floor",
        max_method="ceil",
    )

    period_min, period_max, raw_period_min, raw_period_max = _pmf_period_bounds(
        np.array([0.7947, 1.2, 7.414]),
        0.5,
        10.0,
        bounds,
    )

    assert raw_period_min == 0.7947
    assert raw_period_max == 7.414
    assert period_min == 0.7
    assert period_max == 7.5


def test_run_aftan_config_can_measure_both_symmetric_branches(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_symmetric_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  branch: both",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    results = run_aftan_config(config_file)

    assert [result.branch for result in results] == ["positive", "negative"]
    assert all(result.output_npz.exists() for result in results)
    assert all(result.output_dat.exists() for result in results)


def test_stack_branch_falls_back_to_positive_for_one_sided_trace(tmp_path):
    input_dir = tmp_path / "ccf"
    input_dir.mkdir()
    sac_file = input_dir / "WT.2001_WT.2100_ZR_pws.SAC"
    _write_wave_packet(sac_file)
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "station: WT.2001",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZR_pws.SAC'",
                "aftan:",
                "  branch: stack",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    assert result.branch == "positive"
    assert result.warnings
    assert "not symmetric" in result.warnings[0]
    assert "requested stack branch but trace is not symmetric" in (
        tmp_path / "ftan" / "aftan.log"
    ).read_text()


def test_period_sampling_accepts_explicit_list(tmp_path):
    config = {
        "station": "WT.2001",
        "io": {
            "input_dir": "ccf",
            "output_dir": "ftan",
            "sac_pattern": "*_ZR.SAC",
        },
        "aftan": {
            "branch": "positive",
            "min_period": 2.0,
            "max_period": 8.0,
            "period_sampling": {
                "mode": "list",
                "periods": [1.0, 2.0, 3.0, 5.0, 9.0],
            },
            "velocity_window": {"min": 2.0, "max": 4.5},
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
        },
    }

    built = build_station_aftan_config(config, base_dir=tmp_path)

    assert built.aftan.period_sampling.mode == "list"
    assert built.aftan.period_sampling.periods == (1.0, 2.0, 3.0, 5.0, 9.0)
