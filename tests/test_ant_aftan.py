from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from obspy import Trace
import pytest
import xarray as xr

from seisforge.ant.aftan import (
    AFTANSNRConfig,
    AFTANConfig,
    AFTANPhaseCycleConfig,
    AFTANQCConfig,
    AFTANPeriodSamplingConfig,
    AFTANTimeTaperConfig,
    _align_to_period_grid,
    _aftan_diagram_snr,
    apply_phase_cycle_shift,
    _override_energy_map_config,
    _period_qc,
    _qc_warnings,
    _snr,
    automatic_pi_over_4,
    build_station_aftan_config,
    physical_branch_for_branch,
    physical_station_pair_for_branch,
    override_phase_cycle,
    run_aftan_config,
)
from seisforge.ant.aftan.taper import cosine_window_weights
from seisforge.ant.aftan.core import (
    AFTANMeasurement,
    _build_period_grid,
    _select_phase_cycle,
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
            "pmf": {
                "alpha": {"mode": "constant", "value": 22.0},
                "trig_threshold": 20.0,
                "jump_points": 5,
            },
            "snr": {"noise_mode": "complement"},
        },
        "phase_cycle_selection": {
            "max_shift": 3,
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
    assert built.aftan.basic.trig_threshold == 50.0
    assert built.aftan.basic.jump_points == 3
    assert built.aftan.pmf.alpha.value == 22.0
    assert built.aftan.pmf.trig_threshold == 20.0
    assert built.aftan.pmf.jump_points == 5
    assert built.aftan.snr.noise_mode == "complement"
    assert built.aftan.phase_cycle.max_shift == 3


@pytest.mark.parametrize(
    "phase_cycle_config",
    [
        {"phase_cycle_selection": {"max_reference_residual_cycles": 0.35}},
        {"aftan": {"phase_cycle": {"enabled": True}}},
    ],
)
def test_build_station_aftan_config_rejects_removed_phase_cycle_fields(
    tmp_path,
    phase_cycle_config,
):
    config = {
        "station": "WT.2001",
        "io": {"input_dir": "ccf", "output_dir": "ftan"},
        "aftan": {"branch": "positive"},
    }
    if "aftan" in phase_cycle_config:
        config["aftan"].update(phase_cycle_config["aftan"])
    else:
        config.update(phase_cycle_config)

    with pytest.raises(ValueError):
        build_station_aftan_config(config, base_dir=tmp_path)


def test_build_station_aftan_config_accepts_station_pair_component_list(tmp_path):
    config = {
        "source_station": "WT.2001",
        "receiver_station": "WT.2085",
        "components": ["ZR", "RZ"],
        "io": {
            "input_root_datadir": "CC_ZRT",
            "output_root_datadir": "FTAN",
            "input_template": "{source_station}_{receiver_station}_{component}_pws.SAC",
        },
        "aftan": {
            "branch": "negative",
            "min_period": 2.0,
            "max_period": 8.0,
            "period_sampling": {"mode": "uniform", "step": 0.5},
            "velocity_window": {"min": 2.0, "max": 4.5},
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
        },
    }

    built = build_station_aftan_config(config, base_dir=tmp_path)

    assert built.station == "WT.2001"
    assert built.source_station == "WT.2001"
    assert built.receiver_station == "WT.2085"
    assert built.components == ("ZR", "RZ")
    assert built.input_dir == tmp_path / "CC_ZRT" / "WT.2001" / "WT.2001_WT.2085"
    assert built.output_dir == tmp_path / "FTAN" / "WT.2001" / "WT.2001_WT.2085"
    assert built.sac_pattern == "{source_station}_{receiver_station}_{component}_pws.SAC"
    assert built.input_template == "{source_station}_{receiver_station}_{component}_pws.SAC"
    assert built.aftan.pi_over_4_mode == "auto"


def test_automatic_pi_over_4_uses_reciprocal_component_for_negative_branch():
    assert automatic_pi_over_4("ZR", "positive") == ("ZR", 1.0)
    assert automatic_pi_over_4("RZ", "positive") == ("RZ", -1.0)
    assert automatic_pi_over_4("ZR", "negative") == ("RZ", -1.0)
    assert automatic_pi_over_4("RZ", "negative") == ("ZR", 1.0)
    assert automatic_pi_over_4("ZZ", "negative") == ("ZZ", -1.0)
    assert physical_branch_for_branch("negative") == "positive"
    assert physical_station_pair_for_branch("WT.2001", "WT.2085", "negative") == (
        "WT.2085",
        "WT.2001",
    )


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
            "basic": {
                "alpha": {"mode": "constant", "value": 18.0},
                "trig_threshold": 123.0,
                "jump_points": 7,
            },
            "pmf": {
                "alpha": {"mode": "constant", "value": 28.0},
                "trig_threshold": 21.0,
                "jump_points": 4,
            },
        },
    }
    built = build_station_aftan_config(config, base_dir=tmp_path)

    overridden = _override_energy_map_config(
        built,
        write_energy_map=True,
        plot_energy_map=True,
    )

    assert overridden.aftan.basic.trig_threshold == 123.0
    assert overridden.aftan.basic.jump_points == 7
    assert overridden.aftan.pmf.alpha.value == 28.0
    assert overridden.aftan.pmf.trig_threshold == 21.0
    assert overridden.aftan.pmf.jump_points == 4


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
            "period_sampling": {"mode": "uniform", "step": 0.5},
            "velocity_window": {"min": 2.0, "max": 4.5},
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
        },
        "short_distance_period_guard": {"max_period_nwl": 0.0},
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
    assert result.output_npz is None
    assert result.output_dat.exists()
    dat_lines = result.output_dat.read_text().splitlines()
    data_lines = [line for line in dat_lines if line and not line.startswith("#")]
    assert data_lines
    assert {len(line.split()) for line in data_lines} == {6}
    for line in data_lines:
        fields = line.split()
        assert "e" in fields[4].lower()
        assert not any("e" in field.lower() for field in fields[:4] + fields[5:])
    log_file = tmp_path / "ftan" / "aftan.log"
    assert log_file.exists()
    log_text = log_file.read_text()
    assert "requested_branch: positive" in log_text
    assert "basic_result WT.2001_WT.2100_ZR_pws.SAC [positive]" in log_text
    assert "alpha=20 (constant)" in log_text
    assert "qc WT.2001_WT.2100_ZR_pws.SAC [positive]" in log_text
    assert "period_grid=" in log_text
    assert "outputs WT.2001_WT.2100_ZR_pws.SAC [positive] basic:" in log_text
    assert "npz=" not in log_text
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


def test_run_aftan_config_expands_components_and_logs_auto_phase(tmp_path):
    pair_dir = tmp_path / "CC_ZRT" / "WT.2001" / "WT.2001_WT.2085"
    pair_dir.mkdir(parents=True)
    _write_symmetric_wave_packet(pair_dir / "WT.2001_WT.2085_ZR_pws.SAC")
    _write_symmetric_wave_packet(pair_dir / "WT.2001_WT.2085_RZ_pws.SAC")
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "source_station: WT.2001",
                "receiver_station: WT.2085",
                "components: [ZR, RZ]",
                "io:",
                "  input_root_datadir: CC_ZRT",
                "  output_root_datadir: FTAN",
                "  input_template: '{source_station}_{receiver_station}_{component}_pws.SAC'",
                "aftan:",
                "  branch: negative",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.5",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
            ]
        )
    )

    results = run_aftan_config(config_file)

    assert len(results) == 2
    by_component = {result.input_component: result for result in results}
    assert by_component["ZR"].physical_component == "RZ"
    assert by_component["ZR"].physical_source_station == "WT.2085"
    assert by_component["ZR"].physical_receiver_station == "WT.2001"
    assert by_component["ZR"].physical_branch == "positive"
    assert by_component["ZR"].pi_over_4 == -1.0
    assert by_component["RZ"].physical_component == "ZR"
    assert by_component["RZ"].pi_over_4 == 1.0
    assert by_component["ZR"].output_dat.name == "WT.2001_WT.2085_ZR_pws.negative.dat"
    assert by_component["RZ"].output_dat.name == "WT.2001_WT.2085_RZ_pws.negative.dat"
    log_text = (tmp_path / "FTAN" / "WT.2001" / "WT.2001_WT.2085" / "aftan.log").read_text()
    assert "components: ZR,RZ" in log_text
    assert "stored_pair=WT.2001_WT.2085->WT.2085_WT.2001" in log_text
    assert "branch=negative->positive" in log_text
    assert "component=ZR->RZ, pi_over_4=-1 (auto)" in log_text
    assert "component=RZ->ZR, pi_over_4=1 (auto)" in log_text
    zr_dat_text = by_component["ZR"].output_dat.read_text()
    assert "# input_file: WT.2001_WT.2085_ZR_pws.SAC" in zr_dat_text
    assert "# input_component:" not in zr_dat_text
    assert "# selected_branch: negative" in zr_dat_text
    assert "# stored_source_station:" not in zr_dat_text
    assert "# physical_source_station:" not in zr_dat_text
    assert "# physical_component:" not in zr_dat_text
    assert "# pi_over_4: -1.0" in zr_dat_text
    assert "# pi_over_4_mode: auto" in zr_dat_text


def test_run_aftan_config_falls_back_to_flat_input_root(tmp_path):
    _write_symmetric_wave_packet(tmp_path / "WT.2001_WT.2085_ZR_pws.SAC")
    config_file = tmp_path / "aftan.yml"
    config_file.write_text(
        "\n".join(
            [
                "source_station: 2001",
                "receiver_station: 2085",
                "component: ZR",
                "io:",
                "  input_root_datadir: .",
                "  output_root_datadir: FTAN",
                "  input_template: '*_{component}_pws.SAC'",
                "aftan:",
                "  branch: negative",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.5",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                *_alpha_config_lines(),
            ]
        )
    )

    results = run_aftan_config(config_file)

    assert len(results) == 1
    result = results[0]
    assert result.input_file.name == "WT.2001_WT.2085_ZR_pws.SAC"
    assert result.input_component == "ZR"
    assert result.physical_component == "RZ"
    assert result.output_dat.name == "WT.2001_WT.2085_ZR_pws.negative.dat"
    log_text = (tmp_path / "FTAN" / "2001" / "2001_2085" / "aftan.log").read_text()
    assert "input ZR:" in log_text
    assert "WT.2001_WT.2085_ZR_pws.SAC" in log_text


def test_snr_uses_raw_signal_and_picked_periods(monkeypatch, tmp_path):
    trace = Trace(data=np.linspace(1.0, 2.0, 800).astype(np.float32))
    trace.stats.delta = 0.5
    trace.stats.sac = {"b": 0.0, "dist": 100.0}
    config = AFTANConfig(
        min_period=2.0,
        max_period=8.0,
        period_sampling=AFTANPeriodSamplingConfig(mode="uniform", step=0.5),
        snr=AFTANSNRConfig(
            definition="pyftan",
            output_db=False,
            dsn=20.0,
            nlen=40.0,
            vmax=4.5,
            vmin=1.0,
        ),
    )
    measurement = AFTANMeasurement(trace, tmp_path / "test.SAC", config)
    picked_periods = np.array([3.0, 4.0, 5.0])
    raw_signal = np.arange(trace.stats.npts, dtype=float)
    calls = []

    def fake_ftan_complex(self, signal_data, target_periods, alpha):
        calls.append((signal_data.copy(), target_periods.copy(), alpha))
        return np.ones((self.nfft, target_periods.size), dtype=np.complex128)

    monkeypatch.setattr(AFTANMeasurement, "_ftan_complex", fake_ftan_complex)

    snr = measurement._spectral_snr(
        np.zeros_like(raw_signal),
        picked_periods,
        np.full(picked_periods.size, 3.0),
        config.snr,
        alpha=22.0,
        target_periods=np.array([2.0, 6.0, 8.0]),
        snr_signal_data=raw_signal,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], raw_signal)
    assert np.array_equal(calls[0][1], picked_periods)
    assert calls[0][2] == 22.0
    assert np.allclose(snr, 1.0)

    local_config = AFTANConfig(
        min_period=2.0,
        max_period=8.0,
        period_sampling=AFTANPeriodSamplingConfig(mode="uniform", step=0.5),
        snr=AFTANSNRConfig(
            definition="local",
            output_db=False,
            dsn=20.0,
            nlen=40.0,
            vmax=4.5,
            vmin=1.0,
        ),
    )
    local_measurement = AFTANMeasurement(trace, tmp_path / "test.SAC", local_config)
    calls.clear()

    snr = local_measurement._spectral_snr(
        np.zeros_like(raw_signal),
        picked_periods,
        np.full(picked_periods.size, 3.0),
        local_config.snr,
        alpha=24.0,
        target_periods=np.array([2.0, 6.0, 8.0]),
        snr_signal_data=raw_signal,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0][0], raw_signal)
    assert np.array_equal(calls[0][1], picked_periods)
    assert calls[0][2] == 24.0
    assert np.allclose(snr, 1.0)


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
    assert result.output_npz is None
    dataset = xr.open_dataset(result.output_energy_map, engine="scipy")
    try:
        assert dataset.attrs["debug"]
        assert dataset["picked_hilbert_instant_period"].dims == ("target_period",)
        assert dataset["picked_instant_period_delta"].dims == ("target_period",)
    finally:
        dataset.close()


def test_global_phase_cycle_selection_recovers_reference_branch():
    period = np.array([2.0, 3.0, 4.0, 5.0])
    reference = np.array([2.8, 3.0, 3.1, 3.2])
    distance_km = 100.0
    uncorrected = 1.0 / (1.0 / reference - period / distance_km)

    corrected, diagnostics = _select_phase_cycle(
        period,
        uncorrected,
        reference,
        distance_km,
        AFTANPhaseCycleConfig(max_shift=2),
    )

    np.testing.assert_allclose(corrected, reference)
    assert diagnostics["phase_cycle_shift"].item() == 1
    assert diagnostics["phase_cycle_reference_score_cycles"].item() == pytest.approx(
        0.0
    )
    assert diagnostics["phase_cycle_reference_score_gap_cycles"].item() > 0
    assert diagnostics["phase_cycle_candidate_reference_score_cycles"].shape == (
        5,
    )


def test_global_phase_cycle_selection_tie_prefers_zero_shift():
    period = np.array([1.0, 2.0, 3.0])
    uncorrected = np.array([2.8, 2.9, 3.0])
    distance_km = 50.0
    reference = 1.0 / (1.0 / uncorrected + 0.5 * period / distance_km)

    corrected, diagnostics = _select_phase_cycle(
        period,
        uncorrected,
        reference,
        distance_km,
        AFTANPhaseCycleConfig(max_shift=1),
    )
    np.testing.assert_allclose(corrected, uncorrected)
    assert diagnostics["phase_cycle_shift"].item() == 0
    assert diagnostics["phase_cycle_reference_score_gap_cycles"].item() == pytest.approx(
        0.0
    )


def test_global_phase_cycle_selection_requires_two_reference_points():
    period = np.array([2.0, 3.0])
    uncorrected = np.array([2.8, 2.9])
    reference = np.array([3.1, np.nan])

    corrected, diagnostics = _select_phase_cycle(
        period,
        uncorrected,
        reference,
        80.0,
        AFTANPhaseCycleConfig(max_shift=2),
    )

    np.testing.assert_allclose(corrected, uncorrected)
    assert diagnostics["phase_cycle_shift"].item() == 0
    assert diagnostics["phase_cycle_valid_count"].item() == 1
    assert np.isnan(diagnostics["phase_cycle_reference_score_cycles"].item())
    assert np.all(
        np.isnan(diagnostics["phase_cycle_candidate_reference_score_cycles"])
    )


def test_manual_phase_cycle_override_previews_then_updates_nc_and_dat(tmp_path):
    period = np.array([2.0, 3.0, 4.0])
    uncorrected = np.array([2.8, 2.9, 3.0])
    reference = np.array([3.0, 3.1, 3.2])
    distance_km = 80.0
    phase_map = tmp_path / "pair.negative.pmf.vph.nc"
    dat_file = tmp_path / "pair.negative.pmf.dat"
    dataset = xr.Dataset(
        data_vars={
            "picked_phase_velocity": (("target_period",), uncorrected),
            "picked_phase_velocity_uncorrected": (
                ("target_period",),
                uncorrected,
            ),
            "reference_phase_velocity": (("target_period",), reference),
            "picked_instant_period": (("target_period",), period),
            "phase_cycle_shift": np.asarray(0, dtype=np.int32),
            "phase_cycle_reference_score_cycles": np.asarray(0.5),
            "phase_cycle_reference_score_gap_cycles": np.asarray(0.1),
            "phase_cycle_search_limit_reached": np.asarray(0, dtype=np.int8),
        },
        coords={"target_period": period},
        attrs={"distance_km": distance_km},
    )
    dataset.to_netcdf(phase_map, engine="scipy")
    table = np.column_stack(
        [period, period, np.full(3, 2.7), uncorrected, np.ones(3), np.full(3, 20.0)]
    )
    np.savetxt(
        dat_file,
        table,
        header=(
            "target_period_s instant_period_s group_velocity_km_s "
            "phase_velocity_km_s amplitude snr"
        ),
    )

    preview = override_phase_cycle(phase_map, -1)
    assert not preview.written
    with xr.open_dataset(phase_map) as unchanged:
        assert unchanged["phase_cycle_shift"].item() == 0

    result = override_phase_cycle(
        phase_map,
        -1,
        write=True,
        reason="visual cycle inspection",
    )
    expected = apply_phase_cycle_shift(period, uncorrected, distance_km, -1)
    assert result.written
    with xr.open_dataset(phase_map) as updated:
        np.testing.assert_allclose(updated["picked_phase_velocity"], expected)
        np.testing.assert_allclose(
            updated["picked_phase_velocity_automatic"], uncorrected
        )
        assert updated["phase_cycle_shift"].item() == -1
        assert updated.attrs["phase_cycle_selection_source"] == "manual_override"
        assert updated.attrs["phase_cycle_manual_reason"] == "visual cycle inspection"
    np.testing.assert_allclose(np.loadtxt(dat_file)[:, 3], expected, rtol=2e-4)


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
                "phase_cycle_selection:",
                "  enabled: true",
                "  max_shift: 2",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    assert result.output_phase_map is not None
    assert result.output_phase_map.name == "WT.2001_WT.2100_ZR_pws.positive.vph.nc"
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
        assert dataset["picked_phase_velocity_uncorrected"].dims == (
            "target_period",
        )
        assert dataset["reference_phase_velocity"].dims == ("target_period",)
        assert dataset["phase_cycle_shift"].dims == ()
        assert dataset["phase_cycle_reference_score_cycles"].dims == ()
        assert dataset["phase_cycle_reference_score_gap_cycles"].dims == ()
        assert dataset["phase_cycle_candidate_reference_score_cycles"].dims == (
            "cycle_shift_candidate",
        )
        assert dataset["phase_cycle_valid_count"].dims == ()
        assert dataset["phase_cycle_search_limit_reached"].dims == ()
        assert dataset.attrs["phase_cycle_selection_method"]
        assert dataset.attrs["phase_cycle_implementation"] == "seisforge_extension"
        assert dataset.attrs["phase_cycle_max_shift"] == 2
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
                "    alpha:",
                "      mode: constant",
                "      value: 24.0",
                "    trig_threshold: 25.0",
                "    jump_points: 5",
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )

    result = run_aftan_config(config_file)[0]

    assert result.output_pmf_npz is None
    assert result.output_pmf_dat is not None
    assert result.output_pmf_dat.exists()
    pmf_lines = [
        line
        for line in result.output_pmf_dat.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert pmf_lines
    assert {len(line.split()) for line in pmf_lines} == {6}
    assert result.pmf_alpha == 24.0
    assert result.pmf_period_min >= result.period.min()
    assert result.pmf_period_max <= result.period.max()
    assert result.pmf_raw_period_min == result.pmf_period_min
    assert result.pmf_raw_period_max == result.pmf_period_max


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
    assert all(result.output_npz is None for result in results)
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


def test_period_sampling_geomspace_uses_pyftan_dfreq_formula(tmp_path):
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
                "mode": "geomspace",
                "dfreq": 0.03,
                "min_count": 3,
            },
            "velocity_window": {"min": 2.0, "max": 4.5},
            "basic": {"alpha": {"mode": "constant", "value": 18.0}},
        },
    }

    built = build_station_aftan_config(config, base_dir=tmp_path)
    periods = _build_period_grid(built.aftan.period_sampling, 2.0, 8.0)

    expected_count = int(np.log(8.0 / 2.0) / 0.03)
    assert built.aftan.period_sampling.mode == "geomspace"
    assert built.aftan.period_sampling.dfreq == 0.03
    assert periods.size == expected_count
    assert np.allclose(periods, np.geomspace(2.0, 8.0, expected_count))


def _one_sided_packet_trace(duration_s, distance_km=100.0, velocity_km_s=3.0):
    delta = 0.5
    time = np.arange(0.0, duration_s, delta)
    center = distance_km / velocity_km_s
    data = np.exp(-0.5 * ((time - center) / 10.0) ** 2) * np.cos(2 * np.pi * time / 5.0)
    trace = Trace(data=data.astype(np.float32))
    trace.stats.delta = delta
    trace.stats.sac = {"b": 0.0, "dist": distance_km}
    return trace


def _group_velocity_at_5s(trace, mode):
    config = replace(
        AFTANConfig(
            min_period=4.0,
            max_period=6.0,
            period_sampling=AFTANPeriodSamplingConfig(mode="uniform", step=0.5),
            velocity_min=1.0,
            velocity_max=5.0,
        ),
        time_taper=AFTANTimeTaperConfig(mode=mode),
    )
    measured = AFTANMeasurement(trace, Path("pair.SAC"), config).measure()
    index = int(np.argmin(np.abs(measured["target_period"] - 5.0)))
    return float(measured["group_velocity"][index])


def test_velocity_window_taper_keeps_short_distance_group_velocity():
    # 100 km at 3 km/s arrives at 33 s, inside the first 5% of a 3000 s branch.
    short = _group_velocity_at_5s(_one_sided_packet_trace(400.0), "velocity_window")
    long = _group_velocity_at_5s(_one_sided_packet_trace(3000.0), "velocity_window")
    assert short == pytest.approx(3.0, abs=0.01)
    assert long == pytest.approx(short, abs=1e-6)


def test_legacy_edge_fraction_taper_biases_short_distance_group_velocity():
    biased = _group_velocity_at_5s(
        _one_sided_packet_trace(3000.0), "legacy_edge_fraction"
    )
    assert biased < 2.8


def test_legacy_edge_fraction_reproduces_obspy_taper_exactly():
    trace = _one_sided_packet_trace(3000.0)
    config = replace(
        AFTANConfig(),
        time_taper=AFTANTimeTaperConfig(mode="legacy_edge_fraction"),
    )
    measurement = AFTANMeasurement(trace, Path("pair.SAC"), config)
    expected = trace.copy().taper(0.05).data
    assert measurement.trace.data.dtype == expected.dtype
    np.testing.assert_array_equal(measurement.trace.data, expected)
    np.testing.assert_array_equal(measurement.snr_data, expected)


def test_velocity_window_taper_leaves_snr_input_untapered():
    trace = _one_sided_packet_trace(3000.0)
    measurement = AFTANMeasurement(trace, Path("pair.SAC"), AFTANConfig())
    np.testing.assert_array_equal(
        measurement.snr_data, np.asarray(trace.data, dtype=float)
    )
    attrs = measurement.time_taper_attrs()
    assert attrs["time_taper_mode"] == "velocity_window"
    assert attrs["time_taper_window_start_s"] == pytest.approx(100.0 / 5.5)
    assert attrs["time_taper_window_end_s"] == pytest.approx(100.0 / 0.5)
    assert attrs["time_taper_ramp_s"] == pytest.approx(measurement.max_period)


def test_cosine_window_weights_shape():
    time_s = np.arange(0.0, 100.0, 1.0)
    weights = cosine_window_weights(time_s, start_s=40.0, end_s=60.0, ramp_s=10.0)
    assert np.all(weights[(time_s >= 40.0) & (time_s <= 60.0)] == 1.0)
    assert np.all(weights[(time_s < 30.0) | (time_s > 70.0)] == 0.0)
    assert weights[35] == pytest.approx(0.5)
    assert weights[65] == pytest.approx(0.5)
    assert np.all(np.diff(weights[30:41]) >= 0)
    assert np.all(np.diff(weights[60:71]) <= 0)


def test_time_taper_config_parsing_and_validation(tmp_path):
    def build(time_taper):
        return build_station_aftan_config(
            {
                "station": "STA",
                "io": {"input_dir": "in", "output_dir": "out"},
                "aftan": {
                    "branch": "positive",
                    "period_sampling": {"mode": "uniform", "step": 0.5},
                    "basic": {"alpha": {"mode": "constant", "value": 20.0}},
                    "time_taper": time_taper,
                },
            },
            base_dir=tmp_path,
        ).aftan.time_taper

    assert build({}).mode == "velocity_window"
    assert build({"mode": "velocity_window", "velocity_window": {"ramp_s": 8}}).ramp_s == 8.0
    legacy = build({"mode": "legacy_edge_fraction", "legacy_edge_fraction": {"fraction": 0.1}})
    assert legacy.legacy_edge_fraction == pytest.approx(0.1)
    with pytest.raises(ValueError, match="time_taper.mode"):
        build({"mode": "hann"})
    with pytest.raises(ValueError, match="ramp_s"):
        build({"velocity_window": {"ramp_s": -1}})
