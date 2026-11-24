from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import yaml

from seisforge.cli.main import main
from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.driver import run_inversion
from seisforge.inv.io import dump_yaml, read_dispersion_dat, read_hv_dat
from seisforge.inv.setup import build_inversion_setup


def _config():
    return {
        "inversion": {
            "parameters": [
                {
                    "name": "vs0",
                    "initial": 1.5,
                    "prior": {"type": "uniform", "bounds": [1.2, 1.8]},
                    "proposal": {"sigma": 0.01},
                },
                {
                    "name": "vs1",
                    "initial": 2.5,
                    "prior": {"type": "uniform", "bounds": [2.2, 2.8]},
                    "proposal": {"sigma": 0.01},
                },
                {
                    "name": "vs2",
                    "initial": 3.2,
                    "prior": {"type": "uniform", "bounds": [3.0, 3.5]},
                    "proposal": {"sigma": 0.01},
                },
            ]
        },
        "model": {
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 1.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "vs0"},
                        "bottom": {"parameter": "vs1"},
                    },
                },
                {
                    "top_km": 1.0,
                    "bottom_km": 3.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "vs1"},
                        "bottom": {"parameter": "vs2"},
                    },
                },
            ]
        },
        "discretization": {"dz": 1.0, "zmax": 3.0},
        "constraints": {
            "constraints": [
                {"type": "positive_velocity"},
                {"type": "monotonic_vs"},
            ]
        },
        "observations": {
            "dispersion": {"wave": "rayleigh", "kind": "phase", "mode": 0},
            "hv": {"wave": "rayleigh", "mode": 0},
        },
        "sampler": {
            "n_steps": 4,
            "burn_in": 1,
            "thin": 1,
            "n_chains": 1,
            "seed": 11,
            "executor": "serial",
            "initial_strategy": "theta0",
        },
    }


def _write_observations(tmp_path, config):
    prior_config = dict(config)
    prior_config.pop("observations", None)
    setup = build_inversion_setup(prior_config)
    layered = setup.prior.evaluate(setup.parameterization.theta0()).layered_model
    periods = np.array([2.0, 3.0])
    dispersion = predict_dispersion(layered, DispersionRequest(periods=periods))
    hv = predict_rayleigh_hv(layered, RayleighHVRequest(periods=periods))
    disp_file = tmp_path / "vph_disp.dat"
    hv_file = tmp_path / "hv.dat"
    np.savetxt(
        disp_file,
        np.column_stack([periods, dispersion.velocity, np.full_like(periods, 0.05)]),
    )
    np.savetxt(
        hv_file,
        np.column_stack([periods, hv.hv, np.full_like(periods, 0.1)]),
    )
    return disp_file, hv_file


def _obs_config():
    return {
        "observations": {
            "dispersion": {
                "file": "vph_disp.dat",
                "wave": "rayleigh",
                "kind": "phase",
                "mode": 0,
            },
            "hv": {"file": "hv.dat", "wave": "rayleigh", "mode": 0},
        },
        "forward": {
            "dispersion": {"algorithm": "dunkin", "dc": 0.005, "dt": 0.025},
            "hv": {"algorithm": "dunkin", "dc": 0.005},
        },
    }


def test_read_observation_dat_sorts_periods(tmp_path):
    disp_file = tmp_path / "disp.dat"
    hv_file = tmp_path / "hv.dat"
    disp_file.write_text("3.0 3.2 0.1\n# comment\n2.0 3.0 0.2\n")
    hv_file.write_text("3.0 0.8 0.1\n2.0 0.9 0.2\n")

    dispersion = read_dispersion_dat(disp_file)
    hv = read_hv_dat(hv_file)

    np.testing.assert_allclose(dispersion["periods"], [2.0, 3.0])
    np.testing.assert_allclose(dispersion["velocity"], [3.0, 3.2])
    np.testing.assert_allclose(hv["hv"], [0.9, 0.8])


def test_disp_hv_driver_writes_xarray_outputs(tmp_path, assert_provenance):
    config = _config()
    config["soft_priors"] = {
        "priors": [
            {
                "type": "vs_curvature",
                "depth_range": [0.0, 3.0],
                "dz": 0.1,
                "sigma": 0.2,
            }
        ]
    }
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)
    output = tmp_path / "disp_hv_posterior"

    result = run_inversion(
        kind="disp_hv",
        inv_file=inv_file,
        obs_file=obs_file,
        output=output,
        progress=False,
    )

    assert result.posterior_mcmc == output / "disp_hv_mcmc.nc"
    assert result.posterior_vs == output / "disp_hv_vs.nc"
    assert result.posterior_predictive == output / "disp_hv_predictive.nc"
    assert (output / "disp_hv.log").exists()
    with xr.open_dataset(result.posterior_mcmc) as dataset:
        assert "theta" in dataset
        assert list(dataset.coords["parameter"].values) == ["vs0", "vs1", "vs2"]
        for name in (
            "log_prior",
            "log_parameter_prior",
            "log_physical_hard_prior",
            "log_soft_prior",
            "log_likelihood",
            "vs_curvature_rms",
        ):
            assert name in dataset
        np.testing.assert_allclose(
            dataset["log_prob"], dataset["log_prior"] + dataset["log_likelihood"]
        )
        assert dataset["split_rhat"].dims == ("parameter",)
        assert dataset["effective_sample_size"].dims == ("parameter",)
        assert "max_split_rhat" in dataset.attrs and "lagging_chains" in dataset.attrs
        assert_provenance(dataset.attrs)
    with xr.open_dataset(result.posterior_predictive) as dataset:
        assert "dispersion_predicted_velocity" in dataset
        assert "hv_predicted_hv" in dataset
        assert_provenance(dataset.attrs)
    with xr.open_dataset(result.posterior_vs) as dataset:
        assert_provenance(dataset.attrs)
    log_text = result.log_file.read_text()
    assert "seisforge_version: " in log_text and "seisforge_git_commit: " in log_text
    for name in ("resolved_config.yaml", "resolved_inv.yaml", "resolved_obs.yaml"):
        resolved = yaml.safe_load((output / name).read_text())
        assert_provenance(resolved["run"])


def test_disp_hv_driver_reads_split_inv_and_obs_configs(tmp_path):
    config = _config()
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)
    output = tmp_path / "disp_hv_split"

    result = run_inversion(
        kind="disp_hv",
        inv_file=inv_file,
        obs_file=obs_file,
        output=output,
        progress=False,
    )

    assert result.posterior_mcmc == output / "disp_hv_mcmc.nc"
    assert (output / "resolved_inv.yaml").exists()
    assert (output / "resolved_obs.yaml").exists()
    with xr.open_dataset(result.posterior_predictive) as dataset:
        assert "dispersion_predicted_velocity" in dataset
        assert "hv_predicted_hv" in dataset


def test_driver_with_tempering_writes_swap_acceptance(tmp_path):
    config = _config()
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_config["sampler"] = dict(
        config["sampler"], n_steps=6, n_chains=2, tempering={"temperatures": [1, 3, 9]}
    )
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)

    result = run_inversion(
        kind="disp_hv",
        inv_file=inv_file,
        obs_file=obs_file,
        output=tmp_path / "tempered",
        with_prior=True,
        progress=False,
    )

    with xr.open_dataset(result.posterior_mcmc) as dataset:
        assert dataset["swap_acceptance"].shape == (2, 2)
        assert dataset.attrs["temperatures"] == "1, 3, 9"
        # 2 chains x 3 replicas, each evaluating its start and <= 6 proposals.
        assert dataset["forward_attempts"].dims == ("chain",)
        assert np.all(dataset["forward_attempts"] >= dataset["forward_attempts_cold"])
        assert np.all(dataset["forward_attempts_cold"] >= 1)
        assert np.all(dataset["forward_failure_rate"] == 0)
    with xr.open_dataset(result.prior_mcmc) as dataset:
        # The prior has no likelihood to temper and is sampled by plain chains.
        assert "swap_acceptance" not in dataset
        assert "forward_attempts" not in dataset  # no forward solver in the prior run
    assert "posterior forward failures per chain" in result.log_file.read_text()


@pytest.mark.parametrize("tempering", [None, {"temperatures": [1, 3]}])
def test_driver_with_adaptation_writes_frozen_proposals(tmp_path, tempering):
    config = _config()
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_config["sampler"] = dict(
        config["sampler"],
        n_steps=14,
        burn_in=10,
        n_chains=2,
        adaptation={"initial_diagonal_steps": 6, "update_interval_steps": 2},
    )
    if tempering is not None:
        inv_config["sampler"]["tempering"] = tempering
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)

    result = run_inversion(
        kind="disp_hv",
        inv_file=inv_file,
        obs_file=obs_file,
        output=tmp_path / "adapted",
        with_prior=True,
        progress=False,
    )
    names = ["vs0", "vs1", "vs2"]
    for path in (result.posterior_mcmc, result.prior_mcmc):
        with xr.open_dataset(path) as dataset:
            covariance = dataset["proposal_covariance"]
            assert covariance.dims == ("chain", "parameter", "parameter_column")
            assert list(covariance["parameter"].values) == names
            assert list(covariance["parameter_column"].values) == names
            np.testing.assert_allclose(covariance.values, np.swapaxes(covariance.values, 1, 2))
            assert dataset.attrs["adaptation_initial_diagonal_steps"] == 6
            assert dataset.attrs["adaptation_update_interval_steps"] == 2
            assert dataset.attrs["adaptation_last_update_step"] == 10
            assert dataset.attrs["adaptation_scale_factor"] == pytest.approx(2.38**2 / 3)
            assert dataset.attrs["sampler_adaptation"].startswith("adaptive Metropolis")
    with xr.open_dataset(result.posterior_mcmc) as dataset:
        if tempering is None:
            assert "proposal_covariance_tempered" not in dataset
        else:
            tempered = dataset["proposal_covariance_tempered"]
            assert tempered.dims == ("chain", "temperature", "parameter", "parameter_column")
            np.testing.assert_array_equal(tempered.isel(temperature=0).values, dataset["proposal_covariance"].values)
    with xr.open_dataset(result.prior_mcmc) as dataset:
        assert "proposal_covariance_tempered" not in dataset  # prior chains are not tempered
    log = result.log_file.read_text()
    assert "sampler adaptation_update_interval_steps: 2" in log
    assert "posterior chain 1 frozen proposal sd: vs0=" in log
    assert "prior chain 0 frozen proposal sd: vs0=" in log


def test_driver_without_adaptation_records_none(tmp_path):
    config = _config()
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)
    result = run_inversion(
        kind="disp_hv", inv_file=inv_file, obs_file=obs_file, output=tmp_path / "fixed", progress=False
    )
    with xr.open_dataset(result.posterior_mcmc) as dataset:
        assert "proposal_covariance" not in dataset
        assert dataset.attrs["sampler_adaptation"] == "none"
        assert "adaptation_initial_diagonal_steps" not in dataset.attrs
    assert "frozen proposal sd" not in result.log_file.read_text()


def test_inv_cli_help_runs(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["inv", "disp-hv", "--help"])

    assert exc_info.value.code == 0
    assert "disp-hv" in capsys.readouterr().out


def test_driver_records_discarded_history_and_acceptance(tmp_path):
    config = _config()
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_config["sampler"] = dict(
        config["sampler"],
        n_steps=14,
        burn_in=10,
        n_chains=2,
        adaptation={"initial_diagonal_steps": 6, "update_interval_steps": 2, "discard_early_history_fraction": 0.5},
    )
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)
    result = run_inversion(
        kind="disp_hv", inv_file=inv_file, obs_file=obs_file, output=tmp_path / "recent", progress=False
    )
    with xr.open_dataset(result.posterior_mcmc) as dataset:
        assert dataset.attrs["adaptation_discard_early_history_fraction"] == 0.5
        assert dataset.attrs["sampler_adaptation_discard_early_history_fraction"] == 0.5
    log = result.log_file.read_text()
    assert "sampler adaptation_discard_early_history_fraction: 0.5" in log
    # A 10-step burn-in for 3 parameters is far too short: both warnings.
    assert "sampler adaptation: burn_in (10) leaves 4 adapted steps" in log
    assert "fewer than 300 (100 per parameter" in log
    assert "posterior post-burn-in acceptance rates: " in log


def test_driver_records_graded_layering_in_attrs_and_log(tmp_path):
    config = _config()
    config["discretization"] = {"dz": 0.25, "zmax": 3.0, "dz_growth_ratio": 1.3, "dz_max": 1.0}
    inv_config = dict(config)
    inv_config.pop("observations", None)
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)

    result = run_inversion(
        kind="disp_hv", inv_file=inv_file, obs_file=obs_file, output=tmp_path / "out", progress=False
    )
    with xr.open_dataset(result.posterior_mcmc) as dataset:
        assert dataset.attrs["discretization_layering"] == "graded"
        assert dataset.attrs["discretization_dz_growth_ratio"] == 1.3
        assert dataset.attrs["discretization_dz_max_km"] == 1.0
        assert dataset.attrs["discretization_n_layers_at_initial"] > 0
        assert dataset.attrs["discretization_hv_layering_error_sigma_ratio"] >= 0.0
    log = result.log_file.read_text()
    assert "discretization dz_growth_ratio: 1.3" in log
    assert "discretization layering_error_at_initial:" in log
