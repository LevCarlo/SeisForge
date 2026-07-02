from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from seisforge.cli.main import main
from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.io import dump_yaml, read_dispersion_dat, read_hv_dat
from seisforge.inv.runner import run_dispersion_hv_inversion
from seisforge.inv.workflow import joint_inversion_setup_from_config


def _config():
    return {
        "Inversion": {
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
        "Model": {
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
        "Discretization": {"dz": 1.0, "zmax": 3.0},
        "Constraints": {
            "constraints": [
                {"type": "positive_velocity"},
                {"type": "monotonic_vs", "dz": 0.5},
            ]
        },
        "Observations": {
            "dispersion": {"wave": "rayleigh", "kind": "phase", "mode": 0},
            "hv": {"wave": "rayleigh", "mode": 0},
        },
        "Sampler": {
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
    prior_config.pop("Observations", None)
    setup = joint_inversion_setup_from_config(prior_config)
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
        "Observations": {
            "dispersion": {
                "file": "vph_disp.dat",
                "wave": "rayleigh",
                "kind": "phase",
                "mode": 0,
            },
            "hv": {"file": "hv.dat", "wave": "rayleigh", "mode": 0},
        },
        "Forward": {
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


def test_disp_hv_runner_writes_xarray_outputs(tmp_path):
    config = _config()
    config_file = tmp_path / "config.yaml"
    dump_yaml(config, config_file)
    disp_file, hv_file = _write_observations(tmp_path, config)
    output = tmp_path / "disp_hv_posterior"

    result = run_dispersion_hv_inversion(
        config_file=config_file,
        dispersion_file=disp_file,
        hv_file=hv_file,
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
    with xr.open_dataset(result.posterior_predictive) as dataset:
        assert "dispersion_predicted_velocity" in dataset
        assert "hv_predicted_hv" in dataset


def test_disp_hv_runner_reads_split_inv_and_obs_configs(tmp_path):
    config = _config()
    inv_config = dict(config)
    inv_config.pop("Observations", None)
    inv_file = tmp_path / "inv.yaml"
    obs_file = tmp_path / "obs.yaml"
    dump_yaml(inv_config, inv_file)
    dump_yaml(_obs_config(), obs_file)
    _write_observations(tmp_path, config)
    output = tmp_path / "disp_hv_split"

    result = run_dispersion_hv_inversion(
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


def test_inv_cli_help_runs(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["inv", "disp-hv", "--help"])

    assert exc_info.value.code == 0
    assert "disp-hv" in capsys.readouterr().out
