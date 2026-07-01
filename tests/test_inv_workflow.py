import numpy as np

from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.workflow import (
    joint_inversion_setup_from_config,
    likelihood_weights_from_config,
    observations_from_config,
)


def _base_config():
    return {
        "Inversion": {
            "parameters": [
                {
                    "name": "vs0",
                    "initial": 1.5,
                    "prior": {"type": "uniform", "bounds": [1.0, 2.0]},
                    "proposal": {"sigma": 0.03},
                },
                {
                    "name": "vs1",
                    "initial": 2.5,
                    "prior": {"type": "uniform", "bounds": [2.0, 3.0]},
                    "proposal": {"sigma": 0.03},
                },
                {
                    "name": "vs2",
                    "initial": 3.2,
                    "prior": {"type": "uniform", "bounds": [2.8, 3.6]},
                    "proposal": {"sigma": 0.03},
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
        "Discretization": {"dz": 1.0},
        "Constraints": {
            "constraints": [
                {"type": "positive_velocity"},
                {"type": "monotonic_vs", "dz": 0.5},
            ]
        },
        "Sampler": {
            "n_steps": 8,
            "burn_in": 2,
            "thin": 1,
            "n_chains": 2,
            "seed": 11,
            "executor": "serial",
            "initial_strategy": "theta0",
        },
    }


def _add_self_consistent_observations(config):
    setup = joint_inversion_setup_from_config(config)
    layered = setup.prior.evaluate(setup.parameterization.theta0()).layered_model
    periods = np.array([5.0, 10.0, 20.0])
    dispersion = predict_dispersion(layered, DispersionRequest(periods=periods))
    hv = predict_rayleigh_hv(layered, RayleighHVRequest(periods=periods))
    config = dict(config)
    config["Observations"] = {
        "dispersion": {
            "periods": periods.tolist(),
            "velocity": dispersion.velocity.tolist(),
            "sigma": [0.05, 0.05, 0.05],
        },
        "hv": {
            "periods": periods.tolist(),
            "hv": hv.hv.tolist(),
            "sigma": [0.1, 0.1, 0.1],
        },
    }
    config["Likelihood"] = {"dispersion_weight": 1.0, "hv_weight": 2.0}
    return config


def test_joint_inversion_setup_from_config_builds_prior_only_workflow():
    setup = joint_inversion_setup_from_config(_base_config())

    assert setup.target is None
    assert setup.sampler.n_chains == 2
    np.testing.assert_allclose(setup.parameterization.theta0(), [1.5, 2.5, 3.2])
    assert setup.prior.evaluate(setup.parameterization.theta0()).success


def test_joint_inversion_setup_can_run_prior_sampler():
    config = _base_config()
    config["Sampler"]["initial_strategy"] = "prior_uniform"
    setup = joint_inversion_setup_from_config(config)

    result = setup.run_prior()

    assert result.samples.shape == (2, 6, 3)
    assert np.all(np.isfinite(result.log_prob))


def test_joint_inversion_setup_builds_posterior_target_from_observations():
    setup = joint_inversion_setup_from_config(_add_self_consistent_observations(_base_config()))
    theta0 = setup.parameterization.theta0()

    detail = setup.target.evaluate(theta0)

    assert detail.success
    assert detail.log_posterior == 0.0
    assert detail.likelihood.n_data == 6
    assert setup.target.hv_weight == 2.0


def test_joint_inversion_setup_can_run_posterior_sampler():
    setup = joint_inversion_setup_from_config(_add_self_consistent_observations(_base_config()))

    result = setup.run_posterior()

    assert result.samples.shape == (2, 6, 3)
    assert np.all(np.isfinite(result.log_prob))


def test_observations_from_config_accepts_aliases():
    dispersion, hv = observations_from_config(
        {
            "dispersion": {
                "periods": [5.0],
                "values": [3.0],
                "sigma": [0.1],
            },
            "rayleigh_hv": {
                "periods": [5.0],
                "values": [1.2],
                "sigma": [0.2],
            },
        }
    )

    np.testing.assert_allclose(dispersion.velocity, [3.0])
    np.testing.assert_allclose(hv.hv, [1.2])


def test_likelihood_weights_accepts_legacy_joint_inversion_key():
    weights = likelihood_weights_from_config({"Joint_Inversion": {"weight": [1.0, 10.0]}})

    assert weights == (1.0, 10.0)
