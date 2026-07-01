import numpy as np

from seisforge.inv.constraints import ConstraintSuite, MonotonicVsConstraint
from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.likelihood import DispersionObservation, RayleighHVObservation
from seisforge.inv.model import DiscretizationConfig
from seisforge.inv.parameterization import ModelParameterization, ParameterSpec, UniformPrior
from seisforge.inv.posterior import JointInversionTarget
from seisforge.inv.prior import GeophysicalPrior
from seisforge.inv.samplers import MetropolisConfig, run_metropolis_chain


def _parameterization():
    return ModelParameterization(
        parameters=(
            ParameterSpec("vs0", 1.5, UniformPrior(1.0, 2.0), proposal_sigma=0.03),
            ParameterSpec("vs1", 2.5, UniformPrior(2.0, 3.0), proposal_sigma=0.03),
            ParameterSpec("vs2", 3.2, UniformPrior(2.8, 3.6), proposal_sigma=0.03),
        ),
        model_config={
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
    )


def _prior(parameterization):
    return GeophysicalPrior(
        parameterization=parameterization,
        constraints=ConstraintSuite(constraints=(MonotonicVsConstraint(dz=0.5),)),
        discretization=DiscretizationConfig(dz=1.0),
    )


def _self_consistent_observations(layered_model):
    periods = np.array([5.0, 10.0, 20.0])
    dispersion_prediction = predict_dispersion(
        layered_model,
        DispersionRequest(periods=periods),
    )
    hv_prediction = predict_rayleigh_hv(
        layered_model,
        RayleighHVRequest(periods=periods),
    )
    return (
        DispersionObservation(
            periods=periods,
            velocity=dispersion_prediction.velocity,
            sigma=np.full_like(periods, 0.05),
        ),
        RayleighHVObservation(
            periods=periods,
            hv=hv_prediction.hv,
            sigma=np.full_like(periods, 0.1),
        ),
    )


def _target():
    parameterization = _parameterization()
    prior = _prior(parameterization)
    prior_result = prior.evaluate(parameterization.theta0())
    dispersion, hv = _self_consistent_observations(prior_result.layered_model)
    return JointInversionTarget(
        prior=prior,
        dispersion=dispersion,
        hv=hv,
        dispersion_weight=1.0,
        hv_weight=2.0,
    )


def test_joint_posterior_is_zero_for_self_consistent_observations():
    target = _target()
    theta0 = target.prior.parameterization.theta0()

    result = target.evaluate(theta0)

    assert result.success
    assert result.error is None
    assert result.log_prior == 0.0
    assert result.log_likelihood == 0.0
    assert result.log_posterior == 0.0
    assert result.chi2 == 0.0
    assert result.raw_chi2 == 0.0
    assert result.n_data == 6
    assert result.likelihood.dispersion is not None
    assert result.likelihood.hv is not None


def test_joint_posterior_skips_likelihood_when_prior_fails():
    target = _target()
    theta = target.prior.parameterization.theta0()
    theta[0] = 0.5

    result = target.evaluate(theta)

    assert not result.success
    assert result.log_posterior == -np.inf
    assert result.log_likelihood == -np.inf
    assert result.likelihood is None
    assert result.error == "prior: parameter_prior"


def test_joint_posterior_tracks_likelihood_weighting():
    target = _target()
    theta0 = target.prior.parameterization.theta0()
    periods = target.dispersion.periods
    perturbed_dispersion = DispersionObservation(
        periods=periods,
        velocity=target.dispersion.velocity + 0.05,
        sigma=np.full_like(periods, 0.05),
    )
    target = JointInversionTarget(
        prior=target.prior,
        dispersion=perturbed_dispersion,
        hv=None,
        dispersion_weight=3.0,
    )

    result = target.evaluate(theta0)

    assert result.success
    np.testing.assert_allclose(result.likelihood.dispersion.chi2, 3.0)
    np.testing.assert_allclose(result.likelihood.dispersion.weighted_chi2, 9.0)
    np.testing.assert_allclose(result.log_likelihood, -4.5)
    np.testing.assert_allclose(result.log_posterior, -4.5)


def test_joint_inversion_target_requires_observations():
    parameterization = _parameterization()
    prior = _prior(parameterization)

    try:
        JointInversionTarget(prior=prior)
    except ValueError as exc:
        assert "At least one observation" in str(exc)
    else:
        raise AssertionError("JointInversionTarget should require observations.")


def test_joint_posterior_log_callable_runs_with_sampler():
    target = _target()
    theta0 = target.prior.parameterization.theta0()
    config = MetropolisConfig(
        n_steps=6,
        proposal_sigma=target.prior.parameterization.proposal_sigmas(),
        burn_in=2,
        bounds=target.prior.parameterization.bounds(),
    )

    result = run_metropolis_chain(target.log_posterior, theta0, config, seed=1)

    assert result.samples.shape == (4, 3)
    assert np.all(np.isfinite(result.log_prob))
