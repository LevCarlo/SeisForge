import numpy as np

from seisforge.inv.constraints import ConstraintSuite, MonotonicVsConstraint
from seisforge.inv.forward import DispersionRequest, RayleighHVRequest, predict_dispersion, predict_rayleigh_hv
from seisforge.inv.likelihood import DispersionObservation, RayleighHVObservation
from seisforge.inv.model import DiscretizationConfig
from seisforge.inv.parameterization import ModelParameterization, ParameterSpec, UniformPrior
from seisforge.inv.posterior import JointInversionTarget
from seisforge.inv.predictive import extract_prediction_ensemble_from_result
from seisforge.inv.prior import GeophysicalPrior
from seisforge.inv.samplers import MetropolisConfig, run_metropolis


def _target():
    parameterization = ModelParameterization(
        parameters=(
            ParameterSpec("vs0", 1.5, UniformPrior(1.0, 2.0), proposal_sigma=0.02),
            ParameterSpec("vs1", 2.5, UniformPrior(2.0, 3.0), proposal_sigma=0.02),
            ParameterSpec("vs2", 3.2, UniformPrior(2.8, 3.6), proposal_sigma=0.02),
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
    prior = GeophysicalPrior(
        parameterization=parameterization,
        constraints=ConstraintSuite((MonotonicVsConstraint(dz=0.5),)),
        discretization=DiscretizationConfig(dz=1.0),
    )
    layered = prior.evaluate(parameterization.theta0()).layered_model
    periods = np.array([5.0, 10.0, 20.0])
    disp = predict_dispersion(layered, DispersionRequest(periods=periods))
    hv = predict_rayleigh_hv(layered, RayleighHVRequest(periods=periods))
    target = JointInversionTarget(
        prior=prior,
        dispersion=DispersionObservation(periods, disp.velocity, np.full_like(periods, 0.05)),
        hv=RayleighHVObservation(periods, hv.hv, np.full_like(periods, 0.1)),
    )
    return parameterization, target


def test_extract_prediction_ensemble_from_result_tracks_observables():
    parameterization, target = _target()
    config = MetropolisConfig(
        n_steps=5,
        proposal_sigma=parameterization.proposal_sigmas(),
        burn_in=1,
        bounds=parameterization.bounds(),
    )
    result = run_metropolis(target.log_posterior, parameterization.theta0(), config, n_chains=2, seeds=1)

    predictions = extract_prediction_ensemble_from_result(result, target=target)

    assert predictions.dispersion.predicted.shape == (2, 4, 3)
    assert predictions.hv.predicted.shape == (2, 4, 3)
    np.testing.assert_allclose(predictions.dispersion.periods, [5.0, 10.0, 20.0])
    assert predictions.dispersion.mean.shape == (3,)
    assert predictions.hv.p16.shape == (3,)
