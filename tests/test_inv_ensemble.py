import numpy as np

from seisforge.inv.ensemble import extract_vs_ensemble, extract_vs_ensemble_from_result
from seisforge.inv.parameterization import ModelParameterization, ParameterSpec, UniformPrior
from seisforge.inv.samplers import MetropolisConfig, run_metropolis


def _parameterization():
    return ModelParameterization(
        parameters=(
            ParameterSpec("vs0", 1.0, UniformPrior(0.5, 2.0), proposal_sigma=0.05),
            ParameterSpec("vs1", 3.0, UniformPrior(2.0, 4.0), proposal_sigma=0.05),
        ),
        model_config={
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 10.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "vs0"},
                        "bottom": {"parameter": "vs1"},
                    },
                }
            ]
        },
    )


def _box_log_target(theta):
    theta = np.asarray(theta, dtype=float)
    if np.any(theta < [0.5, 2.0]) or np.any(theta > [2.0, 4.0]):
        return -np.inf
    return 0.0


def test_extract_vs_ensemble_preserves_chain_draw_shape():
    parameterization = _parameterization()
    samples = np.array(
        [
            [[1.0, 3.0], [1.2, 3.2]],
            [[0.8, 2.8], [1.4, 3.4]],
        ]
    )
    log_prob = np.array([[-2.0, -1.0], [-3.0, 0.0]])
    z = np.array([0.0, 5.0, 10.0])

    ensemble = extract_vs_ensemble(
        samples,
        parameterization=parameterization,
        z=z,
        log_prob=log_prob,
    )

    assert ensemble.vs.shape == (2, 2, 3)
    assert ensemble.theta.shape == samples.shape
    np.testing.assert_allclose(ensemble.vs[0, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ensemble.vs[1, 1], [1.4, 2.4, 3.4])
    np.testing.assert_allclose(ensemble.best_theta, [1.4, 3.4])
    np.testing.assert_allclose(ensemble.best_vs, [1.4, 2.4, 3.4])


def test_vs_ensemble_summary_statistics_are_depthwise():
    parameterization = _parameterization()
    samples = np.array(
        [
            [[1.0, 3.0], [1.2, 3.2]],
            [[0.8, 2.8], [1.4, 3.4]],
        ]
    )
    ensemble = extract_vs_ensemble(
        samples,
        parameterization=parameterization,
        z=[0.0, 10.0],
    )

    np.testing.assert_allclose(ensemble.mean, [1.1, 3.1])
    np.testing.assert_allclose(ensemble.median, [1.1, 3.1])
    np.testing.assert_allclose(ensemble.minimum, [0.8, 2.8])
    np.testing.assert_allclose(ensemble.maximum, [1.4, 3.4])
    assert set(["z", "mean", "std", "min", "max", "q05", "q50", "q95"]).issubset(
        ensemble.summary()
    )


def test_extract_vs_ensemble_from_mcmc_result():
    parameterization = _parameterization()
    config = MetropolisConfig(
        n_steps=8,
        proposal_sigma=parameterization.proposal_sigmas(),
        burn_in=2,
        bounds=parameterization.bounds(),
    )
    result = run_metropolis(
        _box_log_target,
        parameterization.theta0(),
        config,
        n_chains=2,
        seeds=1,
    )

    ensemble = extract_vs_ensemble_from_result(
        result,
        parameterization=parameterization,
        z=[0.0, 5.0, 10.0],
    )

    assert ensemble.vs.shape == (2, 6, 3)
    assert ensemble.log_prob.shape == (2, 6)
    assert ensemble.n_samples == 12


def test_extract_vs_ensemble_validates_depth_grid_and_parameter_count():
    parameterization = _parameterization()

    try:
        extract_vs_ensemble([[1.0, 3.0]], parameterization=parameterization, z=[1.0, 0.0])
    except ValueError as exc:
        assert "strictly increasing" in str(exc)
    else:
        raise AssertionError("extract_vs_ensemble should validate depth grid.")

    try:
        extract_vs_ensemble([[1.0]], parameterization=parameterization, z=[0.0, 1.0])
    except ValueError as exc:
        assert "last dimension" in str(exc)
    else:
        raise AssertionError("extract_vs_ensemble should validate parameter count.")


def test_vs_ensemble_can_export_xarray_dataset():
    parameterization = _parameterization()
    ensemble = extract_vs_ensemble(
        np.array([[[1.0, 3.0]]]),
        parameterization=parameterization,
        z=[0.0, 10.0],
    )

    dataset = ensemble.to_xarray()

    assert dataset["vs"].dims == ("chain", "draw", "depth")
    np.testing.assert_allclose(dataset["depth"].values, [0.0, 10.0])
