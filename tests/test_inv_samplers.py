import numpy as np

from seisforge.inv.constraints import (
    BoundaryNonDecreasingVsConstraint,
    ConstraintSuite,
    MonotonicVsConstraint,
)
from seisforge.inv.parameterization import ModelParameterization, ParameterSpec, UniformPrior
from seisforge.inv.prior import GeophysicalPrior
from seisforge.inv.samplers import (
    MetropolisConfig,
    draw_valid_initial_thetas,
    run_metropolis,
    run_metropolis_chain,
)


def gaussian_log_target(theta):
    theta = np.asarray(theta, dtype=float)
    return -0.5 * np.sum(theta**2)


def box_log_target(theta):
    theta = np.asarray(theta, dtype=float)
    if np.any(theta < 0.0) or np.any(theta > 1.0):
        return -np.inf
    return 0.0


def _prior_target():
    parameterization = ModelParameterization(
        parameters=(
            ParameterSpec("vs0", 1.0, UniformPrior(0.5, 2.0), proposal_sigma=0.05),
            ParameterSpec("vs1", 2.0, UniformPrior(1.0, 3.0), proposal_sigma=0.05),
        ),
        model_config={
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 5.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "vs0"},
                        "bottom": {"parameter": "vs1"},
                    },
                }
            ]
        },
    )
    prior = GeophysicalPrior(
        parameterization=parameterization,
        constraints=ConstraintSuite(
            constraints=(
                BoundaryNonDecreasingVsConstraint(),
                MonotonicVsConstraint(dz=0.5),
            )
        ),
    )
    return parameterization, prior


def test_single_chain_respects_burn_in_and_thin():
    config = MetropolisConfig(n_steps=20, proposal_sigma=0.1, burn_in=5, thin=3)

    result = run_metropolis_chain(gaussian_log_target, [0.0, 0.0], config, seed=1)

    assert result.samples.shape == (5, 2)
    assert result.log_prob.shape == (5,)
    assert result.accepted.shape == (20,)
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_sampler_rejects_nonfinite_initial_theta():
    config = MetropolisConfig(n_steps=10, proposal_sigma=0.1)

    try:
        run_metropolis_chain(box_log_target, [-1.0], config, seed=1)
    except ValueError as exc:
        assert "Initial theta" in str(exc)
    else:
        raise AssertionError("Sampler should reject invalid initial theta.")


def test_multichain_sampler_stacks_independent_chains():
    initial_thetas = np.array([[0.0, 0.0], [1.0, -1.0], [-1.0, 1.0]])
    config = MetropolisConfig(n_steps=15, proposal_sigma=[0.1, 0.2], burn_in=5, thin=2)

    result = run_metropolis(
        gaussian_log_target,
        initial_thetas,
        config,
        seeds=[1, 2, 3],
    )

    assert result.samples.shape == (3, 5, 2)
    assert result.log_prob.shape == (3, 5)
    assert result.accepted.shape == (3, 15)
    assert result.acceptance_rates.shape == (3,)


def test_multichain_sampler_can_use_thread_executor():
    config = MetropolisConfig(n_steps=12, proposal_sigma=0.1, burn_in=2)

    result = run_metropolis(
        gaussian_log_target,
        [0.0, 0.0],
        config,
        n_chains=2,
        seeds=7,
        executor="thread",
        max_workers=2,
    )

    assert result.samples.shape == (2, 10, 2)
    assert np.all(np.isfinite(result.log_prob))


def test_global_jump_uses_uniform_bounded_proposals():
    config = MetropolisConfig(
        n_steps=30,
        proposal_sigma=2.0,
        bounds=([0.0, 0.0], [1.0, 1.0]),
        global_jump_interval=3,
    )

    result = run_metropolis_chain(box_log_target, [0.5, 0.5], config, seed=4)

    assert np.all((result.samples >= 0.0) & (result.samples <= 1.0))
    assert np.any(result.accepted)


def test_bounds_prevent_accepting_out_of_range_gaussian_proposals():
    config = MetropolisConfig(
        n_steps=30,
        proposal_sigma=5.0,
        bounds=([-0.1], [0.1]),
    )

    result = run_metropolis_chain(gaussian_log_target, [0.0], config, seed=10)

    assert np.all((result.samples >= -0.1) & (result.samples <= 0.1))


def test_draw_valid_initial_thetas_samples_prior_support():
    starts = draw_valid_initial_thetas(
        box_log_target,
        bounds=([0.0, 0.0], [1.0, 1.0]),
        n_chains=4,
        seed=5,
    )

    assert starts.shape == (4, 2)
    assert np.all((starts >= 0.0) & (starts <= 1.0))


def test_sampler_runs_geophysical_prior_only_target():
    parameterization, prior = _prior_target()
    lower, upper = parameterization.bounds()
    starts = draw_valid_initial_thetas(
        prior.log_prior,
        bounds=(lower, upper),
        n_chains=2,
        seed=6,
    )
    config = MetropolisConfig(
        n_steps=20,
        proposal_sigma=parameterization.proposal_sigmas(),
        burn_in=5,
        bounds=(lower, upper),
        global_jump_interval=10,
    )

    result = run_metropolis(prior.log_prior, starts, config, seeds=[8, 9])

    assert result.samples.shape == (2, 15, 2)
    assert np.all(np.isfinite(result.log_prob))
    assert np.all(result.samples[:, :, 0] <= result.samples[:, :, 1])
