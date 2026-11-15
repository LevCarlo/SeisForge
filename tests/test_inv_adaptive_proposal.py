"""Adaptive Metropolis during burn-in (``MetropolisConfig.adaptation``).

Independent checks: the running moments against ``numpy.cov``; sample mean
and covariance against the analytic moments of Gaussian targets; ESS against
the best fixed isotropic step; the frozen covariance against a run that stops
right after burn-in. The bit-for-bit check with adaptation off is in
`test_inv_sampler_baseline.py`.
"""

import math

import numpy as np
import pytest

from seisforge.inv.diagnostics import effective_sample_size
from seisforge.inv.samplers import (
    OPTIMAL_RW_SCALE,
    AdaptationConfig,
    MetropolisConfig,
    _ProposalAdapter,
    adaptive_proposal_covariance,
    run_metropolis,
    running_covariance,
    update_running_moments,
)
from seisforge.inv.setup import build_inversion_setup
from test_inv_setup import _base_config

RHO = 0.99
CORRELATED = np.array([[1.0, RHO], [RHO, 1.0]])
_CORRELATED_PRECISION = np.linalg.inv(CORRELATED)

# Three parameters six orders of magnitude apart, two of them correlated.
WIDE_SD = np.array([1e-3, 1.0, 1e3])
_WIDE_CORRELATION = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, -0.3], [0.0, -0.3, 1.0]])
WIDE_COVARIANCE = WIDE_SD[:, None] * _WIDE_CORRELATION * WIDE_SD[None, :]
_WIDE_PRECISION = np.linalg.inv(WIDE_COVARIANCE)


def correlated_log_target(theta) -> float:
    return float(-0.5 * theta @ _CORRELATED_PRECISION @ theta)


def correlated_split_target(theta) -> tuple[float, float]:
    return 0.0, correlated_log_target(theta)


def wide_log_target(theta) -> float:
    return float(-0.5 * theta @ _WIDE_PRECISION @ theta)


def _moments_tolerance(ess: float) -> tuple[float, float]:
    """Five Monte Carlo standard errors for unit-variance Gaussian moments.

    Mean: sd / sqrt(ESS). Covariance entries: about sqrt((1 + rho^2) / ESS)
    <= sqrt(2 / ESS) (Gaussian fourth moments). ESS is the minimum over the
    parameters, so the bound is conservative.
    """
    return 5 / math.sqrt(ess), 5 * math.sqrt(2 / ess)


# --- pure functions -----------------------------------------------------


def test_running_moments_match_numpy_cov_across_scales():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(200, 4)) * np.array([1e-4, 1.0, 1e2, 1e5]) + np.array([3.0, -1.0, 5e3, 1e6])
    count, mean, scatter = 0, np.zeros(4), np.zeros((4, 4))
    for point in points:
        count, mean, scatter = update_running_moments(count, mean, scatter, point)
    assert count == 200
    # Welford's recursion and numpy's two-pass formula differ by rounding
    # only: relative 1e-10 of each entry's scale sd_i * sd_j.
    scale = np.outer(points.std(axis=0), points.std(axis=0))
    np.testing.assert_allclose(mean, points.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(running_covariance(count, scatter) / scale, np.cov(points.T) / scale, atol=1e-10)
    with pytest.raises(ValueError, match="two points"):
        running_covariance(1, np.zeros((4, 4)))


def test_adaptive_proposal_covariance_formula():
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    sigma = np.array([0.1, 10.0])
    expected = (2.38**2 / 2) * (covariance + 1e-3 * np.diag([0.01, 100.0]))
    np.testing.assert_allclose(adaptive_proposal_covariance(covariance, sigma, 1e-3), expected, rtol=1e-15)


def test_adapter_freezes_after_last_update():
    rng = np.random.default_rng(1)
    adaptation = AdaptationConfig(initial_diagonal_steps=5, update_interval_steps=4)
    assert adaptation.update_steps(burn_in=17) == (5, 9, 13, 17)
    sigma = np.array([0.5, 2.0])
    adapter = _ProposalAdapter(adaptation, 17, np.zeros(2), sigma)
    states = rng.normal(size=(40, 2))
    for step, state in enumerate(states[:17]):
        adapter.observe(step, state)
    # History after 17 steps: the start and the 17 states.
    history = np.vstack([np.zeros(2), states[:17]])
    expected = adaptive_proposal_covariance(np.cov(history.T), sigma, adaptation.regularization_fraction)
    # Welford vs two-pass: rounding only.
    np.testing.assert_allclose(adapter.covariance(), expected, rtol=1e-12)
    frozen = adapter.covariance()
    for step, state in enumerate(states[17:], start=17):
        adapter.observe(step, state)
    np.testing.assert_array_equal(adapter.covariance(), frozen)


# --- validation -----------------------------------------------------------


def test_adaptation_needs_burn_in():
    adaptation = AdaptationConfig(initial_diagonal_steps=100, update_interval_steps=10)
    with pytest.raises(ValueError, match=r"burn_in \(0\) must be at least adaptation.initial_diagonal_steps \(100\)"):
        MetropolisConfig(n_steps=1000, proposal_sigma=0.1, adaptation=adaptation)
    with pytest.raises(ValueError, match=r"burn_in \(50\)"):
        MetropolisConfig(n_steps=1000, proposal_sigma=0.1, burn_in=50, adaptation=adaptation)
    MetropolisConfig(n_steps=1000, proposal_sigma=0.1, burn_in=100, adaptation=adaptation)
    for kwargs in (
        dict(initial_diagonal_steps=0, update_interval_steps=1),
        dict(initial_diagonal_steps=5, update_interval_steps=0),
        dict(initial_diagonal_steps=5, update_interval_steps=1, regularization_fraction=0.0),
    ):
        with pytest.raises(ValueError):
            AdaptationConfig(**kwargs)
    # Fewer starting steps than parameters cannot give a covariance.
    config = MetropolisConfig(
        n_steps=20, proposal_sigma=0.1, burn_in=10, adaptation=AdaptationConfig(2, 1)
    )
    with pytest.raises(ValueError, match="must exceed the number of parameters"):
        run_metropolis(correlated_log_target, np.zeros(2), config, seeds=0)


def test_adaptation_from_config():
    config = _base_config()
    assert build_inversion_setup(config).sampler.config.adaptation is None
    config["sampler"].update(
        n_steps=200,
        burn_in=50, adaptation={"initial_diagonal_steps": 20, "update_interval_steps": 10}
    )
    adaptation = build_inversion_setup(config).sampler.config.adaptation
    assert adaptation == AdaptationConfig(20, 10)
    config["sampler"]["adaptation"]["regularization_fraction"] = 1e-3
    assert build_inversion_setup(config).sampler.config.adaptation.regularization_fraction == 1e-3
    config["sampler"]["adaptation"]["interval"] = 3
    with pytest.raises(ValueError, match="interval"):
        build_inversion_setup(config)
    config["sampler"]["adaptation"] = {"initial_diagonal_steps": 20}
    with pytest.raises(ValueError, match="update_interval_steps"):
        build_inversion_setup(config)
    config["sampler"]["adaptation"] = {"initial_diagonal_steps": 80, "update_interval_steps": 10}
    with pytest.raises(ValueError, match="burn_in"):
        build_inversion_setup(config)


# --- sampling -------------------------------------------------------------


def _correlated_run(sigma, adaptation=None, **kwargs):
    config = MetropolisConfig(n_steps=4000, proposal_sigma=sigma, burn_in=1000, adaptation=adaptation, **kwargs)
    return run_metropolis(correlated_log_target, np.zeros((4, 2)), config, seeds=5)


def test_adaptation_raises_ess_on_a_correlated_gaussian():
    adapted = _correlated_run(0.17, AdaptationConfig(initial_diagonal_steps=300, update_interval_steps=50))
    adapted_ess = effective_sample_size(adapted.samples).min()
    # Best fixed isotropic step: a scan over 0.3-2.0 (10 seeds each) gave the
    # largest ESS at the widest steps, ~150 of 12000 draws; adaptation gave
    # 1300-1660, a ratio of 8.7 or more on every seed. Require 4.
    fixed_ess = max(effective_sample_size(_correlated_run(s).samples).min() for s in (0.5, 1.0, 2.0, 3.0))
    assert adapted_ess > 4 * fixed_ess, (adapted_ess, fixed_ess)

    mean_tol, cov_tol = _moments_tolerance(adapted_ess)
    draws = adapted.samples.reshape(-1, 2)
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=mean_tol)
    np.testing.assert_allclose(np.cov(draws.T), CORRELATED, atol=cov_tol)
    for chain in adapted.chains:
        assert chain.proposal_covariance.shape == (1, 2, 2)
        # The frozen proposal follows the target: correlation near 0.99
        # (the burn-in history starts away from equilibrium, so not exact).
        c = chain.proposal_covariance[0]
        assert c[0, 1] / math.sqrt(c[0, 0] * c[1, 1]) > 0.95
    post_burn_in = adapted.accepted[:, 1000:].mean()
    assert 0.15 < post_burn_in < 0.5  # Gaussian optimum ~0.35 for d = 2


def test_proposal_is_frozen_after_burn_in():
    adaptation = AdaptationConfig(initial_diagonal_steps=100, update_interval_steps=30)
    config = dict(proposal_sigma=0.2, burn_in=400, adaptation=adaptation)
    short = run_metropolis(correlated_log_target, np.zeros((2, 2)), MetropolisConfig(n_steps=401, **config), seeds=9)
    long = run_metropolis(correlated_log_target, np.zeros((2, 2)), MetropolisConfig(n_steps=3000, **config), seeds=9)
    for a, b in zip(short.chains, long.chains, strict=True):
        assert not np.allclose(a.proposal_covariance[0], np.diag([0.04, 0.04]))  # it did adapt
        np.testing.assert_array_equal(a.proposal_covariance, b.proposal_covariance)
    np.testing.assert_array_equal(short.samples[:, 0], long.samples[:, 0])


def test_adaptation_across_scales():
    # Start with steps 10x too small but scaled per parameter, as users give
    # proposal.sigma in each parameter's own unit.
    common = dict(n_steps=5000, proposal_sigma=0.1 * WIDE_SD, burn_in=2000)
    adaptation = AdaptationConfig(initial_diagonal_steps=500, update_interval_steps=100)
    result = run_metropolis(
        wide_log_target, np.zeros((4, 3)), MetropolisConfig(adaptation=adaptation, **common), seeds=3
    )
    fixed = run_metropolis(wide_log_target, np.zeros((4, 3)), MetropolisConfig(**common), seeds=3)
    ess = effective_sample_size(result.samples).min()
    # Seeds 0-4: adapted ESS 825-1171, fixed 4-8 (ratio >= 100); require 20.
    assert ess > 20 * effective_sample_size(fixed.samples).min()
    scale = OPTIMAL_RW_SCALE**2 / 3
    for chain in result.chains:
        c = chain.proposal_covariance[0] / scale
        sd = np.sqrt(np.diag(c))
        # Seeds 0-4: sd within 0.82-1.11 of the target sd, correlation within
        # 0.30 (2000 strongly autocorrelated burn-in states, starting with
        # small steps). Failing to adapt across scales would be off by orders
        # of magnitude, so 0.3 and 0.4 still discriminate.
        np.testing.assert_allclose(sd / WIDE_SD, 1.0, atol=0.3)
        np.testing.assert_allclose(c / np.outer(sd, sd), _WIDE_CORRELATION, atol=0.4)
    draws = result.samples.reshape(-1, 3) / WIDE_SD
    mean_tol, cov_tol = _moments_tolerance(ess)
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=mean_tol)
    np.testing.assert_allclose(np.cov(draws.T), _WIDE_CORRELATION, atol=cov_tol)


def _in_box_target(theta) -> float:
    if np.any(np.abs(theta) > 1.0):
        raise AssertionError("evaluated outside the bounds")
    return correlated_log_target(theta)


def test_adaptive_proposals_outside_bounds_are_rejected_unevaluated():
    config = MetropolisConfig(
        n_steps=2000,
        proposal_sigma=0.3,
        burn_in=500,
        bounds=([-1.0, -1.0], [1.0, 1.0]),
        adaptation=AdaptationConfig(initial_diagonal_steps=100, update_interval_steps=50),
    )
    result = run_metropolis(_in_box_target, np.zeros((2, 2)), config, seeds=4)
    assert np.all(np.abs(result.samples) <= 1.0)


def test_global_jumps_do_not_use_the_adapted_proposal():
    box = ([-3.0, -3.0], [3.0, 3.0])
    adaptation = AdaptationConfig(initial_diagonal_steps=50, update_interval_steps=10)
    # Every step a global jump: the adapted proposal is never drawn, so the
    # chains equal those without adaptation draw for draw.
    common = dict(n_steps=500, proposal_sigma=0.3, burn_in=100, bounds=box, global_jump_interval=1)
    plain = run_metropolis(correlated_log_target, np.zeros((2, 2)), MetropolisConfig(**common), seeds=8)
    adapted = run_metropolis(
        correlated_log_target, np.zeros((2, 2)), MetropolisConfig(adaptation=adaptation, **common), seeds=8
    )
    np.testing.assert_array_equal(plain.samples, adapted.samples)
    # Mixed schedule: the kernel still samples the target.
    mixed = run_metropolis(
        correlated_log_target,
        np.zeros((4, 2)),
        MetropolisConfig(
            n_steps=4000,
            proposal_sigma=0.3,
            burn_in=1000,
            bounds=([-6.0, -6.0], [6.0, 6.0]),  # > 6 sd: truncation negligible
            global_jump_interval=7,
            adaptation=AdaptationConfig(initial_diagonal_steps=300, update_interval_steps=50),
        ),
        seeds=8,
    )
    mean_tol, cov_tol = _moments_tolerance(effective_sample_size(mixed.samples).min())
    draws = mixed.samples.reshape(-1, 2)
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=mean_tol)
    np.testing.assert_allclose(np.cov(draws.T), CORRELATED, atol=cov_tol)


def test_tempered_replicas_adapt_separately():
    ladder = (1.0, 4.0, 16.0)
    config = MetropolisConfig(
        n_steps=4000,
        proposal_sigma=0.17,
        burn_in=1500,
        temperatures=ladder,
        adaptation=AdaptationConfig(initial_diagonal_steps=300, update_interval_steps=50),
    )
    result = run_metropolis(correlated_split_target, np.zeros((4, 2)), config, seeds=6)
    scale = OPTIMAL_RW_SCALE**2 / 2
    for chain in result.chains:
        assert chain.proposal_covariance.shape == (3, 2, 2)
        variance = np.array([np.diag(c).mean() / scale for c in chain.proposal_covariance])
        # Replica i samples N(0, T_i C): its proposal variance grows with T_i
        # and stays within a factor 2 of T_i (seeds 0-11 gave 0.75-1.51; the
        # burn-in histories are short and swaps make them noisier).
        assert np.all(np.diff(variance) > 0)
        ratio = variance / np.array(ladder)
        assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio
    ess = effective_sample_size(result.samples).min()
    mean_tol, cov_tol = _moments_tolerance(ess)
    draws = result.samples.reshape(-1, 2)
    np.testing.assert_allclose(draws.mean(axis=0), 0.0, atol=mean_tol)
    np.testing.assert_allclose(np.cov(draws.T), CORRELATED, atol=cov_tol)


@pytest.mark.parametrize("temperatures", [None, (1.0, 3.0)])
def test_executors_agree_with_adaptation(temperatures):
    target = correlated_log_target if temperatures is None else correlated_split_target
    config = MetropolisConfig(
        n_steps=600,
        proposal_sigma=0.2,
        burn_in=200,
        bounds=([-5.0, -5.0], [5.0, 5.0]),
        global_jump_interval=11,
        temperatures=temperatures,
        adaptation=AdaptationConfig(initial_diagonal_steps=50, update_interval_steps=25),
    )
    starts = np.array([[0.5, 0.5], [-0.5, 0.0], [0.0, 1.0]])
    results = {e: run_metropolis(target, starts, config, seeds=21, executor=e) for e in ("serial", "thread", "process")}
    reference = results["serial"]
    for executor, result in results.items():
        np.testing.assert_array_equal(result.samples, reference.samples, err_msg=executor)
        np.testing.assert_array_equal(result.log_prob, reference.log_prob, err_msg=executor)
        for a, b in zip(result.chains, reference.chains, strict=True):
            np.testing.assert_array_equal(a.proposal_covariance, b.proposal_covariance, err_msg=executor)
    # Chains adapt independently: different histories, different proposals.
    covs = [c.proposal_covariance for c in reference.chains]
    assert not np.array_equal(covs[0], covs[1])


def test_adaptive_chain_matches_fixed_chain_before_first_update():
    """Before t_0 an adaptive chain draws exactly what a fixed chain draws."""
    common = dict(n_steps=301, proposal_sigma=0.3, burn_in=300)
    fixed = run_metropolis(correlated_log_target, np.zeros(2), MetropolisConfig(**common), seeds=2)
    adaptive = run_metropolis(
        correlated_log_target,
        np.zeros(2),
        MetropolisConfig(adaptation=AdaptationConfig(300, 50), **common),
        seeds=2,
    )
    # Steps 0-299 use the diagonal proposal; the first update follows step 299.
    np.testing.assert_array_equal(fixed.accepted[0, :300], adaptive.accepted[0, :300])
    assert fixed.accepted[0, :300].sum() > 50
