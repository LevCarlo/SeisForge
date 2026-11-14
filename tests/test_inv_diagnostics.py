import numpy as np
import pytest

from seisforge.inv import (
    convergence_diagnostics,
    effective_sample_size,
    lagging_chains,
    split_rhat,
)


def _ar1(rng, n_chains, n_draws, rho):
    x = np.empty((n_chains, n_draws))
    x[:, 0] = rng.standard_normal(n_chains)
    noise = rng.standard_normal((n_chains, n_draws)) * np.sqrt(1 - rho**2)
    for i in range(1, n_draws):
        x[:, i] = rho * x[:, i - 1] + noise[:, i]
    return x


def test_independent_chains_have_rhat_one_and_full_ess():
    x = np.random.default_rng(1).standard_normal((4, 4000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)
    # ESS of 16000 independent draws; the estimator scatters by a few percent.
    assert effective_sample_size(x) == pytest.approx(16000, rel=0.1)


def test_ess_of_ar1_chains_matches_the_analytic_value():
    # Integrated autocorrelation time of AR(1) is (1 + rho) / (1 - rho), so
    # ESS = N (1 - rho) / (1 + rho): 80000 * 0.1 / 1.9 = 4211.
    rho = 0.9
    x = _ar1(np.random.default_rng(2), 4, 20000, rho)
    expected = x.size * (1 - rho) / (1 + rho)
    # Relative standard error of the ESS estimate is a few percent here.
    assert effective_sample_size(x) == pytest.approx(expected, rel=0.15)


def test_chain_in_another_mode_raises_rhat_and_is_flagged():
    rng = np.random.default_rng(3)
    samples = rng.standard_normal((4, 1000, 2))
    samples[2, :, 0] += 5.0
    rhat = split_rhat(samples)
    assert rhat[0] > 1.5 and rhat[1] == pytest.approx(1.0, abs=0.02)

    log_posterior = rng.standard_normal((4, 1000))
    log_posterior[2] -= 50.0
    assert lagging_chains(log_posterior) == (2,)
    assert lagging_chains(rng.standard_normal((4, 1000))) == ()


def test_convergence_diagnostics_reports_and_warns():
    rng = np.random.default_rng(4)
    samples = rng.standard_normal((4, 500, 2))
    samples[1, :, 1] += 4.0
    log_posterior = rng.standard_normal((4, 500))
    result = convergence_diagnostics(samples, log_posterior)
    assert result.rhat.shape == (2,) and result.ess.shape == (2,)
    warnings = result.warnings(["a", "b"])
    assert any(w.startswith("b: split R-hat") for w in warnings)
    assert not any(w.startswith("a:") for w in warnings)


def test_convergence_diagnostics_too_few_draws_gives_nan():
    result = convergence_diagnostics(np.zeros((1, 3, 2)), np.zeros((1, 3)))
    assert np.all(np.isnan(result.rhat)) and result.lagging_chains == ()
