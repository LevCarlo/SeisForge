"""Forward-solver failure policy and per-chain forward-failure counts."""

from dataclasses import replace

import disba
import numpy as np
import pytest

from seisforge.hvsr.dispersion import approximate_phase_velocity_km_s, phase_velocity_seeds_km_s
from seisforge.hvsr.models import ElasticLayerModel
from seisforge.inv.forward import (
    DispersionRequest,
    ForwardError,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.io import mcmc_result_to_xarray
from seisforge.inv.likelihood import DispersionObservation
from seisforge.inv.model import LayeredModel
from seisforge.inv.posterior import (
    CountingTarget,
    JointInversionTarget,
    forward_failure_summary,
)
from seisforge.inv.samplers import (
    MetropolisConfig,
    _chain_seeds,
    run_metropolis,
    run_metropolis_chain,
    run_tempered_chain,
)
from seisforge.inv.setup import SamplerSettings, build_inversion_setup
from test_inv_setup import _base_config  # the shared three-parameter test model

MODEL = LayeredModel(
    thickness=[2.0, 10.0, 0.0], vp=[4.0, 6.0, 8.0], vs=[2.0, 3.5, 4.5], rho=[2.2, 2.6, 3.3]
)


class _Raises:
    """Stand-in for a disba solver class whose call raises ``error``."""

    error: Exception

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise self.error


def _patch_disba(monkeypatch, error):
    solver = type("Solver", (_Raises,), {"error": error})
    for name in ("PhaseDispersion", "GroupDispersion", "Ellipticity"):
        monkeypatch.setattr(disba, name, solver)


# --- which exceptions become ForwardError -------------------------------------


@pytest.mark.parametrize("error", [disba.DispersionError("no root"), ZeroDivisionError("x")])
def test_solver_failures_become_forward_error(monkeypatch, error):
    _patch_disba(monkeypatch, error)
    for kind in ("phase", "group"):
        with pytest.raises(ForwardError):
            predict_dispersion(MODEL, DispersionRequest(periods=[5.0], kind=kind))
    with pytest.raises(ForwardError):
        predict_rayleigh_hv(MODEL, RayleighHVRequest(periods=[5.0]))


@pytest.mark.parametrize("error", [TypeError("bug"), AttributeError("bug"), ValueError("bug")])
def test_other_exceptions_propagate_from_forward_wrappers(monkeypatch, error):
    _patch_disba(monkeypatch, error)
    with pytest.raises(type(error), match="bug"):
        predict_dispersion(MODEL, DispersionRequest(periods=[5.0]))
    with pytest.raises(type(error), match="bug"):
        predict_rayleigh_hv(MODEL, RayleighHVRequest(periods=[5.0]))


def test_real_root_failure_is_a_forward_error():
    # A half-space slower than the layer above it: disba's fundamental-mode
    # root search fails with DispersionError (probe in the forward module docstring).
    model = LayeredModel(
        thickness=[0.5, 5.0, 0.0], vp=[3.5, 7.0, 2.625], vs=[2.0, 4.0, 1.5], rho=[2.2, 2.7, 2.0]
    )
    with pytest.raises(ForwardError, match="failed to find root"):
        predict_dispersion(model, DispersionRequest(periods=[5.0, 20.0]))


def test_unsorted_periods_are_reported_not_turned_into_rejections():
    # A caller error: before, every proposal was rejected silently.
    with pytest.raises(ValueError, match="sorted"):
        predict_dispersion(MODEL, DispersionRequest(periods=[5.0, 1.0]))


def test_hvsr_seed_functions_keep_failure_behaviour_but_raise_on_bugs(monkeypatch):
    model = ElasticLayerModel([2.0, 10.0], [4.0, 6.0, 8.0], [2.0, 3.5, 4.5], [2.2, 2.6, 3.3])
    _patch_disba(monkeypatch, disba.DispersionError("no root"))
    assert approximate_phase_velocity_km_s(model, 1.0, 0, "rayleigh") is None
    assert np.all(np.isnan(phase_velocity_seeds_km_s(model, [1.0, 2.0], 0, "rayleigh")))
    _patch_disba(monkeypatch, TypeError("bug"))
    with pytest.raises(TypeError):
        approximate_phase_velocity_km_s(model, 1.0, 0, "rayleigh")
    with pytest.raises(TypeError):
        phase_velocity_seeds_km_s(model, [1.0, 2.0], 0, "rayleigh")


def test_sampler_lets_value_errors_of_a_target_propagate():
    def target(theta):
        raise ValueError("bug in target")

    with pytest.raises(ValueError, match="bug in target"):
        run_metropolis_chain(target, [0.0], MetropolisConfig(n_steps=3, proposal_sigma=0.1))


# --- a sampling run with an injected programming error -------------------------


def _target(mode=0, periods=(5.0, 10.0), sigma=0.05):
    setup = build_inversion_setup(_base_config())
    layered = setup.prior.evaluate(setup.parameterization.theta0()).layered_model
    request = DispersionRequest(periods=np.asarray(periods), mode=mode)
    observed = predict_dispersion(layered, request).velocity
    observation = DispersionObservation(
        periods=periods, velocity=observed, sigma=np.full(len(periods), sigma), mode=mode
    )
    return setup, JointInversionTarget(prior=setup.prior, dispersion=observation)


def _with_sampler(setup, target, config, *, executor="serial", n_chains=2, seed=11):
    sampler = SamplerSettings(config=config, n_chains=n_chains, executor=executor, seed=seed)
    return replace(setup, sampler=sampler, target=target)


@pytest.mark.parametrize("executor", ["serial", "thread"])
@pytest.mark.parametrize("temperatures", [None, (1.0, 4.0)])
@pytest.mark.parametrize("error", [TypeError, AttributeError])
def test_programming_error_in_forward_stops_sampling(monkeypatch, executor, temperatures, error):
    # Raised inside the solver call, where `except Exception` used to turn it
    # into a rejected proposal and the run carried on.
    setup, target = _target()
    config = MetropolisConfig(
        n_steps=20, proposal_sigma=0.03, bounds=setup.parameterization.bounds(), temperatures=temperatures
    )
    setup = _with_sampler(setup, target, config, executor=executor)
    real = disba.PhaseDispersion
    calls = []

    class Broken(real):
        def __call__(self, *args, **kwargs):
            calls.append(1)
            if len(calls) > 3:  # let the starting points evaluate, then fail mid-run
                raise error("injected bug")
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(disba, "PhaseDispersion", Broken)
    with pytest.raises(error, match="injected bug"):
        setup.run_posterior()
    assert len(calls) >= 4  # threads: the other chain may run a little further


# --- counts of real solver failures --------------------------------------------

# The first higher mode exists at 1.0 and 1.2 s for the starting model but is
# cut off for about a fifth of the uniform prior box; global jumps reach that
# part, so the chains meet disba's truncated mode-1 results (a ForwardError
# through the period check) on a fair share of their proposals.
FAILING_PERIODS = (1.0, 1.2)


def _failing_setup(executor, temperatures):
    setup, target = _target(mode=1, periods=FAILING_PERIODS, sigma=0.3)
    config = MetropolisConfig(
        n_steps=60,
        burn_in=10,
        proposal_sigma=0.05,
        bounds=setup.parameterization.bounds(),
        global_jump_interval=3,
        temperatures=temperatures,
    )
    return _with_sampler(setup, target, config, executor=executor)


def _solver_fails(setup, theta) -> bool | None:
    """Independent classification: None outside the prior, else whether disba fails."""
    prior = setup.prior.evaluate(theta)
    if not prior.success:
        return None
    solver = disba.PhaseDispersion(*prior.layered_model.as_disba(), dc=0.005)
    try:
        result = solver(np.asarray(FAILING_PERIODS), mode=1, wave="rayleigh")
    except disba.DispersionError:
        return True
    return len(result.velocity) != len(FAILING_PERIODS)


def _true_counts(setup):
    """Re-run every chain serially with recording targets and classify each call."""
    config = setup.sampler.config
    n_replicas = 1 if config.temperatures is None else len(config.temperatures)
    seeds = _chain_seeds(setup.sampler.seed, setup.sampler.n_chains)
    starts = setup.initial_thetas(target="posterior")
    totals, colds = [], []
    for chain, seed in enumerate(seeds):
        recorded = [[] for _ in range(n_replicas)]

        def recorder(i, split):
            def call(theta):
                recorded[i].append(np.array(theta))
                value = setup.target.evaluate(theta)
                if not split:
                    return value.log_posterior
                if not np.isfinite(value.log_prior):
                    return -np.inf, -np.inf
                return value.log_prior, value.log_likelihood

            return call

        if n_replicas == 1:
            run_metropolis_chain(recorder(0, False), starts[chain], config, seed=seed)
        else:
            targets = [recorder(i, True) for i in range(n_replicas)]
            run_tempered_chain(targets, starts[chain], config, seed=seed)
        per_replica = []
        for thetas in recorded:
            labels = [_solver_fails(setup, theta) for theta in thetas]
            per_replica.append(
                (sum(label is not None for label in labels), sum(label is True for label in labels))
            )
        totals.append(tuple(map(sum, zip(*per_replica, strict=True))))
        colds.append(per_replica[0])
    return np.array(totals), np.array(colds)


@pytest.mark.parametrize("temperatures", [None, (1.0, 3.0, 9.0)])
def test_forward_failure_counts_match_independent_count(temperatures):
    expected_total, expected_cold = _true_counts(_failing_setup("serial", temperatures))
    assert np.all(expected_total[:, 1] > 0)  # the test needs failures to count
    assert np.all(expected_total[:, 1] < expected_total[:, 0])

    reference = None
    for executor in ("serial", "thread", "process"):
        result = _failing_setup(executor, temperatures).run_posterior()
        summary = forward_failure_summary(result.chains)
        np.testing.assert_array_equal(summary.attempts, expected_total[:, 0], err_msg=executor)
        np.testing.assert_array_equal(summary.failures, expected_total[:, 1], err_msg=executor)
        np.testing.assert_array_equal(summary.attempts_cold, expected_cold[:, 0], err_msg=executor)
        np.testing.assert_array_equal(summary.failures_cold, expected_cold[:, 1], err_msg=executor)
        assert summary.n_replicas == (1 if temperatures is None else len(temperatures))
        assert summary.most_common_errors(1)[0][0].startswith("likelihood: dispersion periods")
        assert all(np.isfinite(result.log_prob).ravel())
        if reference is None:
            reference = result
        else:  # executors draw the same chains
            np.testing.assert_array_equal(result.samples, reference.samples)


def test_forward_failure_outputs_and_warning():
    setup = _failing_setup("serial", (1.0, 3.0))
    result = setup.run_posterior()
    summary = forward_failure_summary(result.chains)
    dataset = mcmc_result_to_xarray(
        result, parameter_names=setup.parameterization.names, forward_failures=summary
    )
    rate = dataset["forward_failures"] / dataset["forward_attempts"]
    np.testing.assert_allclose(dataset["forward_failure_rate"], rate)
    for name in ("forward_attempts_cold", "forward_failures_cold", "forward_failure_rate_cold"):
        assert dataset[name].dims == ("chain",)
    assert "summed over all temperature replicas" in dataset["forward_attempts"].attrs["description"]
    assert "dispersion periods" in dataset.attrs["forward_failure_top_errors"]

    warnings = summary.warnings(threshold=0.0)
    assert len(warnings) == result.n_chains and "most frequent errors" in warnings[0]
    assert summary.warnings(threshold=1.0) == []


def test_prior_runs_and_direct_calls_are_not_counted():
    setup, target = _target()
    config = MetropolisConfig(n_steps=5, proposal_sigma=0.03, bounds=setup.parameterization.bounds())
    setup = _with_sampler(setup, target, config)
    assert forward_failure_summary(setup.run_prior().chains) is None
    assert mcmc_result_to_xarray(
        setup.run_prior(), parameter_names=setup.parameterization.names
    ).get("forward_failures") is None
    counting = CountingTarget(target)
    theta0 = setup.parameterization.theta0()
    assert counting(theta0) == target.log_posterior(theta0)
    evaluator = counting.chain_evaluator()
    evaluator(theta0)
    assert evaluator.report().n_forward_attempts == 1
    assert counting.chain_evaluator().report().n_forward_attempts == 0


# --- counting changes no number -------------------------------------------------


def test_counted_posterior_run_equals_plain_target_run():
    for temperatures in (None, (1.0, 4.0)):
        setup_t = _failing_setup("serial", temperatures)
        counted = setup_t.run_posterior()
        plain_target = (
            setup_t.target.log_posterior
            if temperatures is None
            else setup_t.target.log_prior_and_likelihood
        )
        plain = run_metropolis(
            plain_target,
            setup_t.initial_thetas(target="posterior"),
            setup_t.sampler.config,
            seeds=setup_t.sampler.seed,
        )
        assert plain.chains[0].target_reports is None
        np.testing.assert_array_equal(counted.samples, plain.samples)
        np.testing.assert_array_equal(counted.log_prob, plain.log_prob)
        np.testing.assert_array_equal(counted.accepted, plain.accepted)


class _GenericCountingTarget:
    """A ChainLocalTarget without geophysics, for the sampler baseline below."""

    def __init__(self, split):
        self.split = split

    def __call__(self, theta):
        log_prior = 0.0 if theta[0] < 1.2 else -np.inf
        log_likelihood = float(-0.5 * np.sum((theta - 0.3) ** 2 / 0.25))
        return (log_prior, log_likelihood) if self.split else log_prior + log_likelihood

    def chain_evaluator(self):
        target = self
        calls = []

        class Evaluator:
            def __call__(self, theta):
                calls.append(1)
                return target(theta)

            def report(self):
                return len(calls)

        return Evaluator()


# Generated with the sampler before chain-local evaluators existed (commit
# f52dadf) from the same target, config and seed: counting must not change a
# single bit. Pure-Python arithmetic, so the values are platform independent.
_BASE = dict(
    n_steps=12,
    proposal_sigma=[0.4, 0.2],
    burn_in=2,
    thin=2,
    bounds=([-2.0, -2.0], [2.0, 2.0]),
    global_jump_interval=5,
)
_STARTS = np.array([[0.0, 0.0], [0.5, -0.5]])
_PLAIN_SAMPLES = np.array([[[0.3958869845193686, -0.020238493332012875], [0.2341246557382261, 0.23339043480650434], [0.10506473773949424, 0.5624642728070101], [0.9713377047056081, 0.7980006153340807], [0.8221354362257494, 0.937516573285637]], [[0.33600144545660615, -0.5938285081960967], [0.2682549485376078, 0.14840848097704962], [0.6424470579866487, 0.1387995637501489], [0.9426547168968707, 0.4009858740494342], [0.9426547168968707, 0.4009858740494342]]])
_PLAIN_LOG_PROB = np.array([[-0.22349401282355058, -0.017552790313747544, -0.21377450194536954, -1.3973978532650349, -1.3581055899530288], [-1.6004510122780489, -0.0479754738640709, -0.2865111363417069, -0.8464064638146507, -0.8464064638146507]])
_PLAIN_ACCEPTED = np.array([[1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]])
_TEMPERED_SAMPLES = np.array([[[0.8836283229083539, -0.35379485165290914], [0.3185615369775533, 0.27268112389186505], [0.593188739489684, 0.47515286350767844], [0.3337087231245167, 0.5040523572675988], [0.3337087231245167, 0.5040523572675988]], [[0.6332303923353553, -0.24680541440790046], [0.07328061855660994, -0.07338399705317072], [0.038034232169112364, 0.13855591159692693], [0.37781700997575024, 0.24692482706191674], [0.2505705693450443, 0.407325051217643]]])
_TEMPERED_LOG_PROB = np.array([[-1.5361394546973344, -0.0021817032935614092, -0.23327632511697843, -0.08554728504229821, -0.08554728504229821], [-0.820077311203541, -0.3816345743549512, -0.1893805143910516, -0.017744922047946917, -0.027923870467485543]])
_TEMPERED_ACCEPTED = np.array([[1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]])


@pytest.mark.parametrize("executor", ["serial", "thread"])
def test_chain_local_evaluators_reproduce_pre_change_baseline(executor):
    plain = run_metropolis(
        _GenericCountingTarget(split=False), _STARTS, MetropolisConfig(**_BASE), seeds=7, executor=executor
    )
    np.testing.assert_array_equal(plain.samples, _PLAIN_SAMPLES)
    np.testing.assert_array_equal(plain.log_prob, _PLAIN_LOG_PROB)
    np.testing.assert_array_equal(plain.accepted, _PLAIN_ACCEPTED.astype(bool))
    assert all(len(c.target_reports) == 1 and c.target_reports[0] > 0 for c in plain.chains)

    tempered = run_metropolis(
        _GenericCountingTarget(split=True),
        _STARTS,
        MetropolisConfig(**_BASE, temperatures=(1.0, 3.0, 9.0)),
        seeds=7,
        executor=executor,
    )
    np.testing.assert_array_equal(tempered.samples, _TEMPERED_SAMPLES)
    np.testing.assert_array_equal(tempered.log_prob, _TEMPERED_LOG_PROB)
    np.testing.assert_array_equal(tempered.accepted, _TEMPERED_ACCEPTED.astype(bool))
    assert all(len(c.target_reports) == 3 for c in tempered.chains)
