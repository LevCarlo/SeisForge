"""Small MCMC samplers for inversion targets.

The samplers operate on generic log-probability targets. Geophysical meaning
lives in `prior.py` and `likelihood.py`; this module only moves chains through
parameter space and records diagnostics.

Random-walk Metropolis with Gaussian proposals, an optional uniform "global
jump" every ``global_jump_interval`` steps (an independence proposal,
symmetric inside the bounds; the ``step4uwalk`` idea of pymcinv), burn-in,
thinning, and independent chains run serially, in threads, or in processes.

Parallel tempering (Geyer 1991, Computing Science and Statistics 23,
156-163; Earl & Deem 2005, Phys. Chem. Chem. Phys. 7, 3910-3916; adapted from
ComplyBay): with ``temperatures = (1, T_2, ..., T_max)`` every chain is a
ladder of replicas stepped together in one worker. The target then returns
``(log prior, log likelihood)`` and replica ``i`` accepts with
``d(log prior) + d(log likelihood) / T_i``. Every ``swap_interval`` steps,
adjacent replicas (even and odd pairs in turn) exchange states with
``log alpha = (1/T_i - 1/T_{i+1}) (L_{i+1} - L_i)``; the prior cancels. This
leaves the product of the tempered posteriors invariant, so the ``T = 1``
replica, the only one stored, samples the posterior, while hot replicas cross
between modes the cold one cannot leave. Without ``temperatures`` a chain
draws exactly the random numbers it drew before tempering existed.

Adaptive proposals during burn-in (``MetropolisConfig.adaptation``): the
adaptive Metropolis algorithm of Haario, Saksman & Tamminen (2001, Bernoulli
7, 223-242), restricted to burn-in. With ``d`` parameters, chain states
``x_0, ..., x_t`` (``x_0`` the starting point; a rejected step repeats the
state) and the diagonal starting proposal ``sigma`` (``proposal_sigma``), the
Gaussian random-walk proposal has covariance

    C_t = diag(sigma**2)                                  for t < t_0
    C_t = s_d * (Cov(x_0, ..., x_t) + eps * diag(sigma**2)) otherwise

where ``t_0 = adaptation.initial_diagonal_steps`` and
``s_d = 2.38**2 / d``, the scaling that is optimal for Gaussian targets
(Gelman, Roberts & Gilks 1996, Bayesian Statistics 5, 599-607; the value
Haario et al. use). ``C_t`` is recomputed every
``adaptation.update_interval_steps`` steps from a running mean and scatter
matrix (Welford's recursion, `update_running_moments`), never from the full
history; the last update falls at or before the end of burn-in, after which
``C`` is frozen. The stored draws therefore come from a Metropolis chain with
one fixed symmetric Gaussian kernel, whose stationary distribution is the
target whatever ``C`` is: adaptation only changes how fast the chain mixes,
and no ergodicity argument for a continuously adapted chain is needed.

Leaving the early history out (``adaptation.discard_early_history_fraction``,
``f``, default 0 = the whole history as above). A chain started far from the
high-probability region first drifts towards it; those transient states
inflate ``Cov(x_0, ..., x_t)`` along the drift direction, so the learned
steps come out too wide and post-burn-in acceptance too low (seen on the
STA001 example: acceptance about 0.11). With ``0 <= f < 1`` the update after
``n`` completed steps uses only the most recent states,

    C_n = s_d * (Cov(x_k, ..., x_n) + eps * diag(sigma**2)),  k = floor(f * (n + 1)),

i.e. the earliest fraction ``f`` of the ``n + 1`` states so far is left out
and the window grows with the chain (``f = 0.5``: the most recent half, the
"discard the first half" rule of thumb also used for burn-in itself, e.g.
Gelman et al. 2013, Bayesian Data Analysis, Sec. 11.4). Every state still
enters some window, so as burn-in lengthens the estimate converges to the
target covariance as in Haario et al.; the frozen-after-burn-in argument
above is unchanged. The covariance of the window is the two-pass sample
covariance of the stored burn-in states (`recent_history_covariance`); with
``f = 0`` the running moments are used exactly as before, so ``f = 0``
reproduces earlier adaptive runs bit for bit. The window at the first update
must hold more than ``d`` states (checked when a chain starts).

``eps`` is ``adaptation.regularization_fraction`` (default
``DEFAULT_REGULARIZATION_FRACTION = 1e-4``, assumed) times the starting
variance of each parameter, not times the identity as in Haario et al.:
parameters with different units and scales (for example km/s next to a
log10 noise scale) then each get a floor of ``sqrt(s_d * eps) * sigma_j``,
about 1% of their own starting step times ``2.38 / sqrt(d)``, which keeps
``C`` positive definite when a parameter has not moved without distorting
any well-sampled direction. A chain that accepted nothing before its first
update therefore keeps a tiny but valid proposal; check the burn-in acceptance.

Relation to the other moves:

- Proposals outside ``bounds`` are rejected without evaluating the target, as
  without adaptation; the repeated state enters the history.
- Global-jump steps keep their uniform proposal and do not use ``C``. The
  state after such a step is a state of the chain like any other and enters
  the history: the history is the chain itself (as in Haario et al.), and
  leaving states out would make it depend on the jump schedule.
- With tempering every temperature slot of the ladder adapts its own
  ``C_i``, starting from ``sigma * sqrt(T_i)`` with ``eps`` relative to that
  start. The history of slot ``i`` is the sequence of states held at
  temperature ``T_i`` after each step, swaps included: after burn-in that
  sequence is the chain that samples the tempered target ``i``, which is the
  covariance the slot's proposal should match. A swap therefore moves states,
  not histories, between slots.
- Every chain adapts from its own history only, with the random numbers of
  its own generator, so the serial, thread and process executors draw the
  same chains.
- Without adaptation the sampler draws exactly the random numbers it drew
  before adaptation existed (pinned by ``tests/test_inv_sampler_baseline.py``);
  before ``t_0`` an adaptive chain draws the same steps as a fixed one.

Failure policy: a target signals an invalid point by returning ``-inf`` (or
NaN, read as ``-inf``). Floating-point overflow raised during an evaluation
(``FloatingPointError``, ``OverflowError``) also counts as ``-inf``; every
other exception propagates and stops the run, so a programming error in a
target is never mistaken for a rejected proposal.

Per-chain evaluation reports: a target may provide ``chain_evaluator()``
returning a fresh callable with a ``report()`` method (see
`ChainLocalTarget`). Each chain then evaluates through its own evaluator, one
per replica of a tempered ladder, and stores the reports in
`ChainResult.target_reports`. The evaluators are created inside the worker, so
no state is shared between threads and reports come back with the chain
result from a worker process. The sampler does not read the reports and the
evaluators must return the same values as the target itself, so random
numbers and log probabilities are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np


LogTarget = Callable[[np.ndarray], float]
SplitLogTarget = Callable[[np.ndarray], tuple[float, float]]
ProgressCallback = Callable[[int, int, float, float], None]
ExecutorKind = Literal["serial", "process", "thread"]


class ReportingEvaluator(Protocol):
    """A log target that also reports what happened during its evaluations."""

    def __call__(self, theta: np.ndarray) -> Any: ...

    def report(self) -> Any: ...


@runtime_checkable
class ChainLocalTarget(Protocol):
    """A log target that gives every chain (replica) its own evaluator.

    ``chain_evaluator()`` must return a new `ReportingEvaluator` that returns
    exactly what ``target(theta)`` returns. Calling the target directly (for
    example to draw starting points) is not reported.
    """

    def __call__(self, theta: np.ndarray) -> Any: ...

    def chain_evaluator(self) -> ReportingEvaluator: ...


# Assumed: regularization of the adapted proposal relative to the starting
# variance of each parameter (see the module docstring).
DEFAULT_REGULARIZATION_FRACTION = 1e-4
# Gelman, Roberts & Gilks (1996): s_d = OPTIMAL_RW_SCALE**2 / d.
OPTIMAL_RW_SCALE = 2.38


@dataclass(frozen=True)
class AdaptationConfig:
    """Adaptive Metropolis during burn-in (Haario et al. 2001), frozen afterwards.

    Parameters
    ----------
    initial_diagonal_steps
        Steps run with the diagonal ``proposal_sigma`` before the first update
        (``t_0``); the empirical covariance is first computed from the
        starting point and these states. Must exceed the number of
        parameters (checked when a chain starts) and must not exceed
        ``burn_in``.
    update_interval_steps
        Steps between proposal updates after ``t_0``.
    regularization_fraction
        ``eps``: fraction of the starting variance added to the empirical
        covariance of each parameter.
    discard_early_history_fraction
        ``f`` in ``[0, 1)``: the earliest fraction of the states so far left
        out of each covariance estimate (module docstring). 0 (default) uses
        the whole history, as before this option existed.
    """

    initial_diagonal_steps: int
    update_interval_steps: int
    regularization_fraction: float = DEFAULT_REGULARIZATION_FRACTION
    discard_early_history_fraction: float = 0.0

    def __post_init__(self):
        if self.initial_diagonal_steps < 1:
            raise ValueError("adaptation.initial_diagonal_steps must be positive.")
        if self.update_interval_steps < 1:
            raise ValueError("adaptation.update_interval_steps must be positive.")
        if not (np.isfinite(self.regularization_fraction) and self.regularization_fraction > 0):
            raise ValueError("adaptation.regularization_fraction must be positive and finite.")
        f = self.discard_early_history_fraction
        if not (np.isfinite(f) and 0.0 <= f < 1.0):
            raise ValueError("adaptation.discard_early_history_fraction must be in [0, 1).")

    def history_window_start(self, n_completed_steps: int) -> int:
        """``k = floor(f * (n + 1))``: index of the first state ``x_k`` used after ``n`` steps."""
        return int(np.floor(self.discard_early_history_fraction * (n_completed_steps + 1)))

    def history_window_size(self, n_completed_steps: int) -> int:
        """Number of states ``x_k, ..., x_n`` in the covariance window after ``n`` steps."""
        return n_completed_steps + 1 - self.history_window_start(n_completed_steps)

    def update_steps(self, burn_in: int) -> tuple[int, ...]:
        """Completed step counts after which the proposal is recomputed.

        The proposal computed after ``n`` completed steps is used from step
        ``n`` (0-based) on; the last entry is at most ``burn_in``.
        """
        return tuple(range(self.initial_diagonal_steps, burn_in + 1, self.update_interval_steps))


@dataclass(frozen=True)
class MetropolisConfig:
    """Configuration for random-walk Metropolis sampling."""

    n_steps: int
    proposal_sigma: float | Sequence[float]
    burn_in: int = 0
    thin: int = 1
    bounds: tuple[Sequence[float], Sequence[float]] | None = None
    global_jump_interval: int | None = None
    temperatures: Sequence[float] | None = None  # parallel tempering ladder, first = 1
    swap_interval: int = 1
    adaptation: AdaptationConfig | None = None  # adaptive proposal during burn-in

    def __post_init__(self):
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must be smaller than n_steps.")
        if self.thin <= 0:
            raise ValueError("thin must be positive.")
        if self.global_jump_interval is not None and self.global_jump_interval <= 0:
            raise ValueError("global_jump_interval must be positive when provided.")
        if self.global_jump_interval is not None and self.bounds is None:
            raise ValueError("bounds are required when global_jump_interval is used.")
        if self.temperatures is not None:
            ladder = np.asarray(self.temperatures, dtype=float)
            if (
                ladder.ndim != 1
                or ladder.size < 2
                or ladder[0] != 1.0
                or not np.all(np.diff(ladder) > 0)
                or not np.all(np.isfinite(ladder))
            ):
                raise ValueError("temperatures must start at 1 and increase, with at least two values.")
            object.__setattr__(self, "temperatures", tuple(float(t) for t in ladder))
        if self.swap_interval < 1:
            raise ValueError("swap_interval must be positive.")
        if self.adaptation is not None:
            if not isinstance(self.adaptation, AdaptationConfig):
                raise TypeError("adaptation must be an AdaptationConfig or None.")
            if self.burn_in < self.adaptation.initial_diagonal_steps:
                raise ValueError(
                    "Adaptive proposals are fitted during burn-in only and frozen afterwards: "
                    f"burn_in ({self.burn_in}) must be at least "
                    f"adaptation.initial_diagonal_steps ({self.adaptation.initial_diagonal_steps})."
                )


def geometric_temperatures(n_temperatures: int, max_temperature: float) -> tuple[float, ...]:
    """``T_i = T_max**(i / (n - 1))``, i = 0..n-1: a geometric ladder from 1 to ``max_temperature``.

    Ratios of about 2-3 between neighbours keep swaps accepted for a few to ten
    parameters (assumed, from ComplyBay's tests); check ``swap_acceptance``.
    """
    if n_temperatures < 2 or max_temperature <= 1:
        raise ValueError("Need n_temperatures >= 2 and max_temperature > 1.")
    exponents = np.arange(n_temperatures) / (n_temperatures - 1)
    return tuple(float(t) for t in max_temperature**exponents)


@dataclass(frozen=True)
class ChainResult:
    """Samples and diagnostics from one Metropolis chain."""

    samples: np.ndarray
    log_prob: np.ndarray
    accepted: np.ndarray
    initial_theta: np.ndarray
    initial_log_prob: float
    final_theta: np.ndarray
    final_log_prob: float
    seed: int | None = None
    chain_id: int = 0
    swap_acceptance: np.ndarray | None = None  # per adjacent temperature pair
    # ``report()`` of each chain-local evaluator, one per replica in ladder
    # order (one for a plain chain); None unless the target is a ChainLocalTarget.
    target_reports: tuple | None = None
    # Frozen adaptive proposal covariance, shape (n_replicas, d, d) in ladder
    # order (n_replicas = 1 for a plain chain), as in use at the end of the
    # run; None without adaptation.
    proposal_covariance: np.ndarray | None = None

    @property
    def n_steps(self) -> int:
        return len(self.accepted)

    @property
    def proposal_sd(self) -> np.ndarray | None:
        """Per-parameter step size of the frozen proposal of the ``T = 1`` chain."""
        if self.proposal_covariance is None:
            return None
        return np.sqrt(np.diag(self.proposal_covariance[0]))

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted)) if self.n_steps else np.nan


@dataclass(frozen=True)
class MCMCResult:
    """Stacked output from one or more independent chains."""

    samples: np.ndarray
    log_prob: np.ndarray
    accepted: np.ndarray
    chains: tuple[ChainResult, ...]
    config: MetropolisConfig

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def acceptance_rates(self) -> np.ndarray:
        return np.array([chain.acceptance_rate for chain in self.chains], dtype=float)

    @property
    def mean_acceptance_rate(self) -> float:
        return float(np.mean(self.acceptance_rates))

    @property
    def post_burn_in_acceptance_rates(self) -> np.ndarray:
        """Acceptance rate of each chain over the steps after ``burn_in``."""
        return self.accepted[:, self.config.burn_in :].mean(axis=1)


def run_metropolis_chain(
    log_target: LogTarget,
    theta0,
    config: MetropolisConfig,
    *,
    seed: int | None = None,
    chain_id: int = 0,
    progress_callback: ProgressCallback | None = None,
    progress_every: int | None = None,
) -> ChainResult:
    """Run a single random-walk Metropolis chain."""

    theta = _as_vector(theta0, name="theta0")
    proposal_sigma = _proposal_sigma(config.proposal_sigma, theta.size)
    bounds = None if config.bounds is None else _bounds(config.bounds, theta.size)
    rng = np.random.default_rng(seed)

    current_log_prob = _safe_log_target(log_target, theta)
    if not np.isfinite(current_log_prob):
        raise ValueError("Initial theta has non-finite log probability.")
    initial_theta = theta.copy()
    initial_log_prob = current_log_prob

    adapter = _adapter(config, theta, proposal_sigma)

    stored_samples = []
    stored_log_prob = []
    accepted = np.zeros(config.n_steps, dtype=bool)

    for step in range(config.n_steps):
        global_jump = _is_global_jump(step, config.global_jump_interval)
        if adapter is not None and not global_jump:
            proposal = adapter.propose(theta, rng)
        else:
            proposal = _propose(theta, proposal_sigma, rng, bounds=bounds, global_jump=global_jump)
        if bounds is not None and not _within_bounds(proposal, bounds):
            proposed_log_prob = -np.inf
        else:
            proposed_log_prob = _safe_log_target(log_target, proposal)
        if _accept(current_log_prob, proposed_log_prob, rng):
            theta = proposal
            current_log_prob = proposed_log_prob
            accepted[step] = True
        if adapter is not None:
            adapter.observe(step, theta)

        if _store_step(step, config.burn_in, config.thin):
            stored_samples.append(theta.copy())
            stored_log_prob.append(current_log_prob)
        if _report_progress(step, config.n_steps, progress_every):
            if progress_callback is not None:
                progress_callback(
                    chain_id,
                    step + 1,
                    float(np.mean(accepted[: step + 1])),
                    float(current_log_prob),
                )

    return ChainResult(
        samples=np.asarray(stored_samples, dtype=float),
        log_prob=np.asarray(stored_log_prob, dtype=float),
        accepted=accepted,
        initial_theta=initial_theta,
        initial_log_prob=float(initial_log_prob),
        final_theta=theta.copy(),
        final_log_prob=float(current_log_prob),
        seed=seed,
        chain_id=chain_id,
        proposal_covariance=None if adapter is None else adapter.covariance()[None],
    )


def run_tempered_chain(
    split_target: SplitLogTarget | Sequence[SplitLogTarget],
    theta0,
    config: MetropolisConfig,
    *,
    seed: int | None = None,
    chain_id: int = 0,
    progress_callback: ProgressCallback | None = None,
    progress_every: int | None = None,
) -> ChainResult:
    """Run one parallel-tempering chain; only the ``T = 1`` replica is stored.

    ``split_target(theta)`` returns ``(log prior, log likelihood)``. Replica
    ``i`` proposes with ``proposal_sigma * sqrt(T_i)``: a likelihood tempered
    by ``1/T`` widens a Gaussian posterior by ``sqrt(T)`` (an assumed scaling;
    the step stays independent of the state, so detailed balance holds). All
    replicas start at ``theta0``. ``accepted`` and the acceptance rate refer to
    the cold replica; ``swap_acceptance`` lists the rate per adjacent pair.
    ``split_target`` may also be a sequence with one callable per temperature
    (all returning the same values): replica ``i`` then evaluates its
    proposals with entry ``i``, and the starting point is evaluated with the
    first.
    """
    if config.temperatures is None:
        raise ValueError("run_tempered_chain needs config.temperatures.")
    theta = _as_vector(theta0, name="theta0")
    proposal_sigma = _proposal_sigma(config.proposal_sigma, theta.size)
    bounds = None if config.bounds is None else _bounds(config.bounds, theta.size)
    rng = np.random.default_rng(seed)
    temperatures = np.asarray(config.temperatures)
    if callable(split_target):
        replica_targets = [split_target] * len(temperatures)
    else:
        replica_targets = list(split_target)
        if len(replica_targets) != len(temperatures):
            raise ValueError("Need one split target per temperature.")

    current = _safe_split_target(replica_targets[0], theta)
    if not np.isfinite(current[0] + current[1]):
        raise ValueError("Initial theta has non-finite log probability.")
    states = [theta.copy() for _ in temperatures]
    values = [current for _ in temperatures]
    sigmas = [proposal_sigma * np.sqrt(t) for t in temperatures]
    adapters = [_adapter(config, theta, sigma) for sigma in sigmas]
    n_pairs = len(temperatures) - 1
    attempts = np.zeros(n_pairs, dtype=int)
    accepts = np.zeros(n_pairs, dtype=int)

    stored_samples = []
    stored_log_prob = []
    accepted = np.zeros(config.n_steps, dtype=bool)

    for step in range(config.n_steps):
        global_jump = _is_global_jump(step, config.global_jump_interval)
        for i, temperature in enumerate(temperatures):
            if adapters[i] is not None and not global_jump:
                proposal = adapters[i].propose(states[i], rng)
            else:
                proposal = _propose(states[i], sigmas[i], rng, bounds=bounds, global_jump=global_jump)
            if bounds is not None and not _within_bounds(proposal, bounds):
                proposed = (-np.inf, -np.inf)
            else:
                proposed = _safe_split_target(replica_targets[i], proposal)
            log_alpha = (proposed[0] - values[i][0]) + (proposed[1] - values[i][1]) / temperature
            if np.isfinite(log_alpha) and (log_alpha >= 0 or np.log(rng.random()) < log_alpha):
                states[i], values[i] = proposal, proposed
                if i == 0:
                    accepted[step] = True
        if n_pairs and (step + 1) % config.swap_interval == 0:
            parity = (step // config.swap_interval) % 2
            for i in range(parity, n_pairs, 2):
                log_alpha = (1 / temperatures[i] - 1 / temperatures[i + 1]) * (
                    values[i + 1][1] - values[i][1]
                )
                attempts[i] += 1
                if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                    states[i], states[i + 1] = states[i + 1], states[i]
                    values[i], values[i + 1] = values[i + 1], values[i]
                    accepts[i] += 1
        for adapter, state in zip(adapters, states, strict=True):
            if adapter is not None:
                adapter.observe(step, state)

        if _store_step(step, config.burn_in, config.thin):
            stored_samples.append(states[0].copy())
            stored_log_prob.append(values[0][0] + values[0][1])
        if _report_progress(step, config.n_steps, progress_every) and progress_callback is not None:
            progress_callback(
                chain_id,
                step + 1,
                float(np.mean(accepted[: step + 1])),
                float(values[0][0] + values[0][1]),
            )

    return ChainResult(
        samples=np.asarray(stored_samples, dtype=float),
        log_prob=np.asarray(stored_log_prob, dtype=float),
        accepted=accepted,
        initial_theta=theta,
        initial_log_prob=float(current[0] + current[1]),
        final_theta=states[0].copy(),
        final_log_prob=float(values[0][0] + values[0][1]),
        seed=seed,
        chain_id=chain_id,
        swap_acceptance=accepts / np.maximum(attempts, 1),
        proposal_covariance=(
            None if config.adaptation is None else np.stack([a.covariance() for a in adapters])
        ),
    )


def run_metropolis(
    log_target: LogTarget,
    initial_thetas,
    config: MetropolisConfig,
    *,
    n_chains: int | None = None,
    seeds: Sequence[int] | int | None = None,
    executor: ExecutorKind = "serial",
    max_workers: int | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_every: int | None = None,
) -> MCMCResult:
    """Run one or more independent Metropolis chains.

    With ``config.temperatures`` each chain is a parallel-tempering ladder and
    ``log_target`` must return ``(log prior, log likelihood)``.
    """

    initial_thetas = _initial_thetas(initial_thetas, n_chains=n_chains)
    chain_seeds = _chain_seeds(seeds, len(initial_thetas))
    args = [
        (
            log_target,
            initial_thetas[i],
            config,
            chain_seeds[i],
            i,
            progress_callback,
            progress_every,
        )
        for i in range(len(initial_thetas))
    ]

    if executor == "serial" or len(args) == 1:
        chains = tuple(_run_chain_worker(arg) for arg in args)
    elif executor == "thread":
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            chains = tuple(pool.map(_run_chain_worker, args))
    elif executor == "process":
        if progress_callback is not None:
            raise ValueError("progress_callback is not supported with process executor.")
        args = [
            (log_target, initial_thetas[i], config, chain_seeds[i], i, None, None)
            for i in range(len(initial_thetas))
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            chains = tuple(pool.map(_run_chain_worker, args))
    else:
        raise ValueError(f"Unknown executor: {executor}")

    return MCMCResult(
        samples=np.stack([chain.samples for chain in chains]),
        log_prob=np.stack([chain.log_prob for chain in chains]),
        accepted=np.stack([chain.accepted for chain in chains]),
        chains=chains,
        config=config,
    )


def draw_valid_initial_thetas(
    log_target: LogTarget,
    bounds: tuple[Sequence[float], Sequence[float]],
    *,
    n_chains: int,
    seed: int | None = None,
    max_attempts: int = 10000,
) -> np.ndarray:
    """Draw valid starting points uniformly from a bounded prior space."""

    if n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    lower, upper = _bounds(bounds, ndim=None)
    rng = np.random.default_rng(seed)
    starts = []
    attempts = 0
    while len(starts) < n_chains and attempts < max_attempts:
        attempts += 1
        theta = rng.uniform(lower, upper)
        if np.isfinite(_safe_log_target(log_target, theta)):
            starts.append(theta)
    if len(starts) < n_chains:
        raise RuntimeError("Could not draw enough valid initial thetas.")
    return np.asarray(starts, dtype=float)


def update_running_moments(
    count: int,
    mean: np.ndarray,
    scatter: np.ndarray,
    x: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Add one point to a running mean and scatter matrix (Welford's recursion).

    With ``n = count + 1``: ``mean_n = mean + (x - mean) / n`` and
    ``scatter_n = scatter + (x - mean) (x - mean_n)^T``, so that
    ``scatter_n = sum_k (x_k - mean_n)(x_k - mean_n)^T`` over all ``n`` points.
    Start from ``count = 0`` and zero arrays. Returns new arrays.
    """
    count = count + 1
    delta = x - mean
    mean = mean + delta / count
    scatter = scatter + np.outer(delta, x - mean)
    return count, mean, scatter


def running_covariance(count: int, scatter: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance ``scatter / (count - 1)``, symmetrized."""
    if count < 2:
        raise ValueError("Need at least two points for a covariance.")
    covariance = scatter / (count - 1)
    return 0.5 * (covariance + covariance.T)


def adaptive_proposal_covariance(
    sample_covariance: np.ndarray,
    reference_sigma: np.ndarray,
    regularization_fraction: float,
) -> np.ndarray:
    """``(2.38**2 / d) * (sample_covariance + eps * diag(reference_sigma**2))``.

    Haario, Saksman & Tamminen (2001) with the regularization ``eps`` taken
    relative to each parameter's starting variance (module docstring).
    """
    reference_sigma = np.asarray(reference_sigma, dtype=float)
    d = reference_sigma.size
    scale = OPTIMAL_RW_SCALE**2 / d
    return scale * (np.asarray(sample_covariance, dtype=float) + regularization_fraction * np.diag(reference_sigma**2))


def recent_history_covariance(history: np.ndarray, discard_early_history_fraction: float) -> np.ndarray:
    """Sample covariance of the most recent states ``x_k, ..., x_n`` of ``history``.

    ``history`` holds the ``n + 1`` states ``x_0, ..., x_n`` (rows);
    ``k = floor(f * (n + 1))`` with ``f = discard_early_history_fraction``.
    Two-pass unbiased covariance (``numpy.cov``), symmetrized.
    """
    history = np.asarray(history, dtype=float)
    f = float(discard_early_history_fraction)
    if not 0.0 <= f < 1.0:
        raise ValueError("discard_early_history_fraction must be in [0, 1).")
    window = history[int(np.floor(f * len(history))) :]
    if len(window) < 2:
        raise ValueError("Need at least two points for a covariance.")
    covariance = np.atleast_2d(np.cov(window.T))
    return 0.5 * (covariance + covariance.T)


# Assumed thresholds for `adaptation_warnings`: at least as many adapted
# burn-in steps as diagonal ones, and at least ADAPTATION_STATES_PER_PARAMETER
# states per parameter in the window of the last update. Chain states are
# strongly autocorrelated (ESS often 1-10 % of the states), so 100 d states
# give only some d to 10 d effectively independent ones for d (d + 1) / 2
# covariance entries.
ADAPTATION_MIN_ADAPTED_TO_DIAGONAL_RATIO = 1.0
ADAPTATION_STATES_PER_PARAMETER = 100


def adaptation_warnings(adaptation: AdaptationConfig | None, *, burn_in: int, n_parameters: int) -> list[str]:
    """Warnings when ``burn_in`` is short for the adaptation it has to hold.

    Thresholds are assumed (`ADAPTATION_MIN_ADAPTED_TO_DIAGONAL_RATIO`,
    `ADAPTATION_STATES_PER_PARAMETER`); no warning means only that these
    rough checks passed.
    """
    if adaptation is None:
        return []
    messages = []
    t0 = adaptation.initial_diagonal_steps
    adapted = burn_in - t0
    if adapted < ADAPTATION_MIN_ADAPTED_TO_DIAGONAL_RATIO * t0:
        messages.append(
            f"burn_in ({burn_in}) leaves {adapted} adapted steps after initial_diagonal_steps ({t0}); "
            "fewer than the diagonal steps (assumed threshold), so the frozen proposal rests on few "
            "updates. Lengthen burn_in."
        )
    last = adaptation.update_steps(burn_in)[-1]
    window = adaptation.history_window_size(last)
    needed = ADAPTATION_STATES_PER_PARAMETER * n_parameters
    if window < needed:
        messages.append(
            f"the last proposal update (after step {last}) uses {window} states for {n_parameters} "
            f"parameters, fewer than {needed} ({ADAPTATION_STATES_PER_PARAMETER} per parameter, assumed "
            "threshold); the frozen covariance may be noisy. Lengthen burn_in."
        )
    return messages


class _ProposalAdapter:
    """Running moments and the current proposal of one chain (or replica)."""

    def __init__(self, adaptation: AdaptationConfig, burn_in: int, theta0: np.ndarray, sigma: np.ndarray):
        if adaptation.initial_diagonal_steps <= theta0.size:
            raise ValueError(
                f"adaptation.initial_diagonal_steps ({adaptation.initial_diagonal_steps}) must exceed "
                f"the number of parameters ({theta0.size}) to estimate a covariance."
            )
        first_window = adaptation.history_window_size(adaptation.initial_diagonal_steps)
        if first_window <= theta0.size:
            raise ValueError(
                f"The first covariance window ({first_window} states after leaving out the earliest "
                f"fraction {adaptation.discard_early_history_fraction}) must exceed the number of "
                f"parameters ({theta0.size}); raise initial_diagonal_steps."
            )
        self._settings = adaptation
        self._sigma = sigma
        self._updates = frozenset(adaptation.update_steps(burn_in))
        self._last_update = max(self._updates)
        self._count, self._mean, self._scatter = update_running_moments(
            0, np.zeros(theta0.size), np.zeros((theta0.size, theta0.size)), theta0
        )
        # The windowed estimate needs the states themselves; f = 0 keeps the
        # running moments only, exactly as before the option existed.
        self._history: np.ndarray | None = None
        if adaptation.discard_early_history_fraction > 0:
            self._history = np.empty((self._last_update + 1, theta0.size))
            self._history[0] = theta0
        self._cholesky: np.ndarray | None = None
        self._covariance = np.diag(sigma**2)

    def propose(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self._cholesky is None:  # the same draw as a fixed diagonal proposal
            return theta + rng.normal(loc=0.0, scale=self._sigma, size=theta.shape)
        return theta + self._cholesky @ rng.standard_normal(theta.size)

    def observe(self, step: int, theta: np.ndarray) -> None:
        """Record the state after ``step`` (0-based); update the proposal on schedule."""
        if step >= self._last_update:  # frozen: nothing more to record
            return
        if self._history is None:
            self._count, self._mean, self._scatter = update_running_moments(
                self._count, self._mean, self._scatter, theta
            )
        else:
            self._history[step + 1] = theta
        if step + 1 in self._updates:
            if self._history is None:
                sample_covariance = running_covariance(self._count, self._scatter)
            else:
                sample_covariance = recent_history_covariance(
                    self._history[: step + 2], self._settings.discard_early_history_fraction
                )
            self._covariance = adaptive_proposal_covariance(
                sample_covariance,
                self._sigma,
                self._settings.regularization_fraction,
            )
            self._cholesky = np.linalg.cholesky(self._covariance)

    def covariance(self) -> np.ndarray:
        return self._covariance.copy()


def _adapter(config: MetropolisConfig, theta0: np.ndarray, sigma: np.ndarray) -> _ProposalAdapter | None:
    if config.adaptation is None:
        return None
    return _ProposalAdapter(config.adaptation, config.burn_in, theta0, sigma)


def _run_chain_worker(args) -> ChainResult:
    log_target, theta0, config, seed, chain_id, progress_callback, progress_every = args
    runner = run_metropolis_chain if config.temperatures is None else run_tempered_chain
    evaluators = None
    if isinstance(log_target, ChainLocalTarget):
        n_replicas = 1 if config.temperatures is None else len(config.temperatures)
        evaluators = [log_target.chain_evaluator() for _ in range(n_replicas)]
        log_target = evaluators[0] if config.temperatures is None else evaluators
    result = runner(
        log_target,
        theta0,
        config,
        seed=seed,
        chain_id=chain_id,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )
    if evaluators is None:
        return result
    return replace(result, target_reports=tuple(e.report() for e in evaluators))


def _propose(
    theta: np.ndarray,
    proposal_sigma: np.ndarray,
    rng: np.random.Generator,
    *,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    global_jump: bool,
) -> np.ndarray:
    if global_jump:
        if bounds is None:
            raise ValueError("bounds are required for global jump proposals.")
        lower, upper = bounds
        return rng.uniform(lower, upper)
    return theta + rng.normal(loc=0.0, scale=proposal_sigma, size=theta.shape)


def _accept(
    current_log_prob: float,
    proposed_log_prob: float,
    rng: np.random.Generator,
) -> bool:
    if not np.isfinite(proposed_log_prob):
        return False
    if proposed_log_prob >= current_log_prob:
        return True
    return np.log(rng.random()) < proposed_log_prob - current_log_prob


def _safe_log_target(log_target: LogTarget, theta: np.ndarray) -> float:
    try:
        value = float(log_target(theta))
    except (FloatingPointError, OverflowError):
        return -np.inf
    if np.isnan(value):
        return -np.inf
    return value


def _safe_split_target(split_target: SplitLogTarget, theta: np.ndarray) -> tuple[float, float]:
    try:
        log_prior, log_likelihood = split_target(theta)
        log_prior, log_likelihood = float(log_prior), float(log_likelihood)
    except (FloatingPointError, OverflowError):
        return -np.inf, -np.inf
    if not np.isfinite(log_prior):
        return -np.inf, -np.inf
    if np.isnan(log_likelihood):
        return log_prior, -np.inf
    return log_prior, log_likelihood


def _is_global_jump(step: int, interval: int | None) -> bool:
    return interval is not None and (step + 1) % interval == 0


def _store_step(step: int, burn_in: int, thin: int) -> bool:
    return step >= burn_in and (step - burn_in) % thin == 0


def _report_progress(step: int, n_steps: int, progress_every: int | None) -> bool:
    if progress_every is None:
        return False
    if progress_every <= 0:
        raise ValueError("progress_every must be positive when provided.")
    return (step + 1) % progress_every == 0 or (step + 1) == n_steps


def _as_vector(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array.copy()


def _initial_thetas(initial_thetas, *, n_chains: int | None) -> np.ndarray:
    array = np.asarray(initial_thetas, dtype=float)
    if array.ndim == 1:
        count = 1 if n_chains is None else n_chains
        if count <= 0:
            raise ValueError("n_chains must be positive.")
        array = np.tile(array, (count, 1))
    elif array.ndim == 2:
        if n_chains is not None and n_chains != len(array):
            raise ValueError("n_chains does not match the number of initial_thetas.")
    else:
        raise ValueError("initial_thetas must be one- or two-dimensional.")
    if not np.all(np.isfinite(array)):
        raise ValueError("initial_thetas must contain only finite values.")
    return array


def _chain_seeds(seeds: Sequence[int] | int | None, n_chains: int) -> list[int | None]:
    if seeds is None:
        return [None] * n_chains
    if isinstance(seeds, int):
        seed_sequence = np.random.SeedSequence(seeds)
        return [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in seed_sequence.spawn(n_chains)
        ]
    if len(seeds) != n_chains:
        raise ValueError("Number of seeds must match number of chains.")
    return [int(seed) for seed in seeds]


def _proposal_sigma(proposal_sigma, ndim: int) -> np.ndarray:
    sigma = np.asarray(proposal_sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(ndim, float(sigma), dtype=float)
    if sigma.shape != (ndim,):
        raise ValueError(f"proposal_sigma must be scalar or have shape ({ndim},).")
    if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("proposal_sigma must contain positive finite values.")
    return sigma


def _bounds(bounds, ndim: int | None) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        raise ValueError("bounds are required.")
    lower = np.asarray(bounds[0], dtype=float)
    upper = np.asarray(bounds[1], dtype=float)
    if lower.shape != upper.shape:
        raise ValueError("Lower and upper bounds must have the same shape.")
    if ndim is not None and lower.shape != (ndim,):
        raise ValueError(f"bounds must have shape ({ndim},).")
    if np.any(upper <= lower) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Bounds must be finite and strictly increasing.")
    return lower, upper


def _within_bounds(theta: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    lower, upper = bounds
    return bool(np.all((theta >= lower) & (theta <= upper)))
