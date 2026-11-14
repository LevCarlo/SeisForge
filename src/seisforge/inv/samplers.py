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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np


LogTarget = Callable[[np.ndarray], float]
SplitLogTarget = Callable[[np.ndarray], tuple[float, float]]
ProgressCallback = Callable[[int, int, float, float], None]
ExecutorKind = Literal["serial", "process", "thread"]


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

    @property
    def n_steps(self) -> int:
        return len(self.accepted)

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

    stored_samples = []
    stored_log_prob = []
    accepted = np.zeros(config.n_steps, dtype=bool)

    for step in range(config.n_steps):
        proposal = _propose(
            theta,
            proposal_sigma,
            rng,
            bounds=bounds,
            global_jump=_is_global_jump(step, config.global_jump_interval),
        )
        if bounds is not None and not _within_bounds(proposal, bounds):
            proposed_log_prob = -np.inf
        else:
            proposed_log_prob = _safe_log_target(log_target, proposal)
        if _accept(current_log_prob, proposed_log_prob, rng):
            theta = proposal
            current_log_prob = proposed_log_prob
            accepted[step] = True

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
    )


def run_tempered_chain(
    split_target: SplitLogTarget,
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
    """
    if config.temperatures is None:
        raise ValueError("run_tempered_chain needs config.temperatures.")
    theta = _as_vector(theta0, name="theta0")
    proposal_sigma = _proposal_sigma(config.proposal_sigma, theta.size)
    bounds = None if config.bounds is None else _bounds(config.bounds, theta.size)
    rng = np.random.default_rng(seed)
    temperatures = np.asarray(config.temperatures)

    current = _safe_split_target(split_target, theta)
    if not np.isfinite(current[0] + current[1]):
        raise ValueError("Initial theta has non-finite log probability.")
    states = [theta.copy() for _ in temperatures]
    values = [current for _ in temperatures]
    sigmas = [proposal_sigma * np.sqrt(t) for t in temperatures]
    n_pairs = len(temperatures) - 1
    attempts = np.zeros(n_pairs, dtype=int)
    accepts = np.zeros(n_pairs, dtype=int)

    stored_samples = []
    stored_log_prob = []
    accepted = np.zeros(config.n_steps, dtype=bool)

    for step in range(config.n_steps):
        global_jump = _is_global_jump(step, config.global_jump_interval)
        for i, temperature in enumerate(temperatures):
            proposal = _propose(states[i], sigmas[i], rng, bounds=bounds, global_jump=global_jump)
            if bounds is not None and not _within_bounds(proposal, bounds):
                proposed = (-np.inf, -np.inf)
            else:
                proposed = _safe_split_target(split_target, proposal)
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


def _run_chain_worker(args) -> ChainResult:
    log_target, theta0, config, seed, chain_id, progress_callback, progress_every = args
    runner = run_metropolis_chain if config.temperatures is None else run_tempered_chain
    return runner(
        log_target,
        theta0,
        config,
        seed=seed,
        chain_id=chain_id,
        progress_callback=progress_callback,
        progress_every=progress_every,
    )


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
    except (FloatingPointError, OverflowError, ValueError):
        return -np.inf
    if np.isnan(value):
        return -np.inf
    return value


def _safe_split_target(split_target: SplitLogTarget, theta: np.ndarray) -> tuple[float, float]:
    try:
        log_prior, log_likelihood = split_target(theta)
        log_prior, log_likelihood = float(log_prior), float(log_likelihood)
    except (FloatingPointError, OverflowError, ValueError):
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
