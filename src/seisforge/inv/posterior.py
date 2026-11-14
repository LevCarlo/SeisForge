"""Posterior targets for joint geophysical inversion.

Forward-failure counts: `CountingTarget` wraps a `JointInversionTarget` for the
sampler and gives every chain replica its own `ForwardFailureCounter` (see
`seisforge.inv.samplers.ChainLocalTarget`). A *forward attempt* is an
evaluation whose parameters lie inside the hard prior support, so the forward
solvers run; a *forward failure* is an attempt whose likelihood failed (the
solver raised `ForwardError`, which the likelihood turns into ``-inf``, so the
proposal is rejected). Points outside the prior never reach the solver and are
not counted. Each chain counts its starting point and every proposal it
evaluates; with parallel tempering the counts are kept per replica, and
`forward_failure_summary` reports both the chain total over all replicas and
the cold (``T = 1``) replica alone. The counter returns exactly the target's
own values, so counting changes no random number or log probability.
Evaluations made outside the sampler (starting-point draws, posterior
predictive ensembles) call the target directly and are not counted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from seisforge.inv.likelihood import (
    DispersionObservation,
    JointLikelihoodResult,
    RayleighHVObservation,
    evaluate_joint_likelihood,
)
from seisforge.inv.model_config import parameter_references
from seisforge.inv.prior import GeophysicalPrior, GeophysicalPriorResult


@dataclass(frozen=True)
class JointPosteriorResult:
    """Detailed target evaluation for one parameter vector."""

    log_posterior: float
    log_prior: float
    log_likelihood: float
    success: bool
    prior: GeophysicalPriorResult
    likelihood: JointLikelihoodResult | None = None
    error: str | None = None
    noise_scales: dict[str, float] | None = None

    @property
    def chi2(self) -> float:
        return np.inf if self.likelihood is None else self.likelihood.chi2

    @property
    def raw_chi2(self) -> float:
        return np.inf if self.likelihood is None else self.likelihood.raw_chi2

    @property
    def n_data(self) -> int:
        return 0 if self.likelihood is None else self.likelihood.n_data


@dataclass(frozen=True)
class JointInversionTarget:
    """Log-posterior target for Rayleigh phase velocity and HV inversion.

    An observation's ``noise_parameter`` names an inversion parameter holding
    ``log10 h``, the factor on that observation's standard deviations (see
    `seisforge.inv.likelihood`). Such a parameter must not appear in the model,
    and the observation's weight must stay 1: the sampled scale replaces it.
    """

    prior: GeophysicalPrior
    dispersion: DispersionObservation | None = None
    hv: RayleighHVObservation | None = None
    dispersion_weight: float = 1.0
    hv_weight: float = 1.0

    def __post_init__(self):
        if self.dispersion is None and self.hv is None:
            raise ValueError("At least one observation must be provided.")
        if self.dispersion_weight < 0 or self.hv_weight < 0:
            raise ValueError("Likelihood weights must be non-negative.")
        names = self.prior.parameterization.names
        model_parameters = parameter_references(self.prior.parameterization.model_config)
        indices = {}
        for label, observation, weight in (
            ("dispersion", self.dispersion, self.dispersion_weight),
            ("hv", self.hv, self.hv_weight),
        ):
            name = None if observation is None else observation.noise_parameter
            if name is None:
                continue
            if name not in names:
                raise ValueError(f"{label} noise_parameter {name!r} is not an inversion parameter.")
            if name in model_parameters:
                raise ValueError(f"{label} noise_parameter {name!r} is also used by the model.")
            if weight != 1.0:
                raise ValueError(
                    f"{label} has a sampled noise scale; leave its likelihood weight at 1."
                )
            indices[label] = names.index(name)
        object.__setattr__(self, "_noise_indices", indices)

    def noise_scales(self, theta) -> dict[str, float]:
        """Noise scale ``h = 10**theta[noise_parameter]`` per observation that has one."""
        theta = np.asarray(theta, dtype=float)
        return {label: float(10.0 ** theta[i]) for label, i in self._noise_indices.items()}

    def evaluate(self, theta) -> JointPosteriorResult:
        prior_result = self.prior.evaluate(theta)
        if not prior_result.success:
            return JointPosteriorResult(
                log_posterior=-np.inf,
                log_prior=prior_result.log_prior,
                log_likelihood=-np.inf,
                success=False,
                prior=prior_result,
                error=f"prior: {prior_result.error}",
            )

        scales = self.noise_scales(theta)
        likelihood_result = evaluate_joint_likelihood(
            prior_result.layered_model,
            dispersion=self.dispersion,
            hv=self.hv,
            dispersion_weight=self.dispersion_weight,
            hv_weight=self.hv_weight,
            dispersion_noise_scale=scales.get("dispersion", 1.0),
            hv_noise_scale=scales.get("hv", 1.0),
        )
        if not likelihood_result.success:
            return JointPosteriorResult(
                log_posterior=-np.inf,
                log_prior=prior_result.log_prior,
                log_likelihood=likelihood_result.loglike,
                success=False,
                prior=prior_result,
                likelihood=likelihood_result,
                error=f"likelihood: {likelihood_result.error}",
                noise_scales=scales,
            )

        log_posterior = prior_result.log_prior + likelihood_result.loglike
        return JointPosteriorResult(
            log_posterior=float(log_posterior),
            log_prior=prior_result.log_prior,
            log_likelihood=likelihood_result.loglike,
            success=True,
            prior=prior_result,
            likelihood=likelihood_result,
            noise_scales=scales,
        )

    def log_posterior(self, theta) -> float:
        return self.evaluate(theta).log_posterior

    def log_prior_and_likelihood(self, theta) -> tuple[float, float]:
        """``(log prior, log likelihood)``; the likelihood is -inf outside the prior."""
        return _target_value(self.evaluate(theta), split=True)


# Assumed: a chain whose forward solver fails on more than 1% of the proposals
# inside the prior samples a visibly truncated prior; above this rate the
# driver logs a warning. Not taken from a published source.
FORWARD_FAILURE_WARNING_RATE = 0.01


@dataclass(frozen=True)
class ForwardFailureReport:
    """Forward attempts and failures of one chain replica."""

    n_forward_attempts: int
    n_forward_failures: int
    error_counts: dict[str, int] = field(default_factory=dict)


class ForwardFailureCounter:
    """Chain-local evaluator that counts forward failures of a target.

    Returns ``target.log_posterior(theta)``, or ``target.log_prior_and_likelihood(theta)``
    when ``split`` is true, from a single ``target.evaluate`` call. One instance
    serves one chain replica, so it needs no lock.
    """

    def __init__(self, target: JointInversionTarget, *, split: bool):
        self._target = target
        self._split = split
        self._attempts = 0
        self._errors: Counter[str] = Counter()

    def __call__(self, theta):
        result = self._target.evaluate(theta)
        if result.prior.success:
            self._attempts += 1
            if not result.success:
                self._errors[str(result.error)] += 1
        return _target_value(result, split=self._split)

    def report(self) -> ForwardFailureReport:
        return ForwardFailureReport(
            n_forward_attempts=self._attempts,
            n_forward_failures=int(sum(self._errors.values())),
            error_counts=dict(self._errors),
        )


@dataclass(frozen=True)
class CountingTarget:
    """Sampler target that counts forward failures per chain replica.

    Calling it directly returns the same value as ``target.log_posterior``
    (or ``log_prior_and_likelihood`` with ``split``) without counting.
    """

    target: JointInversionTarget
    split: bool = False

    def __call__(self, theta):
        return _target_value(self.target.evaluate(theta), split=self.split)

    def chain_evaluator(self) -> ForwardFailureCounter:
        return ForwardFailureCounter(self.target, split=self.split)


@dataclass(frozen=True)
class ForwardFailureSummary:
    """Per-chain forward failures; ``*_cold`` count the ``T = 1`` replica only.

    Without parallel tempering a chain has one replica and the cold counts equal
    the totals.
    """

    attempts: np.ndarray
    failures: np.ndarray
    attempts_cold: np.ndarray
    failures_cold: np.ndarray
    error_counts: dict[str, int]
    n_replicas: int

    @property
    def failure_rate(self) -> np.ndarray:
        return _rate(self.failures, self.attempts)

    @property
    def failure_rate_cold(self) -> np.ndarray:
        return _rate(self.failures_cold, self.attempts_cold)

    def most_common_errors(self, n: int = 2) -> list[tuple[str, int]]:
        return Counter(self.error_counts).most_common(n)

    def warnings(self, threshold: float = FORWARD_FAILURE_WARNING_RATE) -> list[str]:
        """One message per chain whose total failure rate exceeds ``threshold``."""
        common = "; ".join(f"{message!r} x{count}" for message, count in self.most_common_errors(2))
        return [
            f"chain {chain}: forward solver failed on {int(self.failures[chain])} of "
            f"{int(self.attempts[chain])} evaluations inside the prior "
            f"(rate {rate:.3g} > {threshold:g}); most frequent errors: {common}"
            for chain, rate in enumerate(self.failure_rate)
            if rate > threshold
        ]


def forward_failure_summary(chains: Sequence) -> ForwardFailureSummary | None:
    """Sum the `ForwardFailureReport` s of sampler chains (None if not counted)."""
    reports = [getattr(chain, "target_reports", None) for chain in chains]
    if not reports or any(r is None for r in reports):
        return None
    errors: Counter[str] = Counter()
    for chain_reports in reports:
        for report in chain_reports:
            errors.update(report.error_counts)
    return ForwardFailureSummary(
        attempts=np.array([sum(r.n_forward_attempts for r in c) for c in reports], dtype=int),
        failures=np.array([sum(r.n_forward_failures for r in c) for c in reports], dtype=int),
        attempts_cold=np.array([c[0].n_forward_attempts for c in reports], dtype=int),
        failures_cold=np.array([c[0].n_forward_failures for c in reports], dtype=int),
        error_counts=dict(errors),
        n_replicas=len(reports[0]),
    )


def _target_value(result: JointPosteriorResult, *, split: bool):
    if not split:
        return result.log_posterior
    if not np.isfinite(result.log_prior):
        return -np.inf, -np.inf
    return result.log_prior, result.log_likelihood


def _rate(failures: np.ndarray, attempts: np.ndarray) -> np.ndarray:
    return np.where(attempts > 0, failures / np.maximum(attempts, 1), 0.0)
