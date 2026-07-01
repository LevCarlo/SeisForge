"""Posterior targets for joint geophysical inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.likelihood import (
    DispersionObservation,
    JointLikelihoodResult,
    RayleighHVObservation,
    evaluate_joint_likelihood,
)
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
    """Log-posterior target for Rayleigh phase velocity and HV inversion."""

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

        likelihood_result = evaluate_joint_likelihood(
            prior_result.layered_model,
            dispersion=self.dispersion,
            hv=self.hv,
            dispersion_weight=self.dispersion_weight,
            hv_weight=self.hv_weight,
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
            )

        log_posterior = prior_result.log_prior + likelihood_result.loglike
        return JointPosteriorResult(
            log_posterior=float(log_posterior),
            log_prior=prior_result.log_prior,
            log_likelihood=likelihood_result.loglike,
            success=True,
            prior=prior_result,
            likelihood=likelihood_result,
        )

    def log_posterior(self, theta) -> float:
        return self.evaluate(theta).log_posterior
