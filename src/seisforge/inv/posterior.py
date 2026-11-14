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
        result = self.evaluate(theta)
        if not np.isfinite(result.log_prior):
            return -np.inf, -np.inf
        return result.log_prior, result.log_likelihood
