"""Geophysical prior target for model-space sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.constraints import ConstraintEvaluation, ConstraintSuite
from seisforge.inv.model import DiscretizationConfig, LayeredModel, ParameterizedVsModel
from seisforge.inv.parameterization import ModelParameterization
from seisforge.inv.soft_priors import SoftPriorSuite, SoftPriorSuiteEvaluation


@dataclass(frozen=True)
class GeophysicalPriorResult:
    """Detailed prior evaluation for one parameter vector."""

    log_prior: float
    parameter_log_prior: float
    physical_log_prior: float
    soft_log_prior: float
    success: bool
    model: ParameterizedVsModel | None = None
    layered_model: LayeredModel | None = None
    constraint_evaluation: ConstraintEvaluation | None = None
    soft_prior_evaluation: SoftPriorSuiteEvaluation | None = None
    error: str | None = None


@dataclass(frozen=True)
class PriorSampleDiagnostics:
    """Prior decomposition evaluated for an array of retained MCMC samples."""

    log_prior: np.ndarray
    parameter_log_prior: np.ndarray
    physical_log_prior: np.ndarray
    soft_log_prior: np.ndarray
    soft_diagnostics: dict[str, np.ndarray]


@dataclass(frozen=True)
class GeophysicalPrior:
    """Hard-support and soft model-space priors for geophysical MCMC.

    Parameter bounds and physical constraints define hard support. Soft model
    priors add finite log-probability terms after a model is inside that
    support, so a prior-only sampler includes both kinds of prior information.
    """

    parameterization: ModelParameterization
    constraints: ConstraintSuite = ConstraintSuite()
    soft_priors: SoftPriorSuite = SoftPriorSuite()
    discretization: DiscretizationConfig | None = None

    def evaluate(self, theta) -> GeophysicalPriorResult:
        try:
            parameter_log_prior = self.parameterization.log_parameter_prior(theta)
        except ValueError as exc:
            return _failed(f"parameter_prior: {exc}")
        if not np.isfinite(parameter_log_prior):
            return _failed("parameter_prior")

        try:
            model = self.parameterization.vector_to_model(theta)
            layered = model.to_layered_model(self.discretization)
        except ValueError as exc:
            return _failed(f"model_mapping: {exc}", parameter_log_prior=parameter_log_prior)

        constraint_evaluation = self.constraints.evaluate(model, layered)
        physical_log_prior = constraint_evaluation.log_prior()
        if not np.isfinite(physical_log_prior):
            return GeophysicalPriorResult(
                log_prior=-np.inf,
                parameter_log_prior=parameter_log_prior,
                physical_log_prior=physical_log_prior,
                soft_log_prior=0.0,
                success=False,
                model=model,
                layered_model=layered,
                constraint_evaluation=constraint_evaluation,
                error="physical_constraints",
            )

        try:
            soft_prior_evaluation = self.soft_priors.evaluate(model)
        except ValueError as exc:
            return GeophysicalPriorResult(
                log_prior=-np.inf,
                parameter_log_prior=parameter_log_prior,
                physical_log_prior=physical_log_prior,
                soft_log_prior=-np.inf,
                success=False,
                model=model,
                layered_model=layered,
                constraint_evaluation=constraint_evaluation,
                error=f"soft_priors: {exc}",
            )

        return GeophysicalPriorResult(
            log_prior=(
                parameter_log_prior
                + physical_log_prior
                + soft_prior_evaluation.log_prior
            ),
            parameter_log_prior=parameter_log_prior,
            physical_log_prior=physical_log_prior,
            soft_log_prior=soft_prior_evaluation.log_prior,
            success=True,
            model=model,
            layered_model=layered,
            constraint_evaluation=constraint_evaluation,
            soft_prior_evaluation=soft_prior_evaluation,
        )

    def log_prior(self, theta) -> float:
        return self.evaluate(theta).log_prior


def evaluate_prior_samples(samples, prior: GeophysicalPrior) -> PriorSampleDiagnostics:
    """Evaluate the prior decomposition for retained sample arrays.

    This deliberately evaluates only model-space terms. Posterior likelihood
    values are recovered exactly from ``log_prob - log_prior`` and therefore do
    not require a second expensive forward-model pass.
    """

    samples = np.asarray(samples, dtype=float)
    if samples.ndim < 2:
        raise ValueError("samples must have at least sample and parameter dimensions.")
    sample_shape = samples.shape[:-1]
    details = [prior.evaluate(theta) for theta in samples.reshape((-1, samples.shape[-1]))]
    if any(not detail.success for detail in details):
        raise ValueError("Retained MCMC samples must satisfy the geophysical prior.")

    def values(attribute: str) -> np.ndarray:
        return np.asarray([getattr(detail, attribute) for detail in details], dtype=float).reshape(
            sample_shape
        )

    diagnostic_names = sorted(
        {
            name
            for detail in details
            for name in detail.soft_prior_evaluation.diagnostics
        }
    )
    soft_diagnostics = {
        name: np.asarray(
            [detail.soft_prior_evaluation.diagnostics.get(name, np.nan) for detail in details],
            dtype=float,
        ).reshape(sample_shape)
        for name in diagnostic_names
    }
    return PriorSampleDiagnostics(
        log_prior=values("log_prior"),
        parameter_log_prior=values("parameter_log_prior"),
        physical_log_prior=values("physical_log_prior"),
        soft_log_prior=values("soft_log_prior"),
        soft_diagnostics=soft_diagnostics,
    )


def _failed(error: str, parameter_log_prior: float = -np.inf) -> GeophysicalPriorResult:
    return GeophysicalPriorResult(
        log_prior=-np.inf,
        parameter_log_prior=parameter_log_prior,
        physical_log_prior=-np.inf,
        soft_log_prior=-np.inf,
        success=False,
        error=error,
    )
