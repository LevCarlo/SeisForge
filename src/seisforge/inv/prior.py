"""Geophysical prior target for model-space sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.constraints import ConstraintEvaluation, ConstraintSuite
from seisforge.inv.model import DiscretizationConfig, LayeredModel, ParameterizedVsModel
from seisforge.inv.parameterization import ModelParameterization


@dataclass(frozen=True)
class GeophysicalPriorResult:
    """Detailed prior evaluation for one parameter vector."""

    log_prior: float
    parameter_log_prior: float
    physical_log_prior: float
    success: bool
    model: ParameterizedVsModel | None = None
    layered_model: LayeredModel | None = None
    constraint_evaluation: ConstraintEvaluation | None = None
    error: str | None = None


@dataclass(frozen=True)
class GeophysicalPrior:
    """Hard-prior support for geophysical MCMC.

    A valid model receives log prior 0. Any parameter-bound or physical
    constraint violation receives `-inf`, so a prior-only sampler explores the
    allowed model space without using data likelihood.
    """

    parameterization: ModelParameterization
    constraints: ConstraintSuite = ConstraintSuite()
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
                success=False,
                model=model,
                layered_model=layered,
                constraint_evaluation=constraint_evaluation,
                error="physical_constraints",
            )

        return GeophysicalPriorResult(
            log_prior=parameter_log_prior + physical_log_prior,
            parameter_log_prior=parameter_log_prior,
            physical_log_prior=physical_log_prior,
            success=True,
            model=model,
            layered_model=layered,
            constraint_evaluation=constraint_evaluation,
        )

    def log_prior(self, theta) -> float:
        return self.evaluate(theta).log_prior


def _failed(error: str, parameter_log_prior: float = -np.inf) -> GeophysicalPriorResult:
    return GeophysicalPriorResult(
        log_prior=-np.inf,
        parameter_log_prior=parameter_log_prior,
        physical_log_prior=-np.inf,
        success=False,
        error=error,
    )
