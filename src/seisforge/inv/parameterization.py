"""Parameter-vector mapping for geophysical inversion models.

The sampler should only see a numerical vector. This module owns the explicit
mapping between that vector and a `ParameterizedVsModel`, keeping proposal
mechanics separate from the geophysical prior support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seisforge.inv.config_schema import prepare_inv_config
from seisforge.inv.model import ParameterizedVsModel, ScalingConfig
from seisforge.inv.model_config import (
    parameter_references,
    parameterized_model_from_config,
    scaling_from_config,
    validate_model_layout,
)


@dataclass(frozen=True)
class UniformPrior:
    """Uniform hard-bounded parameter prior."""

    lower: float
    upper: float

    def __post_init__(self):
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("Uniform prior bounds must be finite.")
        if self.upper <= self.lower:
            raise ValueError("Uniform prior upper bound must be greater than lower bound.")

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def logpdf(self, value: float) -> float:
        return 0.0 if self.contains(value) else -np.inf


@dataclass(frozen=True)
class NormalPrior:
    """Gaussian prior truncated to ``[lower, upper]`` (unnormalized log density).

    Typical use: an interface depth with an independent estimate, e.g. Moho
    depth from H-k stacking as ``mean +/- sigma``.
    """

    mean: float
    sigma: float
    lower: float
    upper: float

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Normal prior mean must be finite and sigma positive.")
        UniformPrior(self.lower, self.upper)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def logpdf(self, value: float) -> float:
        if not self.contains(value):
            return -np.inf
        return float(-0.5 * ((value - self.mean) / self.sigma) ** 2)


@dataclass(frozen=True)
class HalfNormalPrior:
    """Half-normal prior on ``[lower, upper]`` peaking at ``lower`` (default 0).

    Shrinks a non-negative quantity toward ``lower`` unless the data demand
    more; used for interface transition widths so that a hard step is the
    default and a gradational interface must earn its misfit improvement.
    """

    scale: float
    upper: float
    lower: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Half-normal prior scale must be finite and positive.")
        UniformPrior(self.lower, self.upper)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def logpdf(self, value: float) -> float:
        if not self.contains(value):
            return -np.inf
        return float(-0.5 * ((value - self.lower) / self.scale) ** 2)


ParameterPrior = UniformPrior | NormalPrior | HalfNormalPrior


@dataclass(frozen=True)
class ParameterSpec:
    """One scalar model parameter.

    `prior` defines the support of the model space. `proposal_sigma` is only a
    sampler tuning value and is intentionally not used in prior evaluation.
    """

    name: str
    initial: float
    prior: ParameterPrior
    proposal_sigma: float | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Parameter name cannot be empty.")
        if not np.isfinite(self.initial):
            raise ValueError(f"Initial value for {self.name} must be finite.")
        if not self.prior.contains(self.initial):
            raise ValueError(f"Initial value for {self.name} is outside its prior bounds.")
        if self.proposal_sigma is not None and self.proposal_sigma <= 0:
            raise ValueError(f"Proposal sigma for {self.name} must be positive.")


@dataclass(frozen=True)
class ModelParameterization:
    """Mapping between a flat parameter vector and a segmented Vs model."""

    parameters: tuple[ParameterSpec, ...]
    model_config: dict[str, Any]
    scaling: ScalingConfig = ScalingConfig()

    def __post_init__(self):
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique.")
        if "segments" not in self.model_config:
            raise ValueError("model_config must contain a segments list.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def theta0(self) -> np.ndarray:
        return np.array([parameter.initial for parameter in self.parameters], dtype=float)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array([parameter.prior.lower for parameter in self.parameters], dtype=float)
        upper = np.array([parameter.prior.upper for parameter in self.parameters], dtype=float)
        return lower, upper

    def proposal_sigmas(self) -> np.ndarray:
        values = []
        for parameter in self.parameters:
            if parameter.proposal_sigma is None:
                values.append(0.05 * (parameter.prior.upper - parameter.prior.lower))
            else:
                values.append(parameter.proposal_sigma)
        return np.array(values, dtype=float)

    def log_parameter_prior(self, theta) -> float:
        theta = self._validate_theta(theta)
        total = 0.0
        for value, parameter in zip(theta, self.parameters):
            logp = parameter.prior.logpdf(float(value))
            if not np.isfinite(logp):
                return -np.inf
            total += logp
        return total

    def in_bounds(self, theta) -> bool:
        return np.isfinite(self.log_parameter_prior(theta))

    def vector_to_model(self, theta) -> ParameterizedVsModel:
        theta = self._validate_theta(theta)
        values = dict(zip(self.names, theta))
        return parameterized_model_from_config(
            self.model_config, values, default_scaling=self.scaling
        )

    def _validate_theta(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.parameters),):
            raise ValueError(f"theta must have shape ({len(self.parameters)},).")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain only finite values.")
        return theta


def parameterization_from_config(config: dict) -> ModelParameterization:
    """Build a parameterization from a YAML-like dictionary.

    Reads ``inversion.parameters`` and ``model`` (or ``inversion.model``).
    """

    config = prepare_inv_config(config)
    root = config.get("inversion", config)
    if "parameters" not in root:
        raise ValueError("Configuration must contain inversion.parameters.")
    parameter_cfgs = root["parameters"]
    parameters = tuple(_parameter_from_config(item) for item in parameter_cfgs)
    model_cfg = config.get("model", root.get("model"))
    if model_cfg is None:
        raise ValueError("Configuration must contain model or inversion.model.")
    model_config = {
        key: model_cfg[key]
        for key in ("segments", "interfaces", "zmax_km")
        if key in model_cfg
    }
    validate_model_layout(model_config)
    names = {parameter.name for parameter in parameters}
    unknown = sorted(parameter_references(model_config) - names)
    if unknown:
        raise ValueError(f"Model references undefined parameters: {unknown}")
    parameterization = ModelParameterization(
        parameters=parameters,
        model_config=model_config,
        scaling=scaling_from_config(model_cfg.get("scaling")),
    )
    try:
        parameterization.vector_to_model(parameterization.theta0())
    except ValueError as exc:
        raise ValueError(f"Initial parameter values give an invalid model: {exc}") from exc
    return parameterization


def _parameter_from_config(config: dict) -> ParameterSpec:
    proposal_cfg = config.get("proposal", {})
    proposal_sigma = proposal_cfg.get("sigma", config.get("proposal_sigma"))
    return ParameterSpec(
        name=config["name"],
        initial=float(config["initial"]),
        prior=prior_from_config(config.get("prior", {}), fallback_bounds=config.get("bounds")),
        proposal_sigma=None if proposal_sigma is None else float(proposal_sigma),
    )


def prior_from_config(config: dict, *, fallback_bounds=None) -> ParameterPrior:
    """``uniform`` / ``normal`` / ``half_normal``; ``bounds`` are always required.

    Bounds define the support the sampler explores (and draws starting points
    from); the prior type defines the density inside it.
    """
    prior_type = str(config.get("type", "uniform")).lower()
    bounds = config.get("bounds", fallback_bounds)
    if bounds is None:
        raise ValueError("Every parameter prior needs bounds: [lower, upper].")
    lower, upper = (float(value) for value in bounds)
    if prior_type == "uniform":
        return UniformPrior(lower=lower, upper=upper)
    if prior_type == "normal":
        return NormalPrior(
            mean=float(config["mean"]),
            sigma=float(config["sigma"]),
            lower=lower,
            upper=upper,
        )
    if prior_type == "half_normal":
        return HalfNormalPrior(scale=float(config["scale"]), lower=lower, upper=upper)
    raise ValueError(f"Unsupported prior type: {config.get('type')}")
