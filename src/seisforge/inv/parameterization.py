"""Parameter-vector mapping for geophysical inversion models.

The sampler should only see a numerical vector. This module owns the explicit
mapping between that vector and a `ParameterizedVsModel`, keeping proposal
mechanics separate from the geophysical prior support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from seisforge.inv.model import (
    BSplineVs,
    ConstantVs,
    GradientVs,
    ParameterizedVsModel,
    ScalingConfig,
    VsSegment,
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
class ParameterSpec:
    """One scalar model parameter.

    `prior` defines the support of the model space. `proposal_sigma` is only a
    sampler tuning value and is intentionally not used in prior evaluation.
    """

    name: str
    initial: float
    prior: UniformPrior
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
        segments = tuple(_segment_from_config(item, values) for item in self.model_config["segments"])
        return ParameterizedVsModel(segments=segments, scaling=self.scaling)

    def _validate_theta(self, theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(self.parameters),):
            raise ValueError(f"theta must have shape ({len(self.parameters)},).")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain only finite values.")
        return theta


def parameterization_from_config(config: dict) -> ModelParameterization:
    """Build a parameterization from a YAML-like dictionary."""

    root = config.get("Inversion", config)
    parameter_cfgs = root["parameters"]
    parameters = tuple(_parameter_from_config(item) for item in parameter_cfgs)
    model_cfg = config.get("Model", root.get("model"))
    if model_cfg is None:
        raise ValueError("Configuration must contain Model or Inversion.model.")
    scaling = _scaling_from_model_config(model_cfg)
    return ModelParameterization(
        parameters=parameters,
        model_config={"segments": model_cfg["segments"]},
        scaling=scaling,
    )


def _parameter_from_config(config: dict) -> ParameterSpec:
    prior_cfg = config.get("prior", {})
    prior_type = prior_cfg.get("type", "uniform").lower()
    if prior_type != "uniform":
        raise ValueError(f"Unsupported prior type: {prior_cfg.get('type')}")
    lower, upper = prior_cfg.get("bounds", config.get("bounds"))
    proposal_cfg = config.get("proposal", {})
    proposal_sigma = proposal_cfg.get("sigma", config.get("proposal_sigma"))
    return ParameterSpec(
        name=config["name"],
        initial=float(config["initial"]),
        prior=UniformPrior(lower=float(lower), upper=float(upper)),
        proposal_sigma=None if proposal_sigma is None else float(proposal_sigma),
    )


def _segment_from_config(config: dict, values: dict[str, float]) -> VsSegment:
    return VsSegment(
        top=float(config["top_km"]),
        bottom=float(config["bottom_km"]),
        profile=_profile_from_config(config["profile"], values),
    )


def _profile_from_config(config: dict, values: dict[str, float]):
    profile_type = config["type"].lower()
    if profile_type == "constant":
        return ConstantVs(value=_resolve_value(config["value"], values))
    if profile_type == "gradient":
        return GradientVs(
            top=_resolve_value(config["top"], values),
            bottom=_resolve_value(config["bottom"], values),
        )
    if profile_type == "bspline":
        coefficients = [_resolve_value(item, values) for item in config["coefficients"]]
        return BSplineVs(
            coefficients=coefficients,
            degree=config.get("degree", 3),
            knot_spacing=config.get("knot_spacing", "geometric"),
            knot_alpha=config.get("knot_alpha", 2.0),
        )
    raise ValueError(f"Unknown Vs profile type: {config['type']}")


def _resolve_value(config_value, values: dict[str, float]) -> float:
    if isinstance(config_value, dict):
        name = config_value.get("parameter")
        if name is None:
            raise ValueError("Parameterized values must use {'parameter': name}.")
        try:
            return float(values[name])
        except KeyError as exc:
            raise ValueError(f"Unknown parameter reference: {name}") from exc
    return float(config_value)


def _scaling_from_model_config(model_cfg: dict) -> ScalingConfig:
    scaling_cfg = model_cfg.get("scaling", {})
    vp_cfg = scaling_cfg.get("vp", {})
    rho_cfg = scaling_cfg.get("rho", {})
    return ScalingConfig(
        vp_method=vp_cfg.get("method", "brocher2005"),
        rho_method=rho_cfg.get("method", "brocher2005"),
        vp_kwargs={k: v for k, v in vp_cfg.items() if k != "method"} or None,
        rho_kwargs={k: v for k, v in rho_cfg.items() if k != "method"} or None,
    )
