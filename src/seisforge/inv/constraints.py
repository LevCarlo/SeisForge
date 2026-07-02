"""Physical prior constraints for inversion model spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from seisforge.inv.model import LayeredModel, ParameterizedVsModel


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Result of evaluating geophysical prior constraints."""

    valid: bool
    failed: tuple[str, ...] = ()

    def log_prior(self) -> float:
        return 0.0 if self.valid else -np.inf


class PhysicalConstraint(Protocol):
    name: str

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        ...


@dataclass(frozen=True)
class ConstraintSuite:
    """Collection of hard constraints defining the physical prior support."""

    constraints: tuple[PhysicalConstraint, ...] = ()

    def evaluate(self, model: ParameterizedVsModel, layered: LayeredModel | None = None) -> ConstraintEvaluation:
        try:
            layered = model.to_layered_model() if layered is None else layered
        except ValueError as exc:
            return ConstraintEvaluation(valid=False, failed=(f"layered_model: {exc}",))

        failed = []
        for constraint in self.constraints:
            try:
                valid = constraint.check(model, layered)
            except ValueError:
                valid = False
            if not valid:
                failed.append(constraint.name)
        return ConstraintEvaluation(valid=len(failed) == 0, failed=tuple(failed))

    def log_physical_prior(self, model: ParameterizedVsModel, layered: LayeredModel | None = None) -> float:
        return self.evaluate(model, layered).log_prior()


@dataclass(frozen=True)
class PositiveVelocityConstraint:
    name: str = "positive_velocity"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        return (
            np.all(np.isfinite(layered.vs))
            and np.all(np.isfinite(layered.vp))
            and np.all(np.isfinite(layered.rho))
            and np.all(layered.vs > 0.0)
            and np.all(layered.vp > 0.0)
            and np.all(layered.rho > 0.0)
        )


@dataclass(frozen=True)
class VpGreaterThanVsConstraint:
    margin: float = 0.0
    name: str = "vp_gt_vs"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        return bool(np.all(layered.vp - layered.vs > self.margin))


@dataclass(frozen=True)
class VpVsRangeConstraint:
    """Require the Vp/Vs ratio to stay inside a geophysical range."""

    bounds: tuple[float, float]
    name: str = "vp_vs_range"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        lower, upper = self.bounds
        ratio = layered.vp / layered.vs
        return bool(np.all((ratio >= lower) & (ratio <= upper)))


@dataclass(frozen=True)
class BoundaryNonDecreasingVsConstraint:
    """Require Vs not to drop across segment boundaries."""

    tolerance: float = 0.0
    name: str = "boundary_non_decreasing_vs"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        for left, right in zip(model.segments[:-1], model.segments[1:]):
            left_vs = left.evaluate(np.array([left.bottom]))[0]
            right_vs = right.evaluate(np.array([right.top]))[0]
            if left_vs - right_vs > self.tolerance:
                return False
        return True


@dataclass(frozen=True)
class VsRangeConstraint:
    bounds: tuple[float, float]
    depth_range: tuple[float, float] | None = None
    dz: float = 0.1
    name: str = "vs_range"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        z, vs = _sample_vs(model, self.depth_range, self.dz)
        lower, upper = self.bounds
        return bool(np.all((vs >= lower) & (vs <= upper)))


@dataclass(frozen=True)
class MonotonicVsConstraint:
    """Require Vs to increase with depth, allowing a finite tolerance."""

    depth_range: tuple[float, float] | None = None
    tolerance: float = 0.0
    dz: float = 0.1
    name: str = "monotonic_vs"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        z, vs = _sample_vs(model, self.depth_range, self.dz)
        if len(z) < 2:
            return True
        return bool(np.all(np.diff(vs) >= -self.tolerance))


@dataclass(frozen=True)
class MaxVsGradientConstraint:
    max_abs: float
    depth_range: tuple[float, float] | None = None
    dz: float = 0.1
    name: str = "max_vs_gradient"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        z, vs = _sample_vs(model, self.depth_range, self.dz)
        if len(z) < 2:
            return True
        gradient = np.diff(vs) / np.diff(z)
        return bool(np.all(np.abs(gradient) <= self.max_abs))


def constraints_from_config(config: dict | None) -> ConstraintSuite:
    config = config or {}
    if isinstance(config, list):
        items = config
    else:
        if "Constraints" in config:
            config = config["Constraints"]
        elif "PhysicalConstraints" in config:
            config = config["PhysicalConstraints"]
        items = config if isinstance(config, list) else config.get("constraints", ())
    return ConstraintSuite(constraints=tuple(_constraint_from_config(item) for item in items))


def _constraint_from_config(config: dict) -> PhysicalConstraint:
    constraint_type = config["type"].lower()
    if constraint_type == "positive_velocity":
        return PositiveVelocityConstraint()
    if constraint_type == "vp_gt_vs":
        return VpGreaterThanVsConstraint(margin=float(config.get("margin", 0.0)))
    if constraint_type == "vp_vs_range":
        return VpVsRangeConstraint(bounds=_as_pair(config["bounds"]))
    if constraint_type in {"boundary_non_decreasing_vs", "boundary_vs"}:
        return BoundaryNonDecreasingVsConstraint(tolerance=float(config.get("tolerance", 0.0)))
    if constraint_type == "vs_range":
        return VsRangeConstraint(
            bounds=_as_pair(config["bounds"]),
            depth_range=_optional_pair(config.get("depth_range")),
            dz=float(config.get("dz", 0.1)),
        )
    if constraint_type == "monotonic_vs":
        return MonotonicVsConstraint(
            depth_range=_optional_pair(config.get("depth_range")),
            tolerance=float(config.get("tolerance", 0.0)),
            dz=float(config.get("dz", 0.1)),
        )
    if constraint_type == "max_vs_gradient":
        return MaxVsGradientConstraint(
            max_abs=float(config["max_abs"]),
            depth_range=_optional_pair(config.get("depth_range")),
            dz=float(config.get("dz", 0.1)),
        )
    raise ValueError(f"Unknown physical constraint type: {config['type']}")


def _sample_vs(
    model: ParameterizedVsModel,
    depth_range: tuple[float, float] | None,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dz <= 0:
        raise ValueError("Constraint sampling dz must be positive.")
    if depth_range is None:
        top, bottom = 0.0, model.zmax
    else:
        top, bottom = depth_range
    if top < 0 or bottom < top:
        raise ValueError("Invalid constraint depth_range.")
    bottom = min(bottom, model.zmax)
    if np.isclose(top, bottom):
        z = np.array([top], dtype=float)
    else:
        n = max(1, int(np.ceil((bottom - top) / dz)))
        z = np.linspace(top, bottom, n + 1)
    segment_edges = np.array([segment.top for segment in model.segments] + [model.segments[-1].bottom])
    segment_edges = segment_edges[(segment_edges >= top) & (segment_edges <= bottom)]
    z = np.unique(np.concatenate([z, segment_edges]))
    return z, model.evaluate(z)


def _as_pair(value) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("Expected a two-value pair.")
    return float(value[0]), float(value[1])


def _optional_pair(value) -> tuple[float, float] | None:
    return None if value is None else _as_pair(value)
