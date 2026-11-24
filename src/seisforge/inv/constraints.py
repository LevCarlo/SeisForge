"""Physical prior constraints for inversion model spaces.

The constraints are hard: a model passes all of them or lies outside the
prior support (log prior ``-inf``). Layer checks (``positive_velocity``,
``vp_gt_vs``, ``vp_vs_range``) act on the forward layers; the depth checks act
on the continuous Vs(z) profile and come in two evaluations.

``continuous`` (default)
    Statements about the continuous profile over a closed ``depth_range``,
    evaluated with `seisforge.inv.profile_analysis`, independent of any depth
    spacing and of where an interface falls relative to one:

    - ``vs_range``: ``lower <= Vs(z) <= upper`` for every z, including both
      one-sided values at a hard interface;
    - ``monotonic_vs``: ``dVs/dz >= -max_decrease_km_s_per_km`` wherever Vs is
      smooth; a hard interface is a jump, not a gradient, so a drop across it
      is not seen (``boundary_non_decreasing_vs`` bounds that);
    - ``max_vs_gradient``: ``|dVs/dz| <= max_abs_km_s_per_km`` wherever Vs is
      smooth; inside a gradational transition this is the slope of the blend
      (e.g. ``1.875 dv / (2h) = 0.950 dv / w`` for a smootherstep between flat
      segments), and a hard interface is not a gradient.

``grid`` (compatibility)
    The evaluation of SeisForge before continuous constraints: Vs sampled
    every ``dz`` over ``depth_range`` plus every segment edge, where an edge
    takes the lower segment's value; ``monotonic_vs`` allows a decrease of
    ``tolerance`` per sample step and ``max_vs_gradient`` bounds the
    difference quotient between neighbouring samples (a hard jump divided by
    the distance to the next sample). Decisions therefore depend on ``dz`` and
    on interface positions. Kept to reproduce old results
    (``constraints.evaluation: grid``), pinned by
    ``tests/data/constraint_grid_baseline.npz``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from seisforge.inv.config_schema import (
    CONSTRAINT_EVALUATIONS,
    prepare_inv_config,
    validate_constraints_config,
)
from seisforge.inv.model import LayeredModel, ParameterizedVsModel
from seisforge.inv.profile_analysis import (
    vs_extremes_km_s,
    vs_slope_extremes_km_s_per_km,
)


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
    # How the depth constraints were built ("continuous" or "grid"); recorded
    # for provenance, the constraint objects themselves carry the behaviour.
    evaluation: str = "continuous"

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
    """Continuous Vs(z) within ``bounds`` (km/s) over ``depth_range`` (km)."""

    bounds: tuple[float, float]
    depth_range: tuple[float, float] | None = None
    name: str = "vs_range"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        top, bottom = _depth_range(model, self.depth_range)
        lowest, highest = vs_extremes_km_s(model, top, bottom)
        lower, upper = self.bounds
        return bool(lowest >= lower and highest <= upper)


@dataclass(frozen=True)
class MonotonicVsConstraint:
    """``dVs/dz >= -max_decrease_km_s_per_km`` wherever the profile is smooth.

    Jumps at hard interfaces are not gradients and are not checked here.
    """

    depth_range: tuple[float, float] | None = None
    max_decrease_km_s_per_km: float = 0.0
    name: str = "monotonic_vs"

    def __post_init__(self):
        if not self.max_decrease_km_s_per_km >= 0:
            raise ValueError("monotonic_vs max_decrease_km_s_per_km must be non-negative.")

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        top, bottom = _depth_range(model, self.depth_range)
        lowest, _ = vs_slope_extremes_km_s_per_km(model, top, bottom)
        return bool(lowest >= -self.max_decrease_km_s_per_km)


@dataclass(frozen=True)
class MaxVsGradientConstraint:
    """``|dVs/dz| <= max_abs_km_s_per_km`` wherever the profile is smooth.

    Jumps at hard interfaces are not gradients and are not checked here.
    """

    max_abs_km_s_per_km: float
    depth_range: tuple[float, float] | None = None
    name: str = "max_vs_gradient"

    def __post_init__(self):
        if not self.max_abs_km_s_per_km >= 0:
            raise ValueError("max_vs_gradient max_abs_km_s_per_km must be non-negative.")

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        top, bottom = _depth_range(model, self.depth_range)
        lowest, highest = vs_slope_extremes_km_s_per_km(model, top, bottom)
        return bool(max(-lowest, highest) <= self.max_abs_km_s_per_km)


@dataclass(frozen=True)
class GridVsRangeConstraint:
    """Compatibility (``evaluation: grid``): Vs within ``bounds`` at the grid samples."""

    bounds: tuple[float, float]
    depth_range: tuple[float, float] | None = None
    dz: float = 0.1
    name: str = "vs_range"

    def check(self, model: ParameterizedVsModel, layered: LayeredModel) -> bool:
        z, vs = _sample_vs(model, self.depth_range, self.dz)
        lower, upper = self.bounds
        return bool(np.all((vs >= lower) & (vs <= upper)))


@dataclass(frozen=True)
class GridMonotonicVsConstraint:
    """Compatibility (``evaluation: grid``): a drop of at most ``tolerance`` per sample step."""

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
class GridMaxVsGradientConstraint:
    """Compatibility (``evaluation: grid``): difference quotients between samples."""

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


def constraints_from_config(config: dict | list | None) -> ConstraintSuite:
    """Build hard constraints from a full configuration or a ``constraints`` section.

    The section is a list of constraints, or a mapping ``{constraints: [...]}``
    with an optional ``evaluation`` (``continuous``, the default, or ``grid``;
    see the module docstring).
    """
    config = config or {}
    if isinstance(config, dict) and not _is_constraints_section(config):
        # A full configuration.
        config = prepare_inv_config(config).get("constraints") or []
    validate_constraints_config(config)
    if isinstance(config, list):
        items, evaluation = config, "continuous"
    else:
        items = config.get("constraints") or ()
        evaluation = str(config.get("evaluation", "continuous")).lower()
    if evaluation not in CONSTRAINT_EVALUATIONS:
        raise ValueError(f"constraints.evaluation must be one of {CONSTRAINT_EVALUATIONS}.")
    return ConstraintSuite(
        constraints=tuple(_constraint_from_config(item, evaluation) for item in items),
        evaluation=evaluation,
    )


def _is_constraints_section(config: dict) -> bool:
    return isinstance(config.get("constraints"), list) and set(config) <= {"constraints", "evaluation"}


def _constraint_from_config(config: dict, evaluation: str) -> PhysicalConstraint:
    constraint_type = config["type"].lower()
    if constraint_type == "positive_velocity":
        return PositiveVelocityConstraint()
    if constraint_type == "vp_gt_vs":
        return VpGreaterThanVsConstraint(margin=float(config.get("margin", 0.0)))
    if constraint_type == "vp_vs_range":
        return VpVsRangeConstraint(bounds=_as_pair(config["bounds"]))
    if constraint_type in {"boundary_non_decreasing_vs", "boundary_vs"}:
        return BoundaryNonDecreasingVsConstraint(tolerance=float(config.get("tolerance", 0.0)))
    depth_range = _optional_pair(config.get("depth_range"))
    if evaluation == "grid":
        dz = float(config.get("dz", 0.1))
        if constraint_type == "vs_range":
            return GridVsRangeConstraint(bounds=_as_pair(config["bounds"]), depth_range=depth_range, dz=dz)
        if constraint_type == "monotonic_vs":
            return GridMonotonicVsConstraint(
                depth_range=depth_range, tolerance=float(config.get("tolerance", 0.0)), dz=dz
            )
        if constraint_type == "max_vs_gradient":
            return GridMaxVsGradientConstraint(
                max_abs=float(config["max_abs"]), depth_range=depth_range, dz=dz
            )
    else:
        if constraint_type == "vs_range":
            return VsRangeConstraint(bounds=_as_pair(config["bounds"]), depth_range=depth_range)
        if constraint_type == "monotonic_vs":
            return MonotonicVsConstraint(
                depth_range=depth_range,
                max_decrease_km_s_per_km=float(config.get("max_decrease_km_s_per_km", 0.0)),
            )
        if constraint_type == "max_vs_gradient":
            return MaxVsGradientConstraint(
                max_abs_km_s_per_km=float(config["max_abs_km_s_per_km"]), depth_range=depth_range
            )
    raise ValueError(f"Unknown physical constraint type: {config['type']}")


def _depth_range(model: ParameterizedVsModel, depth_range: tuple[float, float] | None) -> tuple[float, float]:
    return (0.0, model.zmax) if depth_range is None else depth_range


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
