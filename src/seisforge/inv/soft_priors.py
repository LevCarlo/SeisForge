"""Soft geophysical priors evaluated on continuous parameterized models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from seisforge.inv.model import ParameterizedVsModel


@dataclass(frozen=True)
class SoftPriorEvaluation:
    """One soft-prior contribution and its scalar diagnostics."""

    name: str
    log_prior: float
    diagnostics: dict[str, float]


class SoftPhysicalPrior(Protocol):
    """Protocol for a finite model-space log-prior contribution."""

    name: str

    def evaluate(self, model: ParameterizedVsModel) -> SoftPriorEvaluation:
        ...


@dataclass(frozen=True)
class VsCurvaturePrior:
    """Prefer low RMS curvature in a continuous Vs(z) model.

    The contribution is ``-0.5 * (rms_curvature / sigma)**2``, where
    ``rms_curvature`` is the depth-averaged RMS of ``d2Vs/dz2`` evaluated
    independently inside each model segment. Segment boundaries are never
    differenced across, because they may represent a physical interface.

    Units are km/s for Vs and km for depth, so ``sigma`` is in km/s/km^2.
    Depth averaging makes the practical strength approximately independent of
    the diagnostic grid spacing ``dz``.
    """

    sigma: float
    depth_range: tuple[float, float] | None = None
    dz: float = 0.05
    name: str = "vs_curvature"

    def __post_init__(self):
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("VsCurvaturePrior sigma must be finite and positive.")
        if not np.isfinite(self.dz) or self.dz <= 0:
            raise ValueError("VsCurvaturePrior dz must be finite and positive.")
        if self.depth_range is not None:
            top, bottom = self.depth_range
            valid = np.isfinite(top) and np.isfinite(bottom) and top >= 0 and bottom > top
            if not valid:
                raise ValueError("VsCurvaturePrior depth_range must be a positive interval.")

    def evaluate(self, model: ParameterizedVsModel) -> SoftPriorEvaluation:
        energy, length, n_depth = _curvature_energy(model, self.depth_range, self.dz)
        rms_curvature = float(np.sqrt(energy / length))
        log_prior = float(-0.5 * (rms_curvature / self.sigma) ** 2)
        return SoftPriorEvaluation(
            name=self.name,
            log_prior=log_prior,
            diagnostics={
                "vs_curvature_rms": rms_curvature,
                "vs_curvature_energy": float(energy),
                "vs_curvature_depth_km": float(length),
                "vs_curvature_n_depth": float(n_depth),
            },
        )


@dataclass(frozen=True)
class SoftPriorSuiteEvaluation:
    """Combined soft-prior contribution for one model."""

    log_prior: float
    evaluations: tuple[SoftPriorEvaluation, ...]

    @property
    def diagnostics(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for evaluation in self.evaluations:
            duplicates = set(output).intersection(evaluation.diagnostics)
            if duplicates:
                names = ", ".join(sorted(duplicates))
                raise ValueError(f"Duplicate soft-prior diagnostic names: {names}")
            output.update(evaluation.diagnostics)
        return output


@dataclass(frozen=True)
class SoftPriorSuite:
    """Collection of finite soft geophysical priors."""

    priors: tuple[SoftPhysicalPrior, ...] = ()

    def evaluate(self, model: ParameterizedVsModel) -> SoftPriorSuiteEvaluation:
        evaluations = tuple(prior.evaluate(model) for prior in self.priors)
        log_prior = float(sum(evaluation.log_prior for evaluation in evaluations))
        if not np.isfinite(log_prior):
            raise ValueError("Soft priors must return finite log-prior values.")
        return SoftPriorSuiteEvaluation(log_prior=log_prior, evaluations=evaluations)


def soft_priors_from_config(config: dict | None) -> SoftPriorSuite:
    """Build soft model-space priors from the ``SoftPriors`` YAML section."""

    config = config or {}
    if "SoftPriors" in config:
        config = config["SoftPriors"]
    priors: list[SoftPhysicalPrior] = []
    for entry in config.get("priors", ()):
        prior_type = str(entry["type"]).lower()
        if prior_type != "vs_curvature":
            raise ValueError(f"Unknown soft prior type: {entry['type']}")
        depth_range = entry.get("depth_range")
        priors.append(
            VsCurvaturePrior(
                sigma=float(entry["sigma"]),
                depth_range=None
                if depth_range is None
                else (float(depth_range[0]), float(depth_range[1])),
                dz=float(entry.get("dz", 0.05)),
            )
        )
    return SoftPriorSuite(priors=tuple(priors))


def _curvature_energy(
    model: ParameterizedVsModel,
    depth_range: tuple[float, float] | None,
    dz: float,
) -> tuple[float, float, int]:
    if depth_range is None:
        top, bottom = 0.0, model.zmax
    else:
        top, bottom = depth_range
    top = max(0.0, float(top))
    bottom = min(float(bottom), model.zmax)
    if bottom <= top:
        raise ValueError("VsCurvaturePrior depth_range has no overlap with the model.")

    energy = 0.0
    length = 0.0
    n_depth = 0
    for segment in model.segments:
        local_top = max(top, segment.top)
        local_bottom = min(bottom, segment.bottom)
        if local_bottom <= local_top:
            continue
        segment_length = local_bottom - local_top
        n_interval = max(2, int(np.ceil(segment_length / dz)))
        z_km = np.linspace(local_top, local_bottom, n_interval + 1)
        # Evaluate through the active segment so its lower endpoint is not
        # reassigned to the following segment by the whole-model evaluator.
        vs_km_s = segment.evaluate(z_km)
        first_derivative = np.gradient(vs_km_s, z_km, edge_order=2)
        curvature = np.gradient(first_derivative, z_km, edge_order=2)
        energy += float(np.trapezoid(curvature**2, z_km))
        length += segment_length
        n_depth += len(z_km)
    if length <= 0 or n_depth < 3:
        raise ValueError("VsCurvaturePrior could not construct a valid depth grid.")
    return energy, length, n_depth
