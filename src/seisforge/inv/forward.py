"""Forward modeling wrappers for single-station inversion.

The public functions accept SeisForge model objects and hide disba-specific
details. This keeps likelihoods and samplers independent from the forward
solver implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.model import LayeredModel


ArrayLike = float | np.ndarray


class ForwardError(RuntimeError):
    """Raised when a forward calculation fails for a candidate model."""


def _as_periods(periods: ArrayLike, *, name: str = "periods") -> np.ndarray:
    periods = np.asarray(periods, dtype=float)
    if periods.ndim == 0:
        periods = periods.reshape(1)
    if periods.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if len(periods) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.any(~np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError(f"{name} must contain positive finite values.")
    return periods


def _check_finite(values: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ForwardError(f"{name} contains non-finite values.")
    return values


@dataclass(frozen=True)
class DispersionRequest:
    """Request for a surface-wave dispersion prediction."""

    periods: ArrayLike
    mode: int = 0
    wave: str = "rayleigh"
    kind: str = "phase"
    algorithm: str = "dunkin"
    dc: float = 0.005
    dt: float = 0.025

    def __post_init__(self):
        object.__setattr__(self, "periods", _as_periods(self.periods))
        wave = self.wave.lower()
        kind = self.kind.lower()
        if wave not in {"rayleigh", "love"}:
            raise ValueError("wave must be 'rayleigh' or 'love'.")
        if kind not in {"phase", "group"}:
            raise ValueError("kind must be 'phase' or 'group'.")
        if self.mode < 0:
            raise ValueError("mode must be non-negative.")
        object.__setattr__(self, "wave", wave)
        object.__setattr__(self, "kind", kind)


@dataclass(frozen=True)
class RayleighHVRequest:
    """Request for Rayleigh-wave HV/ellipticity prediction."""

    periods: ArrayLike
    mode: int = 0
    algorithm: str = "dunkin"
    dc: float = 0.005

    def __post_init__(self):
        object.__setattr__(self, "periods", _as_periods(self.periods))
        if self.mode < 0:
            raise ValueError("mode must be non-negative.")


@dataclass(frozen=True)
class DispersionPrediction:
    periods: np.ndarray
    velocity: np.ndarray
    mode: int
    wave: str
    kind: str


@dataclass(frozen=True)
class RayleighHVPrediction:
    periods: np.ndarray
    hv: np.ndarray
    mode: int
    observable: str = "rayleigh_ellipticity"


@dataclass(frozen=True)
class JointPrediction:
    dispersion: DispersionPrediction | None = None
    hv: RayleighHVPrediction | None = None


def predict_dispersion(
    model: LayeredModel,
    request: DispersionRequest,
) -> DispersionPrediction:
    """Predict Rayleigh/Love phase or group velocity with disba."""
    try:
        if request.kind == "phase":
            from disba import PhaseDispersion as Dispersion

            solver = Dispersion(
                *model.as_disba(),
                algorithm=request.algorithm,
                dc=request.dc,
            )
        else:
            from disba import GroupDispersion as Dispersion

            solver = Dispersion(
                *model.as_disba(),
                algorithm=request.algorithm,
                dc=request.dc,
                dt=request.dt,
            )

        result = solver(request.periods, mode=request.mode, wave=request.wave)
    except Exception as exc:
        raise ForwardError(f"Dispersion prediction failed: {exc}") from exc

    return DispersionPrediction(
        periods=np.asarray(result.period, dtype=float),
        velocity=_check_finite(result.velocity, name="dispersion velocity"),
        mode=result.mode,
        wave=result.wave,
        kind=result.type,
    )


def predict_rayleigh_hv(
    model: LayeredModel,
    request: RayleighHVRequest,
) -> RayleighHVPrediction:
    """Predict Rayleigh ellipticity used as the first CCF-HV forward observable."""
    try:
        from disba import Ellipticity

        solver = Ellipticity(
            *model.as_disba(),
            algorithm=request.algorithm,
            dc=request.dc,
        )
        result = solver(request.periods, mode=request.mode)
    except Exception as exc:
        raise ForwardError(f"Rayleigh HV prediction failed: {exc}") from exc

    return RayleighHVPrediction(
        periods=np.asarray(result.period, dtype=float),
        hv=_check_finite(result.ellipticity, name="rayleigh ellipticity"),
        mode=result.mode,
    )


def predict_joint(
    model: LayeredModel,
    dispersion_request: DispersionRequest | None = None,
    hv_request: RayleighHVRequest | None = None,
) -> JointPrediction:
    """Run all requested forward calculations for a model."""
    if dispersion_request is None and hv_request is None:
        raise ValueError("At least one forward request must be provided.")

    dispersion = (
        predict_dispersion(model, dispersion_request)
        if dispersion_request is not None
        else None
    )
    hv = predict_rayleigh_hv(model, hv_request) if hv_request is not None else None
    return JointPrediction(dispersion=dispersion, hv=hv)
