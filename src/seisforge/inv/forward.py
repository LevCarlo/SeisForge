"""Forward modeling wrappers for single-station inversion.

The public functions accept SeisForge model objects and hide disba-specific
details. This keeps likelihoods and samplers independent from the forward
solver implementation.

Failure policy: a candidate model for which the solver cannot produce the
requested values raises `ForwardError`; the likelihood turns it into ``-inf``.
Any other exception (wrong input types, unsorted periods, a missing disba,
programming errors) propagates, so it stops a run instead of silently
rejecting every proposal.

How disba 0.7 fails (read from its source and checked with probe models):

- ``disba.DispersionError`` "failed to find root for fundamental mode" from
  ``PhaseDispersion``/``GroupDispersion`` when the fundamental-mode root search
  does not bracket a root. In 3000 random valid models (2-5 layers, Vs
  0.1-4.5 km/s in any order, Vp/Vs 1.0001-3, periods 0.05-60 s) 467 raised
  it and no other exception occurred; 454 of the 467 had a half-space slower
  than an overlying layer, the rest a layer with Vp/Vs < 1.2. None of 500
  models with Vs increasing downwards and Vp/Vs 1.5-2.2 failed
  (``tools/probe_disba_failures.py``). Also raised for periods of 1e4-1e5 s
  and for NaN velocities. Caught here.
- ``ZeroDivisionError`` from the numba-compiled solvers when a denominator
  vanishes; observed for a zero period (excluded by request validation).
  Treated as numerical breakdown for that model and caught here.
- Silent truncation, no exception: periods where a higher mode does not exist
  are dropped from the result, group velocities of models with a strong
  low-velocity layer can be cut short, and ``Ellipticity`` stops at the first
  period whose eigenproblem fails (it catches ``DispersionError`` itself).
  The likelihood detects the period mismatch and raises `ForwardError`.
- Non-finite values without an exception (an infinite layer thickness gives
  NaN ellipticity): `_check_finite` raises `ForwardError`.
- ``ValueError`` "period axis must be sorted" (phase and group) and bare
  ``TypeError``/``ValueError`` from disba's input checks (array lengths,
  algorithm name, ``dc``/``dt`` not float): caller errors, not caught.
- A layer with Vp < Vs made the phase-velocity search run for more than
  90 s without returning; `LayeredModel` rejects such models before any
  solver call.

Strong low-velocity layers, thin layers (1e-6 km), Vp/Vs = 20, very short
periods (1e-4 s) and a water layer (Vs = 0) solved without error in the
probes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.model import LayeredModel


ArrayLike = float | np.ndarray


class ForwardError(RuntimeError):
    """Raised when a forward calculation fails for a candidate model."""


def _solver_failures() -> tuple[type[Exception], ...]:
    """Exceptions of the disba solvers that mean "no solution for this model".

    See the module docstring; every other exception propagates. disba is
    imported lazily, as everywhere in SeisForge, to keep imports fast.
    """
    from disba import DispersionError

    return (DispersionError, ZeroDivisionError)


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
class ReceiverFunctionRequest:
    """Request for a synthetic P receiver function (see `seisforge.rf.synthetic`).

    ``gauss_a`` is the Gaussian parameter of ``exp(-w^2 / (4 a^2))``; the time
    axis is ``n * dt_s - time_shift_s`` (direct P at 0 s).
    """

    ray_parameter_s_per_km: float
    dt_s: float
    n_samples: int
    gauss_a: float
    time_shift_s: float = 5.0
    amplitude: str = "impulse"

    def __post_init__(self):
        if not 0 <= self.ray_parameter_s_per_km < 0.2:
            raise ValueError("ray_parameter_s_per_km must be in [0, 0.2) s/km.")
        if self.dt_s <= 0 or self.gauss_a <= 0 or self.n_samples < 2:
            raise ValueError("dt_s and gauss_a must be positive and n_samples >= 2.")


@dataclass(frozen=True)
class ReceiverFunctionPrediction:
    time_s: np.ndarray
    rf: np.ndarray
    ray_parameter_s_per_km: float
    gauss_a: float


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
                dc=float(request.dc),  # disba requires a Python float
            )
        else:
            from disba import GroupDispersion as Dispersion

            solver = Dispersion(
                *model.as_disba(),
                algorithm=request.algorithm,
                dc=float(request.dc),  # disba requires a Python float
                dt=float(request.dt),
            )

        result = solver(request.periods, mode=request.mode, wave=request.wave)
    except _solver_failures() as exc:
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
            dc=float(request.dc),  # disba requires a Python float
        )
        result = solver(request.periods, mode=request.mode)
    except _solver_failures() as exc:
        raise ForwardError(f"Rayleigh HV prediction failed: {exc}") from exc

    return RayleighHVPrediction(
        periods=np.asarray(result.period, dtype=float),
        hv=_check_finite(result.ellipticity, name="rayleigh ellipticity"),
        mode=result.mode,
    )


def predict_receiver_function(
    model: LayeredModel,
    request: ReceiverFunctionRequest,
) -> ReceiverFunctionPrediction:
    """Synthetic P receiver function using every layer's Vp, Vs, and density."""
    from seisforge.rf.synthetic import receiver_function

    try:
        time, rf = receiver_function(
            model.thickness,
            model.vp,
            model.vs,
            model.rho,
            request.ray_parameter_s_per_km,
            dt_s=request.dt_s,
            n_samples=request.n_samples,
            gauss_a=request.gauss_a,
            time_shift_s=request.time_shift_s,
            amplitude=request.amplitude,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ForwardError(f"Receiver function prediction failed: {exc}") from exc
    return ReceiverFunctionPrediction(
        time_s=time,
        rf=_check_finite(rf, name="receiver function"),
        ray_parameter_s_per_km=float(request.ray_parameter_s_per_km),
        gauss_a=float(request.gauss_a),
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
