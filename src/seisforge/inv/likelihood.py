"""Likelihood helpers for single-station surface-wave inversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.forward import (
    DispersionRequest,
    ForwardError,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.model import LayeredModel


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class DispersionObservation:
    """Observed surface-wave dispersion data with Gaussian uncertainties."""

    periods: ArrayLike
    velocity: ArrayLike
    sigma: ArrayLike
    mode: int = 0
    wave: str = "rayleigh"
    kind: str = "phase"
    algorithm: str = "dunkin"
    dc: float = 0.005
    dt: float = 0.025

    def __post_init__(self):
        periods, velocity, sigma = _validate_observation(
            self.periods,
            self.velocity,
            self.sigma,
            value_name="velocity",
        )
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "sigma", sigma)

    def request(self) -> DispersionRequest:
        return DispersionRequest(
            periods=self.periods,
            mode=self.mode,
            wave=self.wave,
            kind=self.kind,
            algorithm=self.algorithm,
            dc=self.dc,
            dt=self.dt,
        )


@dataclass(frozen=True)
class RayleighHVObservation:
    """Observed Rayleigh-wave HV/ellipticity data with Gaussian uncertainties."""

    periods: ArrayLike
    hv: ArrayLike
    sigma: ArrayLike
    mode: int = 0
    wave: str = "rayleigh"
    algorithm: str = "dunkin"
    dc: float = 0.005

    def __post_init__(self):
        periods, hv, sigma = _validate_observation(
            self.periods,
            self.hv,
            self.sigma,
            value_name="hv",
        )
        object.__setattr__(self, "periods", periods)
        object.__setattr__(self, "hv", hv)
        object.__setattr__(self, "sigma", sigma)
        wave = self.wave.lower()
        if wave != "rayleigh":
            raise ValueError("Rayleigh H/V currently supports only wave='rayleigh'.")
        object.__setattr__(self, "wave", wave)

    def request(self) -> RayleighHVRequest:
        return RayleighHVRequest(
            periods=self.periods,
            mode=self.mode,
            algorithm=self.algorithm,
            dc=self.dc,
        )


@dataclass(frozen=True)
class ObservableFit:
    """Gaussian misfit details for one observable."""

    n_data: int
    periods: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    sigma: np.ndarray
    residual: np.ndarray
    normalized_residual: np.ndarray
    chi2: float
    misfit: float
    rms: float
    weighted_chi2: float
    loglike: float
    weight: float = 1.0


@dataclass(frozen=True)
class JointLikelihoodResult:
    """Joint likelihood result for phase/HV inversion."""

    loglike: float
    chi2: float
    raw_chi2: float
    misfit: float
    n_data: int
    success: bool
    dispersion: ObservableFit | None = None
    hv: ObservableFit | None = None
    error: str | None = None


def evaluate_joint_likelihood(
    model: LayeredModel,
    *,
    dispersion: DispersionObservation | None = None,
    hv: RayleighHVObservation | None = None,
    dispersion_weight: float = 1.0,
    hv_weight: float = 1.0,
) -> JointLikelihoodResult:
    """Forward model requested observables and compute a joint Gaussian loglike."""
    if dispersion is None and hv is None:
        raise ValueError("At least one observation must be provided.")
    if dispersion_weight < 0 or hv_weight < 0:
        raise ValueError("Likelihood weights must be non-negative.")

    try:
        dispersion_fit = None
        hv_fit = None
        if dispersion is not None and dispersion_weight > 0:
            dispersion_prediction = predict_dispersion(model, dispersion.request())
            _check_periods_match(
                dispersion.periods,
                dispersion_prediction.periods,
                name="dispersion periods",
            )
            dispersion_fit = gaussian_observable_fit(
                periods=dispersion.periods,
                observed=dispersion.velocity,
                predicted=dispersion_prediction.velocity,
                sigma=dispersion.sigma,
                weight=dispersion_weight,
            )
        if hv is not None and hv_weight > 0:
            hv_prediction = predict_rayleigh_hv(model, hv.request())
            _check_periods_match(hv.periods, hv_prediction.periods, name="HV periods")
            hv_fit = gaussian_observable_fit(
                periods=hv.periods,
                observed=hv.hv,
                predicted=hv_prediction.hv,
                sigma=hv.sigma,
                weight=hv_weight,
            )
    except ForwardError as exc:
        return JointLikelihoodResult(
            loglike=-np.inf,
            chi2=np.inf,
            raw_chi2=np.inf,
            misfit=np.inf,
            n_data=0,
            success=False,
            error=str(exc),
        )

    total_loglike = 0.0
    weighted_chi2 = 0.0
    raw_chi2 = 0.0
    n_data = 0
    for fit in (dispersion_fit, hv_fit):
        if fit is None:
            continue
        total_loglike += fit.loglike
        weighted_chi2 += fit.weighted_chi2
        raw_chi2 += fit.chi2
        n_data += fit.n_data
    joint_misfit = float(np.sqrt(raw_chi2 / n_data)) if n_data > 0 else 0.0

    return JointLikelihoodResult(
        loglike=float(total_loglike),
        chi2=float(weighted_chi2),
        raw_chi2=float(raw_chi2),
        misfit=joint_misfit,
        n_data=n_data,
        success=True,
        dispersion=dispersion_fit,
        hv=hv_fit,
    )


def gaussian_observable_fit(
    *,
    periods: ArrayLike,
    observed: ArrayLike,
    predicted: ArrayLike,
    sigma: ArrayLike,
    weight: float = 1.0,
) -> ObservableFit:
    """Compute weighted Gaussian log likelihood for one observable."""
    periods, observed, sigma = _validate_observation(
        periods,
        observed,
        sigma,
        value_name="observed",
    )
    predicted = _as_1d_array(predicted, name="predicted")
    if len(predicted) != len(observed):
        raise ValueError("predicted must have the same length as observed.")
    if weight < 0:
        raise ValueError("weight must be non-negative.")

    residual = observed - predicted
    normalized = residual / sigma
    chi2 = float(np.sum(normalized**2))
    n_data = len(observed)
    misfit = float(np.sqrt(chi2 / n_data))
    weighted_chi2 = float(weight * chi2)
    loglike = float(-0.5 * weighted_chi2)
    return ObservableFit(
        n_data=n_data,
        periods=periods,
        observed=observed,
        predicted=predicted,
        sigma=sigma,
        residual=residual,
        normalized_residual=normalized,
        chi2=chi2,
        misfit=misfit,
        rms=misfit,
        weighted_chi2=weighted_chi2,
        loglike=loglike,
        weight=float(weight),
    )


def _validate_observation(
    periods: ArrayLike,
    values: ArrayLike,
    sigma: ArrayLike,
    *,
    value_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    periods = _as_1d_array(periods, name="periods")
    values = _as_1d_array(values, name=value_name)
    sigma = _as_1d_array(sigma, name="sigma")
    if len(periods) != len(values) or len(periods) != len(sigma):
        raise ValueError(f"periods, {value_name}, and sigma must have the same length.")
    if np.any(periods <= 0):
        raise ValueError("periods must be positive.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive.")
    return periods, values, sigma


def _as_1d_array(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _check_periods_match(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    name: str,
) -> None:
    if len(observed) != len(predicted) or not np.allclose(observed, predicted):
        raise ForwardError(f"{name} returned by forward solver do not match observations.")
