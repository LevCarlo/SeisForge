"""Posterior predictive ensembles for inversion observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.posterior import JointInversionTarget
from seisforge.inv.samplers import MCMCResult


@dataclass(frozen=True)
class ObservablePredictionEnsemble:
    periods: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray | None = None
    sigma: np.ndarray | None = None

    @property
    def mean(self) -> np.ndarray:
        return np.nanmean(self._flat(), axis=0)

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.50)

    @property
    def p16(self) -> np.ndarray:
        return self.quantile(0.16)

    @property
    def p84(self) -> np.ndarray:
        return self.quantile(0.84)

    def quantile(self, q) -> np.ndarray:
        return np.nanquantile(self._flat(), q, axis=0)

    def _flat(self) -> np.ndarray:
        return self.predicted.reshape((-1, self.predicted.shape[-1]))


@dataclass(frozen=True)
class JointPredictionEnsemble:
    dispersion: ObservablePredictionEnsemble | None = None
    hv: ObservablePredictionEnsemble | None = None
    log_prob: np.ndarray | None = None


def extract_prediction_ensemble_from_result(
    result: MCMCResult,
    *,
    target: JointInversionTarget,
    fill_invalid: bool = False,
) -> JointPredictionEnsemble:
    """Forward model every sampled theta and collect phase/HV predictions."""

    return extract_prediction_ensemble(
        result.samples,
        target=target,
        log_prob=result.log_prob,
        fill_invalid=fill_invalid,
    )


def extract_prediction_ensemble(
    samples,
    *,
    target: JointInversionTarget,
    log_prob=None,
    fill_invalid: bool = False,
) -> JointPredictionEnsemble:
    samples = np.asarray(samples, dtype=float)
    if samples.ndim < 2:
        raise ValueError("samples must have at least sample and parameter dimensions.")
    sample_shape = samples.shape[:-1]
    flat_samples = samples.reshape((-1, samples.shape[-1]))

    dispersion_values = _empty_predictions(sample_shape, target.dispersion)
    hv_values = _empty_predictions(sample_shape, target.hv)

    flat_dispersion = None if dispersion_values is None else dispersion_values.reshape((-1, dispersion_values.shape[-1]))
    flat_hv = None if hv_values is None else hv_values.reshape((-1, hv_values.shape[-1]))

    for i, theta in enumerate(flat_samples):
        detail = target.evaluate(theta)
        if not detail.success:
            if not fill_invalid:
                raise ValueError(f"Target evaluation failed for sample {i}: {detail.error}")
            if flat_dispersion is not None:
                flat_dispersion[i] = np.nan
            if flat_hv is not None:
                flat_hv[i] = np.nan
            continue
        if flat_dispersion is not None:
            flat_dispersion[i] = detail.likelihood.dispersion.predicted
        if flat_hv is not None:
            flat_hv[i] = detail.likelihood.hv.predicted

    return JointPredictionEnsemble(
        dispersion=_observable_ensemble(target.dispersion, dispersion_values),
        hv=_observable_ensemble(target.hv, hv_values),
        log_prob=None if log_prob is None else np.asarray(log_prob, dtype=float),
    )


def _empty_predictions(sample_shape, observation):
    if observation is None:
        return None
    return np.empty(sample_shape + (len(observation.periods),), dtype=float)


def _observable_ensemble(observation, predicted):
    if observation is None:
        return None
    values = observation.velocity if hasattr(observation, "velocity") else observation.hv
    return ObservablePredictionEnsemble(
        periods=observation.periods,
        predicted=predicted,
        observed=values,
        sigma=observation.sigma,
    )
