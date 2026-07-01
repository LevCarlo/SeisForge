"""Derived model ensembles from MCMC parameter samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.parameterization import ModelParameterization
from seisforge.inv.samplers import MCMCResult


@dataclass(frozen=True)
class VsEnsemble:
    """Vs(z) ensemble derived from sampled parameter vectors."""

    z: np.ndarray
    vs: np.ndarray
    theta: np.ndarray
    log_prob: np.ndarray | None = None
    chain_shape: tuple[int, ...] = ()

    def __post_init__(self):
        z = np.asarray(self.z, dtype=float)
        vs = np.asarray(self.vs, dtype=float)
        theta = np.asarray(self.theta, dtype=float)
        if z.ndim != 1:
            raise ValueError("z must be one-dimensional.")
        if np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError("z must be finite and strictly increasing.")
        if vs.shape[-1] != len(z):
            raise ValueError("The last dimension of vs must match z.")
        if theta.shape[:-1] != vs.shape[:-1]:
            raise ValueError("theta and vs must have matching sample dimensions.")
        if self.log_prob is not None:
            log_prob = np.asarray(self.log_prob, dtype=float)
            if log_prob.shape != vs.shape[:-1]:
                raise ValueError("log_prob must match the sample dimensions.")
            object.__setattr__(self, "log_prob", log_prob)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "vs", vs)
        object.__setattr__(self, "theta", theta)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.vs.shape[:-1]

    @property
    def n_samples(self) -> int:
        return int(np.prod(self.sample_shape))

    @property
    def mean(self) -> np.ndarray:
        return self._nan_stat(np.nanmean)

    @property
    def median(self) -> np.ndarray:
        return self.quantile(0.50)

    @property
    def std(self) -> np.ndarray:
        return self._nan_stat(np.nanstd)

    @property
    def minimum(self) -> np.ndarray:
        return self._nan_stat(np.nanmin)

    @property
    def maximum(self) -> np.ndarray:
        return self._nan_stat(np.nanmax)

    @property
    def p05(self) -> np.ndarray:
        return self.quantile(0.05)

    @property
    def p16(self) -> np.ndarray:
        return self.quantile(0.16)

    @property
    def p84(self) -> np.ndarray:
        return self.quantile(0.84)

    @property
    def p95(self) -> np.ndarray:
        return self.quantile(0.95)

    @property
    def best_index(self) -> tuple[int, ...] | None:
        if self.log_prob is None:
            return None
        flat_index = int(np.nanargmax(self.log_prob))
        return np.unravel_index(flat_index, self.log_prob.shape)

    @property
    def best_theta(self) -> np.ndarray | None:
        index = self.best_index
        return None if index is None else self.theta[index]

    @property
    def best_vs(self) -> np.ndarray | None:
        index = self.best_index
        return None if index is None else self.vs[index]

    def quantile(self, q: float | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        return np.nanquantile(self._flat_vs(), q, axis=0)

    def summary(self, quantiles=(0.05, 0.16, 0.50, 0.84, 0.95)) -> dict[str, np.ndarray]:
        output = {
            "z": self.z,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }
        for q, values in zip(quantiles, self.quantile(quantiles)):
            output[f"q{int(round(q * 100)):02d}"] = values
        return output

    def to_xarray(self, name: str = "vs"):
        import xarray as xr

        dims = _sample_dims(self.sample_shape) + ("depth",)
        coords = {dim: np.arange(size) for dim, size in zip(dims[:-1], self.sample_shape)}
        coords["depth"] = self.z
        data_vars = {name: (dims, self.vs)}
        if self.log_prob is not None:
            data_vars["log_prob"] = (dims[:-1], self.log_prob)
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def _flat_vs(self) -> np.ndarray:
        return self.vs.reshape((-1, self.vs.shape[-1]))

    def _nan_stat(self, func) -> np.ndarray:
        return func(self._flat_vs(), axis=0)


def extract_vs_ensemble(
    samples,
    *,
    parameterization: ModelParameterization,
    z,
    log_prob=None,
    fill_invalid: bool = False,
) -> VsEnsemble:
    """Convert sampled theta values into a Vs(z) ensemble."""

    samples = np.asarray(samples, dtype=float)
    if samples.ndim < 2:
        raise ValueError("samples must have at least sample and parameter dimensions.")
    z = _validate_depth_grid(z)
    n_parameters = len(parameterization.parameters)
    if samples.shape[-1] != n_parameters:
        raise ValueError(f"samples last dimension must have length {n_parameters}.")

    sample_shape = samples.shape[:-1]
    flat_samples = samples.reshape((-1, n_parameters))
    flat_vs = np.empty((len(flat_samples), len(z)), dtype=float)
    for i, theta in enumerate(flat_samples):
        try:
            model = parameterization.vector_to_model(theta)
            flat_vs[i] = model.evaluate(z)
        except ValueError:
            if not fill_invalid:
                raise
            flat_vs[i] = np.nan

    vs = flat_vs.reshape(sample_shape + (len(z),))
    log_prob = None if log_prob is None else np.asarray(log_prob, dtype=float)
    return VsEnsemble(
        z=z,
        vs=vs,
        theta=samples,
        log_prob=log_prob,
        chain_shape=sample_shape,
    )


def extract_vs_ensemble_from_result(
    result: MCMCResult,
    *,
    parameterization: ModelParameterization,
    z,
    fill_invalid: bool = False,
) -> VsEnsemble:
    """Convert an `MCMCResult` into a Vs(z) ensemble."""

    return extract_vs_ensemble(
        result.samples,
        parameterization=parameterization,
        z=z,
        log_prob=result.log_prob,
        fill_invalid=fill_invalid,
    )


def _validate_depth_grid(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError("z must be one-dimensional.")
    if len(z) == 0:
        raise ValueError("z cannot be empty.")
    if np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("z must be finite and strictly increasing.")
    return z


def _sample_dims(sample_shape: tuple[int, ...]) -> tuple[str, ...]:
    if len(sample_shape) == 1:
        return ("draw",)
    if len(sample_shape) == 2:
        return ("chain", "draw")
    return tuple(f"sample_dim_{i}" for i in range(len(sample_shape)))
