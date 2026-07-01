"""I/O helpers for inversion configuration files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from seisforge.inv.ensemble import VsEnsemble
from seisforge.inv.forward import DispersionRequest, RayleighHVRequest
from seisforge.inv.model import (
    BSplineVs,
    ConstantVs,
    DiscretizationConfig,
    GradientVs,
    LayeredVsModel,
    ParameterizedVsModel,
    ScalingConfig,
    VsSegment,
)
from seisforge.inv.predictive import JointPredictionEnsemble, ObservablePredictionEnsemble
from seisforge.inv.samplers import MCMCResult


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def dump_yaml(data: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def scaling_from_config(config: dict | None) -> ScalingConfig:
    config = config or {}
    vp_cfg = config.get("vp", {})
    rho_cfg = config.get("rho", {})
    return ScalingConfig(
        vp_method=vp_cfg.get("method", "brocher2005"),
        rho_method=rho_cfg.get("method", "brocher2005"),
        vp_kwargs={k: v for k, v in vp_cfg.items() if k != "method"} or None,
        rho_kwargs={k: v for k, v in rho_cfg.items() if k != "method"} or None,
    )


def layered_vs_model_from_config(config: dict) -> LayeredVsModel:
    model_cfg = config.get("Model", config)
    return LayeredVsModel(
        thickness=model_cfg["thickness_km"],
        vs=model_cfg["vs_km_s"],
        scaling=scaling_from_config(model_cfg.get("scaling")),
    )


def parameterized_vs_model_from_config(config: dict) -> ParameterizedVsModel:
    model_cfg = config.get("Model", config)
    segments = tuple(_segment_from_config(item) for item in model_cfg["segments"])
    return ParameterizedVsModel(
        segments=segments,
        scaling=scaling_from_config(model_cfg.get("scaling")),
    )


def discretization_from_config(config: dict | None) -> DiscretizationConfig:
    config = config or {}
    return DiscretizationConfig(
        dz=config.get("dz", 0.5),
        zmax=config.get("zmax"),
        force_depths=tuple(config.get("force_depths", ())),
    )


def dispersion_request_from_config(config: dict) -> DispersionRequest:
    return DispersionRequest(
        periods=config["periods"],
        mode=config.get("mode", 0),
        wave=config.get("wave", "rayleigh"),
        kind=config.get("kind", "phase"),
        algorithm=config.get("algorithm", "dunkin"),
        dc=config.get("dc", 0.005),
        dt=config.get("dt", 0.025),
    )


def rayleigh_hv_request_from_config(config: dict) -> RayleighHVRequest:
    return RayleighHVRequest(
        periods=config["periods"],
        mode=config.get("mode", 0),
        algorithm=config.get("algorithm", "dunkin"),
        dc=config.get("dc", 0.005),
    )


def read_three_column_dat(path: str | Path, *, value_name: str) -> dict[str, np.ndarray]:
    """Read period, observed value, and one-sigma uncertainty columns."""

    path = Path(path)
    try:
        data = np.loadtxt(path, comments="#", dtype=float)
    except OSError as exc:
        raise FileNotFoundError(f"Could not read observation file: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid three-column observation file: {path}") from exc

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{path} must contain exactly three columns.")
    if len(data) == 0:
        raise ValueError(f"{path} contains no data.")
    if np.any(~np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite values.")

    periods = data[:, 0]
    sigma = data[:, 2]
    if np.any(periods <= 0):
        raise ValueError(f"{path} periods must be positive.")
    if np.any(sigma <= 0):
        raise ValueError(f"{path} uncertainties must be positive.")

    order = np.argsort(periods)
    return {
        "periods": periods[order],
        value_name: data[:, 1][order],
        "sigma": sigma[order],
    }


def read_dispersion_dat(path: str | Path) -> dict[str, np.ndarray]:
    """Read period, phase/group velocity, and uncertainty columns."""

    return read_three_column_dat(path, value_name="velocity")


def read_hv_dat(path: str | Path) -> dict[str, np.ndarray]:
    """Read period, Rayleigh H/V, and uncertainty columns."""

    return read_three_column_dat(path, value_name="hv")


def mcmc_result_to_xarray(
    result: MCMCResult,
    *,
    parameter_names: Sequence[str],
    attrs: dict | None = None,
) -> xr.Dataset:
    """Convert MCMC samples and diagnostics to an xarray dataset."""

    parameter_names = tuple(parameter_names)
    if result.samples.shape[-1] != len(parameter_names):
        raise ValueError("parameter_names length must match sample parameter dimension.")

    dataset = xr.Dataset(
        data_vars={
            "theta": (("chain", "draw", "parameter"), result.samples),
            "log_prob": (("chain", "draw"), result.log_prob),
            "accepted": (("chain", "step"), result.accepted.astype(np.int8)),
            "acceptance_rate": (("chain",), result.acceptance_rates),
            "initial_theta": (
                ("chain", "parameter"),
                np.stack([chain.initial_theta for chain in result.chains]),
            ),
            "final_theta": (
                ("chain", "parameter"),
                np.stack([chain.final_theta for chain in result.chains]),
            ),
            "initial_log_prob": (
                ("chain",),
                np.array([chain.initial_log_prob for chain in result.chains]),
            ),
            "final_log_prob": (
                ("chain",),
                np.array([chain.final_log_prob for chain in result.chains]),
            ),
        },
        coords={
            "chain": np.arange(result.samples.shape[0]),
            "draw": np.arange(result.samples.shape[1]),
            "parameter": list(parameter_names),
            "step": np.arange(result.accepted.shape[1]),
        },
        attrs=_attrs(attrs),
    )
    dataset.attrs.update(
        {
            "n_steps": int(result.config.n_steps),
            "burn_in": int(result.config.burn_in),
            "thin": int(result.config.thin),
            "mean_acceptance_rate": float(result.mean_acceptance_rate),
        }
    )
    return dataset


def vs_ensemble_to_xarray(
    ensemble: VsEnsemble,
    *,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Convert a Vs(z) ensemble and depth-wise summaries to xarray."""

    dataset = ensemble.to_xarray(name="vs")
    q01, q05, q16, q50, q84, q95, q99 = ensemble.quantile(
        [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]
    )
    dataset["mean"] = (("depth",), ensemble.mean)
    dataset["std"] = (("depth",), ensemble.std)
    dataset["minimum"] = (("depth",), ensemble.minimum)
    dataset["maximum"] = (("depth",), ensemble.maximum)
    dataset["p01"] = (("depth",), q01)
    dataset["p05"] = (("depth",), q05)
    dataset["p16"] = (("depth",), q16)
    dataset["median"] = (("depth",), q50)
    dataset["p84"] = (("depth",), q84)
    dataset["p95"] = (("depth",), q95)
    dataset["p99"] = (("depth",), q99)
    if ensemble.best_vs is not None:
        dataset["best_vs"] = (("depth",), ensemble.best_vs)
    dataset.attrs.update(_attrs(attrs))
    return dataset


def prediction_ensemble_to_xarray(
    prediction: JointPredictionEnsemble,
    *,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Convert posterior predictive dispersion/HV ensembles to xarray."""

    data_vars = {}
    coords = {}
    if prediction.dispersion is not None:
        _add_observable(
            data_vars,
            coords,
            prediction.dispersion,
            prefix="dispersion",
            value_name="velocity",
        )
    if prediction.hv is not None:
        _add_observable(
            data_vars,
            coords,
            prediction.hv,
            prefix="hv",
            value_name="hv",
        )
    if prediction.log_prob is not None:
        data_vars["log_prob"] = (
            _sample_dims(prediction.log_prob.shape),
            prediction.log_prob,
        )
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=_attrs(attrs))


def _segment_from_config(config: dict) -> VsSegment:
    return VsSegment(
        top=config["top_km"],
        bottom=config["bottom_km"],
        profile=_profile_from_config(config["profile"]),
    )


def _profile_from_config(config: dict):
    profile_type = config["type"].lower()
    if profile_type == "constant":
        return ConstantVs(value=config["value"])
    if profile_type == "gradient":
        return GradientVs(top=config["top"], bottom=config["bottom"])
    if profile_type == "bspline":
        return BSplineVs(
            coefficients=config["coefficients"],
            degree=config.get("degree", 3),
            knot_spacing=config.get("knot_spacing", "geometric"),
            knot_alpha=config.get("knot_alpha", 2.0),
        )
    raise ValueError(f"Unknown Vs profile type: {config['type']}")


def _add_observable(
    data_vars: dict,
    coords: dict,
    observable: ObservablePredictionEnsemble,
    *,
    prefix: str,
    value_name: str,
) -> None:
    sample_dims = _sample_dims(observable.predicted.shape[:-1])
    period_dim = f"{prefix}_period"
    coords[period_dim] = observable.periods
    dims = sample_dims + (period_dim,)
    data_vars[f"{prefix}_predicted_{value_name}"] = (dims, observable.predicted)
    data_vars[f"{prefix}_mean"] = ((period_dim,), observable.mean)
    data_vars[f"{prefix}_median"] = ((period_dim,), observable.median)
    q05, q16, q84, q95 = observable.quantile([0.05, 0.16, 0.84, 0.95])
    data_vars[f"{prefix}_p05"] = ((period_dim,), q05)
    data_vars[f"{prefix}_p16"] = ((period_dim,), q16)
    data_vars[f"{prefix}_p84"] = ((period_dim,), q84)
    data_vars[f"{prefix}_p95"] = ((period_dim,), q95)
    if observable.observed is not None:
        data_vars[f"{prefix}_observed_{value_name}"] = (
            (period_dim,),
            observable.observed,
        )
    if observable.sigma is not None:
        data_vars[f"{prefix}_sigma"] = ((period_dim,), observable.sigma)


def _sample_dims(shape: tuple[int, ...]) -> tuple[str, ...]:
    if len(shape) == 1:
        return ("draw",)
    if len(shape) == 2:
        return ("chain", "draw")
    return tuple(f"sample_dim_{i}" for i in range(len(shape)))


def _attrs(attrs: dict | None) -> dict:
    if attrs is None:
        return {}
    output = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, np.integer, np.floating)):
            output[key] = value
        else:
            output[key] = str(value)
    return output
