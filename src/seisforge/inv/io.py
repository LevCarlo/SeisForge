"""I/O helpers for inversion configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from seisforge.inv.forward import DispersionRequest, RayleighHVRequest
from seisforge.inv.model import (
    ConstantVs,
    DiscretizationConfig,
    GradientVs,
    LayeredVsModel,
    ParameterizedVsModel,
    ScalingConfig,
    VsSegment,
)


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
        raise NotImplementedError("B-spline profile config is reserved for a later step.")
    raise ValueError(f"Unknown Vs profile type: {config['type']}")
