"""Build and serialize `ParameterizedVsModel` objects from ``model`` config.

One builder serves literal models (``inv.io``) and sampled models
(``inv.parameterization``): any numeric field may be a parameter reference
``{parameter: name}`` resolved from the current theta values.

Two layouts are accepted.

Interface layout (absolute interface depths, optional gradational widths,
per-segment scaling)::

    model:
      zmax_km: 80
      scaling: {vp: {method: brocher2005}, rho: {method: brocher2005}}
      interfaces:
        - name: basement
          depth_km: {parameter: basement.depth}
          width_km: 0.0            # 10-90 % width; 0 = hard step
          shape: sigmoid           # or smootherstep
        - name: moho
          depth_km: {parameter: moho.depth}
          width_km: {parameter: moho.width}
      segments:                    # one more than interfaces, top to bottom
        - name: sediment
          profile: {type: gradient, top: ..., bottom: ...}
          scaling: {vp: {method: constant_vp_vs, vp_vs: 2.2}, rho: {method: gardner1974}}
        - name: crust
          profile: {type: bspline, coefficients: [...]}
        - name: mantle
          profile: {type: constant, value: 4.5}
          scaling: {vp: {method: constant_vp_vs, vp_vs: 1.8}, rho: {method: constant, rho: 3.35}}

Segment layout (legacy, fixed hard boundaries)::

    model:
      segments:
        - {top_km: 0.0, bottom_km: 2.0, profile: ...}
        - {top_km: 2.0, bottom_km: 30.0, profile: ...}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from seisforge.inv.model import (
    BSplineVs,
    ConstantVs,
    GradientVs,
    Interface,
    ParameterizedVsModel,
    ScalingConfig,
    VsSegment,
)


def parameterized_model_from_config(
    model_cfg: Mapping[str, Any],
    values: Mapping[str, float] | None = None,
    *,
    default_scaling: ScalingConfig | None = None,
) -> ParameterizedVsModel:
    """Build a model from a ``model`` mapping, resolving parameter references."""
    values = values or {}
    scaling = (
        default_scaling
        if default_scaling is not None
        else scaling_from_config(model_cfg.get("scaling"))
    )
    validate_model_layout(model_cfg)
    segment_cfgs = model_cfg["segments"]
    if "interfaces" not in model_cfg:
        segments = tuple(
            VsSegment(
                top=resolve_value(item["top_km"], values),
                bottom=resolve_value(item["bottom_km"], values),
                profile=profile_from_config(item["profile"], values),
                name=item.get("name"),
                scaling=_optional_scaling(item.get("scaling")),
            )
            for item in segment_cfgs
        )
        return ParameterizedVsModel(segments=segments, scaling=scaling)

    interface_cfgs = model_cfg["interfaces"] or []
    interfaces = tuple(
        Interface(
            depth_km=resolve_value(item["depth_km"], values),
            width_km=resolve_value(item.get("width_km", 0.0), values),
            shape=str(item.get("shape", "sigmoid")),
            name=item.get("name"),
        )
        for item in interface_cfgs
    )
    depths = [0.0, *(interface.depth_km for interface in interfaces)]
    depths.append(resolve_value(model_cfg["zmax_km"], values))
    for upper, lower, label in zip(depths[:-1], depths[1:], _boundary_labels(interfaces)):
        if lower <= upper:
            raise ValueError(f"Interface depths must increase downward ({label}).")
    segments = tuple(
        VsSegment(
            top=depths[index],
            bottom=depths[index + 1],
            profile=profile_from_config(item["profile"], values),
            name=item.get("name"),
            scaling=_optional_scaling(item.get("scaling")),
        )
        for index, item in enumerate(segment_cfgs)
    )
    return ParameterizedVsModel(segments=segments, scaling=scaling, interfaces=interfaces)


def validate_model_layout(model_cfg: Mapping[str, Any]) -> None:
    """Check the structure of a ``model`` mapping without resolving values."""
    segment_cfgs = model_cfg.get("segments")
    if not segment_cfgs:
        raise ValueError("model.segments must be a non-empty list.")
    for index, item in enumerate(segment_cfgs):
        if "profile" not in item:
            raise ValueError(f"model.segments[{index}] needs a profile.")
    if "interfaces" not in model_cfg:
        for index, item in enumerate(segment_cfgs):
            if "top_km" not in item or "bottom_km" not in item:
                raise ValueError(
                    f"model.segments[{index}] needs top_km and bottom_km "
                    "(or use model.interfaces with model.zmax_km)."
                )
        return
    interface_cfgs = model_cfg["interfaces"] or []
    if "zmax_km" not in model_cfg:
        raise ValueError("model.zmax_km is required with model.interfaces.")
    if len(segment_cfgs) != len(interface_cfgs) + 1:
        raise ValueError("model.segments must have one more entry than model.interfaces.")
    for index, item in enumerate(interface_cfgs):
        if "depth_km" not in item:
            raise ValueError(f"model.interfaces[{index}] needs depth_km.")
    for item in segment_cfgs:
        if "top_km" in item or "bottom_km" in item:
            raise ValueError(
                "With model.interfaces, segment depths come from the interfaces; "
                "remove top_km/bottom_km from segments."
            )


def parameterized_model_to_config(model: ParameterizedVsModel) -> dict:
    """Serialize a concrete model (literal values only) to a ``model`` mapping.

    Models with only unnamed hard interfaces and no segment scaling keep the
    legacy segment layout so existing configs round-trip unchanged.
    """
    config: dict[str, Any] = {"scaling": scaling_to_config(model.scaling)}
    legacy = all(interface.is_hard and interface.name is None for interface in model.interfaces)
    legacy = legacy and all(segment.scaling is None for segment in model.segments)
    if legacy:
        config["segments"] = [
            {
                **({"name": segment.name} if segment.name else {}),
                "top_km": float(segment.top),
                "bottom_km": float(segment.bottom),
                "profile": profile_to_config(segment.profile),
            }
            for segment in model.segments
        ]
        return config

    config["zmax_km"] = float(model.zmax)
    config["interfaces"] = [
        {
            **({"name": interface.name} if interface.name else {}),
            "depth_km": float(interface.depth_km),
            "width_km": float(interface.width_km),
            "shape": interface.shape,
        }
        for interface in model.interfaces
    ]
    config["segments"] = []
    for segment in model.segments:
        item: dict[str, Any] = {}
        if segment.name:
            item["name"] = segment.name
        item["profile"] = profile_to_config(segment.profile)
        if segment.scaling is not None:
            item["scaling"] = scaling_to_config(segment.scaling)
        config["segments"].append(item)
    return config


def resolve_value(config_value: Any, values: Mapping[str, float]) -> float:
    """A literal number, or ``{parameter: name}`` looked up in ``values``."""
    if isinstance(config_value, Mapping):
        name = config_value.get("parameter")
        if name is None:
            raise ValueError("Parameterized values must use {'parameter': name}.")
        try:
            return float(values[name])
        except KeyError as exc:
            raise ValueError(f"Unknown parameter reference: {name}") from exc
    return float(config_value)


def parameter_references(model_cfg: Any) -> set[str]:
    """All parameter names referenced anywhere in a model config."""
    found: set[str] = set()
    if isinstance(model_cfg, Mapping):
        if set(model_cfg) == {"parameter"}:
            found.add(str(model_cfg["parameter"]))
        else:
            for value in model_cfg.values():
                found |= parameter_references(value)
    elif isinstance(model_cfg, (list, tuple)):
        for value in model_cfg:
            found |= parameter_references(value)
    return found


def profile_from_config(config: Mapping[str, Any], values: Mapping[str, float]):
    profile_type = config["type"].lower()
    if profile_type == "constant":
        return ConstantVs(value=resolve_value(config["value"], values))
    if profile_type == "gradient":
        return GradientVs(
            top=resolve_value(config["top"], values),
            bottom=resolve_value(config["bottom"], values),
        )
    if profile_type == "bspline":
        return BSplineVs(
            coefficients=[resolve_value(item, values) for item in config["coefficients"]],
            degree=config.get("degree", 3),
            knot_spacing=config.get("knot_spacing", "geometric"),
            knot_alpha=config.get("knot_alpha", 2.0),
        )
    raise ValueError(f"Unknown Vs profile type: {config['type']}")


def profile_to_config(profile: ConstantVs | GradientVs | BSplineVs) -> dict:
    if isinstance(profile, ConstantVs):
        return {"type": "constant", "value": float(profile.value)}
    if isinstance(profile, GradientVs):
        return {
            "type": "gradient",
            "top": float(profile.top),
            "bottom": float(profile.bottom),
        }
    if isinstance(profile, BSplineVs):
        return {
            "type": "bspline",
            "coefficients": [float(value) for value in profile.coefficients],
            "degree": int(profile.degree),
            "knot_spacing": profile.knot_spacing,
            "knot_alpha": float(profile.knot_alpha),
        }
    raise TypeError(f"Unsupported Vs profile type: {type(profile).__name__}")


def scaling_from_config(config: Mapping[str, Any] | None) -> ScalingConfig:
    config = config or {}
    vp_cfg = config.get("vp", {})
    rho_cfg = config.get("rho", {})
    return ScalingConfig(
        vp_method=vp_cfg.get("method", "brocher2005"),
        rho_method=rho_cfg.get("method", "brocher2005"),
        vp_kwargs={k: v for k, v in vp_cfg.items() if k != "method"} or None,
        rho_kwargs={k: v for k, v in rho_cfg.items() if k != "method"} or None,
    )


def scaling_to_config(scaling: ScalingConfig) -> dict:
    vp = {"method": scaling.vp_method, **(scaling.vp_kwargs or {})}
    rho = {"method": scaling.rho_method, **(scaling.rho_kwargs or {})}
    return {"vp": vp, "rho": rho}


def _optional_scaling(config: Mapping[str, Any] | None) -> ScalingConfig | None:
    return None if config is None else scaling_from_config(config)


def _boundary_labels(interfaces: tuple[Interface, ...]) -> list[str]:
    labels = [interface.name or f"interface {index}" for index, interface in enumerate(interfaces)]
    return [f"surface -> {name}" for name in labels[:1]] + [
        f"{upper} -> {lower}" for upper, lower in zip(labels[:-1], labels[1:])
    ] + [f"{labels[-1]} -> zmax_km" if labels else "surface -> zmax_km"]
