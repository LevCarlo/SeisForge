"""Allowed keys and section names of the inversion YAML files.

``inv.yaml`` and ``obs.yaml`` are read with ``mapping.get(key, default)``
throughout the inversion config layer, so a misspelled key (``burnin``,
``dispersion_wieght``) would otherwise fall back to its default silently.
This module lists every key those readers understand and rejects anything
else with the location and the closest allowed key. It checks names only;
values are validated by the readers and domain objects.

Section names are lowercase (``model``, ``sampler``, ...). The old capitalized
names (``Model``, ``Sampler``, ...) and the aliases ``PhysicalConstraints``
and ``Data`` are still read, with one `ConfigDeprecationWarning` per load.

The allowed sets were collected from the readers in ``inv.config``,
``inv.setup``, ``inv.model_config``, ``inv.parameterization``,
``inv.constraints``, ``inv.soft_priors`` and ``inv.io``; keep them in step
when a reader learns a new key.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from seisforge.config_keys import (
    check_keys,
    rename_legacy_sections,
    warn_legacy_sections,
)
from seisforge.inv.scaling import RHO_FROM_VP_SCALINGS, RHO_SCALINGS, VP_SCALINGS

#: Deprecated section name -> current name, for both inv.yaml and obs.yaml.
INV_SECTION_RENAMES: dict[str, str] = {
    "Model": "model",
    "Inversion": "inversion",
    "Discretization": "discretization",
    "Constraints": "constraints",
    "PhysicalConstraints": "constraints",
    "SoftPriors": "soft_priors",
    "Likelihood": "likelihood",
    "Sampler": "sampler",
    "Observations": "observations",
    "Data": "observations",
    "Forward": "forward",
    "Run": "run",
}

# Top level of inv.yaml and of the merged run configuration. ``observations``
# holds the merged data (resolved_config.yaml); ``run`` holds provenance
# attributes written by the driver and is not checked. ``parameters`` at the
# top level is the older form of ``inversion.parameters``.
INV_TOP_KEYS = frozenset(
    {
        "model",
        "inversion",
        "parameters",
        "discretization",
        "constraints",
        "soft_priors",
        "likelihood",
        "sampler",
        "observations",
        "run",
    }
)
OBS_TOP_KEYS = frozenset({"observations", "forward", "run"})

# Model. ``thickness_km``/``vs_km_s`` are the layered form read by
# ``layered_vs_model_from_config``.
MODEL_KEYS = frozenset({"segments", "interfaces", "zmax_km", "scaling", "thickness_km", "vs_km_s"})
SEGMENT_KEYS = frozenset({"name", "top_km", "bottom_km", "profile", "scaling"})
INTERFACE_KEYS = frozenset({"name", "depth_km", "width_km", "shape"})
PROFILE_KEYS = {
    "constant": frozenset({"type", "value"}),
    "gradient": frozenset({"type", "top", "bottom"}),
    "bspline": frozenset({"type", "coefficients", "degree", "knot_spacing", "knot_alpha"}),
}
PARAMETER_REFERENCE_KEYS = frozenset({"parameter"})
SCALING_KEYS = frozenset({"vp", "rho"})

# Inversion parameters. ``bounds`` and ``proposal_sigma`` on the parameter
# itself are older forms of ``prior.bounds`` and ``proposal.sigma``.
INVERSION_KEYS = frozenset({"parameters", "model"})
PARAMETER_KEYS = frozenset({"name", "initial", "prior", "proposal", "proposal_sigma", "bounds"})
PROPOSAL_KEYS = frozenset({"sigma"})
PRIOR_KEYS = {
    "uniform": frozenset({"type", "bounds"}),
    "normal": frozenset({"type", "bounds", "mean", "sigma"}),
    "half_normal": frozenset({"type", "bounds", "scale"}),
}

DISCRETIZATION_KEYS = frozenset(
    {
        "dz",
        "zmax",
        "force_depths",
        "layers_per_segment",
        "transition_layers",
        "layer_value",
        "dz_growth_ratio",
        "dz_max",
    }
)

CONSTRAINTS_SECTION_KEYS = frozenset({"constraints"})
CONSTRAINT_KEYS = {
    "positive_velocity": frozenset({"type"}),
    "vp_gt_vs": frozenset({"type", "margin"}),
    "vp_vs_range": frozenset({"type", "bounds"}),
    "boundary_non_decreasing_vs": frozenset({"type", "tolerance"}),
    "boundary_vs": frozenset({"type", "tolerance"}),
    "vs_range": frozenset({"type", "bounds", "depth_range", "dz"}),
    "monotonic_vs": frozenset({"type", "depth_range", "tolerance", "dz"}),
    "max_vs_gradient": frozenset({"type", "max_abs", "depth_range", "dz"}),
}

SOFT_PRIORS_SECTION_KEYS = frozenset({"priors"})
SOFT_PRIOR_KEYS = {"vs_curvature": frozenset({"type", "sigma", "depth_range", "dz"})}

LIKELIHOOD_KEYS = frozenset({"dispersion_weight", "hv_weight"})

SAMPLER_KEYS = frozenset(
    {
        "n_steps",
        "burn_in",
        "thin",
        "n_chains",
        "executor",
        "max_workers",
        "seed",
        "initial_strategy",
        "initial_thetas",
        "max_initial_attempts",
        "proposal_sigma",
        "bounds",
        "use_parameter_bounds",
        "global_jump_interval",
        "tempering",
        "adaptation",
    }
)
# sampler.adaptation: adaptive Metropolis during burn-in (see inv/samplers.py).
ADAPTATION_KEYS = frozenset(
    {
        "initial_diagonal_steps",
        "update_interval_steps",
        "regularization_fraction",
        "discard_early_history_fraction",
    }
)
TEMPERING_KEYS = frozenset({"temperatures", "n_temperatures", "max_temperature", "swap_interval"})

# Observations: ``hv`` and ``rayleigh_hv`` are aliases. An observation entry
# holds its file and metadata (obs.yaml), forward settings merged from
# ``forward``, and, in the merged run configuration, the data arrays
# (``velocity``/``hv`` or the alias ``values``).
OBSERVATION_TYPES = frozenset({"dispersion", "hv", "rayleigh_hv"})
DISPERSION_FORWARD_KEYS = frozenset({"algorithm", "dc", "dt"})
HV_FORWARD_KEYS = frozenset({"algorithm", "dc"})
DISPERSION_OBSERVATION_KEYS = DISPERSION_FORWARD_KEYS | {
    "file",
    "periods",
    "velocity",
    "values",
    "sigma",
    "mode",
    "wave",
    "kind",
    "noise_parameter",
}
HV_OBSERVATION_KEYS = HV_FORWARD_KEYS | {
    "file",
    "periods",
    "hv",
    "values",
    "sigma",
    "mode",
    "wave",
    "noise_parameter",
}


def prepare_inv_config(config: Any, *, where: str = "inversion configuration") -> dict:
    """Rename legacy sections (one warning), check every key, return a copy.

    Accepts an ``inv.yaml`` mapping or the merged run configuration. Already
    prepared configurations pass through without a warning.
    """
    renamed, used = rename_legacy_sections(config, INV_SECTION_RENAMES, where=where)
    warn_legacy_sections(used, where=where)
    validate_inv_config(renamed)
    return renamed


def prepare_obs_config(config: Any, *, where: str = "observation configuration") -> dict:
    """`prepare_inv_config` for an ``obs.yaml`` mapping."""
    renamed, used = rename_legacy_sections(config, INV_SECTION_RENAMES, where=where)
    warn_legacy_sections(used, where=where)
    validate_obs_config(renamed)
    return renamed


def validate_inv_config(config: Any) -> None:
    """Raise `ValueError` for any unknown key in an inv/run configuration.

    Expects the lowercase section names (see `prepare_inv_config`).
    """
    check_keys(config, INV_TOP_KEYS, where="top level")
    inversion = config.get("inversion")
    if inversion is not None:
        check_keys(inversion, INVERSION_KEYS, where="inversion")
        if "parameters" in config:
            raise ValueError(
                "Give the parameters in inversion.parameters, not also at the top level."
            )
        if "model" in config and "model" in inversion:
            raise ValueError("Give the model in model or in inversion.model, not both.")
        _validate_parameters(inversion.get("parameters"), where="inversion.parameters")
        if inversion.get("model") is not None:
            validate_model_config(inversion["model"], where="inversion.model")
    else:
        _validate_parameters(config.get("parameters"), where="parameters")
    if config.get("model") is not None:
        validate_model_config(config["model"])
    if config.get("discretization") is not None:
        check_keys(config["discretization"], DISCRETIZATION_KEYS, where="discretization")
    if config.get("constraints") is not None:
        validate_constraints_config(config["constraints"])
    if config.get("soft_priors") is not None:
        validate_soft_priors_config(config["soft_priors"])
    if config.get("likelihood") is not None:
        check_keys(config["likelihood"], LIKELIHOOD_KEYS, where="likelihood")
    if config.get("sampler") is not None:
        _validate_sampler(config["sampler"])
    if config.get("observations") is not None:
        _validate_observations(config["observations"], where="observations")


def validate_obs_config(config: Any) -> None:
    """Raise `ValueError` for any unknown key in an ``obs.yaml`` mapping."""
    check_keys(config, OBS_TOP_KEYS, where="top level")
    if config.get("observations") is not None:
        _validate_observations(config["observations"], where="observations")
    forward = config.get("forward")
    if forward is not None:
        _check_observation_types(forward, where="forward")
        for name, settings in forward.items():
            allowed = DISPERSION_FORWARD_KEYS if name == "dispersion" else HV_FORWARD_KEYS
            check_keys(settings, allowed, where=f"forward.{name}")


def validate_model_config(model: Any, *, where: str = "model") -> None:
    """Check the keys of a ``model`` section, its segments, interfaces and scaling."""
    check_keys(model, MODEL_KEYS, where=where)
    if model.get("scaling") is not None:
        validate_scaling_config(model["scaling"], where=f"{where}.scaling")
    for key in ("zmax_km",):
        if key in model:
            _check_number(model[key], where=f"{where}.{key}")
    for index, item in enumerate(_as_list(model.get("interfaces"), where=f"{where}.interfaces")):
        label = _item_label(where, "interfaces", index, item)
        check_keys(item, INTERFACE_KEYS, where=label)
        for key in ("depth_km", "width_km"):
            if key in item:
                _check_number(item[key], where=f"{label}.{key}")
    for index, item in enumerate(_as_list(model.get("segments"), where=f"{where}.segments")):
        label = _item_label(where, "segments", index, item)
        check_keys(item, SEGMENT_KEYS, where=label)
        for key in ("top_km", "bottom_km"):
            if key in item:
                _check_number(item[key], where=f"{label}.{key}")
        if item.get("profile") is not None:
            _validate_profile(item["profile"], where=f"{label}.profile")
        if item.get("scaling") is not None:
            validate_scaling_config(item["scaling"], where=f"{label}.scaling")


def validate_scaling_config(scaling: Any, *, where: str = "model.scaling") -> None:
    """Check ``scaling.vp`` / ``scaling.rho``: ``method`` plus that method's parameters."""
    check_keys(scaling, SCALING_KEYS, where=where)
    relations = {
        "vp": VP_SCALINGS,
        # A rho relation from Vp takes precedence, as in ScalingConfig.apply.
        "rho": {**RHO_SCALINGS, **RHO_FROM_VP_SCALINGS},
    }
    for name, registry in relations.items():
        entry = scaling.get(name)
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where}.{name} must be a mapping, got {type(entry).__name__}.")
        method = str(entry.get("method", "brocher2005")).lower()
        func = registry.get(method)
        if func is None:
            continue  # the scaling functions report unknown methods
        check_keys(entry, {"method"} | _keyword_parameters(func), where=f"{where}.{name} ({method})")


def validate_constraints_config(section: Any, *, where: str = "constraints") -> None:
    """Check a ``constraints`` section: a list, or a mapping with ``constraints``."""
    if isinstance(section, Mapping):
        check_keys(section, CONSTRAINTS_SECTION_KEYS, where=where)
        where = f"{where}.constraints"
        section = section.get("constraints")
    for index, item in enumerate(_as_list(section, where=where)):
        _check_typed(item, CONSTRAINT_KEYS, where=f"{where}[{index}]", kind="constraint")


def validate_soft_priors_config(section: Any, *, where: str = "soft_priors") -> None:
    """Check a ``soft_priors`` section (``priors`` list)."""
    check_keys(section, SOFT_PRIORS_SECTION_KEYS, where=where)
    for index, item in enumerate(_as_list(section.get("priors"), where=f"{where}.priors")):
        _check_typed(item, SOFT_PRIOR_KEYS, where=f"{where}.priors[{index}]", kind="soft prior")


def _validate_parameters(parameters: Any, *, where: str) -> None:
    for index, item in enumerate(_as_list(parameters, where=where)):
        label = f"{where}[{index}]"
        if isinstance(item, Mapping) and "name" in item:
            label = f"{label} ({item['name']})"
        check_keys(item, PARAMETER_KEYS, where=label)
        if item.get("proposal") is not None:
            check_keys(item["proposal"], PROPOSAL_KEYS, where=f"{label}.proposal")
        prior = item.get("prior")
        if prior is not None:
            if not isinstance(prior, Mapping):
                raise ValueError(f"{label}.prior must be a mapping.")
            prior_type = str(prior.get("type", "uniform")).lower()
            allowed = PRIOR_KEYS.get(prior_type)
            if allowed is not None:  # prior_from_config reports unknown types
                check_keys(prior, allowed, where=f"{label}.prior ({prior_type})")


def _validate_sampler(sampler: Any) -> None:
    check_keys(sampler, SAMPLER_KEYS, where="sampler")
    tempering = sampler.get("tempering")
    if isinstance(tempering, Mapping):  # non-mappings are rejected by the reader
        check_keys(tempering, TEMPERING_KEYS, where="sampler.tempering")
    adaptation = sampler.get("adaptation")
    if isinstance(adaptation, Mapping):
        check_keys(adaptation, ADAPTATION_KEYS, where="sampler.adaptation")


def _validate_observations(observations: Any, *, where: str) -> None:
    _check_observation_types(observations, where=where)
    for name, entry in observations.items():
        allowed = DISPERSION_OBSERVATION_KEYS if name == "dispersion" else HV_OBSERVATION_KEYS
        check_keys(entry, allowed, where=f"{where}.{name}")


def _check_observation_types(section: Any, *, where: str) -> None:
    check_keys(section, OBSERVATION_TYPES, where=where)
    if "hv" in section and "rayleigh_hv" in section:
        raise ValueError(f"{where} gives both hv and its alias rayleigh_hv; keep only hv.")


def _validate_profile(profile: Any, *, where: str) -> None:
    if not isinstance(profile, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(profile).__name__}.")
    profile_type = str(profile.get("type", "")).lower()
    allowed = PROFILE_KEYS.get(profile_type)
    if allowed is None:
        return  # profile_from_config reports a missing or unknown type
    check_keys(profile, allowed, where=f"{where} ({profile_type})")
    for key in ("value", "top", "bottom"):
        if key in profile:
            _check_number(profile[key], where=f"{where}.{key}")
    coefficients = profile.get("coefficients")
    if isinstance(coefficients, list):
        for index, value in enumerate(coefficients):
            _check_number(value, where=f"{where}.coefficients[{index}]")


def _check_typed(item: Any, allowed_by_type: Mapping[str, frozenset], *, where: str, kind: str) -> None:
    if not isinstance(item, Mapping):
        raise ValueError(f"{where} must be a mapping, got {type(item).__name__}.")
    if "type" not in item:
        raise ValueError(f"{where} needs a type ({', '.join(sorted(allowed_by_type))}).")
    item_type = str(item["type"]).lower()
    allowed = allowed_by_type.get(item_type)
    if allowed is None:
        return  # the reader reports unknown types
    check_keys(item, allowed, where=f"{where} ({item_type})")


def _check_number(value: Any, *, where: str) -> None:
    """A literal, or a parameter reference that has only the key ``parameter``."""
    if isinstance(value, Mapping):
        check_keys(value, PARAMETER_REFERENCE_KEYS, where=where)


def _as_list(value: Any, *, where: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a list, got {type(value).__name__}.")
    return value


def _item_label(where: str, key: str, index: int, item: Any) -> str:
    label = f"{where}.{key}[{index}]"
    if isinstance(item, Mapping) and item.get("name"):
        label = f"{label} ({item['name']})"
    return label


def _keyword_parameters(func: Callable) -> set[str]:
    """Names a scaling relation accepts after its first (Vs or Vp) argument."""
    parameters = list(inspect.signature(func).parameters.values())[1:]
    kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return {parameter.name for parameter in parameters if parameter.kind in kinds}
