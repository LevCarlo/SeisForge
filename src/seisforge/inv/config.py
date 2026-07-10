"""Configuration loading for single-station inversion.

This module turns station YAML files into explicit in-memory dictionaries and
domain observation objects. It owns file-path resolution and observation-file
loading, but it does not run samplers or write inversion outputs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from seisforge.inv.io import load_yaml, read_dispersion_dat, read_hv_dat
from seisforge.inv.likelihood import DispersionObservation, RayleighHVObservation


InversionKind = Literal["disp", "hv", "disp_hv"]


@dataclass(frozen=True)
class InversionInput:
    """Loaded inversion and observation configuration for one run."""

    kind: InversionKind
    inv_file: Path
    obs_file: Path | None
    inv_config: dict
    obs_config: dict | None
    run_config: dict
    dispersion_file: Path | None = None
    hv_file: Path | None = None


def load_inversion_input(
    *,
    kind: InversionKind,
    inv_file: str | Path,
    obs_file: str | Path | None = None,
    require_observations: bool = True,
) -> InversionInput:
    """Load `inv.yaml` and optional `obs.yaml` for a run."""

    inv_path = Path(inv_file)
    obs_path = None if obs_file is None else Path(obs_file)
    inv_config = load_yaml(inv_path)
    obs_config = None if obs_path is None else load_yaml(obs_path)
    run_config, dispersion_file, hv_file = config_with_observations(
        inv_config,
        obs_config=obs_config,
        obs_base_dir=None if obs_path is None else obs_path.parent,
        kind=kind,
        require_observations=require_observations,
    )
    return InversionInput(
        kind=kind,
        inv_file=inv_path,
        obs_file=obs_path,
        inv_config=inv_config,
        obs_config=obs_config,
        run_config=run_config,
        dispersion_file=dispersion_file,
        hv_file=hv_file,
    )


def config_with_observations(
    inv_config: dict,
    *,
    obs_config: dict | None,
    obs_base_dir: Path | None,
    kind: InversionKind,
    require_observations: bool,
) -> tuple[dict, Path | None, Path | None]:
    """Merge observation metadata, forward settings, and .dat arrays."""

    run_config = copy.deepcopy(inv_config)
    observation_specs = observation_specs_from_config(obs_config)
    observations = {}
    dispersion_file = None
    hv_file = None

    if kind in {"disp", "disp_hv"}:
        dispersion_spec = observation_specs.get("dispersion", {})
        dispersion_file = _observation_file(dispersion_spec, base_dir=obs_base_dir)
        if dispersion_file is None:
            if require_observations:
                raise ValueError(
                    "Observations.dispersion.file is required for dispersion runs."
                )
        else:
            dispersion = copy.deepcopy(dispersion_spec)
            dispersion.pop("file", None)
            dispersion.update(read_dispersion_dat(dispersion_file))
            observations["dispersion"] = dispersion

    if kind in {"hv", "disp_hv"}:
        hv_spec = observation_specs.get("hv", observation_specs.get("rayleigh_hv", {}))
        hv_file = _observation_file(hv_spec, base_dir=obs_base_dir)
        if hv_file is None:
            if require_observations:
                raise ValueError("Observations.hv.file is required for H/V runs.")
        else:
            hv = copy.deepcopy(hv_spec)
            hv.pop("file", None)
            hv.update(read_hv_dat(hv_file))
            observations["hv"] = hv

    if observations and require_observations:
        run_config["Observations"] = observations
    else:
        run_config.pop("Observations", None)
    return run_config, dispersion_file, hv_file


def observation_specs_from_config(obs_config: dict | None) -> dict:
    """Return observation specs with forward settings merged in."""

    if obs_config is None:
        return {}
    observations = copy.deepcopy(obs_config.get("Observations", obs_config.get("Data", {})))
    forward = copy.deepcopy(obs_config.get("Forward", {}))
    for key, settings in forward.items():
        if not isinstance(settings, dict):
            continue
        observation = observations.setdefault(key, {})
        if isinstance(observation, dict):
            observation.update(settings)
    return observations


def observations_from_config(
    config: dict | None,
) -> tuple[DispersionObservation | None, RayleighHVObservation | None]:
    """Build domain observation objects from merged observation config."""

    config = config or {}
    dispersion_cfg = config.get("dispersion")
    hv_cfg = config.get("hv", config.get("rayleigh_hv"))
    dispersion = (
        dispersion_observation_from_config(dispersion_cfg)
        if dispersion_cfg is not None
        else None
    )
    hv = rayleigh_hv_observation_from_config(hv_cfg) if hv_cfg is not None else None
    return dispersion, hv


def dispersion_observation_from_config(config: dict) -> DispersionObservation:
    return DispersionObservation(
        periods=config["periods"],
        velocity=config.get("velocity", config.get("values")),
        sigma=config["sigma"],
        mode=config.get("mode", 0),
        wave=config.get("wave", "rayleigh"),
        kind=config.get("kind", "phase"),
        algorithm=config.get("algorithm", "dunkin"),
        dc=config.get("dc", 0.005),
        dt=config.get("dt", 0.025),
    )


def rayleigh_hv_observation_from_config(config: dict) -> RayleighHVObservation:
    return RayleighHVObservation(
        periods=config["periods"],
        hv=config.get("hv", config.get("values")),
        sigma=config["sigma"],
        mode=config.get("mode", 0),
        wave=config.get("wave", "rayleigh"),
        algorithm=config.get("algorithm", "dunkin"),
        dc=config.get("dc", 0.005),
    )


def likelihood_weights_from_config(config: dict) -> tuple[float, float]:
    likelihood_cfg = config.get("Likelihood", {})
    if likelihood_cfg:
        return (
            float(likelihood_cfg.get("dispersion_weight", 1.0)),
            float(likelihood_cfg.get("hv_weight", 1.0)),
        )
    return 1.0, 1.0


def _observation_file(metadata: dict, *, base_dir: Path | None) -> Path | None:
    path = metadata.get("file")
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path
