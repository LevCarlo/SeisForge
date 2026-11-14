"""Strict inversion config keys and lowercase section names.

Independent check for the renaming: copies of the examples as they were
before the rename (``tests/data/legacy_inv_config``) must load to exactly the
same configuration as the converted examples.
"""

from __future__ import annotations

import copy
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from seisforge.config_keys import ConfigDeprecationWarning, check_keys
from seisforge.inv.config import (
    load_inversion_input,
    observation_specs_from_config,
    observations_from_config,
)
from seisforge.inv.config_schema import (
    prepare_inv_config,
    prepare_obs_config,
    validate_inv_config,
    validate_obs_config,
)
from seisforge.inv.constraints import constraints_from_config
from seisforge.inv.driver import run_inversion
from seisforge.inv.io import dump_yaml, load_yaml
from seisforge.inv.parameterization import parameterization_from_config
from seisforge.inv.setup import build_inversion_setup

ROOT = Path(__file__).parents[1]
LEGACY = Path(__file__).parent / "data" / "legacy_inv_config"
STA001 = ROOT / "examples" / "STA001"
INTERFACE_MODEL = ROOT / "examples" / "configs" / "interface_model.yml"


def _deprecation_warnings(record) -> list[warnings.WarningMessage]:
    return [item for item in record if issubclass(item.category, ConfigDeprecationWarning)]


def _copy_obs(tmp_path: Path, obs_yaml: Path) -> Path:
    for name in ("vph_disp.dat", "hv.dat"):
        shutil.copy(STA001 / name, tmp_path / name)
    target = tmp_path / "obs.yaml"
    shutil.copy(obs_yaml, target)
    return target


@pytest.mark.parametrize(
    ("legacy_inv", "current_inv"),
    [
        (LEGACY / "STA001_inv.yaml", STA001 / "inv.yaml"),
        (LEGACY / "interface_model.yml", INTERFACE_MODEL),
    ],
)
def test_legacy_example_loads_identically_with_one_warning(tmp_path, legacy_inv, current_inv):
    legacy_dir = tmp_path / "legacy"
    current_dir = tmp_path / "current"
    legacy_dir.mkdir()
    current_dir.mkdir()
    legacy_obs = _copy_obs(legacy_dir, LEGACY / "STA001_obs.yaml")
    current_obs = _copy_obs(current_dir, STA001 / "obs.yaml")

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        legacy = load_inversion_input(kind="disp_hv", inv_file=legacy_inv, obs_file=legacy_obs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        current = load_inversion_input(kind="disp_hv", inv_file=current_inv, obs_file=current_obs)

    deprecations = _deprecation_warnings(record)
    assert len(deprecations) == 1, "one warning per load, covering both files"
    message = str(deprecations[0].message)
    assert "Sampler -> sampler" in message and "Observations -> observations" in message
    assert legacy.inv_config == current.inv_config
    assert legacy.obs_config == current.obs_config
    np.testing.assert_equal(legacy.run_config, current.run_config)

    legacy_setup = build_inversion_setup(legacy.run_config)
    current_setup = build_inversion_setup(current.run_config)
    np.testing.assert_array_equal(
        legacy_setup.parameterization.theta0(), current_setup.parameterization.theta0()
    )
    np.testing.assert_equal(vars(legacy_setup.sampler.config), vars(current_setup.sampler.config))


def _example():
    """The interface example plus the sections it does not use."""
    config = load_yaml(INTERFACE_MODEL)
    config["soft_priors"] = {"priors": [{"type": "vs_curvature", "sigma": 0.2, "dz": 0.1}]}
    config["sampler"]["tempering"] = {"n_temperatures": 3, "swap_interval": 1}
    config["constraints"]["constraints"].append(
        {"type": "monotonic_vs", "depth_range": [0.0, 2.0], "tolerance": 0.05}
    )
    config["observations"] = {
        "dispersion": {"periods": [1.0], "velocity": [3.0], "sigma": [0.1]},
        "hv": {"periods": [1.0], "hv": [1.2], "sigma": [0.1]},
    }
    return config


def _rename(config, path, old, new):
    node = config
    for key in path:
        node = node[key]
    node[new] = node.pop(old)


def test_example_with_every_section_is_valid():
    validate_inv_config(_example())
    validate_obs_config(load_yaml(STA001 / "obs.yaml"))


@pytest.mark.parametrize(
    ("path", "old", "new", "where", "suggestion"),
    [
        ((), "sampler", "sampelr", "top level", "sampler"),
        (("model",), "zmax_km", "zmax", "model", "zmax_km"),
        (("model", "scaling"), "rho", "rhoo", "model.scaling", "rho"),
        (("model", "interfaces", 1), "width_km", "width", "model.interfaces[1] (moho)", "width_km"),
        (("model", "segments", 0), "scaling", "scalling", "model.segments[0] (sediment)", "scaling"),
        (("model", "segments", 0, "profile"), "bottom", "botom", "profile (gradient)", "bottom"),
        (("model", "segments", 1, "profile"), "knot_spacing", "knot_spaceing", "(bspline)", "knot_spacing"),
        (("model", "segments", 0, "scaling", "vp"), "vp_vs", "vpvs", "vp (constant_vp_vs)", "vp_vs"),
        (("model", "interfaces", 0, "depth_km"), "parameter", "paramter", "depth_km", "parameter"),
        (("inversion",), "parameters", "parameter", "inversion", "parameters"),
        (("inversion", "parameters", 0), "initial", "inital", "(basement.depth)", "initial"),
        (("inversion", "parameters", 0, "proposal"), "sigma", "sigm", "proposal", "sigma"),
        (("inversion", "parameters", 1, "prior"), "sigma", "sigam", "prior (normal)", "sigma"),
        (("inversion", "parameters", 2, "prior"), "scale", "scal", "prior (half_normal)", "scale"),
        (("discretization",), "transition_layers", "transition_layer", "discretization", "transition_layers"),
        (("constraints",), "constraints", "constraint", "constraints", "constraints"),
        (("constraints", "constraints", 1), "tolerance", "tolerence", "(boundary_non_decreasing_vs)", "tolerance"),
        (("constraints", "constraints", 3), "depth_range", "depth_ranges", "(monotonic_vs)", "depth_range"),
        (("soft_priors",), "priors", "prior", "soft_priors", "priors"),
        (("soft_priors", "priors", 0), "sigma", "sgima", "(vs_curvature)", "sigma"),
        (("likelihood",), "dispersion_weight", "dispersion_wieght", "likelihood", "dispersion_weight"),
        (("sampler",), "burn_in", "burnin", "sampler", "burn_in"),
        (("sampler",), "n_steps", "n_step", "sampler", "n_steps"),
        (("sampler", "tempering"), "swap_interval", "swap_intervall", "sampler.tempering", "swap_interval"),
        (("observations",), "dispersion", "dispersions", "observations", "dispersion"),
        (("observations", "hv"), "sigma", "sigmas", "observations.hv", "sigma"),
    ],
)
def test_misspelled_inv_key_is_rejected_with_suggestion(path, old, new, where, suggestion):
    config = _example()
    _rename(config, path, old, new)

    with pytest.raises(ValueError) as excinfo:
        validate_inv_config(config)

    message = str(excinfo.value)
    assert f"{new!r}" in message
    assert where in message
    assert f"did you mean {suggestion!r}" in message


@pytest.mark.parametrize(
    ("path", "old", "new", "where", "suggestion"),
    [
        ((), "forward", "foward", "top level", "forward"),
        (("observations", "dispersion"), "kind", "knd", "observations.dispersion", "kind"),
        (("observations", "hv"), "mode", "mod", "observations.hv", "mode"),
        (("forward", "dispersion"), "dt", "dtt", "forward.dispersion", "dt"),
        (("forward", "hv"), "algorithm", "algorith", "forward.hv", "algorithm"),
    ],
)
def test_misspelled_obs_key_is_rejected_with_suggestion(path, old, new, where, suggestion):
    config = load_yaml(STA001 / "obs.yaml")
    _rename(config, path, old, new)

    with pytest.raises(ValueError, match=f"did you mean {suggestion!r}") as excinfo:
        validate_obs_config(config)
    assert where in str(excinfo.value)


def test_forward_section_takes_only_numerical_settings():
    config = load_yaml(STA001 / "obs.yaml")
    config["forward"]["hv"]["dt"] = 0.025  # dt is a dispersion-only setting

    with pytest.raises(ValueError, match=r"forward\.hv"):
        validate_obs_config(config)


def test_load_reports_the_file_with_the_unknown_key(tmp_path):
    config = load_yaml(INTERFACE_MODEL)
    _rename(config, ("sampler",), "burn_in", "burnin")
    path = tmp_path / "inv.yaml"
    dump_yaml(config, path)

    with pytest.raises(ValueError, match=r"inv\.yaml: Unknown key in sampler: 'burnin'"):
        load_inversion_input(kind="disp", inv_file=path, require_observations=False)


def test_unknown_key_reaches_every_public_reader():
    config = _example()
    config["sampler"]["burnin"] = 10

    for reader in (build_inversion_setup, parameterization_from_config, constraints_from_config):
        with pytest.raises(ValueError, match="burn_in"):
            reader(config)


def test_aliases_are_accepted():
    config = _example()
    # Parameter-level bounds and proposal_sigma (older forms of prior/proposal).
    parameter = config["inversion"]["parameters"][5]
    parameter["bounds"] = parameter.pop("prior")["bounds"]
    parameter["proposal_sigma"] = parameter.pop("proposal")["sigma"]
    # boundary_vs alias and a bare list of constraints.
    config["constraints"] = [{"type": "boundary_vs", "tolerance": 0.0}, {"type": "positive_velocity"}]
    # rayleigh_hv and values aliases in merged observations.
    config["observations"] = {
        "dispersion": {"periods": [1.0], "values": [3.0], "sigma": [0.1]},
        "rayleigh_hv": {"periods": [1.0], "values": [1.2], "sigma": [0.1]},
    }
    validate_inv_config(config)
    dispersion, hv = observations_from_config(config["observations"])
    assert dispersion.velocity[0] == 3.0 and hv.hv[0] == 1.2

    # The model may sit under inversion.model, and parameters at the top level.
    nested = _example()
    nested["inversion"]["model"] = nested.pop("model")
    flat = _example()
    flat["parameters"] = flat.pop("inversion")["parameters"]
    for variant in (nested, flat):
        parameterization = parameterization_from_config(variant)
        assert parameterization.names == parameterization_from_config(_example()).names


def test_legacy_aliases_data_and_physical_constraints_are_renamed():
    obs = load_yaml(STA001 / "obs.yaml")
    legacy_obs = {"Data": obs["observations"], "Forward": obs["forward"]}
    with pytest.warns(ConfigDeprecationWarning, match="Data -> observations"):
        assert prepare_obs_config(legacy_obs) == obs

    inv = _example()
    legacy_inv = {("PhysicalConstraints" if k == "constraints" else k): v for k, v in inv.items()}
    with pytest.warns(ConfigDeprecationWarning, match="PhysicalConstraints -> constraints"):
        assert prepare_inv_config(legacy_inv) == inv


@pytest.mark.parametrize(
    "config",
    [
        {"Sampler": {"n_steps": 10}, "sampler": {"n_steps": 10}},
        {"Constraints": [], "PhysicalConstraints": []},
    ],
)
def test_section_given_under_old_and_new_name_is_rejected(config):
    with pytest.raises(ValueError, match="twice"):
        prepare_inv_config(config)


def test_observations_under_both_hv_names_are_rejected():
    obs = load_yaml(STA001 / "obs.yaml")
    obs["observations"]["rayleigh_hv"] = dict(obs["observations"]["hv"])
    with pytest.raises(ValueError, match="rayleigh_hv"):
        observation_specs_from_config(obs)


def test_model_in_two_places_is_rejected():
    config = _example()
    config["inversion"]["model"] = copy.deepcopy(config["model"])
    with pytest.raises(ValueError, match="not both"):
        validate_inv_config(config)


def test_check_keys_message_lists_allowed_keys():
    with pytest.raises(ValueError, match=r"Allowed keys: a, b\.") as excinfo:
        check_keys({"a": 1, "zzz": 2}, {"a", "b"}, where="demo")
    assert "did you mean" not in str(excinfo.value)
    with pytest.raises(ValueError, match="must be a mapping"):
        check_keys([1], {"a"}, where="demo")


def test_resolved_yaml_files_are_read_back(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    obs_path = _copy_obs(inputs, STA001 / "obs.yaml")
    obs = load_yaml(obs_path)
    for entry in obs["observations"].values():
        entry["file"] = str(inputs / entry["file"])  # resolved_obs.yaml moves folder
    dump_yaml(obs, obs_path)
    inv = load_yaml(STA001 / "inv.yaml")
    inv["sampler"].update(
        n_steps=3, burn_in=0, thin=1, n_chains=1, initial_strategy="theta0", global_jump_interval=None
    )
    inv_path = inputs / "inv.yaml"
    dump_yaml(inv, inv_path)
    output = tmp_path / "run"
    run_inversion(kind="disp_hv", inv_file=inv_path, obs_file=obs_path, output=output, progress=False)

    resolved = {name: load_yaml(output / name) for name in (
        "resolved_inv.yaml", "resolved_obs.yaml", "resolved_config.yaml"
    )}
    for config in resolved.values():
        assert "run" in config and not any(key[0].isupper() for key in config)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        snapshot = load_inversion_input(
            kind="disp_hv",
            inv_file=output / "resolved_inv.yaml",
            obs_file=output / "resolved_obs.yaml",
        )
        merged = load_inversion_input(
            kind="disp_hv",
            inv_file=output / "resolved_config.yaml",
            require_observations=False,
        )
        assert build_inversion_setup(snapshot.run_config).target is not None
        build_inversion_setup(merged.run_config)
