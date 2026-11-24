"""Models and constraint items pinning the grid evaluation of the hard constraints.

`tools/make_constraint_grid_baseline.py` records, with the code version before
continuous constraint evaluation (commit 8b6babf), whether every model passes
every constraint item; `tests/test_inv_constraints_continuous.py` checks that
``constraints.evaluation: grid`` still gives the same decisions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seisforge.inv.model import GradientVs, Interface, ParameterizedVsModel, VsSegment
from seisforge.inv.parameterization import parameterization_from_config

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "data" / "constraint_grid_baseline.npz"
# Model and parameters of examples/06_inversion/inv.yaml at commit 8b6babf.
EXAMPLE_CONFIG = ROOT / "tests" / "data" / "constraint_grid_baseline_model.yaml"
N_EXAMPLE_DRAWS = 300
SEED = 20261124


def constraint_items() -> list[dict]:
    items = []
    for dz in (0.1, 0.25, 1.0):
        items.append({"type": "vs_range", "bounds": [2.0, 3.8], "dz": dz})
        items.append({"type": "vs_range", "bounds": [1.0, 4.6], "depth_range": [0.5, 12.0], "dz": dz})
        for tolerance in (0.0, 0.03, 0.06):
            items.append({"type": "monotonic_vs", "tolerance": tolerance, "dz": dz})
        items.append({"type": "monotonic_vs", "tolerance": 0.06, "depth_range": [0.0, 15.0], "dz": dz})
        for max_abs in (0.3, 1.2):
            items.append({"type": "max_vs_gradient", "max_abs": max_abs, "dz": dz})
        items.append({"type": "max_vs_gradient", "max_abs": 1.2, "depth_range": [2.0, 15.0], "dz": dz})
    return items


def example_thetas() -> np.ndarray:
    import yaml

    config = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    parameterization = parameterization_from_config(config)
    lower, upper = parameterization.bounds()
    rng = np.random.default_rng(SEED)
    return lower + (upper - lower) * rng.random((N_EXAMPLE_DRAWS, len(lower)))


def example_models() -> list[ParameterizedVsModel]:
    import yaml

    config = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    parameterization = parameterization_from_config(config)
    return [parameterization.vector_to_model(theta) for theta in example_thetas()]


def interface_models() -> list[ParameterizedVsModel]:
    """Two gradient segments over 15 km, with the interface swept across grid cells."""
    models = []
    for depth in np.linspace(0.8, 1.2, 17):
        for jump in (-0.2, 0.1, 0.5):
            for width, shape in ((0.0, "sigmoid"), (0.1, "sigmoid"), (0.2, "smootherstep")):
                upper = VsSegment(0.0, depth, GradientVs(2.4, 2.6))
                lower = VsSegment(depth, 15.0, GradientVs(2.6 + jump, 3.9))
                models.append(
                    ParameterizedVsModel(
                        (upper, lower),
                        interfaces=(Interface(depth, width, shape),),
                    )
                )
    return models


def all_models() -> list[ParameterizedVsModel]:
    return example_models() + interface_models()


def decisions(models, items, *, evaluation: str | None = None) -> np.ndarray:
    """``valid[i_model, i_item]`` from one single-item suite per constraint item."""
    from seisforge.inv.constraints import constraints_from_config

    suites = []
    for item in items:
        section = {"constraints": [item]}
        if evaluation is not None:
            section["evaluation"] = evaluation
        suites.append(constraints_from_config(section))
    return np.array([[suite.evaluate(model).valid for suite in suites] for model in models], dtype=bool)
