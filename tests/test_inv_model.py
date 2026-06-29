import numpy as np

from seisforge.inv.io import (
    discretization_from_config,
    layered_vs_model_from_config,
    parameterized_vs_model_from_config,
)
from seisforge.inv.model import (
    ConstantVs,
    DiscretizationConfig,
    GradientVs,
    LayeredModel,
    LayeredVsModel,
    ParameterizedVsModel,
    VsSegment,
)


def test_layered_model_requires_halfspace():
    try:
        LayeredModel(
            thickness=[1.0, 2.0],
            vp=[3.0, 4.0],
            vs=[2.0, 3.0],
            rho=[2.5, 2.7],
        )
    except ValueError as exc:
        assert "half-space" in str(exc)
    else:
        raise AssertionError("LayeredModel should require final thickness 0")


def test_layered_vs_model_converts_to_elastic_model():
    model = LayeredVsModel(
        thickness=[1.0, 2.0, 0.0],
        vs=[1.5, 2.5, 3.5],
    )

    layered = model.to_layered_model()

    assert layered.n_layers == 3
    assert layered.thickness[-1] == 0.0
    assert layered.vp.shape == layered.vs.shape
    assert np.all(layered.vp > layered.vs)


def test_parameterized_model_preserves_segment_boundary():
    model = ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 5.0, ConstantVs(2.0)),
            VsSegment(5.0, 10.0, ConstantVs(4.0)),
        )
    )

    layered = model.to_layered_model(DiscretizationConfig(dz=2.0))

    assert any(np.isclose(layered.interfaces, 5.0))
    assert 3.0 not in layered.vs


def test_gradient_segment_evaluates_with_local_depth():
    segment = VsSegment(10.0, 20.0, GradientVs(top=3.0, bottom=4.0))

    values = segment.evaluate(np.array([10.0, 15.0, 20.0]))

    np.testing.assert_allclose(values, np.array([3.0, 3.5, 4.0]))


def test_layered_vs_model_from_config():
    config = {
        "Model": {
            "thickness_km": [1.0, 0.0],
            "vs_km_s": [2.0, 3.5],
            "scaling": {
                "vp": {"method": "constant_vp_vs", "vp_vs": 1.8},
                "rho": {"method": "constant", "rho": 2.7},
            },
        }
    }

    model = layered_vs_model_from_config(config).to_layered_model()

    np.testing.assert_allclose(model.vp, np.array([3.6, 6.3]))
    np.testing.assert_allclose(model.rho, np.array([2.7, 2.7]))


def test_parameterized_model_from_config_and_discretization_config():
    config = {
        "Model": {
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 2.0,
                    "profile": {"type": "constant", "value": 2.0},
                },
                {
                    "top_km": 2.0,
                    "bottom_km": 4.0,
                    "profile": {"type": "gradient", "top": 2.5, "bottom": 3.5},
                },
            ],
        },
        "Discretization": {"dz": 1.0, "force_depths": [1.5]},
    }

    model = parameterized_vs_model_from_config(config)
    disc = discretization_from_config(config["Discretization"])
    layered = model.to_layered_model(disc)

    assert any(np.isclose(layered.interfaces, 1.5))
    assert any(np.isclose(layered.interfaces, 2.0))
