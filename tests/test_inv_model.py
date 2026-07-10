import numpy as np

from seisforge.inv.io import (
    discretization_from_config,
    layered_vs_model_from_config,
    parameterized_vs_model_from_config,
    parameterized_vs_model_to_config,
)
from seisforge.inv.model import (
    BSplineVs,
    ConstantVs,
    DiscretizationConfig,
    GradientVs,
    LayeredModel,
    LayeredVsModel,
    ParameterizedVsModel,
    ScalingConfig,
    VsSegment,
    layer_values_from_depth_profile,
    layered_model_from_depth_profiles,
    layered_model_from_layer_top_depths,
    layered_vs_model_from_depth_profile,
    layered_vs_model_from_layer_top_depths,
    parameterized_vs_model_from_depth_profile,
)
from seisforge.inv.plotting import layer_stairs, model_property_stairs


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


def test_layer_values_from_depth_profile_uses_midpoint_interpolation():
    thickness, values = layer_values_from_depth_profile(
        z=[0.0, 2.0, 4.0],
        values=[2.0, 4.0, 6.0],
        config=DiscretizationConfig(dz=1.0),
    )

    np.testing.assert_allclose(thickness, np.array([1.0, 1.0, 1.0, 1.0, 0.0]))
    np.testing.assert_allclose(values, np.array([2.5, 3.5, 4.5, 5.5, 6.0]))


def test_layered_vs_model_from_depth_profile_accepts_custom_boundaries():
    model = layered_vs_model_from_depth_profile(
        z=[0.0, 1.0, 3.0],
        vs=[2.0, 3.0, 4.0],
        boundaries=[0.0, 0.25, 0.5, 1.0, 3.0],
    )

    np.testing.assert_allclose(model.thickness, np.array([0.25, 0.25, 0.5, 2.0, 0.0]))
    assert model.vs[0] < model.vs[-2]


def test_layered_model_from_depth_profiles_can_derive_vp_and_rho():
    model = layered_model_from_depth_profiles(
        z=[0.0, 2.0, 4.0],
        vs=[2.0, 3.0, 4.0],
        config=DiscretizationConfig(dz=2.0),
    )

    assert isinstance(model, LayeredModel)
    np.testing.assert_allclose(model.vs, np.array([2.5, 3.5, 4.0]))
    assert np.all(model.vp > model.vs)
    assert np.all(model.rho > 0.0)


def test_layered_model_from_depth_profiles_accepts_explicit_vp_and_rho():
    model = layered_model_from_depth_profiles(
        z=[0.0, 2.0, 4.0],
        vs=[2.0, 3.0, 4.0],
        vp=[4.0, 5.0, 6.0],
        rho=[2.5, 2.6, 2.7],
        config=DiscretizationConfig(dz=2.0),
    )

    np.testing.assert_allclose(model.vs, np.array([2.5, 3.5, 4.0]))
    np.testing.assert_allclose(model.vp, np.array([4.5, 5.5, 6.0]))
    np.testing.assert_allclose(model.rho, np.array([2.55, 2.65, 2.7]))


def test_layered_vs_model_from_layer_top_depths_keeps_layer_values():
    model = layered_vs_model_from_layer_top_depths(
        z=[0.0, 2.0, 5.0],
        vs=[1.5, 2.5, 3.5],
    )

    np.testing.assert_allclose(model.thickness, np.array([2.0, 3.0, 0.0]))
    np.testing.assert_allclose(model.vs, np.array([1.5, 2.5, 3.5]))


def test_layered_model_from_layer_top_depths_accepts_explicit_properties():
    model = layered_model_from_layer_top_depths(
        z=[0.0, 2.0, 5.0],
        vs=[2.0, 3.0, 4.0],
        vp=[4.0, 5.0, 6.0],
        rho=[2.5, 2.6, 2.7],
    )

    np.testing.assert_allclose(model.thickness, np.array([2.0, 3.0, 0.0]))
    np.testing.assert_allclose(model.vp, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(model.rho, np.array([2.5, 2.6, 2.7]))


def test_parameterized_vs_model_from_depth_profile_builds_gradient_segments():
    model = parameterized_vs_model_from_depth_profile(
        z=[0.0, 2.0, 4.0],
        vs=[2.0, 4.0, 5.0],
    )

    np.testing.assert_allclose(model.evaluate(np.array([0.0, 1.0, 3.0])), [2.0, 3.0, 4.5])


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


def test_bspline_segment_evaluates_with_local_depth():
    segment = VsSegment(
        10.0,
        20.0,
        BSplineVs(coefficients=[3.0, 3.2, 3.6, 4.0], degree=3),
    )

    values = segment.evaluate(np.array([10.0, 15.0, 20.0]))

    assert values.shape == (3,)
    np.testing.assert_allclose(values[[0, -1]], np.array([3.0, 4.0]))
    assert 3.0 < values[1] < 4.0


def test_parameterized_model_combines_gradient_and_bspline_segments():
    model = ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 2.0, GradientVs(top=1.5, bottom=2.5)),
            VsSegment(
                2.0,
                10.0,
                BSplineVs(
                    coefficients=[2.6, 3.0, 3.4, 3.8],
                    degree=3,
                    knot_spacing="uniform",
                ),
            ),
        )
    )

    values = model.evaluate(np.array([0.0, 1.0, 2.0, 6.0, 10.0]))
    layered = model.to_layered_model(DiscretizationConfig(dz=2.0))

    np.testing.assert_allclose(values[:3], np.array([1.5, 2.0, 2.6]))
    np.testing.assert_allclose(values[-1], 3.8)
    assert any(np.isclose(layered.interfaces, 2.0))
    assert np.all(layered.vs > 0.0)


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


def test_parameterized_model_from_config_accepts_bspline_profile():
    config = {
        "Model": {
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 1.0,
                    "profile": {"type": "gradient", "top": 1.0, "bottom": 2.0},
                },
                {
                    "top_km": 1.0,
                    "bottom_km": 6.0,
                    "profile": {
                        "type": "bspline",
                        "coefficients": [2.0, 2.4, 2.8, 3.2],
                        "degree": 3,
                        "knot_spacing": "uniform",
                    },
                },
            ],
        },
    }

    model = parameterized_vs_model_from_config(config)
    values = model.evaluate(np.array([0.0, 1.0, 6.0]))

    np.testing.assert_allclose(values, np.array([1.0, 2.0, 3.2]))


def test_parameterized_model_config_round_trip():
    model = ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 0.5, ConstantVs(1.2)),
            VsSegment(0.5, 2.0, GradientVs(1.4, 2.3)),
            VsSegment(
                2.0,
                8.0,
                BSplineVs(
                    coefficients=[2.4, 2.7, 3.1, 3.4],
                    degree=3,
                    knot_spacing="geometric",
                    knot_alpha=1.5,
                ),
            ),
        ),
        scaling=ScalingConfig(
            vp_method="constant_vp_vs",
            vp_kwargs={"vp_vs": 1.75},
            rho_method="constant",
            rho_kwargs={"rho": 2.7},
        ),
    )
    discretization = DiscretizationConfig(dz=0.2, zmax=8.0, force_depths=(1.0,))

    config = parameterized_vs_model_to_config(model, discretization=discretization)
    restored_model = parameterized_vs_model_from_config(config)
    restored_disc = discretization_from_config(config["Discretization"])

    np.testing.assert_allclose(
        restored_model.evaluate(np.linspace(0.0, 8.0, 31)),
        model.evaluate(np.linspace(0.0, 8.0, 31)),
    )
    assert restored_model.scaling.vp_method == "constant_vp_vs"
    assert restored_model.scaling.vp_kwargs == {"vp_vs": 1.75}
    assert restored_model.scaling.rho_method == "constant"
    assert restored_model.scaling.rho_kwargs == {"rho": 2.7}
    assert restored_disc == discretization


def test_layer_stairs_clip_to_zmax():
    values, edges = layer_stairs(
        thickness=[1.0, 1.0, 0.0],
        values=[2.0, 3.0, 4.0],
        zmax=1.5,
    )

    np.testing.assert_allclose(values, np.array([2.0, 3.0]))
    np.testing.assert_allclose(edges, np.array([0.0, 1.0, 1.5]))


def test_model_property_stairs_can_plot_vp():
    model = LayeredModel(
        thickness=[1.0, 1.0, 0.0],
        vp=[4.0, 5.0, 6.0],
        vs=[2.0, 3.0, 4.0],
        rho=[2.5, 2.6, 2.7],
    )

    values, edges = model_property_stairs(model, model.vp, zmax=2.5)

    np.testing.assert_allclose(values, np.array([4.0, 5.0, 6.0]))
    np.testing.assert_allclose(edges, np.array([0.0, 1.0, 2.0, 2.5]))
