import numpy as np
import pytest

from seisforge.inv import (
    ConstantVs,
    DiscretizationConfig,
    GradientVs,
    HalfNormalPrior,
    Interface,
    LayeredModel,
    NormalPrior,
    ParameterizedVsModel,
    ScalingConfig,
    VsSegment,
    layer_geometry,
    layering_report,
    parameterization_from_config,
    parameterized_model_from_config,
    parameterized_model_to_config,
    prior_from_config,
    transition_half_support_km,
    transition_weight,
)
from seisforge.inv.prior import GeophysicalPrior


MANTLE_SCALING = ScalingConfig(
    vp_method="constant_vp_vs",
    rho_method="constant",
    vp_kwargs={"vp_vs": 1.8},
    rho_kwargs={"rho": 3.35},
)


def _three_layer_model(
    *,
    basement_km=1.0,
    moho_km=35.0,
    basement_width_km=0.0,
    moho_width_km=0.0,
    shape="sigmoid",
):
    return ParameterizedVsModel(
        segments=(
            VsSegment(0.0, basement_km, GradientVs(0.4, 1.2), name="sediment"),
            VsSegment(basement_km, moho_km, GradientVs(3.2, 3.9), name="crust"),
            VsSegment(moho_km, 80.0, GradientVs(4.4, 4.6), name="mantle", scaling=MANTLE_SCALING),
        ),
        interfaces=(
            Interface(basement_km, basement_width_km, shape, name="basement"),
            Interface(moho_km, moho_width_km, shape, name="moho"),
        ),
    )


@pytest.mark.parametrize("shape", ["sigmoid", "smootherstep"])
def test_transition_weight_is_a_compact_monotone_step(shape):
    depth, width = 10.0, 2.0
    half = transition_half_support_km(width, shape)
    z = np.linspace(depth - 2 * half, depth + 2 * half, 2001)
    weight = transition_weight(z, depth, width, shape)

    assert transition_weight(np.array([depth]), depth, width, shape)[0] == pytest.approx(0.5)
    assert transition_weight(np.array([depth - width / 2]), depth, width, shape)[0] == pytest.approx(0.1, abs=1e-3)
    assert transition_weight(np.array([depth + width / 2]), depth, width, shape)[0] == pytest.approx(0.9, abs=1e-3)
    assert np.all(weight[z <= depth - half] == 0.0)
    assert np.all(weight[z >= depth + half] == 1.0)
    assert np.all(np.diff(weight) >= 0)


def test_zero_width_transition_is_the_hard_step():
    z = np.array([0.5, 1.0, 1.5])
    np.testing.assert_array_equal(transition_weight(z, 1.0, 0.0), [0.0, 1.0, 1.0])


def test_interface_validation():
    with pytest.raises(ValueError, match="depth"):
        Interface(0.0)
    with pytest.raises(ValueError, match="width"):
        Interface(1.0, width_km=-0.1)
    with pytest.raises(ValueError, match="shape"):
        Interface(1.0, shape="erf")


def test_explicit_hard_interfaces_match_legacy_segments_exactly():
    legacy = ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 1.0, GradientVs(0.4, 1.2)),
            VsSegment(1.0, 30.0, GradientVs(3.2, 3.9)),
        )
    )
    explicit = ParameterizedVsModel(segments=legacy.segments, interfaces=(Interface(1.0),))
    z = np.linspace(0.0, 30.0, 301)
    np.testing.assert_array_equal(explicit.evaluate(z), legacy.evaluate(z))
    config = DiscretizationConfig(dz=0.3)
    a = explicit.to_layered_model(config)
    b = legacy.to_layered_model(config)
    for name in ("thickness", "vp", "vs", "rho"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize("shape", ["sigmoid", "smootherstep"])
def test_gradational_interface_is_smooth_and_matches_segments_outside_support(shape):
    model = _three_layer_model(moho_width_km=4.0, shape=shape)
    moho = model.interfaces[1]
    top, bottom = moho.support_km
    hard = _three_layer_model()

    outside = np.r_[np.linspace(2.0, top, 50), np.linspace(bottom, 79.0, 50)]
    np.testing.assert_allclose(model.evaluate(outside), hard.evaluate(outside))

    z = np.linspace(top - 1.0, bottom + 1.0, 4001)
    vs = model.evaluate(z)
    assert vs.min() >= 3.2 and vs.max() <= 4.6
    assert np.all(np.diff(vs) > 0)
    # No jumps in Vs or in its slope across the transition.
    step = z[1] - z[0]
    assert np.max(np.abs(np.diff(vs))) < 0.01
    slope = np.diff(vs) / step
    assert np.max(np.abs(np.diff(slope))) < 0.05


def test_transition_supports_may_not_overlap_surface_neighbours_or_bottom():
    with pytest.raises(ValueError, match="surface"):
        _three_layer_model(basement_width_km=0.8)
    with pytest.raises(ValueError, match="previous interface"):
        _three_layer_model(basement_km=10.0, moho_km=12.0, basement_width_km=1.0, moho_width_km=1.0)
    with pytest.raises(ValueError, match="model bottom"):
        _three_layer_model(moho_km=75.0, moho_width_km=4.0)


def test_per_segment_scaling_allows_mantle_velocities():
    model = _three_layer_model()
    vs, vp, rho = model.evaluate_properties(np.array([0.5, 20.0, 50.0]))
    assert vp[2] == pytest.approx(1.8 * vs[2])
    assert rho[2] == pytest.approx(3.35)
    layered = model.to_layered_model(DiscretizationConfig(dz=1.0))
    mantle = layered.segment_index == 2
    np.testing.assert_allclose(layered.vp[mantle], 1.8 * layered.vs[mantle])
    np.testing.assert_allclose(layered.rho[mantle], 3.35)

    brocher_everywhere = ParameterizedVsModel(segments=model.segments[:2] + (
        VsSegment(35.0, 80.0, GradientVs(4.4, 4.6), name="mantle"),
    ))
    with pytest.raises(ValueError, match="mantle: Brocher"):
        brocher_everywhere.to_layered_model()


def test_gardner_density_uses_scaled_vp():
    model = ParameterizedVsModel(
        segments=(
            VsSegment(
                0.0,
                2.0,
                ConstantVs(0.8),
                scaling=ScalingConfig(
                    vp_method="constant_vp_vs", rho_method="gardner1974", vp_kwargs={"vp_vs": 2.5}
                ),
            ),
        )
    )
    _, vp, rho = model.evaluate_properties(np.array([1.0]))
    assert vp[0] == pytest.approx(2.0)
    assert rho[0] == pytest.approx(1.741 * 2.0**0.25)


def test_hard_interfaces_are_exact_layer_boundaries_at_any_depth():
    model = _three_layer_model(basement_km=0.737, moho_km=33.31)
    layered = model.to_layered_model(DiscretizationConfig(dz=0.5))
    report = layering_report(model, layered)
    assert report.hard_interfaces_preserved
    assert np.any(np.isclose(layered.layer_tops, 0.737, atol=0, rtol=0))
    assert np.any(np.isclose(layered.layer_tops, 33.31, atol=0, rtol=0))


def test_transition_support_gets_the_configured_layers():
    model = _three_layer_model(moho_width_km=3.0)
    layered = model.to_layered_model(DiscretizationConfig(dz=2.0, transition_layers=12))
    report = layering_report(model, layered)
    assert report.transition_layer_counts == (12,)
    top, bottom = model.interfaces[1].support_km
    inside = (layered.layer_tops[:-1] >= top - 1e-9) & (layered.layer_tops[:-1] < bottom - 1e-9)
    np.testing.assert_allclose(layered.thickness[:-1][inside], (bottom - top) / 12)


def test_fixed_layer_counts_keep_geometry_continuous_in_interface_depth():
    config = DiscretizationConfig(
        layers_per_segment={"sediment": 6, "crust": 20, "mantle": 8},
    )
    a = layer_geometry(_three_layer_model(basement_km=1.000), config)
    b = layer_geometry(_three_layer_model(basement_km=1.001), config)
    assert a.tops_km.size == b.tops_km.size
    assert np.max(np.abs(a.tops_km - b.tops_km)) <= 1.001e-3

    by_dz = DiscretizationConfig(dz=0.25)
    c = layer_geometry(_three_layer_model(basement_km=1.00), by_dz)
    d = layer_geometry(_three_layer_model(basement_km=1.04), by_dz)
    assert c.tops_km.size != d.tops_km.size


def test_layering_report_converges_and_detects_straddled_interfaces():
    model = _three_layer_model(moho_width_km=4.0)
    coarse = layering_report(model, model.to_layered_model(DiscretizationConfig(dz=2.0)))
    fine = layering_report(model, model.to_layered_model(DiscretizationConfig(dz=0.1)))
    assert abs(fine.vs_vertical_time_relative_error) < abs(coarse.vs_vertical_time_relative_error)
    assert abs(fine.vs_vertical_time_relative_error) < 1e-4

    # A hand-made layering whose boundary misses the hard basement interface.
    straddled = LayeredModel(
        thickness=[0.75, 0.5, 33.75, 0.0],
        vs=[0.7, 2.0, 3.5, 4.5],
        vp=[1.8, 3.8, 6.0, 8.1],
        rho=[1.9, 2.2, 2.7, 3.35],
    )
    report = layering_report(model, straddled)
    assert not report.hard_interfaces_preserved
    assert report.max_interface_offset_km == pytest.approx(0.25)


def test_layer_mean_equals_midpoint_for_linear_profiles():
    model = _three_layer_model()
    midpoint = model.to_layered_model(DiscretizationConfig(dz=1.0))
    mean = model.to_layered_model(DiscretizationConfig(dz=1.0, layer_value="mean"))
    np.testing.assert_allclose(mean.vs, midpoint.vs, rtol=1e-12)


def test_halfspace_and_segment_provenance():
    model = _three_layer_model()
    layered = model.to_layered_model(DiscretizationConfig(dz=5.0, zmax=60.0))
    assert layered.thickness[-1] == 0.0
    assert layered.vs[-1] == pytest.approx(model.evaluate(np.array([60.0]))[0])
    assert layered.segment_index[0] == 0 and layered.segment_index[-1] == 2
    hvsr = layered.to_hvsr_model()
    assert hvsr.thickness_km.size == layered.n_layers - 1


def _interface_config():
    return {
        "inversion": {
            "parameters": [
                {"name": "basement.depth", "initial": 1.0, "prior": {"bounds": [0.3, 2.0]}},
                {
                    "name": "moho.depth",
                    "initial": 35.0,
                    "prior": {"type": "normal", "mean": 35.0, "sigma": 2.0, "bounds": [28.0, 45.0]},
                },
                {
                    "name": "moho.width",
                    "initial": 0.0,
                    "prior": {"type": "half_normal", "scale": 2.0, "bounds": [0.0, 8.0]},
                },
                {"name": "crust.vs_top", "initial": 3.2, "prior": {"bounds": [2.8, 3.6]}},
            ]
        },
        "model": {
            "zmax_km": 80.0,
            "interfaces": [
                {"name": "basement", "depth_km": {"parameter": "basement.depth"}},
                {
                    "name": "moho",
                    "depth_km": {"parameter": "moho.depth"},
                    "width_km": {"parameter": "moho.width"},
                    "shape": "smootherstep",
                },
            ],
            "segments": [
                {"name": "sediment", "profile": {"type": "gradient", "top": 0.4, "bottom": 1.2}},
                {
                    "name": "crust",
                    "profile": {"type": "gradient", "top": {"parameter": "crust.vs_top"}, "bottom": 3.9},
                },
                {
                    "name": "mantle",
                    "profile": {"type": "gradient", "top": 4.4, "bottom": 4.6},
                    "scaling": {
                        "vp": {"method": "constant_vp_vs", "vp_vs": 1.8},
                        "rho": {"method": "constant", "rho": 3.35},
                    },
                },
            ],
        },
    }


def test_interface_config_maps_theta_to_depths_widths_and_scaling():
    parameterization = parameterization_from_config(_interface_config())
    model = parameterization.vector_to_model([0.8, 32.0, 3.0, 3.3])
    np.testing.assert_allclose(model.interface_depths(), [0.8, 32.0])
    assert model.interfaces[1].width_km == 3.0
    assert model.interfaces[1].shape == "smootherstep"
    assert model.segments[1].top == 0.8 and model.segments[1].bottom == 32.0
    assert model.segments[2].scaling.vp_kwargs == {"vp_vs": 1.8}
    assert model.evaluate(np.array([0.8]))[0] == pytest.approx(3.3)


def test_invalid_interface_geometry_is_rejected_by_the_prior_not_raised():
    config = _interface_config()
    parameterization = parameterization_from_config(config)
    prior = GeophysicalPrior(parameterization=parameterization)
    ok = prior.evaluate([1.0, 35.0, 0.0, 3.2])
    assert ok.success
    # A 8 km wide transition needs ~8 km of room; fine at 35 km.
    assert prior.evaluate([1.0, 35.0, 8.0, 3.2]).success
    # Too wide for a sediment-thin model: support reaches the basement.
    config["inversion"]["parameters"][0]["prior"]["bounds"] = [0.3, 40.0]
    parameterization = parameterization_from_config(config)
    prior = GeophysicalPrior(parameterization=parameterization)
    result = prior.evaluate([30.0, 35.0, 8.0, 3.2])
    assert not result.success and "model_mapping" in result.error


def test_interface_config_errors_are_explicit():
    config = _interface_config()
    config["model"]["segments"][0]["top_km"] = 0.0
    with pytest.raises(ValueError, match="remove top_km"):
        parameterization_from_config(config)
    config = _interface_config()
    config["model"]["interfaces"][0]["depth_km"] = {"parameter": "missing"}
    with pytest.raises(ValueError, match="undefined parameters"):
        parameterization_from_config(config)
    model_cfg = _interface_config()["model"]
    model_cfg["interfaces"][0]["depth_km"] = 50.0
    model_cfg["interfaces"][1]["depth_km"] = 35.0
    model_cfg["interfaces"][1]["width_km"] = 0.0
    model_cfg["segments"][1]["profile"]["top"] = 3.2
    with pytest.raises(ValueError, match="increase downward"):
        parameterized_model_from_config(model_cfg)


def test_interface_model_config_round_trip():
    model = _three_layer_model(moho_width_km=3.0, shape="smootherstep")
    rebuilt = parameterized_model_from_config(parameterized_model_to_config(model))
    z = np.linspace(0.0, 80.0, 801)
    np.testing.assert_array_equal(rebuilt.evaluate(z), model.evaluate(z))
    assert rebuilt.interfaces == model.interfaces
    assert rebuilt.segments[2].scaling == MANTLE_SCALING


def test_normal_and_half_normal_priors():
    normal = prior_from_config({"type": "normal", "mean": 35.0, "sigma": 2.0, "bounds": [28, 45]})
    assert isinstance(normal, NormalPrior)
    assert normal.logpdf(37.0) == pytest.approx(-0.5)
    assert normal.logpdf(50.0) == -np.inf
    half = prior_from_config({"type": "half_normal", "scale": 2.0, "bounds": [0, 8]})
    assert isinstance(half, HalfNormalPrior)
    assert half.logpdf(0.0) == 0.0
    assert half.logpdf(2.0) == pytest.approx(-0.5)
    assert half.logpdf(-0.1) == -np.inf
    with pytest.raises(ValueError, match="bounds"):
        prior_from_config({"type": "normal", "mean": 1.0, "sigma": 1.0})


def test_example_interface_config_builds_a_valid_initial_model():
    from pathlib import Path

    from seisforge.inv import build_inversion_setup, load_yaml

    path = Path(__file__).parents[1] / "examples" / "reference_configs" / "inversion_interface_model.yml"
    setup = build_inversion_setup(load_yaml(path))
    theta0 = setup.parameterization.theta0()
    result = setup.prior.evaluate(theta0)
    assert result.success
    report = layering_report(result.model, result.layered_model)
    assert report.hard_interfaces_preserved
    assert abs(report.vs_vertical_time_relative_error) < 1e-3
