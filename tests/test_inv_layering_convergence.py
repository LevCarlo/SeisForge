"""Graded (depth-dependent) layering: geometry, defaults, and convergence.

The independent check is convergence: as the graded layers get thinner, the
disba phase velocity and ellipticity of the layered model approach those of
a fine uniform layering of the same continuous model.
"""

from dataclasses import replace

import numpy as np
import pytest

from seisforge.inv.forward import DispersionRequest, RayleighHVRequest
from seisforge.inv.io import discretization_from_config, discretization_to_config
from seisforge.inv.layering import (
    graded_layer_boundaries,
    layer_geometry,
    layered_model_from_parameterized,
    layering_observable_error,
    reference_discretization,
    target_layer_thickness_km,
)
from seisforge.inv.likelihood import DispersionObservation, RayleighHVObservation
from seisforge.inv.model import (
    BSplineVs,
    ConstantVs,
    DiscretizationConfig,
    GradientVs,
    Interface,
    ParameterizedVsModel,
    VsSegment,
    _segment_boundaries,
    layer_values_from_depth_profile,
)

PERIODS_S = np.array([0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])


def _sta001_like():
    """The STA001 truth model (tools/make_example_06_inversion.py)."""
    return ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 1.0, GradientVs(1.70, 2.60)),
            VsSegment(
                1.0,
                15.0,
                BSplineVs(
                    [2.65, 2.85, 3.05, 3.20, 3.30, 3.45, 3.60, 3.70],
                    knot_alpha=1.6,
                ),
            ),
        )
    )


def _gradient_with_hard_step():
    return ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 2.0, GradientVs(1.2, 2.2)),
            VsSegment(2.0, 12.0, GradientVs(3.0, 3.6)),
        )
    )


def _gradational_interface():
    return ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 3.0, GradientVs(1.5, 2.4), name="sediment"),
            VsSegment(3.0, 15.0, ConstantVs(3.4), name="crust"),
        ),
        interfaces=(Interface(3.0, width_km=0.6),),
    )


MODELS = {
    "sta001_truth": _sta001_like,
    "gradient_hard_step": _gradient_with_hard_step,
    "gradational_interface": _gradational_interface,
}


# --- geometry -------------------------------------------------------------


def test_graded_layers_are_geometric_from_the_surface():
    # Without a cap, a region from 0 to dz (r^n - 1) / (r - 1) holds exactly
    # n layers of thickness dz r^k (analytic, from the module docstring).
    dz, r, n = 0.1, 1.25, 9
    bottom = dz * (r**n - 1.0) / (r - 1.0)
    boundaries = graded_layer_boundaries(0.0, bottom, dz_km=dz, growth_ratio=r)
    # rtol: log1p/expm1 round trip, a few ulp per layer.
    np.testing.assert_allclose(np.diff(boundaries), dz * r ** np.arange(n), rtol=1e-12)


def test_graded_layers_respect_cap_ends_and_force_depths():
    dz, r, cap = 0.125, 1.2, 0.8
    boundaries = graded_layer_boundaries(
        1.3, 14.0, dz_km=dz, growth_ratio=r, max_thickness_km=cap, force_depths=(0.5, 4.4, 20.0)
    )
    assert boundaries[0] == 1.3 and boundaries[-1] == 14.0
    assert 4.4 in boundaries
    assert np.all(np.diff(boundaries) > 0)
    thickness = np.diff(boundaries)
    # No layer is thicker than the target thickness at its bottom (1e-12 for rounding).
    target = target_layer_thickness_km(boundaries[1:], dz_km=dz, growth_ratio=r, max_thickness_km=cap)
    assert np.all(thickness <= target + 1e-12)
    assert thickness.max() <= cap + 1e-12


def test_graded_geometry_keeps_segment_boundaries_and_transition_support():
    model = _gradational_interface()
    config = DiscretizationConfig(dz=0.1, dz_growth_ratio=1.3, dz_max=1.5, force_depths=(7.0,))
    geometry = layer_geometry(model, config)
    boundaries = np.r_[geometry.tops_km, geometry.bottoms_km[-1]]
    top, bottom = model.interfaces[0].support_km
    assert np.any(np.isclose(boundaries, top)) and np.any(np.isclose(boundaries, bottom))
    assert np.any(boundaries == 7.0)
    inside = (geometry.tops_km >= top - 1e-12) & (geometry.bottoms_km <= bottom + 1e-12)
    assert np.count_nonzero(inside) == config.transition_layers
    hard = layer_geometry(_gradient_with_hard_step(), config)
    assert np.any(hard.tops_km == 2.0)


def test_growth_ratio_one_is_bit_for_bit_the_uniform_layering():
    for build in MODELS.values():
        model = build()
        default = layered_model_from_parameterized(model, DiscretizationConfig(dz=0.25, zmax=12.0))
        explicit = layered_model_from_parameterized(
            model, DiscretizationConfig(dz=0.25, zmax=12.0, dz_growth_ratio=1.0, dz_max=3.0)
        )
        for name in ("thickness", "vp", "vs", "rho"):
            np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))
    z = np.linspace(0.0, 10.0, 11)
    np.testing.assert_array_equal(
        layer_values_from_depth_profile(z, z + 1.0, config=DiscretizationConfig(dz=0.3))[0],
        np.r_[np.diff(_segment_boundaries(0.0, 10.0, dz=0.3)), 0.0],
    )


def test_depth_profile_sampling_uses_graded_layering():
    z = np.linspace(0.0, 10.0, 11)
    config = DiscretizationConfig(dz=0.2, dz_growth_ratio=1.2, dz_max=2.0)
    thickness, _ = layer_values_from_depth_profile(z, z + 1.0, config=config)
    np.testing.assert_allclose(
        thickness[:-1],
        np.diff(graded_layer_boundaries(0.0, 10.0, dz_km=0.2, growth_ratio=1.2, max_thickness_km=2.0)),
    )


@pytest.mark.parametrize(
    "kwargs", [dict(dz_growth_ratio=0.9), dict(dz_growth_ratio=np.inf), dict(dz=0.5, dz_max=0.4)]
)
def test_invalid_graded_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DiscretizationConfig(**kwargs)


def test_graded_settings_round_trip_through_config():
    config = DiscretizationConfig(dz=0.0625, zmax=15.0, dz_growth_ratio=1.1, dz_max=1.0)
    assert discretization_from_config(discretization_to_config(config)) == config
    assert "dz_growth_ratio" not in discretization_to_config(DiscretizationConfig())


# --- convergence ------------------------------------------------------------

# Graded layerings that get finer step by step (thinner first layer, slower
# growth, smaller cap), against a uniform reference of 1/64 km.
REFINEMENTS = [
    dict(dz=0.25, dz_growth_ratio=1.4, dz_max=2.0),
    dict(dz=0.125, dz_growth_ratio=1.2, dz_max=1.0),
    dict(dz=0.0625, dz_growth_ratio=1.1, dz_max=0.5),
    dict(dz=0.03125, dz_growth_ratio=1.05, dz_max=0.25),
]
REFERENCE_DZ_KM = 1.0 / 64.0


@pytest.mark.parametrize("name", sorted(MODELS))
def test_graded_layering_converges_to_fine_uniform_layering(name):
    model = MODELS[name]()
    dispersion = DispersionRequest(periods=PERIODS_S)
    hv = RayleighHVRequest(periods=PERIODS_S)
    errors = []
    for kwargs in REFINEMENTS:
        result = layering_observable_error(
            model,
            DiscretizationConfig(zmax=15.0, **kwargs),
            dispersion=dispersion,
            hv=hv,
            reference_dz_km=REFERENCE_DZ_KM,
        )
        assert result.dispersion.periods_s.size == PERIODS_S.size
        errors.append((result.dispersion.max_abs_difference, result.hv.max_abs_difference))
    errors = np.array(errors)
    # Each refinement at least halves the error (piecewise-constant layers
    # converge at second order in thickness, measured about 4x per step;
    # halving leaves room for the changing growth ratio and cap).
    assert np.all(errors[1:] <= 0.5 * errors[:-1]), errors
    # Finest graded layering: well below the STA001 noise (0.03 km/s, 0.02).
    # 0.1 sigma: assumed "negligible" level, half the 0.2 sigma working
    # level of docs/inv/forward.md.
    assert errors[-1, 0] < 0.1 * 0.03
    assert errors[-1, 1] < 0.1 * 0.02


def test_observable_error_is_zero_against_itself_and_scales_with_sigma():
    model = _sta001_like()
    config = DiscretizationConfig(dz=0.25, zmax=15.0)
    same = layering_observable_error(
        model, config, dispersion=DispersionRequest(periods=PERIODS_S), reference_dz_km=0.25
    )
    assert same.dispersion.max_abs_difference == 0.0
    assert same.hv is None and same.dispersion.sigma_ratio is None

    sigma = np.full(PERIODS_S.size, 0.03)
    observation = DispersionObservation(PERIODS_S, np.full(PERIODS_S.size, 3.0), sigma)
    hv_observation = RayleighHVObservation(PERIODS_S, np.ones(PERIODS_S.size), np.full(PERIODS_S.size, 0.02))
    result = layering_observable_error(model, config, dispersion=observation, hv=hv_observation)
    assert result.reference_dz_km == pytest.approx(0.0625)
    assert result.dispersion.sigma_ratio == pytest.approx(result.dispersion.max_abs_difference / 0.03)
    assert result.hv.sigma_ratio == pytest.approx(result.hv.max_abs_difference / 0.02)
    assert "sigma" in result.summary()


def test_reference_discretization_is_uniform_and_keeps_features():
    config = DiscretizationConfig(
        dz=0.2, zmax=9.0, force_depths=(2.5,), layers_per_segment=3, dz_growth_ratio=1.3, dz_max=1.0
    )
    reference = reference_discretization(config)
    assert reference == replace(
        config, dz=0.05, dz_growth_ratio=1.0, dz_max=None, layers_per_segment=None
    )
