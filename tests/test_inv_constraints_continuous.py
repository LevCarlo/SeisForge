"""Hard depth constraints on the continuous Vs(z) profile.

The constraints are statements about the continuous model, so a decision must
not change with the old grid spacing ``dz`` or with where an interface falls
on that grid. Each case below is a model on which the grid evaluation (kept as
``constraints.evaluation: grid``) changes its decision with ``dz`` or with a
sub-grid shift; the continuous evaluation must not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from seisforge.inv.constraints import constraints_from_config
from seisforge.inv.model import (
    BSplineVs,
    ConstantVs,
    GradientVs,
    Interface,
    ParameterizedVsModel,
    VsSegment,
)
from seisforge.inv.profile_analysis import (
    vs_extremes_km_s,
    vs_slope_extremes_km_s_per_km,
    vs_turning_points,
)
from seisforge.inv.transitions import transition_half_support_km

sys.path.insert(0, str(Path(__file__).parent))
from inv_constraint_grid_cases import (  # noqa: E402
    BASELINE,
    all_models,
    constraint_items,
    decisions,
    example_models,
)

GRID_SPACINGS_KM = (0.1, 0.25, 1.0)
# Sub-grid shifts of the interface (km), all smaller than every dz above.
SHIFTS_KM = (0.0, 0.03, 0.09)


def _valid(model, item, *, evaluation="continuous"):
    return constraints_from_config({"evaluation": evaluation, "constraints": [item]}).evaluate(model).valid


def _two_segments(depth_km, upper, lower, *, zmax_km=5.0, width_km=0.0, shape="sigmoid"):
    return ParameterizedVsModel(
        (VsSegment(0.0, depth_km, upper), VsSegment(depth_km, zmax_km, lower)),
        interfaces=(Interface(depth_km, width_km, shape),),
    )


# --- max_vs_gradient ------------------------------------------------------


def test_grid_gradient_check_depends_on_where_a_hard_jump_falls():
    # The documented failure: a 0.1 km/s jump, dz 0.25, max_abs 1.2 (the
    # example): 0.1/0.25 = 0.4 passes, 0.1/0.03 = 3.3 fails, 0.1/0.09 = 1.1 passes.
    decisions_by_depth = [
        _valid(
            _two_segments(depth, ConstantVs(2.0), ConstantVs(2.1)),
            {"type": "max_vs_gradient", "max_abs": 1.2, "dz": 0.25},
            evaluation="grid",
        )
        for depth in (1.0, 1.03, 1.09)
    ]
    assert decisions_by_depth == [True, False, True]


@pytest.mark.parametrize("dz", GRID_SPACINGS_KM)
@pytest.mark.parametrize("shift", SHIFTS_KM)
def test_hard_jump_is_not_a_gradient(dz, shift):
    model = _two_segments(1.0 + shift, ConstantVs(2.0), ConstantVs(2.1))
    assert _valid(model, {"type": "max_vs_gradient", "max_abs_km_s_per_km": 0.3})
    # Bounded by the segment slopes alone: 0.4 km/s per km below the jump.
    sloped = _two_segments(1.0 + shift, ConstantVs(2.0), GradientVs(2.1, 2.1 + 0.4 * (4.0 - shift)))
    assert _valid(sloped, {"type": "max_vs_gradient", "max_abs_km_s_per_km": 0.41})
    assert not _valid(sloped, {"type": "max_vs_gradient", "max_abs_km_s_per_km": 0.39})


@pytest.mark.parametrize("shape", ["sigmoid", "smootherstep"])
@pytest.mark.parametrize("shift", SHIFTS_KM)
def test_transition_gradient_is_the_analytic_maximum_slope(shape, shift):
    # Flat neighbours: max dVs/dz = dv * max dS/dz, derived independently in
    # docs/inv/model.md Appendix A: k / 4 / (l(kh) - l(-kh)) for the truncated
    # logistic, 1.875 / (2h) for the smootherstep.
    dv, width = 0.6, 0.8
    half = transition_half_support_km(width, shape)
    if shape == "sigmoid":
        k = 2.0 * np.log(9.0) / width
        logistic = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
        expected = dv * (k / 4.0) / (logistic(k * half) - logistic(-k * half))
    else:
        expected = dv * 1.875 / (2.0 * half)
    model = _two_segments(2.0 + shift, ConstantVs(2.0), ConstantVs(2.0 + dv), width_km=width, shape=shape)

    lowest, highest = vs_slope_extremes_km_s_per_km(model, 0.0, 5.0)

    # The interface depth is a piece edge, so the peak is sampled exactly;
    # 1e-12 allows for rounding only.
    assert highest == pytest.approx(expected, rel=1e-12)
    assert lowest == pytest.approx(0.0, abs=1e-12)
    item = {"type": "max_vs_gradient", "max_abs_km_s_per_km": expected * 1.001}
    assert _valid(model, item)
    assert not _valid(model, {**item, "max_abs_km_s_per_km": expected * 0.999})


# --- monotonic_vs ---------------------------------------------------------


def _slow_decrease():
    # Vs drops 0.5 km/s over 5 km: -0.1 km/s per km everywhere.
    return ParameterizedVsModel((VsSegment(0.0, 5.0, GradientVs(3.0, 2.5)),))


def test_grid_monotonic_tolerance_is_per_step():
    item = {"type": "monotonic_vs", "tolerance": 0.06}
    assert [_valid(_slow_decrease(), {**item, "dz": dz}, evaluation="grid") for dz in GRID_SPACINGS_KM] == [
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("dz", GRID_SPACINGS_KM)
def test_monotonic_rate_does_not_depend_on_spacing(dz):
    assert _valid(_slow_decrease(), {"type": "monotonic_vs", "max_decrease_km_s_per_km": 0.12})
    assert not _valid(_slow_decrease(), {"type": "monotonic_vs", "max_decrease_km_s_per_km": 0.08})


@pytest.mark.parametrize("shift", SHIFTS_KM)
def test_monotonic_checks_the_bottom_of_the_upper_segment(shift):
    # Vs falls 0.5 km/s inside the upper segment, which ends on a grid point
    # when shift = 0; the grid (dz 1) sees only z = 0 and the lower segment.
    depth = 1.0 + shift
    model = _two_segments(depth, GradientVs(3.0, 2.5), ConstantVs(3.0))
    grid = {"type": "monotonic_vs", "tolerance": 0.0, "dz": 1.0}
    if shift == 0.0:
        assert _valid(model, grid, evaluation="grid")
    assert not _valid(model, {"type": "monotonic_vs", "max_decrease_km_s_per_km": 0.1})
    assert _valid(model, {"type": "monotonic_vs", "max_decrease_km_s_per_km": 0.5 / depth + 1e-9})


def test_monotonic_rate_does_not_see_a_hard_drop():
    # Documented blind spot: a drop at a hard interface is a jump, not a slope.
    model = _two_segments(1.0, ConstantVs(3.0), ConstantVs(2.5))
    assert _valid(model, {"type": "monotonic_vs", "max_decrease_km_s_per_km": 0.0})


# --- vs_range -------------------------------------------------------------


@pytest.mark.parametrize("shift", SHIFTS_KM)
def test_vs_range_checks_both_sides_of_a_boundary(shift):
    # The upper segment reaches 3.0 km/s at its bottom; the grid samples the
    # boundary with the lower segment's 2.5 km/s.
    depth = 1.0 + shift
    model = _two_segments(depth, GradientVs(2.0, 3.0), ConstantVs(2.5))
    if shift == 0.0:
        for dz in (0.25, 1.0):
            assert _valid(model, {"type": "vs_range", "bounds": [2.0, 2.9], "dz": dz}, evaluation="grid")
    assert not _valid(model, {"type": "vs_range", "bounds": [2.0, 2.9]})
    assert _valid(model, {"type": "vs_range", "bounds": [2.0, 3.0]})
    assert vs_extremes_km_s(model, 0.0, 5.0) == (2.0, pytest.approx(3.0, abs=1e-15))


def test_jump_at_the_end_of_a_range_counts_only_from_inside():
    model = _two_segments(1.0, ConstantVs(2.0), ConstantVs(3.0))
    assert vs_extremes_km_s(model, 0.0, 1.0) == (2.0, 2.0)
    assert vs_extremes_km_s(model, 1.0, 5.0) == (3.0, 3.0)


# --- the analysis against dense sampling ----------------------------------


@pytest.mark.parametrize("index", range(0, 300, 30))
def test_profile_extremes_match_dense_sampling(index):
    # Independent check: brute-force sampling every 0.5 m. The analysis
    # returns attained values, so it can only be more extreme than the dense
    # grid; the dense grid misses a smooth extreme by at most
    # |v''| (dz/2)^2 / 2 ~ 1e-7 km/s for |v''| < 10 km/s per km^2.
    model = example_models()[index]
    z = np.linspace(0.0, model.zmax, 30001)
    vs = model.evaluate(z)
    # One-sided value at the 1 km hard boundary, from the upper segment.
    upper_bottom = model.segments[0].evaluate(np.array([model.segments[0].bottom]))[0]
    lowest, highest = vs_extremes_km_s(model, 0.0, model.zmax)
    assert lowest <= min(vs.min(), upper_bottom) + 1e-12
    assert highest >= max(vs.max(), upper_bottom) - 1e-12
    assert lowest >= min(vs.min(), upper_bottom) - 1e-6
    assert highest <= max(vs.max(), upper_bottom) + 1e-6

    # Slopes from centred differences inside each segment (jump excluded).
    slopes = []
    for segment in model.segments:
        zs = np.linspace(segment.top, segment.bottom, 20001)
        slopes.append(np.gradient(segment.evaluate(zs), zs, edge_order=2))
    slopes = np.concatenate(slopes)
    low, high = vs_slope_extremes_km_s_per_km(model, 0.0, model.zmax)
    # Second-order differences (also at the ends): error ~ |v'''| h^2 < 1e-6 here.
    assert low == pytest.approx(slopes.min(), abs=1e-5)
    assert high == pytest.approx(slopes.max(), abs=1e-5)


def test_turning_points_give_the_total_variation():
    model = ParameterizedVsModel(
        (VsSegment(0.0, 6.0, BSplineVs([2.0, 3.0, 2.2, 3.4, 2.8], knot_spacing="uniform")),)
    )
    points = vs_turning_points(model, 0.0, 6.0)
    z = np.linspace(0.0, 6.0, 600001)
    drops = -np.diff(model.evaluate(z))
    # Dense total decrease; same sampling bound as above.
    assert points.total_decrease_km_s() == pytest.approx(drops[drops > 0].sum(), abs=1e-6)


# --- compatibility and configuration --------------------------------------


def test_grid_evaluation_reproduces_the_decisions_before_continuous_constraints():
    # Baseline written with commit 8b6babf by tools/make_constraint_grid_baseline.py.
    expected = np.load(BASELINE)["valid"]
    np.testing.assert_array_equal(decisions(all_models(), constraint_items(), evaluation="grid"), expected)


@pytest.mark.parametrize(
    "item, hint",
    [
        ({"type": "monotonic_vs", "tolerance": 0.06, "dz": 0.25}, "0.24 km/s per km"),
        ({"type": "max_vs_gradient", "max_abs": 1.2}, "max_abs_km_s_per_km"),
        ({"type": "vs_range", "bounds": [1.0, 4.0], "dz": 0.25}, "drop dz"),
    ],
)
def test_grid_keys_under_the_continuous_default_name_both_ways_forward(item, hint):
    with pytest.raises(ValueError, match="evaluation: grid") as error:
        constraints_from_config({"constraints": [item]})
    assert hint in str(error.value)


def test_continuous_keys_are_rejected_by_the_grid_evaluation():
    with pytest.raises(ValueError, match="max_decrease_km_s_per_km"):
        constraints_from_config(
            {"evaluation": "grid", "constraints": [{"type": "monotonic_vs", "max_decrease_km_s_per_km": 0.1}]}
        )


def test_constraint_section_may_be_a_bare_list():
    suite = constraints_from_config([{"type": "positive_velocity"}, {"type": "vs_range", "bounds": [1, 4]}])
    assert len(suite.constraints) == 2
    assert suite.evaluation == "continuous"


def test_unknown_evaluation_is_rejected():
    with pytest.raises(ValueError, match="evaluation"):
        constraints_from_config({"evaluation": "midpoint", "constraints": []})
