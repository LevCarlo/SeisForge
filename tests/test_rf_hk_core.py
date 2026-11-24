import os

from joblib import load
import numpy as np
from obspy import Trace
import pytest

from seisforge.rf.hk import HkParam, HkStack, hkSeq
from seisforge.rf.hk_core import (
    direct_p_mask,
    hk_grid,
    hk_phase_delays_s,
    hk_pick,
    interpolate_at_times,
    median_and_robust_std,
    velocities_from_vpvs_ratio,
)
from seisforge.rf.traces import RFtrace


def test_linear_interpolation_is_exact_for_a_ramp():
    # Independent check: linear interpolation reproduces a linear function.
    begin_s, delta_s = -10.0, 0.1
    t_axis = begin_s + delta_s * np.arange(201)
    data = 2.0 * t_axis + 1.0
    rng = np.random.default_rng(1)
    times = rng.uniform(-10.0, 10.0, size=(30, 20))
    amplitude, valid = interpolate_at_times(data, times, begin_s, delta_s)
    assert valid.all()
    np.testing.assert_allclose(amplitude, 2.0 * times + 1.0, rtol=0, atol=1e-12)


def test_out_of_range_and_nan_delays_contribute_zero():
    data = np.arange(1.0, 11.0)
    times = np.array([-0.5, 0.0, 4.4, 9.0, 9.6, 30.0, np.nan])
    amplitude, valid = interpolate_at_times(data, times, begin_s=0.0, delta_s=1.0)
    np.testing.assert_array_equal(valid, [False, True, True, True, False, False, False])
    np.testing.assert_allclose(amplitude, [0.0, 1.0, 5.4, 10.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("vmin", "vmax", "step", "expected"),
    [
        (0.0, 1.0, 0.3, [0.0, 0.3, 0.6, 0.9]),        # vmax off the grid: last point below it
        (0.0, 1.0, 0.25, [0.0, 0.25, 0.5, 0.75, 1.0]),  # vmax on the grid: included
        (0.1, 0.3, 0.1, [0.1, 0.2, 0.3]),              # (0.3-0.1)/0.1 = 1.9999999999999998
        (2.0, 2.0, 0.05, [2.0]),
    ],
)
def test_hk_grid_step_is_exactly_the_requested_step(vmin, vmax, step, expected):
    np.testing.assert_allclose(hk_grid(vmin, vmax, step), expected, rtol=0, atol=1e-12)


def test_hk_param_step_equals_dv():
    # Before 0.2.0, linspace(0, 1, int(1 / 0.3) + 1) gave a step of 1/3.
    values = HkParam("H", 0.0, 1.0, dv=0.3).values
    np.testing.assert_allclose(np.diff(values), 0.3, rtol=0, atol=1e-12)
    assert values[-1] <= 1.0


def test_pick_skips_excluded_points_and_flags_the_edge():
    stack = np.array([[5.0, 1.0, 2.0, 0.5]])
    t_ps = np.array([[0.0, 0.2, 0.4, 0.6]])
    excluded = direct_p_mask(t_ps, 0.3)
    np.testing.assert_array_equal(excluded, [[True, True, False, False]])
    assert hk_pick(stack, excluded) == (0, 2, True)
    assert hk_pick(stack, np.zeros_like(excluded)) == (0, 0, False)
    with pytest.raises(ValueError, match="min_ps_delay_s"):
        hk_pick(stack, np.ones_like(excluded))


def test_robust_std_estimates_gaussian_sigma():
    rng = np.random.default_rng(3)
    center, spread = median_and_robust_std(rng.normal(5.0, 2.0, 40000))
    # Standard error of the scaled MAD is about 1.17 sigma / sqrt(n) = 0.012.
    assert abs(center - 5.0) < 0.05
    assert abs(spread - 2.0) < 0.05
    assert median_and_robust_std(np.array([3.0])) == (3.0, 0.0)


def test_vertical_incidence_delays_match_closed_form():
    k = np.array([1.70, 1.75, 1.80])
    vp, vs = velocities_from_vpvs_ratio(k, vp_km_s=6.3)
    thickness = np.array([30.0, 35.0])
    t_ps, t_ppps, t_psps = hk_phase_delays_s(thickness, vp, vs, 0.0)
    np.testing.assert_allclose(t_ps, np.outer(1 / vs - 1 / vp, thickness))
    np.testing.assert_allclose(t_ppps, np.outer(1 / vs + 1 / vp, thickness))
    np.testing.assert_allclose(t_psps, np.outer(2 / vs, thickness))


def test_two_layer_delays_reduce_to_one_layer_for_identical_upper_layer():
    k = np.linspace(1.6, 2.0, 5)
    vp, vs = velocities_from_vpvs_ratio(k, vs_km_s=3.6)
    thickness = np.linspace(20.0, 40.0, 4)
    one = hk_phase_delays_s(thickness, vp, vs, 0.06)
    two = hk_phase_delays_s(
        thickness, vp, vs, 0.06,
        upper_thickness_km=5.0, upper_vp_km_s=vp, upper_vs_km_s=vs,
    )
    for single, layered in zip(one, two):
        np.testing.assert_allclose(layered, single)


def _rf_trace(npts=301, slowness=0.065, seed=0):
    rng = np.random.default_rng(seed)
    trace = Trace(data=rng.normal(size=npts).astype(np.float32))
    trace.stats.delta = 0.05
    trace.stats.sac = {"b": -10.0, "delta": 0.05, "npts": npts, "user0": slowness}
    return trace


def test_hk_stack_matches_np_interp_and_reports_invalid_points():
    rf = RFtrace(trace=_rf_trace(), slowness=0.065)
    hk = HkStack(
        RFtrace=rf,
        H=HkParam("H", 20.0, 60.0, dv=1.0),
        k=HkParam("k", 1.6, 2.0, dv=0.05),
        min_ps_delay_s=0.0,
    )
    ds = hk.stack()
    data = hk._amp_correct(method="seispy")
    t_axis = -10.0 + 0.05 * np.arange(301)
    for phase in ("Ps", "PpPs", "PsPs"):
        delays = ds[f"t_{phase}"].values
        reference = np.interp(delays, t_axis, data)
        inside = delays <= t_axis[-1]
        np.testing.assert_allclose(
            ds[f"amp_{phase}"].values[inside], reference[inside], rtol=0, atol=1e-6
        )  # float32 trace data
        assert np.all(ds[f"amp_{phase}"].values[~inside] == 0)
        assert ds.attrs[f"n_invalid_{phase}"] == int(np.count_nonzero(~inside))
    assert ds.attrs["n_invalid_PsPs"] > 0


def _hk_config(root, save="hkSeq"):
    grid = {
        "H": {"min": 20.0, "max": 30.0, "delta": 2.0},
        "k": {"min": 1.7, "max": 1.8, "delta": 0.05},
        "Vp": 6.3,
        "Vs": None,
        "weight": [0.5, 0.2, 0.3],
        "min_ps_delay_s": 0.3,
    }
    qc = {"mahalanobis_dist_threshold": 10.0}
    return {
        "IO": {
            "ROOT": str(root),
            "HighFreq_RF": "high",
            "LowFreq_RF": "low",
            "SAVE": save,
            "FIGURE": "figures",
            "LOG": "logs",
        },
        "Meta": {"station": "STA", "network": "NET"},
        "Model": {"Sediment": {**grid, "two_layer_min_thickness_km": 0.4}, "Crust": grid},
        "QC": {"HighFreq": qc, "LowFreq": qc},
    }


def _write_rfs(root):
    for name in ("high", "low"):
        os.makedirs(root / name)
        for seed in range(3):
            _rf_trace(npts=1201, seed=seed).write(str(root / name / f"rf{seed}.sac"), format="SAC")


@pytest.mark.parametrize("save", [".", "high", "high/out", ".."])
def test_hkseq_refuses_output_dir_overlapping_root_or_inputs(tmp_path, save):
    root = tmp_path / "NET.STA"
    _write_rfs(root)
    with pytest.raises(ValueError, match="IO.SAVE"):
        hkSeq(_hk_config(root, save=save), mode="low")
    assert sorted(os.listdir(root / "high")) == ["rf0.sac", "rf1.sac", "rf2.sac"]


def test_hkseq_requires_overwrite_and_keeps_other_stage_outputs(tmp_path, assert_provenance):
    root = tmp_path / "NET.STA"
    _write_rfs(root)
    config = _hk_config(root)
    hkSeq(config, mode="high")
    high_dat = root / "hkSeq" / "NET.STA_HFreq_Hk_QC.dat"
    assert high_dat.exists()
    assert (root / "logs" / "NET.STA_hkSeq.log").read_text().count("Starting H-k") == 1
    assert_provenance((root / "logs" / "NET.STA_hkSeq.log").read_text())
    assert_provenance(high_dat.read_text())
    high_results = load(root / "hkSeq" / "NET.STA_HFreq_Hk_results.joblib")
    assert high_results
    for result in high_results:
        assert_provenance(result.attrs)

    with pytest.raises(FileExistsError, match="overwrite"):
        hkSeq(config, mode="high")

    hkSeq(config, mode="low")
    assert high_dat.exists()
    assert (root / "hkSeq" / "NET.STA_LFreq_Hk_QC.dat").exists()

    hkSeq(config, mode="high", overwrite=True)
