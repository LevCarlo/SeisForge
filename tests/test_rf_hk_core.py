import os

import numpy as np
from obspy import Trace
import pytest

from seisforge.rf.hk import HkParam, HkStack, hkSeq
from seisforge.rf.hk_core import (
    hk_phase_delays_s,
    nearest_sample_index,
    sample_at_times,
    velocities_from_vpvs_ratio,
)
from seisforge.rf.traces import RFtrace


def _argmin_reference(time_s, begin_s, delta_s, npts):
    t_axis = begin_s + delta_s * np.arange(npts)
    flat = np.asarray(time_s, dtype=float).ravel()
    return np.abs(t_axis[:, None] - flat[None, :]).argmin(axis=0).reshape(np.shape(time_s))


def test_nearest_sample_index_matches_argmin_including_ties():
    rng = np.random.default_rng(1)
    begin_s, delta_s, npts = -10.0, 0.05, 1201
    times = rng.uniform(-10.0, 50.0, size=(40, 30))
    # Exact midpoints between samples: argmin keeps the earlier sample.
    times[0, :10] = begin_s + delta_s * (np.arange(10) + 100.5)
    index, valid = nearest_sample_index(times, begin_s, delta_s, npts)
    assert valid.all()
    np.testing.assert_array_equal(index, _argmin_reference(times, begin_s, delta_s, npts))


def test_out_of_range_and_nan_delays_contribute_zero():
    data = np.arange(1.0, 11.0)
    times = np.array([-5.0, 0.0, 4.4, 9.0, 9.6, 30.0, np.nan])
    amplitude, valid = sample_at_times(data, times, begin_s=0.0, delta_s=1.0)
    np.testing.assert_array_equal(valid, [False, True, True, True, False, False, False])
    np.testing.assert_array_equal(amplitude, [0.0, 1.0, 5.0, 10.0, 0.0, 0.0, 0.0])


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


def test_hk_stack_matches_argmin_reference_and_reports_invalid_points():
    rf = RFtrace(trace=_rf_trace(), slowness=0.065)
    hk = HkStack(
        RFtrace=rf,
        H=HkParam("H", 20.0, 60.0, dv=1.0),
        k=HkParam("k", 1.6, 2.0, dv=0.05),
    )
    ds = hk.stack()
    data = hk._amp_correct(method="seispy")
    t_axis_end = -10.0 + 0.05 * 300
    for phase in ("Ps", "PpPs", "PsPs"):
        delays = ds[f"t_{phase}"].values
        reference = data[_argmin_reference(delays, -10.0, 0.05, 301)]
        inside = delays <= t_axis_end + 0.025
        np.testing.assert_array_equal(ds[f"amp_{phase}"].values[inside], reference[inside])
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
    }
    qc = {"energy_threshold": 0.1, "ell_mad_sigma": 5.0, "mahalanobis_dist_threshold": 10.0}
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
        "Model": {"Sediment": grid, "Crust": grid},
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


def test_hkseq_requires_overwrite_and_keeps_other_stage_outputs(tmp_path):
    root = tmp_path / "NET.STA"
    _write_rfs(root)
    config = _hk_config(root)
    hkSeq(config, mode="high")
    high_dat = root / "hkSeq" / "NET.STA_HFreq_Hk_QC.dat"
    assert high_dat.exists()
    assert (root / "logs" / "NET.STA_hkSeq.log").read_text().count("Starting H-k") == 1

    with pytest.raises(FileExistsError, match="overwrite"):
        hkSeq(config, mode="high")

    hkSeq(config, mode="low")
    assert high_dat.exists()
    assert (root / "hkSeq" / "NET.STA_LFreq_Hk_QC.dat").exists()

    hkSeq(config, mode="high", overwrite=True)
