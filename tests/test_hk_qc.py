import numpy as np
import pytest
from obspy import Trace
import xarray as xr

from seisforge.rf.hk import HkParam, HkQController, HkStack
from seisforge.rf.hk_workflow import _estimate_hk_summary
from seisforge.rf.traces import RFtrace


def _hk_result(h=10.0, k=1.8, energy=1.0):
    return xr.Dataset(
        data_vars={"amp_stack": (["k", "H"], np.array([[energy]]))},
        coords={"H": [h], "k": [k]},
        attrs={"H_best": h, "k_best": k, "amp_max": energy},
    )


def test_hk_qc_handles_empty_results():
    controller = HkQController([], {"mahalanobis_dist_threshold": 1.0})

    mask, selected = controller.mahalanobis_dist_select(return_mask_only=False)

    assert mask.size == 0
    assert selected == []


def test_mahalanobis_select_keeps_single_result():
    controller = HkQController(
        [_hk_result()],
        {"mahalanobis_dist_threshold": 1.0},
    )

    mask, selected = controller.mahalanobis_dist_select(return_mask_only=False)

    np.testing.assert_array_equal(mask, np.array([True]))
    assert len(selected) == 1


def test_hk_param_requires_grid_definition():
    try:
        HkParam("H", 0.0, 1.0)
    except ValueError as exc:
        assert "Either dv or values" in str(exc)
    else:
        raise AssertionError("HkParam should require either dv or values")


def test_hk_summary_returns_nan_for_empty_mask():
    summary = _estimate_hk_summary([_hk_result()], np.array([False]))

    assert all(np.isnan(value) for value in summary)


@pytest.mark.parametrize("n", [5, 14, 15, 40])
def test_hk_summary_is_median_and_scaled_mad_for_every_n(n):
    # Before 0.2.0: mean and population std for n < 15, median and unscaled
    # MAD for n >= 15 (a factor of about 1.5 apart for Gaussian scatter).
    rng = np.random.default_rng(n)
    h = rng.normal(35.0, 0.5, n)
    k = rng.normal(1.75, 0.02, n)
    results = [_hk_result(hi, ki) for hi, ki in zip(h, k)]
    h_best, k_best, h_spread, k_spread = _estimate_hk_summary(results, np.ones(n, bool))
    assert h_best == np.median(h) and k_best == np.median(k)
    np.testing.assert_allclose(h_spread, 1.4826 * np.median(np.abs(h - np.median(h))))
    np.testing.assert_allclose(k_spread, 1.4826 * np.median(np.abs(k - np.median(k))))


def test_hk_stack_max_amp_correction_handles_zero_window_amplitude():
    tr = Trace(data=np.zeros(11))
    tr.stats.sac = {"b": -5.0, "delta": 1.0, "npts": 11}
    rf = RFtrace(trace=tr, slowness=0.06)
    hk = HkStack(
        RFtrace=rf,
        H=HkParam("H", 1.0, 2.0, dv=1.0),
        k=HkParam("k", 1.7, 1.8, dv=0.1),
        min_ps_delay_s=0.0,
    )

    corrected = hk._amp_correct(method="max_amp")

    np.testing.assert_allclose(corrected, np.zeros(11))
