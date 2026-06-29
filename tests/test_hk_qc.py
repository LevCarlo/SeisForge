import numpy as np
import xarray as xr

from seisforge.rf.hk import HkQController


def _hk_result(h=10.0, k=1.8, energy=1.0):
    return xr.Dataset(
        data_vars={"amp_stack": (["k", "H"], np.array([[energy]]))},
        coords={"H": [h], "k": [k]},
        attrs={"H_best": h, "k_best": k, "amp_max": energy},
    )


def test_hk_qc_handles_empty_results():
    controller = HkQController([], {"energy_threshold": 0.5})

    mask, selected = controller.hk_energy_select(return_mask_only=False)

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
