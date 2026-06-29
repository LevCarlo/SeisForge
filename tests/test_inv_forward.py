import numpy as np

from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_joint,
    predict_rayleigh_hv,
)
from seisforge.inv.io import (
    dispersion_request_from_config,
    rayleigh_hv_request_from_config,
)
from seisforge.inv.model import LayeredModel


def _model():
    return LayeredModel(
        thickness=[1.0, 2.0, 0.0],
        vp=[3.0, 4.0, 5.0],
        vs=[1.5, 2.5, 3.2],
        rho=[2.4, 2.6, 2.8],
    )


def test_predict_rayleigh_phase_dispersion():
    request = DispersionRequest(periods=[5.0, 10.0, 20.0])

    prediction = predict_dispersion(_model(), request)

    np.testing.assert_allclose(prediction.periods, request.periods)
    assert prediction.mode == 0
    assert prediction.wave == "rayleigh"
    assert prediction.kind == "phase"
    assert np.all(np.isfinite(prediction.velocity))
    assert np.all(prediction.velocity > 0)


def test_predict_rayleigh_hv():
    request = RayleighHVRequest(periods=[5.0, 10.0, 20.0])

    prediction = predict_rayleigh_hv(_model(), request)

    np.testing.assert_allclose(prediction.periods, request.periods)
    assert prediction.mode == 0
    assert prediction.observable == "rayleigh_ellipticity"
    assert np.all(np.isfinite(prediction.hv))
    assert np.all(prediction.hv > 0)


def test_predict_joint_accepts_subset_requests():
    joint = predict_joint(_model(), dispersion_request=DispersionRequest([5.0]))

    assert joint.dispersion is not None
    assert joint.hv is None


def test_forward_requests_from_config():
    disp = dispersion_request_from_config(
        {"periods": [5.0, 10.0], "kind": "group", "wave": "rayleigh"}
    )
    hv = rayleigh_hv_request_from_config({"periods": [5.0, 10.0], "mode": 0})

    assert disp.kind == "group"
    assert hv.mode == 0
