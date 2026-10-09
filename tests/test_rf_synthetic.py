import numpy as np
import pytest
from scipy.signal import find_peaks

from seisforge.inv import (
    DiscretizationConfig,
    ForwardError,
    GradientVs,
    Interface,
    LayeredModel,
    ParameterizedVsModel,
    ReceiverFunctionRequest,
    ScalingConfig,
    VsSegment,
    predict_receiver_function,
)
from seisforge.rf.hk_core import hk_phase_delays_s
from seisforge.rf.synthetic import p_wave_surface_response, receiver_function


CRUST = dict(
    thickness_km=np.array([35.0, 0.0]),
    vp_km_s=np.array([6.3, 8.1]),
    vs_km_s=np.array([3.6, 4.5]),
)
CRUST["rho_g_cm3"] = 0.77 + 0.32 * CRUST["vp_km_s"]


def _rf(model=CRUST, p=0.06, **kwargs):
    options = dict(dt_s=0.05, n_samples=1200, gauss_a=2.5)
    options.update(kwargs)
    return receiver_function(
        model["thickness_km"],
        model["vp_km_s"],
        model["vs_km_s"],
        model["rho_g_cm3"],
        p,
        **options,
    )


def _amplitude_near(time, rf, target_s, half_window=4):
    index = int(np.argmin(np.abs(time - target_s)))
    window = slice(index - half_window, index + half_window + 1)
    return rf[window][np.argmax(np.abs(rf[window]))]


def test_half_space_ratio_matches_free_surface_solution():
    vp, vs, rho, p = 6.3, 3.6, 2.8, 0.06
    omega = np.array([0.0, 1.0, 30.0])
    radial, vertical = p_wave_surface_response(
        np.array([0.0]), np.array([vp]), np.array([vs]), np.array([rho]), p, omega
    )
    q_s = np.sqrt(1 / vs**2 - p**2)
    expected = 2 * p * q_s / (1 / vs**2 - 2 * p**2)
    np.testing.assert_allclose(radial / vertical, expected, rtol=1e-12)


def test_converted_phase_times_and_polarities_match_hk_formulas():
    p = 0.06
    time, rf = _rf(p=p)
    t_ps, t_ppps, t_psps = (
        float(delay[0, 0])
        for delay in hk_phase_delays_s(
            np.array([35.0]), np.array([6.3]), np.array([3.6]), p
        )
    )
    peaks, _ = find_peaks(np.abs(rf), height=0.1 * np.abs(rf).max())
    arrivals = time[peaks]
    for target in (0.0, t_ps, t_ppps, t_psps):
        assert np.min(np.abs(arrivals - target)) <= 0.05
    assert _amplitude_near(time, rf, 0.0) > 0
    assert _amplitude_near(time, rf, t_ps) > 0
    assert _amplitude_near(time, rf, t_ppps) > 0
    assert _amplitude_near(time, rf, t_psps) < 0


def test_pygrt_reference_amplitudes():
    # PyGRT 0.17.2 (grt rcvfn, independent generalized R/T code) for this
    # model, p = 0.06 s/km, a = 2.5, dt = 0.05 s: direct P 0.6562, and
    # Ps/P 0.2962, PpPs/P 0.3311, PsPs/P -0.2755.
    time, rf = _rf(p=0.06)
    direct = _amplitude_near(time, rf, 0.0)
    assert direct == pytest.approx(0.6562, abs=2e-4)
    assert _amplitude_near(time, rf, 4.35) / direct == pytest.approx(0.2962, abs=1e-3)
    assert _amplitude_near(time, rf, 14.65) / direct == pytest.approx(0.3311, abs=1e-3)
    assert _amplitude_near(time, rf, 19.0) / direct == pytest.approx(-0.2755, abs=1e-3)


def test_homogeneous_layers_reduce_to_the_half_space():
    model = dict(
        thickness_km=np.array([2.0, 5.0, 10.0, 0.0]),
        vp_km_s=np.full(4, 6.3),
        vs_km_s=np.full(4, 3.6),
        rho_g_cm3=np.full(4, 2.8),
    )
    half_space = dict(
        thickness_km=np.array([0.0]),
        vp_km_s=np.array([6.3]),
        vs_km_s=np.array([3.6]),
        rho_g_cm3=np.array([2.8]),
    )
    np.testing.assert_allclose(_rf(model)[1], _rf(half_space)[1], atol=1e-10)


def test_vanishing_layer_converges_to_model_without_it():
    reference = _rf()[1]
    for thickness, tolerance in ((0.01, 0.01), (0.0001, 1e-4)):
        model = dict(
            thickness_km=np.array([thickness, 35.0, 0.0]),
            vp_km_s=np.array([2.0, 6.3, 8.1]),
            vs_km_s=np.array([1.0, 3.6, 4.5]),
            rho_g_cm3=np.array([2.0, *CRUST["rho_g_cm3"]]),
        )
        assert np.max(np.abs(_rf(model)[1] - reference)) < tolerance * np.abs(reference).max()


def test_density_enters_the_conversions():
    lighter_mantle = dict(CRUST, rho_g_cm3=np.array([2.8, 3.0]))
    heavier_mantle = dict(CRUST, rho_g_cm3=np.array([2.8, 3.5]))
    time, a = _rf(lighter_mantle)
    _, b = _rf(heavier_mantle)
    assert _amplitude_near(time, a, 0.0) == pytest.approx(_amplitude_near(time, b, 0.0), rel=1e-6)
    assert abs(_amplitude_near(time, b, 4.35)) > abs(_amplitude_near(time, a, 4.35))


def test_gaussian_peak_amplitude_recovers_the_spike_amplitude():
    vp, vs, p = 6.3, 3.6, 0.06
    half_space = dict(
        thickness_km=np.array([0.0]),
        vp_km_s=np.array([vp]),
        vs_km_s=np.array([vs]),
        rho_g_cm3=np.array([2.8]),
    )
    q_s = np.sqrt(1 / vs**2 - p**2)
    ratio = 2 * p * q_s / (1 / vs**2 - 2 * p**2)
    _, impulse = _rf(half_space, p=p, gauss_a=2.5, dt_s=0.01, n_samples=2000)
    _, peak = _rf(half_space, p=p, gauss_a=2.5, dt_s=0.01, n_samples=2000, amplitude="gaussian_peak")
    assert impulse.max() == pytest.approx(ratio * 2.5 / np.sqrt(np.pi), rel=1e-3)
    assert peak.max() == pytest.approx(ratio, rel=1e-3)


def test_padding_does_not_change_the_window():
    _, default = _rf()
    _, padded = _rf(n_fft=16384)
    np.testing.assert_allclose(default, padded, atol=1e-6)


def test_evanescent_p_is_rejected():
    with pytest.raises(ValueError, match="evanescent"):
        _rf(p=0.13)


def test_predict_receiver_function_uses_layered_model_and_reports_errors():
    mantle = ScalingConfig(
        vp_method="constant_vp_vs", vp_kwargs={"vp_vs": 1.8},
        rho_method="constant", rho_kwargs={"rho": 3.35},
    )
    model = ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 35.0, GradientVs(3.4, 3.8), name="crust"),
            VsSegment(35.0, 80.0, GradientVs(4.4, 4.5), name="mantle", scaling=mantle),
        ),
        interfaces=(Interface(35.0, width_km=4.0),),
    )
    layered = model.to_layered_model(DiscretizationConfig(layers_per_segment=20))
    request = ReceiverFunctionRequest(ray_parameter_s_per_km=0.06, dt_s=0.1, n_samples=400, gauss_a=1.0)
    prediction = predict_receiver_function(layered, request)
    assert prediction.rf.shape == (400,)
    assert prediction.time_s[0] == pytest.approx(-5.0)
    assert np.all(np.isfinite(prediction.rf))

    # p = 0.075 s/km is evanescent in a 14 km/s half-space (1/14 = 0.071).
    too_fast = LayeredModel(thickness=[1.0, 0.0], vp=[6.0, 14.0], vs=[3.5, 7.0], rho=[2.7, 3.5])
    steep = ReceiverFunctionRequest(ray_parameter_s_per_km=0.075, dt_s=0.1, n_samples=400, gauss_a=1.0)
    with pytest.raises(ForwardError, match="evanescent"):
        predict_receiver_function(too_fast, steep)
