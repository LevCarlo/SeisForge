import numpy as np
import pytest

from seisforge.hvsr import (
    BodyWaveSettings,
    ElasticLayerModel,
    predict_body_wave_hvsr,
)
from seisforge.hvsr.propagator import (
    psv_surface_compliance,
    sh_surface_compliance,
    vertical_decay_wavenumber,
)


def test_radiation_branch_decays_or_propagates_downward() -> None:
    evanescent = vertical_decay_wavenumber(2.0, 2.0, 2.0)
    propagating = vertical_decay_wavenumber(0.0, 2.0, 2.0)

    assert evanescent.real > 0.0
    assert propagating.imag < 0.0


def test_halfspace_normal_incidence_compliance_is_analytic() -> None:
    model = ElasticLayerModel([], [5.5], [3.2], [2.7])
    frequency_hz = 1.7
    omega = 2.0 * np.pi * frequency_hz

    psv = psv_surface_compliance(model, frequency_hz, 0.0)
    sh = sh_surface_compliance(model, frequency_hz, 0.0)

    np.testing.assert_allclose(psv[0, 0], -1j / (2.7 * 3.2 * omega))
    np.testing.assert_allclose(psv[1, 1], -1j / (2.7 * 5.5 * omega))
    np.testing.assert_allclose(psv[0, 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(psv[1, 0], 0.0, atol=1e-15)
    np.testing.assert_allclose(sh, -1j / (2.7 * 3.2 * omega))


def test_halfspace_body_wave_hv_is_frequency_independent() -> None:
    model = ElasticLayerModel([], [5.5], [3.2], [2.7])
    result = predict_body_wave_hvsr(model, [0.1, 1.0, 10.0])

    np.testing.assert_allclose(result.hvsr, result.hvsr[0], rtol=1e-12)
    assert result.approximation == "body_waves_only"


def test_boundary_system_remains_finite_for_thick_high_frequency_layer() -> None:
    model = ElasticLayerModel([100.0], [2.0, 6.0], [1.0, 3.5], [2.0, 2.8])

    psv = psv_surface_compliance(model, 50.0, 400.0 - 0.01j)
    sh = sh_surface_compliance(model, 50.0, 400.0 - 0.01j)

    assert np.all(np.isfinite(psv))
    assert np.isfinite(sh)


def test_layered_body_wave_result_matches_hvdfa_reference_at_converged_points() -> None:
    model = ElasticLayerModel(
        [0.5, 1.0, 3.0],
        [1.8, 2.2, 3.2, 5.5],
        [0.5, 1.0, 1.8, 3.2],
        [1.9, 2.1, 2.4, 2.7],
    )
    result = predict_body_wave_hvsr(model, [0.2, 1.0], BodyWaveSettings())

    # HV-DFA with 10,000 real-axis samples gives 5.34253 and 1.21937. The
    # converged integral agrees to 2e-4 (was 3e-3 with the fixed rule); the
    # rest is HV-DFA's own sampling and precision (-prec 0.1 %).
    np.testing.assert_allclose(result.hvsr, [5.34253, 1.21937], rtol=5e-4)


# --- T21: the body-wave integral near a nearly real pole -------------------
#
# Model of examples/05_hvsr_forward. Below its cut-off (~1.08 Hz) Rayleigh
# mode 1 continues as a leaky pole inside 0 < k < omega/Vs_N whose leakage
# vanishes near 0.9523 Hz (half width ~ (f - 0.9523)^2): the integrand has a
# peak narrower than 1e-6 of the interval, which a fixed Gauss rule cannot
# resolve (it gave errors of 4-90 % here, and 7-16 % at 3-10 Hz on the
# four-layer model).

def _t21_model() -> ElasticLayerModel:
    return ElasticLayerModel([0.05], [0.6, 2.0], [0.2, 1.0], [1.8, 2.2])


def _quadpack_body_terms(model, frequency_hz, shift_fraction):
    """Independent reference: scipy's QAGS (21-point rule with extrapolation)
    on each term and segment, to a relative 1e-11."""

    from scipy.integrate import quad

    from seisforge.hvsr.body_waves import body_wave_segments

    k_limit = 2.0 * np.pi * frequency_hz / model.vs_km_s[-1]
    shift = 1.0 - 1j * shift_fraction

    def term(unit, which):
        k = k_limit * unit * shift
        if which == 2:
            value = sh_surface_compliance(model, frequency_hz, k)
        else:
            value = psv_surface_compliance(model, frequency_hz, k)[which, which]
        return -np.imag(k_limit * shift * k * value)

    boundaries = body_wave_segments(model)
    totals = []
    for which in range(3):
        total = 0.0
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            total += quad(
                term, left, right, args=(which,), limit=4000, epsabs=0.0, epsrel=1e-11
            )[0]
        totals.append(total)
    return np.array(totals) * np.array([0.5, 1.0, 0.5])


@pytest.mark.parametrize("frequency_hz", [0.955, 0.97])
def test_body_integral_matches_independent_quadrature_near_leaky_pole(frequency_hz):
    from seisforge.hvsr.body_waves import body_wave_green_contributions

    model = _t21_model()
    result = body_wave_green_contributions(model, [frequency_hz])
    actual = np.array(
        [result.psv_horizontal[0], result.psv_vertical[0], result.sh_horizontal[0]]
    )
    expected = _quadpack_body_terms(model, frequency_hz, 1.0e-8)
    # The requested tolerance (1e-6) bounds the Gauss-Kronrod error estimate;
    # the measured error is ~1e-8. Before: 4 % (0.97 Hz) and 90 % (0.955 Hz).
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert result.issues == ()


def test_contour_shift_bias_is_linear_and_small_by_default():
    model = _t21_model()
    exact = _quadpack_body_terms(model, 2.0, 0.0)  # real axis: no pole at 2 Hz
    for shift, bound in ((1.0e-5, 2e-3), (1.0e-8, 2e-6)):
        from seisforge.hvsr.body_waves import body_wave_green_contributions

        result = body_wave_green_contributions(
            model, [2.0], settings=BodyWaveSettings(contour_shift_fraction=shift)
        )
        actual = np.array(
            [result.psv_horizontal[0], result.psv_vertical[0], result.sh_horizontal[0]]
        )
        # Measured bias 8e-4 at eps = 1e-5 and 1e-6 at 1e-8 (|dI/I| ~ 80 eps).
        assert np.max(np.abs(actual / exact - 1.0)) < bound


def test_unconverged_body_quadrature_is_reported():
    from seisforge.hvsr.body_waves import body_wave_green_contributions

    result = body_wave_green_contributions(
        _t21_model(), [0.955, 2.0], settings=BodyWaveSettings(max_intervals=3)
    )
    assert [issue.frequency_hz for issue in result.issues] == [0.955, 2.0]
    assert all(issue.term == "body_waves" and issue.mode is None for issue in result.issues)
    assert "not converged" in result.issues[0].reason


@pytest.mark.slow
def test_peak_height_converges_near_f0():
    from seisforge.hvsr import NativeDiffuseFieldSolver

    frequencies = np.linspace(0.94, 0.97, 31)
    default = NativeDiffuseFieldSolver()(_t21_model(), frequencies)
    tight = NativeDiffuseFieldSolver(body_quadrature_rtol=1e-9)(_t21_model(), frequencies)
    # Before: H/V changed by up to a factor of 3 with the quadrature order.
    np.testing.assert_allclose(default.hvsr, tight.hvsr, rtol=1e-6)
    assert all(d.status.value == "ok" for d in default.diagnostics)
    # No spikes: the second difference is small compared with the values.
    second = np.abs(np.diff(default.hvsr, 2))
    assert np.max(second) < 0.05 * np.min(default.hvsr)
