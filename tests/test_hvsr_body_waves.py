import numpy as np

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
    settings = BodyWaveSettings(
        quadrature_order=96,
        contour_shift_fraction=1.0e-8,
    )
    result = predict_body_wave_hvsr(model, [0.2, 1.0], settings)

    # HV-DFA with 10,000 real-axis samples gives 5.34253 and 1.21937.
    np.testing.assert_allclose(result.hvsr, [5.34253, 1.21937], rtol=3e-3)
