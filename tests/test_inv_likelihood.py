import numpy as np

from seisforge.inv.forward import (
    DispersionRequest,
    RayleighHVRequest,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.likelihood import (
    DispersionObservation,
    RayleighHVObservation,
    evaluate_joint_likelihood,
    gaussian_observable_fit,
)
from seisforge.inv.model import LayeredModel


def _model():
    return LayeredModel(
        thickness=[1.0, 2.0, 0.0],
        vp=[3.0, 4.0, 5.0],
        vs=[1.5, 2.5, 3.2],
        rho=[2.4, 2.6, 2.8],
    )


def test_gaussian_observable_fit_computes_weighted_loglike():
    fit = gaussian_observable_fit(
        periods=[5.0, 10.0],
        observed=[3.0, 4.0],
        predicted=[2.9, 4.2],
        sigma=[0.1, 0.2],
        weight=2.0,
    )

    np.testing.assert_allclose(fit.residual, np.array([0.1, -0.2]))
    np.testing.assert_allclose(fit.normalized_residual, np.array([1.0, -1.0]))
    np.testing.assert_allclose(fit.chi2, 2.0)
    np.testing.assert_allclose(fit.misfit, 1.0)
    np.testing.assert_allclose(fit.rms, fit.misfit)
    np.testing.assert_allclose(fit.weighted_chi2, 4.0)
    np.testing.assert_allclose(fit.loglike, -2.0)
    assert fit.n_data == 2


def test_joint_likelihood_is_zero_for_self_consistent_observations():
    model = _model()
    periods = np.array([5.0, 10.0, 20.0])
    dispersion_prediction = predict_dispersion(
        model,
        DispersionRequest(periods=periods),
    )
    hv_prediction = predict_rayleigh_hv(model, RayleighHVRequest(periods=periods))

    result = evaluate_joint_likelihood(
        model,
        dispersion=DispersionObservation(
            periods=periods,
            velocity=dispersion_prediction.velocity,
            sigma=np.full_like(periods, 0.05),
        ),
        hv=RayleighHVObservation(
            periods=periods,
            hv=hv_prediction.hv,
            sigma=np.full_like(periods, 0.1),
        ),
        dispersion_weight=1.0,
        hv_weight=2.0,
    )

    assert result.success
    assert result.error is None
    assert result.chi2 == 0.0
    assert result.raw_chi2 == 0.0
    assert result.misfit == 0.0
    assert result.n_data == 6
    assert result.loglike == 0.0
    assert result.dispersion is not None
    assert result.hv is not None


def test_joint_likelihood_accepts_single_observable():
    model = _model()
    periods = np.array([5.0, 10.0])
    prediction = predict_dispersion(model, DispersionRequest(periods=periods))

    result = evaluate_joint_likelihood(
        model,
        dispersion=DispersionObservation(
            periods=periods,
            velocity=prediction.velocity + 0.05,
            sigma=np.full_like(periods, 0.05),
        ),
    )

    assert result.success
    assert result.hv is None
    assert result.dispersion is not None
    np.testing.assert_allclose(result.dispersion.normalized_residual, np.ones(2))
    np.testing.assert_allclose(result.chi2, 2.0)
    np.testing.assert_allclose(result.raw_chi2, 2.0)
    np.testing.assert_allclose(result.misfit, 1.0)
    assert result.n_data == 2
    np.testing.assert_allclose(result.loglike, -1.0)


def test_joint_likelihood_tracks_raw_and_weighted_chi2_separately():
    fit = gaussian_observable_fit(
        periods=[5.0, 10.0],
        observed=[3.0, 4.0],
        predicted=[2.9, 4.2],
        sigma=[0.1, 0.2],
        weight=3.0,
    )

    np.testing.assert_allclose(fit.chi2, 2.0)
    np.testing.assert_allclose(fit.weighted_chi2, 6.0)
    np.testing.assert_allclose(fit.loglike, -3.0)


def test_observation_requires_matching_lengths():
    try:
        DispersionObservation(periods=[5.0, 10.0], velocity=[3.0], sigma=[0.1, 0.1])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("DispersionObservation should validate array lengths")
