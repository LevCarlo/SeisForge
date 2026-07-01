import numpy as np

from seisforge.inv.bspline import (
    bspline_basis,
    bspline_knots,
    evaluate_bspline,
    fit_bspline_coefficients,
)


def test_bspline_basis_is_partition_of_unity():
    z = np.linspace(0.0, 30.0, 61)

    basis, knots = bspline_basis(
        z,
        n_coefficients=5,
        degree=3,
        domain=(0.0, 30.0),
    )

    assert basis.shape == (61, 5)
    assert len(knots) == 9
    np.testing.assert_allclose(basis.sum(axis=1), np.ones_like(z))
    np.testing.assert_allclose(basis[0], np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(basis[-1], np.array([0.0, 0.0, 0.0, 0.0, 1.0]))


def test_geometric_knots_are_denser_near_top():
    knots = bspline_knots(
        n_coefficients=7,
        degree=3,
        spacing="geometric",
        alpha=2.0,
    )
    interior = knots[4:-4]
    intervals = np.diff(np.r_[0.0, interior, 1.0])

    assert np.all(np.diff(intervals) > 0)


def test_evaluate_bspline_uses_coefficients():
    z = np.array([0.0, 5.0, 10.0])

    values = evaluate_bspline(
        z,
        coefficients=[2.0, 4.0],
        degree=3,
        domain=(0.0, 10.0),
    )

    np.testing.assert_allclose(values, np.array([2.0, 3.0, 4.0]))


def test_fit_bspline_coefficients_recovers_smooth_profile():
    z = np.linspace(0.0, 40.0, 81)
    reference = 3.0 + 0.02 * z + 0.12 * np.sin(np.pi * z / 40.0)

    fit = fit_bspline_coefficients(
        z,
        reference,
        n_coefficients=6,
        degree=3,
        domain=(0.0, 40.0),
        spacing="uniform",
    )

    assert fit.coefficients.shape == (6,)
    assert fit.predicted.shape == reference.shape
    assert fit.rms < 0.01


def test_fit_bspline_coefficients_accepts_bounds_and_smoothing():
    z = np.linspace(0.0, 30.0, 31)
    reference = 2.5 + 0.04 * z

    fit = fit_bspline_coefficients(
        z,
        reference,
        n_coefficients=5,
        degree=3,
        domain=(0.0, 30.0),
        bounds=(2.0, 4.0),
        smoothing=0.1,
    )

    assert np.all(fit.coefficients >= 2.0)
    assert np.all(fit.coefficients <= 4.0)
    assert fit.rms < 0.05
