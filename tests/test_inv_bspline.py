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


def _uncached_basis(z, *, n_coefficients, degree, domain, spacing, alpha):
    """The pre-cache construction: one single-coefficient spline per column."""
    from scipy.interpolate import BSpline

    z = np.asarray(z, dtype=float)
    x = np.clip((z - domain[0]) / (domain[1] - domain[0]), 0.0, 1.0)
    degree = min(degree, n_coefficients - 1)
    knots = bspline_knots(n_coefficients, degree=degree, spacing=spacing, alpha=alpha)
    basis = np.empty((len(z), n_coefficients))
    for i in range(n_coefficients):
        coefficients = np.zeros(n_coefficients)
        coefficients[i] = 1.0
        basis[:, i] = BSpline(knots, coefficients, degree, extrapolate=False)(x)
    return basis, knots


BASIS_CASES = [
    # (z, n_coefficients, degree, domain, spacing, alpha)
    (np.linspace(0.0, 14.0, 57), 8, 3, (0.0, 14.0), "geometric", 1.6),
    (np.r_[0.125:14.0:0.25], 8, 3, (0.0, 14.0), "geometric", 1.6),
    (np.random.default_rng(3).uniform(0.0, 30.0, 41), 5, 2, (0.0, 30.0), "uniform", 2.0),
    (np.array([0.0, 2.5, 7.0]), 12, 3, (0.0, 7.0), "geometric", 3.0),
    (np.array([1.0]), 3, 3, (0.0, 1.0), "geometric", 2.0),
    (np.linspace(0.0, 1.0, 11), 1, 3, (0.0, 1.0), "uniform", 2.0),
]


def test_cached_basis_is_bit_identical_to_uncached_construction():
    from seisforge.inv.bspline import clear_bspline_basis_cache

    clear_bspline_basis_cache()
    for z, n, degree, domain, spacing, alpha in BASIS_CASES:
        expected, expected_knots = _uncached_basis(
            z, n_coefficients=n, degree=degree, domain=domain, spacing=spacing, alpha=alpha
        )
        coefficients = np.linspace(2.0, 3.5, n)
        for _ in range(2):  # first call fills the cache, second one hits it
            basis, knots = bspline_basis(
                z, n_coefficients=n, degree=degree, domain=domain, spacing=spacing, alpha=alpha
            )
            np.testing.assert_array_equal(basis, expected)
            np.testing.assert_array_equal(knots, expected_knots)
            values = evaluate_bspline(
                z, coefficients, degree=degree, domain=domain, spacing=spacing, alpha=alpha
            )
            np.testing.assert_array_equal(values, expected @ coefficients)


def test_basis_cache_cannot_be_changed_by_callers():
    from seisforge.inv.bspline import _cached_basis

    z = np.linspace(0.0, 10.0, 21)
    z_before = z.copy()
    basis, knots = bspline_basis(z, n_coefficients=6, domain=(0.0, 10.0))
    assert basis.flags.writeable and knots.flags.writeable
    basis[:] = -1.0
    knots[:] = -1.0
    again, again_knots = bspline_basis(z, n_coefficients=6, domain=(0.0, 10.0))
    assert np.all(again >= 0.0) and np.all(again_knots >= 0.0)
    np.testing.assert_array_equal(z, z_before)

    # The cached matrices themselves are read-only.
    x = (z / 10.0).tobytes()
    cached, cached_knots = _cached_basis(x, 6, 3, "geometric", 2.0)
    assert not cached.flags.writeable and not cached_knots.flags.writeable


def test_basis_cache_is_bounded():
    from seisforge.inv.bspline import (
        BSPLINE_BASIS_CACHE_SIZE,
        _cached_basis,
        clear_bspline_basis_cache,
    )

    clear_bspline_basis_cache()
    for i in range(BSPLINE_BASIS_CACHE_SIZE + 20):
        evaluate_bspline(np.array([0.0, 0.5 + 1e-4 * i, 1.0]), np.ones(4), domain=(0.0, 1.0))
    assert _cached_basis.cache_info().currsize == BSPLINE_BASIS_CACHE_SIZE


def test_basis_cache_is_safe_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    from seisforge.inv.bspline import clear_bspline_basis_cache

    clear_bspline_basis_cache()
    z = np.linspace(0.0, 14.0, 57)
    expected, _ = _uncached_basis(
        z, n_coefficients=8, degree=3, domain=(0.0, 14.0), spacing="geometric", alpha=1.6
    )
    coefficients = np.random.default_rng(0).uniform(2.0, 4.0, (64, 8))

    def run(c):
        return evaluate_bspline(z, c, domain=(0.0, 14.0), alpha=1.6)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, coefficients))
    for c, values in zip(coefficients, results, strict=True):
        np.testing.assert_array_equal(values, expected @ c)


def _sta001_prior():
    from pathlib import Path

    from seisforge.inv.config import load_inversion_input
    from seisforge.inv.setup import build_inversion_setup

    example = Path(__file__).resolve().parents[1] / "examples" / "STA001"
    inversion_input = load_inversion_input(
        kind="disp_hv",
        inv_file=example / "inv.yaml",
        obs_file=example / "obs.yaml",
        require_observations=False,
    )
    return build_inversion_setup(inversion_input.run_config)


def test_sta001_prior_chains_match_uncached_basis_for_every_executor(monkeypatch):
    """The cache changes no prior value and no chain, whatever the executor."""
    import dataclasses

    import seisforge.inv.bspline as bspline_module
    from seisforge.inv.samplers import run_metropolis

    setup = _sta001_prior()
    config = dataclasses.replace(setup.sampler.config, n_steps=40, burn_in=10, thin=1,
                                 adaptation=None, global_jump_interval=None)
    theta0 = np.tile(setup.parameterization.theta0(), (2, 1))
    results = {
        executor: run_metropolis(
            setup.prior.log_prior, theta0, config, seeds=7, executor=executor, max_workers=2
        )
        for executor in ("serial", "thread", "process")
    }
    for executor in ("thread", "process"):
        np.testing.assert_array_equal(results[executor].samples, results["serial"].samples)
        np.testing.assert_array_equal(results[executor].log_prob, results["serial"].log_prob)

    cached = [setup.prior.log_prior(theta) for theta in results["serial"].samples.reshape(-1, 10)]

    def uncached(z, *, n_coefficients, degree, domain, spacing, alpha):
        basis, knots = _uncached_basis(
            z, n_coefficients=n_coefficients, degree=degree, domain=domain,
            spacing=spacing, alpha=alpha,
        )
        return basis, knots

    monkeypatch.setattr(bspline_module, "_basis_and_knots", uncached)
    reference = [setup.prior.log_prior(theta) for theta in results["serial"].samples.reshape(-1, 10)]
    assert cached == reference
