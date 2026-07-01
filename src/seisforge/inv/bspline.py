"""B-spline basis utilities for 1-D Vs parameterizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class BSplineFit:
    """Result of fitting B-spline coefficients to a depth profile."""

    coefficients: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    rms: float
    knots: np.ndarray
    degree: int


def bspline_knots(
    n_coefficients: int,
    *,
    degree: int = 3,
    spacing: str = "geometric",
    alpha: float = 2.0,
) -> np.ndarray:
    """Return an open clamped knot vector on the normalized interval [0, 1]."""
    if n_coefficients <= 0:
        raise ValueError("n_coefficients must be positive.")
    degree = _effective_degree(n_coefficients, degree)
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    n_interior = n_coefficients - degree - 1
    if n_interior <= 0:
        interior = np.array([], dtype=float)
    else:
        spacing = spacing.lower()
        if spacing == "uniform" or np.isclose(alpha, 1.0):
            interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
        elif spacing == "geometric":
            lengths = alpha ** np.arange(n_interior + 1, dtype=float)
            interior = np.cumsum(lengths / lengths.sum())[:-1]
        else:
            raise ValueError("spacing must be 'uniform' or 'geometric'.")

    return np.r_[
        np.zeros(degree + 1),
        interior,
        np.ones(degree + 1),
    ]


def bspline_basis(
    z: ArrayLike,
    *,
    n_coefficients: int,
    degree: int = 3,
    domain: tuple[float, float] | None = None,
    spacing: str = "geometric",
    alpha: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate B-spline basis functions at depth samples.

    Returns
    -------
    basis, knots
        ``basis`` has shape ``(len(z), n_coefficients)``. ``knots`` is the
        normalized open clamped knot vector used to build the basis.
    """
    z = _as_1d_array(z, name="z")
    zmin, zmax = _domain_from_z(z, domain)
    x = (z - zmin) / (zmax - zmin)
    if np.any((x < -1e-12) | (x > 1.0 + 1e-12)):
        raise ValueError("z samples must fall inside domain.")
    x = np.clip(x, 0.0, 1.0)

    degree = _effective_degree(n_coefficients, degree)
    knots = bspline_knots(
        n_coefficients,
        degree=degree,
        spacing=spacing,
        alpha=alpha,
    )
    basis = np.empty((len(z), n_coefficients), dtype=float)
    for i in range(n_coefficients):
        coefficients = np.zeros(n_coefficients, dtype=float)
        coefficients[i] = 1.0
        spline = BSpline(knots, coefficients, degree, extrapolate=False)
        basis[:, i] = spline(x)

    if np.any(~np.isfinite(basis)):
        raise ValueError("B-spline basis evaluation produced non-finite values.")
    return basis, knots


def evaluate_bspline(
    z: ArrayLike,
    coefficients: ArrayLike,
    *,
    degree: int = 3,
    domain: tuple[float, float] | None = None,
    spacing: str = "geometric",
    alpha: float = 2.0,
) -> np.ndarray:
    """Evaluate a B-spline profile from coefficients."""
    coefficients = _as_1d_array(coefficients, name="coefficients")
    basis, _ = bspline_basis(
        z,
        n_coefficients=len(coefficients),
        degree=degree,
        domain=domain,
        spacing=spacing,
        alpha=alpha,
    )
    return basis @ coefficients


def fit_bspline_coefficients(
    z: ArrayLike,
    values: ArrayLike,
    *,
    n_coefficients: int,
    degree: int = 3,
    domain: tuple[float, float] | None = None,
    spacing: str = "geometric",
    alpha: float = 2.0,
    weights: ArrayLike | None = None,
    bounds: tuple[ArrayLike, ArrayLike] | None = None,
    smoothing: float = 0.0,
) -> BSplineFit:
    """Fit B-spline coefficients to a sampled depth profile.

    ``smoothing`` adds a second-difference penalty to the coefficients. This is
    useful when the reference profile is noisy or when the number of
    coefficients is high relative to the number of samples.
    """
    z = _as_1d_array(z, name="z")
    values = _as_1d_array(values, name="values")
    if len(z) != len(values):
        raise ValueError("z and values must have the same length.")
    if len(z) < 2:
        raise ValueError("At least two samples are required.")
    if n_coefficients <= 0:
        raise ValueError("n_coefficients must be positive.")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")

    basis, knots = bspline_basis(
        z,
        n_coefficients=n_coefficients,
        degree=degree,
        domain=domain,
        spacing=spacing,
        alpha=alpha,
    )
    matrix = basis
    rhs = values

    if weights is not None:
        weights = _as_1d_array(weights, name="weights")
        if len(weights) != len(z):
            raise ValueError("weights must have the same length as z.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        matrix = matrix * weights[:, None]
        rhs = rhs * weights

    if smoothing > 0 and n_coefficients >= 3:
        regularizer = np.diff(np.eye(n_coefficients), n=2, axis=0)
        matrix = np.vstack([matrix, np.sqrt(smoothing) * regularizer])
        rhs = np.r_[rhs, np.zeros(regularizer.shape[0])]

    if bounds is None:
        coefficients, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    else:
        lower, upper = bounds
        lower = _broadcast_bounds(lower, n_coefficients, name="lower")
        upper = _broadcast_bounds(upper, n_coefficients, name="upper")
        if np.any(lower >= upper):
            raise ValueError("Each lower bound must be smaller than upper bound.")
        result = lsq_linear(matrix, rhs, bounds=(lower, upper))
        if not result.success:
            raise ValueError(f"B-spline fit failed: {result.message}")
        coefficients = result.x

    predicted = basis @ coefficients
    residual = values - predicted
    rms = float(np.sqrt(np.mean(residual**2)))
    return BSplineFit(
        coefficients=np.asarray(coefficients, dtype=float),
        predicted=predicted,
        residual=residual,
        rms=rms,
        knots=knots,
        degree=_effective_degree(n_coefficients, degree),
    )


def _as_1d_array(values: ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if len(array) == 0:
        raise ValueError(f"{name} cannot be empty.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _broadcast_bounds(values: ArrayLike, length: int, *, name: str) -> np.ndarray:
    values = _as_1d_array(values, name=name)
    if len(values) == 1:
        return np.full(length, values[0], dtype=float)
    if len(values) != length:
        raise ValueError(f"{name} must be scalar or have length {length}.")
    return values


def _domain_from_z(
    z: np.ndarray,
    domain: tuple[float, float] | None,
) -> tuple[float, float]:
    if domain is None:
        zmin = float(z.min())
        zmax = float(z.max())
    else:
        zmin, zmax = map(float, domain)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("domain must be a finite increasing (zmin, zmax) pair.")
    return zmin, zmax


def _effective_degree(n_coefficients: int, degree: int) -> int:
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    return min(int(degree), n_coefficients - 1)
