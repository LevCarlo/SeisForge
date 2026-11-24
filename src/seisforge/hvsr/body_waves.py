"""Diffuse P--SV and SH body-wave Green-function contributions.

The body-wave terms are (eq. 2 of `docs/hvsr/index.md`)

    -Im integral_0^{omega/Vs_N} k C(k) dk

for ``C = C_xx, C_zz, C_yy``, evaluated on the path ``k (1 - i eps)`` just
below the real axis (the outgoing side of the branch cuts, ``exp(-i omega t)``).

Quadrature: adaptive Gauss--Kronrod (7-point Gauss, 15-point Kronrod; the
QUADPACK QK15 rule) in ``u = k Vs_N / omega``, starting from segments split at
every branch point ``u = Vs_N / v_j`` of the model. The interval with the
largest error relative to the target is bisected until, for each of the three
terms, the summed estimate ``|K15 - G7|`` is below ``relative_tolerance``
times the magnitude of that term. The error is measured on the complex
integrand, not only on its imaginary part: a pole close to the real axis (a
weakly leaking mode) shows up in the real part as a ``1/(k - k_p)`` tail at
any distance, while its imaginary part is a narrow peak that a coarse rule can
miss entirely. The Fortran kernel runs the same algorithm (same rule, the
same choice of interval, the same order of the interval list).

Failure policy: if ``max_intervals`` is reached first, the integral is kept
and the frequency carries a `ContributionIssue` (status ``INCOMPLETE``).
Negative energy beyond round-off raises ``RuntimeError``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import assemble_diffuse_field_prediction
from .models import (
    ContributionIssue,
    DiffuseFieldPrediction,
    ElasticLayerModel,
    GreenFunctionContributions,
    HorizontalDefinition,
)
from .propagator import psv_surface_compliance, sh_surface_compliance

# QUADPACK QK15 abscissae (x >= 0, descending) and weights (Piessens et al.
# 1983); the 7-point Gauss nodes are the odd entries and the centre.
_KRONROD_NODES = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
)
_KRONROD_WEIGHTS = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583938629992,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_GAUSS_WEIGHTS = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)
# Full 15-point rule on [-1, 1], ascending.
_GK_NODES = np.concatenate((-_KRONROD_NODES[:-1], _KRONROD_NODES[::-1]))
_GK_KRONROD = np.concatenate((_KRONROD_WEIGHTS[:-1], _KRONROD_WEIGHTS[::-1]))
_GK_GAUSS = np.zeros(15)
_GK_GAUSS[[1, 3, 5, 7, 9, 11, 13]] = np.concatenate(
    (_GAUSS_WEIGHTS[:-1], [_GAUSS_WEIGHTS[-1]], _GAUSS_WEIGHTS[-2::-1])
)


@dataclass(frozen=True)
class BodyWaveSettings:
    """Numerical controls for the body-wave integral.

    ``relative_tolerance`` bounds the Gauss--Kronrod error estimate of each
    term relative to the term (the actual error is typically 10-100 times
    smaller); ``max_intervals`` caps the adaptive subdivision;
    ``contour_shift_fraction`` is ``eps`` of the path ``k (1 - i eps)``. The
    result approaches the real-axis limit linearly in ``eps`` (measured
    ``|dI/I| ~ 10-100 eps`` on the documented models), so the default
    ``1e-8`` keeps that bias below the default tolerance.
    """

    relative_tolerance: float = 1.0e-6
    max_intervals: int = 4000
    contour_shift_fraction: float = 1.0e-8

    def __post_init__(self) -> None:
        if not np.isfinite(self.relative_tolerance) or not (
            1.0e-12 <= self.relative_tolerance <= 1.0e-1
        ):
            raise ValueError("relative_tolerance must lie in [1e-12, 0.1]")
        if self.max_intervals < 1:
            raise ValueError("max_intervals must be at least one")
        if (
            not np.isfinite(self.contour_shift_fraction)
            or self.contour_shift_fraction <= 0.0
            or self.contour_shift_fraction >= 0.1
        ):
            raise ValueError("contour_shift_fraction must lie between 0 and 0.1")


def body_wave_segments(model: ElasticLayerModel) -> np.ndarray:
    """Boundaries in ``u = k Vs_N / omega`` of the initial quadrature segments.

    Every P/S vertical wavenumber has a branch point at ``k = omega / v``,
    where the integrand has a square-root kink; splitting there lets the
    rule converge quickly on each piece.
    """

    vs_halfspace = model.vs_km_s[-1]
    critical = np.concatenate((vs_halfspace / model.vp_km_s, vs_halfspace / model.vs_km_s))
    return np.unique(
        np.concatenate(([0.0], critical[(critical > 0.0) & (critical < 1.0)], [1.0]))
    )


def _integrand(
    model: ElasticLayerModel,
    frequency_hz: float,
    unit_nodes: np.ndarray,
    shift: complex,
    k_limit: float,
) -> np.ndarray:
    """``k C dk/du`` for ``(C_xx, C_zz, C_yy)`` at the nodes; shape ``(n, 3)``."""

    values = np.empty((unit_nodes.size, 3), dtype=np.complex128)
    for sample, unit in enumerate(unit_nodes):
        k = k_limit * unit * shift
        psv = psv_surface_compliance(model, frequency_hz, k)
        values[sample, 0] = psv[0, 0]
        values[sample, 1] = psv[1, 1]
        values[sample, 2] = sh_surface_compliance(model, frequency_hz, k)
        values[sample] *= k * k_limit * shift
    return values


def _gauss_kronrod_interval(integrand, left: float, right: float):
    half = 0.5 * (right - left)
    values = integrand(left + half * (_GK_NODES + 1.0))
    kronrod = half * (_GK_KRONROD @ values)
    gauss = half * (_GK_GAUSS @ values)
    return kronrod, np.abs(kronrod - gauss)


def _adaptive_body_integral(integrand, boundaries, relative_tolerance, max_intervals):
    """Return ``(integral, error, intervals, converged)`` for three terms.

    ``integral`` is complex with shape ``(3,)``; ``error`` the summed
    ``|K15 - G7|``. The interval list keeps its order: a bisected interval is
    replaced by its left half and the right half is appended (as in the
    Fortran kernel, so both sum in the same order).
    """

    lefts, rights, integrals, errors = [], [], [], []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        value, error = _gauss_kronrod_interval(integrand, left, right)
        lefts.append(left)
        rights.append(right)
        integrals.append(value)
        errors.append(error)
    while True:
        total = np.sum(integrals, axis=0)
        error = np.sum(errors, axis=0)
        target = relative_tolerance * np.abs(total.imag)
        if np.all(error <= target):
            return total, error, len(lefts), True
        if len(lefts) >= max_intervals:
            return total, error, len(lefts), False
        scale = np.where(target > 0.0, target, np.finfo(float).tiny)
        worst = int(np.argmax(np.max(np.asarray(errors) / scale, axis=1)))
        left, right = lefts[worst], rights[worst]
        middle = 0.5 * (left + right)
        left_value, left_error = _gauss_kronrod_interval(integrand, left, middle)
        right_value, right_error = _gauss_kronrod_interval(integrand, middle, right)
        rights[worst], integrals[worst], errors[worst] = middle, left_value, left_error
        lefts.append(middle)
        rights.append(right)
        integrals.append(right_value)
        errors.append(right_error)


def body_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: BodyWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Integrate body-wave terms over ``0 < k < omega / Vs_halfspace``.

    See the module docstring for the quadrature. The horizontal terms are
    already halved (azimuthal average); the H/V definition (sum or mean) is
    applied later, in `combine_green_function_contributions`. Frequencies at
    which the tolerance was not reached carry a ``body_waves`` issue.
    Returned arrays contain no surface-wave pole residues.
    """

    settings = settings or BodyWaveSettings()
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")

    shift = 1.0 - 1j * settings.contour_shift_fraction
    boundaries = body_wave_segments(model)
    psv_horizontal = np.empty(frequencies.size)
    psv_vertical = np.empty(frequencies.size)
    sh_horizontal = np.empty(frequencies.size)
    issues = []

    for index, frequency in enumerate(frequencies):
        k_limit = 2.0 * np.pi * frequency / model.vs_km_s[-1]

        def integrand(unit_nodes, frequency=frequency, k_limit=k_limit):
            return _integrand(model, frequency, unit_nodes, shift, k_limit)

        total, error, intervals, converged = _adaptive_body_integral(
            integrand, boundaries, settings.relative_tolerance, settings.max_intervals
        )
        # The common Fourier factor cancels in H/V. The angular average gives
        # 1/2 for each horizontal polarization and 1 for the vertical term.
        psv_horizontal[index] = -total[0].imag / 2.0
        psv_vertical[index] = -total[1].imag
        sh_horizontal[index] = -total[2].imag / 2.0
        if not converged:
            with np.errstate(divide="ignore", invalid="ignore"):
                relative_error = float(np.max(error / np.abs(total.imag)))
            issues.append(
                _unconverged_issue(
                    float(frequency), relative_error, intervals, settings.relative_tolerance
                )
            )

    return _checked_body_contributions(
        psv_horizontal, psv_vertical, sh_horizontal, tuple(issues)
    )


def _unconverged_issue(frequency_hz, relative_error, intervals, relative_tolerance):
    """The ``body_waves`` issue for a frequency that missed the tolerance."""

    return ContributionIssue(
        frequency_hz=frequency_hz,
        term="body_waves",
        mode=None,
        reason=(
            f"quadrature not converged: estimated relative error {relative_error:.1e} "
            f"> {relative_tolerance:g} after {intervals} intervals"
        ),
    )


def _checked_body_contributions(psv_horizontal, psv_vertical, sh_horizontal, issues):
    """Reject negative energy beyond round-off; clip round-off negatives to 0."""

    contribution_matrix = np.column_stack((psv_horizontal, psv_vertical, sh_horizontal))
    scale = np.max(np.abs(contribution_matrix), axis=1, keepdims=True)
    roundoff = 100.0 * np.finfo(float).eps * np.maximum(scale, 1.0)
    if np.any(contribution_matrix < -roundoff):
        raise RuntimeError(
            "body-wave integration produced negative spectral energy; "
            "lower relative_tolerance or check the model"
        )
    contribution_matrix[contribution_matrix < 0.0] = 0.0
    psv_horizontal, psv_vertical, sh_horizontal = contribution_matrix.T
    zeros = np.zeros_like(psv_horizontal)
    return GreenFunctionContributions(
        rayleigh_horizontal=zeros,
        rayleigh_vertical=zeros,
        love_horizontal=zeros,
        psv_horizontal=psv_horizontal,
        psv_vertical=psv_vertical,
        sh_horizontal=sh_horizontal,
        issues=issues,
    )


def predict_body_wave_hvsr(
    model: ElasticLayerModel,
    frequencies_hz,
    settings: BodyWaveSettings | None = None,
    *,
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM,
) -> DiffuseFieldPrediction:
    """Predict the explicitly labelled body-wave-only DFA approximation."""

    contributions = body_wave_green_contributions(
        model, frequencies_hz, settings=settings or BodyWaveSettings()
    )
    return assemble_diffuse_field_prediction(
        frequencies_hz,
        surface=None,
        body=contributions,
        horizontal_definition=horizontal_definition,
        backend="native_body_waves",
    )
