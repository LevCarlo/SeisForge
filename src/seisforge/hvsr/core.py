"""Backend-independent diffuse-field H/V relations.

`combine_green_function_contributions` turns the six Green-function terms
into H/V (Sánchez-Sesma et al. 2011, eq. 1 of `docs/hvsr/index.md`);
`assemble_diffuse_field_prediction` is the step every solver shares: add the
surface- and body-wave parts that were computed, label the approximation, and
combine. Neither knows how the terms were computed.
"""

from __future__ import annotations

import numpy as np

from .models import (
    GREEN_FUNCTION_TERMS,
    ContributionIssue,
    DiffuseFieldPrediction,
    ForwardStatus,
    FrequencyDiagnostic,
    GreenFunctionContributions,
    HorizontalDefinition,
)

_INVALID_ENERGY = "non-finite/negative horizontal or non-positive vertical energy"


def _frequency_diagnostic(
    frequency_hz: float,
    valid: bool,
    issues: tuple[ContributionIssue, ...],
) -> FrequencyDiagnostic:
    """OK only when the energies are valid and no term was left out."""

    notes = [issue.describe() for issue in issues]
    if not valid:
        status = ForwardStatus.INVALID
        notes.insert(0, _INVALID_ENERGY)
    elif issues:
        status = ForwardStatus.INCOMPLETE
    else:
        status = ForwardStatus.OK
    return FrequencyDiagnostic(
        frequency_hz=frequency_hz,
        status=status,
        message="; ".join(notes),
        issues=issues,
    )


def combine_green_function_contributions(
    frequency_hz,
    contributions: GreenFunctionContributions,
    *,
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM,
    backend: str = "native",
    approximation: str = "full_dfa",
) -> DiffuseFieldPrediction:
    """Combine surface- and body-wave terms using the DFA H/V definition.

    ``SUM`` reproduces HV-DFA and Sánchez-Sesma et al. (2011): the two
    horizontal components are summed. ``MEAN`` divides that horizontal energy
    by two, and therefore gives an H/V smaller by ``sqrt(2)``.

    Each frequency's diagnostic is ``INVALID`` (H/V is NaN) when the energies
    are not usable, ``INCOMPLETE`` when ``contributions.issues`` lists a term
    left out or not converged at that frequency, and ``OK`` otherwise.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if frequency.ndim != 1 or frequency.size != contributions.rayleigh_horizontal.size:
        raise ValueError("frequency_hz must be one-dimensional and match contributions")
    horizontal_definition = HorizontalDefinition(horizontal_definition)

    horizontal = 2.0 * (
        contributions.rayleigh_horizontal
        + contributions.love_horizontal
        + contributions.psv_horizontal
        + contributions.sh_horizontal
    )
    if horizontal_definition is HorizontalDefinition.MEAN:
        horizontal = horizontal / 2.0
    vertical = contributions.rayleigh_vertical + contributions.psv_vertical

    valid = (
        np.isfinite(horizontal)
        & np.isfinite(vertical)
        & (horizontal >= 0.0)
        & (vertical > 0.0)
    )
    hvsr = np.full(frequency.shape, np.nan, dtype=np.float64)
    hvsr[valid] = np.sqrt(horizontal[valid] / vertical[valid])
    diagnostics = tuple(
        _frequency_diagnostic(
            float(value),
            bool(ok),
            tuple(i for i in contributions.issues if i.frequency_hz == float(value)),
        )
        for value, ok in zip(frequency, valid, strict=True)
    )
    return DiffuseFieldPrediction(
        frequency_hz=frequency,
        hvsr=hvsr,
        horizontal_energy=horizontal,
        vertical_energy=vertical,
        contributions=contributions,
        diagnostics=diagnostics,
        backend=backend,
        approximation=approximation,
        horizontal_definition=horizontal_definition,
    )


def assemble_diffuse_field_prediction(
    frequency_hz,
    *,
    surface: GreenFunctionContributions | None,
    body: GreenFunctionContributions | None,
    horizontal_definition: HorizontalDefinition | str,
    backend: str,
) -> DiffuseFieldPrediction:
    """Add the computed parts and combine them into H/V.

    ``surface`` and ``body`` are None when that part was not requested. The
    approximation label follows from which parts are present: ``full_dfa``,
    ``surface_waves_only`` or ``body_waves_only``. Issues of both parts are
    kept.
    """

    parts = [part for part in (surface, body) if part is not None]
    if not parts:
        raise ValueError("at least one of surface and body contributions is required")
    total = GreenFunctionContributions(
        **{name: sum(getattr(part, name) for part in parts) for name in GREEN_FUNCTION_TERMS},
        issues=tuple(issue for part in parts for issue in part.issues),
    )
    if surface is not None and body is not None:
        approximation = "full_dfa"
    elif surface is not None:
        approximation = "surface_waves_only"
    else:
        approximation = "body_waves_only"
    return combine_green_function_contributions(
        frequency_hz,
        total,
        horizontal_definition=horizontal_definition,
        backend=backend,
        approximation=approximation,
    )
