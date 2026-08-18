"""Backend-independent diffuse-field H/V relations."""

from __future__ import annotations

import numpy as np

from .models import (
    DiffuseFieldPrediction,
    ForwardStatus,
    FrequencyDiagnostic,
    GreenFunctionContributions,
    HorizontalDefinition,
)


def combine_green_function_contributions(
    frequency_hz,
    contributions: GreenFunctionContributions,
    *,
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM,
    backend: str = "native",
) -> DiffuseFieldPrediction:
    """Combine surface- and body-wave terms using the DFA H/V definition.

    ``SUM`` reproduces HV-DFA and Sánchez-Sesma et al. (2011): the two
    horizontal components are summed. ``MEAN`` divides that horizontal energy
    by two, and therefore gives an H/V smaller by ``sqrt(2)``.
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
        FrequencyDiagnostic(
            frequency_hz=float(value),
            status=ForwardStatus.OK if ok else ForwardStatus.INVALID,
            message="" if ok else "non-finite/negative horizontal or non-positive vertical energy",
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
        horizontal_definition=horizontal_definition,
    )
