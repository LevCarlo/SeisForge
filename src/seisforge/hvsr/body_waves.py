"""Diffuse P--SV and SH body-wave Green-function contributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import combine_green_function_contributions
from .models import (
    DiffuseFieldPrediction,
    ElasticLayerModel,
    GreenFunctionContributions,
    HorizontalDefinition,
)
from .propagator import psv_surface_compliance, sh_surface_compliance


@dataclass(frozen=True)
class BodyWaveSettings:
    """Numerical controls for the finite body-wave contour integral."""

    quadrature_order: int = 24
    contour_shift_fraction: float = 1.0e-5
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM

    def __post_init__(self) -> None:
        if self.quadrature_order < 8:
            raise ValueError("quadrature_order must be at least 8 per segment")
        if (
            not np.isfinite(self.contour_shift_fraction)
            or self.contour_shift_fraction <= 0.0
            or self.contour_shift_fraction >= 0.1
        ):
            raise ValueError("contour_shift_fraction must lie between 0 and 0.1")
        object.__setattr__(
            self,
            "horizontal_definition",
            HorizontalDefinition(self.horizontal_definition),
        )


def body_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: BodyWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Integrate body-wave terms over ``0 < k < omega / Vs_halfspace``.

    Gauss--Legendre nodes avoid both branch points. The small fourth-quadrant
    contour shift enforces the outgoing-wave side of the real-axis cuts.
    Returned arrays contain no surface-wave pole residues.
    """

    settings = settings or BodyWaveSettings()
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")

    nodes, weights = np.polynomial.legendre.leggauss(settings.quadrature_order)
    shift = 1.0 - 1j * settings.contour_shift_fraction

    psv_horizontal = np.empty(frequencies.size)
    psv_vertical = np.empty(frequencies.size)
    sh_horizontal = np.empty(frequencies.size)
    vs_halfspace = model.vs_km_s[-1]

    for index, frequency in enumerate(frequencies):
        omega = 2.0 * np.pi * frequency
        k_limit = omega / vs_halfspace
        # Every P/S vertical wavenumber has a branch point at omega/velocity.
        # Splitting there makes fixed Gaussian quadrature converge rapidly.
        critical = np.concatenate(
            (vs_halfspace / model.vp_km_s, vs_halfspace / model.vs_km_s)
        )
        boundaries = np.unique(
            np.concatenate(([0.0], critical[(critical > 0.0) & (critical < 1.0)], [1.0]))
        )
        unit_nodes_parts = []
        unit_weight_parts = []
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            unit_nodes_parts.append(0.5 * ((right - left) * nodes + right + left))
            unit_weight_parts.append(0.5 * (right - left) * weights)
        unit_nodes = np.concatenate(unit_nodes_parts)
        unit_weights = np.concatenate(unit_weight_parts)
        k = k_limit * unit_nodes * shift
        dk_weights = k_limit * unit_weights * shift
        radial = np.empty(k.size, dtype=np.complex128)
        vertical = np.empty(k.size, dtype=np.complex128)
        transverse = np.empty(k.size, dtype=np.complex128)
        for sample, wavenumber in enumerate(k):
            psv = psv_surface_compliance(model, frequency, wavenumber)
            radial[sample] = psv[0, 0]
            vertical[sample] = psv[1, 1]
            transverse[sample] = sh_surface_compliance(
                model, frequency, wavenumber
            )

        # The common Fourier factor cancels in H/V. The angular average gives
        # 1/2 for each horizontal polarization and 1 for the vertical term.
        psv_horizontal[index] = (
            -np.imag(np.sum(dk_weights * k * radial)) / 2.0
        )
        psv_vertical[index] = -np.imag(np.sum(dk_weights * k * vertical))
        sh_horizontal[index] = (
            -np.imag(np.sum(dk_weights * k * transverse)) / 2.0
        )

    contribution_matrix = np.column_stack(
        (psv_horizontal, psv_vertical, sh_horizontal)
    )
    scale = np.max(np.abs(contribution_matrix), axis=1, keepdims=True)
    roundoff = 100.0 * np.finfo(float).eps * np.maximum(scale, 1.0)
    if np.any(contribution_matrix < -roundoff):
        raise RuntimeError(
            "body-wave integration produced negative spectral energy; "
            "increase quadrature_order or contour_shift_fraction"
        )
    contribution_matrix[contribution_matrix < 0.0] = 0.0
    psv_horizontal, psv_vertical, sh_horizontal = contribution_matrix.T

    zeros = np.zeros_like(frequencies)
    return GreenFunctionContributions(
        rayleigh_horizontal=zeros,
        rayleigh_vertical=zeros,
        love_horizontal=zeros,
        psv_horizontal=psv_horizontal,
        psv_vertical=psv_vertical,
        sh_horizontal=sh_horizontal,
    )


def predict_body_wave_hvsr(
    model: ElasticLayerModel,
    frequencies_hz,
    settings: BodyWaveSettings | None = None,
) -> DiffuseFieldPrediction:
    """Predict the explicitly labelled body-wave-only DFA approximation."""

    settings = settings or BodyWaveSettings()
    contributions = body_wave_green_contributions(
        model, frequencies_hz, settings=settings
    )
    result = combine_green_function_contributions(
        frequencies_hz,
        contributions,
        horizontal_definition=settings.horizontal_definition,
        backend="native_body_waves",
    )
    return DiffuseFieldPrediction(
        frequency_hz=result.frequency_hz,
        hvsr=result.hvsr,
        backend=result.backend,
        approximation="body_waves_only",
        horizontal_definition=result.horizontal_definition,
        horizontal_energy=result.horizontal_energy,
        vertical_energy=result.vertical_energy,
        contributions=result.contributions,
        diagnostics=result.diagnostics,
    )
