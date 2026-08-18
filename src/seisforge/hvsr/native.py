"""Complete native diffuse-field H/V solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_waves import BodyWaveSettings, body_wave_green_contributions
from .core import combine_green_function_contributions
from .models import (
    DiffuseFieldPrediction,
    DiffuseFieldSettings,
    ElasticLayerModel,
    GreenFunctionContributions,
)
from .modes import SurfaceWaveSettings, surface_wave_green_contributions


def _sum_contributions(
    left: GreenFunctionContributions,
    right: GreenFunctionContributions,
) -> GreenFunctionContributions:
    return GreenFunctionContributions(
        **{
            name: getattr(left, name) + getattr(right, name)
            for name in left.__dataclass_fields__
        }
    )


@dataclass(frozen=True)
class NativeDiffuseFieldSolver:
    """Native solver combining pole residues and finite body-wave integrals."""

    body_quadrature_order: int = 48
    contour_shift_fraction: float = 1.0e-5
    dispersion_step_km_s: float = 0.001
    residue_step_fraction: float = 1.0e-5

    def __call__(
        self,
        model: ElasticLayerModel,
        frequencies_hz,
        settings: DiffuseFieldSettings | None = None,
    ) -> DiffuseFieldPrediction:
        settings = settings or DiffuseFieldSettings()
        if settings.max_rayleigh_modes > 0 or settings.max_love_modes > 0:
            surface = surface_wave_green_contributions(
                model,
                frequencies_hz,
                settings=SurfaceWaveSettings(
                    max_rayleigh_modes=settings.max_rayleigh_modes,
                    max_love_modes=settings.max_love_modes,
                    dispersion_step_km_s=self.dispersion_step_km_s,
                    residue_step_fraction=self.residue_step_fraction,
                ),
            )
        else:
            zeros = np.zeros_like(np.asarray(frequencies_hz, dtype=float))
            surface = GreenFunctionContributions(
                rayleigh_horizontal=zeros,
                rayleigh_vertical=zeros,
                love_horizontal=zeros,
                psv_horizontal=zeros,
                psv_vertical=zeros,
                sh_horizontal=zeros,
            )
        if settings.body_wave_wavenumbers > 0:
            # Retain the old setting as a computational-effort hint while the
            # native backend uses much faster per-segment Gaussian quadrature.
            requested_order = max(8, round(settings.body_wave_wavenumbers / 2))
            order = max(self.body_quadrature_order, requested_order)
            body = body_wave_green_contributions(
                model,
                frequencies_hz,
                settings=BodyWaveSettings(
                    quadrature_order=order,
                    contour_shift_fraction=self.contour_shift_fraction,
                    horizontal_definition=settings.horizontal_definition,
                ),
            )
            contributions = _sum_contributions(surface, body)
            approximation = "full_dfa"
        else:
            contributions = surface
            approximation = "surface_waves_only"

        result = combine_green_function_contributions(
            frequencies_hz,
            contributions,
            horizontal_definition=settings.horizontal_definition,
            backend="native",
        )
        return DiffuseFieldPrediction(
            frequency_hz=result.frequency_hz,
            hvsr=result.hvsr,
            backend=result.backend,
            approximation=approximation,
            horizontal_definition=result.horizontal_definition,
            horizontal_energy=result.horizontal_energy,
            vertical_energy=result.vertical_energy,
            contributions=result.contributions,
            diagnostics=result.diagnostics,
        )
