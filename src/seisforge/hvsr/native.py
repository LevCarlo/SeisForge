"""Complete native diffuse-field H/V solver."""

from __future__ import annotations

from dataclasses import dataclass

from .body_waves import BodyWaveSettings, body_wave_green_contributions
from .core import assemble_diffuse_field_prediction
from .models import (
    DiffuseFieldPrediction,
    DiffuseFieldSettings,
    ElasticLayerModel,
)
from .modes import SurfaceWaveSettings, surface_wave_green_contributions


@dataclass(frozen=True)
class _KernelControls:
    """Accuracy controls shared by the native and Fortran solvers.

    See `docs/hvsr/green_function.md`, sections 3-4:

    - ``body_quadrature_rtol``: Gauss--Kronrod error estimate of each
      body-wave term, relative to the term.
    - ``body_quadrature_max_intervals``: cap on the adaptive subdivision; a
      frequency that reaches it is labelled ``INCOMPLETE``.
    - ``contour_shift_fraction``: ``eps`` of the path ``k (1 - i eps)``; the
      bias towards the real-axis limit is linear in ``eps``.
    - ``dispersion_step_km_s``: disba phase-velocity step for mode seeds.
    - ``residue_step_fraction``: central-difference step of the residues.
    """

    body_quadrature_rtol: float = 1.0e-6
    body_quadrature_max_intervals: int = 4000
    contour_shift_fraction: float = 1.0e-8
    dispersion_step_km_s: float = 0.001
    residue_step_fraction: float = 1.0e-5

    def surface_wave_settings(self, settings: DiffuseFieldSettings) -> SurfaceWaveSettings:
        return SurfaceWaveSettings(
            max_rayleigh_modes=settings.max_rayleigh_modes,
            max_love_modes=settings.max_love_modes,
            dispersion_step_km_s=self.dispersion_step_km_s,
            residue_step_fraction=self.residue_step_fraction,
        )

    def body_wave_settings(self) -> BodyWaveSettings:
        return BodyWaveSettings(
            relative_tolerance=self.body_quadrature_rtol,
            max_intervals=self.body_quadrature_max_intervals,
            contour_shift_fraction=self.contour_shift_fraction,
        )


@dataclass(frozen=True)
class NativeDiffuseFieldSolver(_KernelControls):
    """Native solver combining pole residues and adaptive body-wave integrals.

    Fields are the accuracy controls listed in `_KernelControls` (defaults:
    ``body_quadrature_rtol=1e-6``, ``body_quadrature_max_intervals=4000``,
    ``contour_shift_fraction=1e-8``, ``dispersion_step_km_s=0.001``,
    ``residue_step_fraction=1e-5``).
    """

    def __call__(
        self,
        model: ElasticLayerModel,
        frequencies_hz,
        settings: DiffuseFieldSettings | None = None,
    ) -> DiffuseFieldPrediction:
        settings = settings or DiffuseFieldSettings()
        surface = body = None
        if settings.max_rayleigh_modes > 0 or settings.max_love_modes > 0:
            surface = surface_wave_green_contributions(
                model, frequencies_hz, settings=self.surface_wave_settings(settings)
            )
        if settings.include_body_waves:
            body = body_wave_green_contributions(
                model, frequencies_hz, settings=self.body_wave_settings()
            )
        return assemble_diffuse_field_prediction(
            frequencies_hz,
            surface=surface,
            body=body,
            horizontal_definition=settings.horizontal_definition,
            backend="native",
        )
