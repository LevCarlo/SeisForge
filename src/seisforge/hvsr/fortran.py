"""Python boundary for the optional f2py HVSR numerical kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_waves import BodyWaveSettings
from .core import combine_green_function_contributions
from .dispersion import phase_velocity_seeds_km_s
from .models import (
    DiffuseFieldPrediction,
    DiffuseFieldSettings,
    ElasticLayerModel,
    GreenFunctionContributions,
)
from .modes import SurfaceWaveSettings


class FortranExtensionUnavailable(ImportError):
    """Raised when the optional f2py extension has not been built."""


def fortran_extension_available() -> bool:
    """Return whether the compiled f2py extension can be imported."""

    try:
        from . import _hvsr_fortran  # noqa: F401
    except ImportError:
        return False
    return True


def fortran_body_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: BodyWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Evaluate the DFA body-wave contour integral in the Fortran kernel."""

    try:
        from . import _hvsr_fortran
    except ImportError as error:
        raise FortranExtensionUnavailable(
            "the f2py HVSR extension is not built; run "
            "`python tools/build_hvsr_fortran.py` from the project root"
        ) from error

    settings = settings or BodyWaveSettings()
    frequencies = np.ascontiguousarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")

    psv_horizontal, psv_vertical, sh_horizontal, status = (
        _hvsr_fortran.hvsr_core.body_wave_contributions(
            model.thickness_km,
            model.vp_km_s,
            model.vs_km_s,
            model.density_g_cm3,
            frequencies,
            settings.quadrature_order,
            settings.contour_shift_fraction,
        )
    )
    if status != 0:
        messages = {
            1: "invalid model dimensions or quadrature order",
            2: "singular P-SV boundary system",
            3: "singular SH boundary system",
        }
        raise RuntimeError(
            "Fortran body-wave calculation failed: "
            f"{messages.get(int(status), f'unknown status {status}')}"
        )

    contribution_matrix = np.column_stack(
        (psv_horizontal, psv_vertical, sh_horizontal)
    )
    scale = np.max(np.abs(contribution_matrix), axis=1, keepdims=True)
    roundoff = 100.0 * np.finfo(float).eps * np.maximum(scale, 1.0)
    if np.any(contribution_matrix < -roundoff):
        raise RuntimeError("Fortran body-wave integration produced negative energy")
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


def fortran_surface_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: SurfaceWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Track modal seeds with disba and evaluate pole residues in Fortran."""

    try:
        from . import _hvsr_fortran
    except ImportError as error:
        raise FortranExtensionUnavailable(
            "the f2py HVSR extension is not built; run "
            "`python tools/build_hvsr_fortran.py` from the project root"
        ) from error

    settings = settings or SurfaceWaveSettings()
    frequencies = np.ascontiguousarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")

    def seeds(wave: str, modes: int) -> np.ndarray:
        result = np.empty((frequencies.size, modes), dtype=np.float64, order="F")
        for mode in range(modes):
            result[:, mode] = phase_velocity_seeds_km_s(
                model,
                frequencies,
                mode,
                wave,
                dc_km_s=settings.dispersion_step_km_s,
            )
        return result

    rayleigh_seeds = seeds("rayleigh", settings.max_rayleigh_modes)
    love_seeds = seeds("love", settings.max_love_modes)
    rayleigh_horizontal, rayleigh_vertical, love_horizontal, _, status = (
        _hvsr_fortran.hvsr_core.surface_wave_contributions(
            model.thickness_km,
            model.vp_km_s,
            model.vs_km_s,
            model.density_g_cm3,
            frequencies,
            rayleigh_seeds,
            love_seeds,
            3.0 * settings.dispersion_step_km_s,
            settings.residue_step_fraction,
        )
    )
    if status != 0:
        raise RuntimeError(f"Fortran surface-wave calculation failed with status {status}")

    zeros = np.zeros_like(frequencies)
    return GreenFunctionContributions(
        rayleigh_horizontal=rayleigh_horizontal,
        rayleigh_vertical=rayleigh_vertical,
        love_horizontal=love_horizontal,
        psv_horizontal=zeros,
        psv_vertical=zeros,
        sh_horizontal=zeros,
    )


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
class FortranDiffuseFieldSolver:
    """Fast DFA backend using disba seeds and the optional f2py kernel."""

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
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        zeros = np.zeros_like(frequencies)
        empty = GreenFunctionContributions(
            rayleigh_horizontal=zeros,
            rayleigh_vertical=zeros,
            love_horizontal=zeros,
            psv_horizontal=zeros,
            psv_vertical=zeros,
            sh_horizontal=zeros,
        )

        if settings.max_rayleigh_modes > 0 or settings.max_love_modes > 0:
            surface = fortran_surface_wave_green_contributions(
                model,
                frequencies,
                settings=SurfaceWaveSettings(
                    max_rayleigh_modes=settings.max_rayleigh_modes,
                    max_love_modes=settings.max_love_modes,
                    dispersion_step_km_s=self.dispersion_step_km_s,
                    residue_step_fraction=self.residue_step_fraction,
                ),
            )
        else:
            surface = empty

        if settings.body_wave_wavenumbers > 0:
            requested_order = max(8, round(settings.body_wave_wavenumbers / 2))
            body = fortran_body_wave_green_contributions(
                model,
                frequencies,
                settings=BodyWaveSettings(
                    quadrature_order=max(self.body_quadrature_order, requested_order),
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
            frequencies,
            contributions,
            horizontal_definition=settings.horizontal_definition,
            backend="fortran",
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
