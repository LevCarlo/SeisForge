"""Python boundary for the optional f2py HVSR numerical kernel.

The kernel (`_fortran_src/hvsr_core.f90`) runs the same algorithms as the
native Python code: the same radiation-branch rule, the same adaptive
Gauss--Kronrod body-wave quadrature on segments passed from
`body_wave_segments`, and residues at roots refined from the same disba
seeds (its root refinement differs: 65 samples and bisection). It returns a
skip code per frequency and mode and a convergence flag per frequency; this
module turns them into `ContributionIssue` records, so both solvers report
left-out terms the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .body_waves import (
    BodyWaveSettings,
    _checked_body_contributions,
    _unconverged_issue,
    body_wave_segments,
)
from .core import assemble_diffuse_field_prediction
from .dispersion import phase_velocity_seeds_km_s
from .models import (
    ContributionIssue,
    DiffuseFieldPrediction,
    DiffuseFieldSettings,
    ElasticLayerModel,
    GreenFunctionContributions,
)
from .modes import SKIP_REASONS, SurfaceWaveSettings
from .native import _KernelControls


class FortranExtensionUnavailable(ImportError):
    """Raised when the optional f2py extension has not been built."""


def fortran_extension_available() -> bool:
    """Return whether the compiled f2py extension can be imported."""

    try:
        from . import _hvsr_fortran  # noqa: F401
    except ImportError:
        return False
    return True


def _kernel():
    try:
        from . import _hvsr_fortran
    except ImportError as error:
        raise FortranExtensionUnavailable(
            "the f2py HVSR extension is not built; run "
            "`python tools/build_hvsr_fortran.py` from the project root"
        ) from error
    return _hvsr_fortran.hvsr_core


def _validated_frequencies(frequencies_hz) -> np.ndarray:
    frequencies = np.ascontiguousarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")
    return frequencies


def fortran_body_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: BodyWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Evaluate the DFA body-wave integral in the Fortran kernel.

    Same quadrature and failure policy as `body_wave_green_contributions`.
    """

    kernel = _kernel()
    settings = settings or BodyWaveSettings()
    frequencies = _validated_frequencies(frequencies_hz)
    boundaries = body_wave_segments(model)

    (
        psv_horizontal,
        psv_vertical,
        sh_horizontal,
        relative_error,
        intervals,
        converged,
        status,
    ) = kernel.body_wave_contributions(
        model.thickness_km,
        model.vp_km_s,
        model.vs_km_s,
        model.density_g_cm3,
        frequencies,
        boundaries,
        settings.relative_tolerance,
        max(settings.max_intervals, boundaries.size - 1),
        settings.contour_shift_fraction,
    )
    if status != 0:
        messages = {
            1: "invalid model dimensions or quadrature controls",
            2: "singular P-SV boundary system",
            3: "singular SH boundary system",
        }
        raise RuntimeError(
            "Fortran body-wave calculation failed: "
            f"{messages.get(int(status), f'unknown status {status}')}"
        )

    issues = [
        _unconverged_issue(
            float(frequencies[index]),
            float(relative_error[index]),
            int(intervals[index]),
            settings.relative_tolerance,
        )
        for index in np.flatnonzero(converged == 0)
    ]
    return _checked_body_contributions(
        np.array(psv_horizontal), np.array(psv_vertical), np.array(sh_horizontal), tuple(issues)
    )


def fortran_surface_wave_green_contributions(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: SurfaceWaveSettings | None = None,
) -> GreenFunctionContributions:
    """Track modal seeds with disba and evaluate pole residues in Fortran."""

    kernel = _kernel()
    settings = settings or SurfaceWaveSettings()
    frequencies = _validated_frequencies(frequencies_hz)

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
    (
        rayleigh_horizontal,
        rayleigh_vertical,
        love_horizontal,
        rayleigh_skip,
        love_skip,
        status,
    ) = kernel.surface_wave_contributions(
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
    if status != 0:
        raise RuntimeError(f"Fortran surface-wave calculation failed with status {status}")

    # Same order as the native loop: Rayleigh modes, then Love, each by mode
    # and then by frequency.
    issues = [
        ContributionIssue(float(frequencies[index]), wave, int(mode), SKIP_REASONS[int(code)])
        for wave, codes in (("rayleigh", rayleigh_skip), ("love", love_skip))
        for mode in range(np.shape(codes)[1])
        for index, code in enumerate(np.asarray(codes)[:, mode])
        if code != 0
    ]
    zeros = np.zeros_like(frequencies)
    return GreenFunctionContributions(
        rayleigh_horizontal=rayleigh_horizontal,
        rayleigh_vertical=rayleigh_vertical,
        love_horizontal=love_horizontal,
        psv_horizontal=zeros,
        psv_vertical=zeros,
        sh_horizontal=zeros,
        issues=tuple(issues),
    )


@dataclass(frozen=True)
class FortranDiffuseFieldSolver(_KernelControls):
    """Fast DFA backend using disba seeds and the optional f2py kernel.

    Same accuracy controls and defaults as `NativeDiffuseFieldSolver`.
    """

    def __call__(
        self,
        model: ElasticLayerModel,
        frequencies_hz,
        settings: DiffuseFieldSettings | None = None,
    ) -> DiffuseFieldPrediction:
        settings = settings or DiffuseFieldSettings()
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        surface = body = None
        if settings.max_rayleigh_modes > 0 or settings.max_love_modes > 0:
            surface = fortran_surface_wave_green_contributions(
                model, frequencies, settings=self.surface_wave_settings(settings)
            )
        if settings.include_body_waves:
            body = fortran_body_wave_green_contributions(
                model, frequencies, settings=self.body_wave_settings()
            )
        return assemble_diffuse_field_prediction(
            frequencies,
            surface=surface,
            body=body,
            horizontal_definition=settings.horizontal_definition,
            backend="fortran",
        )
