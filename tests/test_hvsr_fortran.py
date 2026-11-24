from __future__ import annotations

import numpy as np
import pytest

import seisforge.hvsr.fortran as fortran_module
from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    FortranDiffuseFieldSolver,
    ForwardStatus,
    NativeDiffuseFieldSolver,
)
from seisforge.hvsr.body_waves import BodyWaveSettings, body_wave_green_contributions
from seisforge.hvsr.fortran import (
    fortran_body_wave_green_contributions,
    fortran_extension_available,
    fortran_surface_wave_green_contributions,
)
from seisforge.hvsr.modes import SurfaceWaveSettings, surface_wave_green_contributions


pytestmark = pytest.mark.skipif(
    not fortran_extension_available(),
    reason="build the optional f2py extension to run Fortran regression tests",
)


def _reference_model() -> ElasticLayerModel:
    return ElasticLayerModel(
        [0.5, 1.0, 3.0],
        [1.8, 2.2, 3.2, 5.5],
        [0.5, 1.0, 1.8, 3.2],
        [1.9, 2.1, 2.4, 2.7],
    )


def test_fortran_body_terms_match_python_native_kernel() -> None:
    frequencies = np.array([0.2, 1.0, 3.0])
    settings = BodyWaveSettings()
    expected = body_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )
    actual = fortran_body_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )

    # Same rule, same interval choices and summation order; what remains is
    # BLAS vs loop summation inside each 15-point rule (measured <= 2e-11).
    np.testing.assert_allclose(actual.psv_horizontal, expected.psv_horizontal, rtol=1e-10)
    np.testing.assert_allclose(actual.psv_vertical, expected.psv_vertical, rtol=1e-10)
    np.testing.assert_allclose(actual.sh_horizontal, expected.sh_horizontal, rtol=1e-10)
    assert actual.issues == expected.issues == ()


def test_fortran_surface_terms_match_python_native_kernel() -> None:
    frequencies = np.geomspace(0.2, 3.0, 8)
    settings = SurfaceWaveSettings(max_rayleigh_modes=3, max_love_modes=3)
    expected = surface_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )
    actual = fortran_surface_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )

    # Both refine roots to 1e-12 (Brent vs bisection); the residue step
    # 1e-5 k turns that into ~1e-10 (measured <= 7e-11 here). Was 2e-4.
    for name in ("rayleigh_horizontal", "rayleigh_vertical", "love_horizontal"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=1e-9, atol=1e-12
        )
    assert actual.issues == expected.issues == ()


def test_fortran_full_solver_matches_python_native_solver() -> None:
    frequencies = np.geomspace(0.2, 3.0, 8)
    settings = DiffuseFieldSettings(
        max_rayleigh_modes=3,
        max_love_modes=3,
    )
    expected = NativeDiffuseFieldSolver()(_reference_model(), frequencies, settings)
    actual = FortranDiffuseFieldSolver()(_reference_model(), frequencies, settings)

    assert actual.backend == "fortran"
    np.testing.assert_allclose(actual.hvsr, expected.hvsr, rtol=1e-9)


def test_fortran_records_the_modes_it_leaves_out() -> None:
    # Rayleigh mode 7 at this frequency: the Fortran refinement (65 samples,
    # bisection) finds no sign change near the disba seed, the Python one
    # (with its seed-neighbour search) does. Before: dropped silently, status
    # OK, H/V 2e-4 off the native value with nothing to show why.
    frequency = [3.5719274595219535]
    settings = DiffuseFieldSettings(max_rayleigh_modes=8, max_love_modes=0)
    actual = FortranDiffuseFieldSolver()(_reference_model(), frequency, settings)
    expected = NativeDiffuseFieldSolver()(_reference_model(), frequency, settings)

    (issue,) = actual.diagnostics[0].issues
    assert (issue.term, issue.mode) == ("rayleigh", 7)
    assert actual.diagnostics[0].status is ForwardStatus.INCOMPLETE
    assert expected.diagnostics[0].status is ForwardStatus.OK


def test_fortran_skip_codes_carry_frequency_mode_and_reason(monkeypatch) -> None:
    seeds = fortran_module.phase_velocity_seeds_km_s

    def shifted(model, frequencies, mode, wave, **kwargs):
        values = seeds(model, frequencies, mode, wave, **kwargs)
        if wave == "love" and mode == 1:
            values = values + 0.05  # far outside the +-3 dc refinement window
        return values

    monkeypatch.setattr(fortran_module, "phase_velocity_seeds_km_s", shifted)
    frequencies = np.array([2.0, 3.0])
    result = fortran_surface_wave_green_contributions(
        _reference_model(),
        frequencies,
        settings=SurfaceWaveSettings(max_rayleigh_modes=1, max_love_modes=2),
    )
    assert [(i.frequency_hz, i.term, i.mode) for i in result.issues] == [
        (2.0, "love", 1),
        (3.0, "love", 1),
    ]
    assert all("root" in i.reason for i in result.issues)


def test_fortran_reports_unconverged_body_quadrature() -> None:
    settings = BodyWaveSettings(max_intervals=4)
    expected = body_wave_green_contributions(_reference_model(), [1.0], settings=settings)
    actual = fortran_body_wave_green_contributions(_reference_model(), [1.0], settings=settings)
    assert len(actual.issues) == 1 and actual.issues[0].term == "body_waves"
    assert actual.issues[0].frequency_hz == expected.issues[0].frequency_hz
    np.testing.assert_allclose(actual.psv_vertical, expected.psv_vertical, rtol=1e-10)
