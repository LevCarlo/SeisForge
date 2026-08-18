from __future__ import annotations

import numpy as np
import pytest

from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    FortranDiffuseFieldSolver,
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
    settings = BodyWaveSettings(
        quadrature_order=48,
        contour_shift_fraction=1.0e-5,
    )
    expected = body_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )
    actual = fortran_body_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )

    np.testing.assert_allclose(actual.psv_horizontal, expected.psv_horizontal, rtol=1e-11)
    np.testing.assert_allclose(actual.psv_vertical, expected.psv_vertical, rtol=1e-11)
    np.testing.assert_allclose(actual.sh_horizontal, expected.sh_horizontal, rtol=1e-11)


def test_fortran_surface_terms_match_python_native_kernel() -> None:
    frequencies = np.geomspace(0.2, 3.0, 8)
    settings = SurfaceWaveSettings(max_rayleigh_modes=3, max_love_modes=3)
    expected = surface_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )
    actual = fortran_surface_wave_green_contributions(
        _reference_model(), frequencies, settings=settings
    )

    np.testing.assert_allclose(
        actual.rayleigh_horizontal, expected.rayleigh_horizontal, rtol=2e-4, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.rayleigh_vertical, expected.rayleigh_vertical, rtol=2e-4, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.love_horizontal, expected.love_horizontal, rtol=2e-4, atol=1e-12
    )


def test_fortran_full_solver_matches_python_native_solver() -> None:
    frequencies = np.geomspace(0.2, 3.0, 8)
    settings = DiffuseFieldSettings(
        max_rayleigh_modes=3,
        max_love_modes=3,
        body_wave_wavenumbers=100,
    )
    expected = NativeDiffuseFieldSolver()(_reference_model(), frequencies, settings)
    actual = FortranDiffuseFieldSolver()(_reference_model(), frequencies, settings)

    assert actual.backend == "fortran"
    np.testing.assert_allclose(actual.hvsr, expected.hvsr, rtol=2e-4)
