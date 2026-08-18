import numpy as np

from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    NativeDiffuseFieldSolver,
    predict_diffuse_field_hvsr,
)
from seisforge.hvsr.dispersion import phase_velocity_seeds_km_s


def _reference_model() -> ElasticLayerModel:
    return ElasticLayerModel(
        [0.5, 1.0, 3.0],
        [1.8, 2.2, 3.2, 5.5],
        [0.5, 1.0, 1.8, 3.2],
        [1.9, 2.1, 2.4, 2.7],
    )


def test_native_full_dfa_matches_upstream_reference() -> None:
    result = predict_diffuse_field_hvsr(
        _reference_model(),
        [0.2, 1.0],
        solver=NativeDiffuseFieldSolver(contour_shift_fraction=1.0e-8),
    )

    # HV-DFA 1.0, 20 Rayleigh/Love modes and 10,000 body-wave samples.
    np.testing.assert_allclose(result.hvsr, [4.20325, 1.36831], rtol=1.0e-2)
    assert result.backend == "native"
    assert result.approximation == "full_dfa"
    assert result.contributions is not None
    assert np.all(result.horizontal_energy > 0.0)
    assert np.all(result.vertical_energy > 0.0)


def test_native_solver_can_calculate_body_waves_without_surface_modes() -> None:
    settings = DiffuseFieldSettings(
        max_rayleigh_modes=0,
        max_love_modes=0,
        body_wave_wavenumbers=100,
    )
    result = NativeDiffuseFieldSolver(contour_shift_fraction=1.0e-8)(
        _reference_model(), [0.2, 1.0], settings
    )

    np.testing.assert_allclose(result.hvsr, [5.34253, 1.21937], rtol=5.0e-3)
    np.testing.assert_allclose(result.contributions.rayleigh_horizontal, 0.0)
    np.testing.assert_allclose(result.contributions.love_horizontal, 0.0)


def test_native_horizontal_mean_is_sum_divided_by_sqrt_two() -> None:
    solver = NativeDiffuseFieldSolver(contour_shift_fraction=1.0e-8)
    body_sum = DiffuseFieldSettings(
        max_rayleigh_modes=0,
        max_love_modes=0,
        body_wave_wavenumbers=100,
        horizontal_definition="sum",
    )
    body_mean = DiffuseFieldSettings(
        max_rayleigh_modes=0,
        max_love_modes=0,
        body_wave_wavenumbers=100,
        horizontal_definition="mean",
    )

    sum_result = solver(_reference_model(), [1.0], body_sum)
    mean_result = solver(_reference_model(), [1.0], body_mean)
    np.testing.assert_allclose(mean_result.hvsr, sum_result.hvsr / np.sqrt(2.0))


def test_native_solver_tracks_narrow_high_mode_roots_over_frequency_band() -> None:
    frequencies = np.geomspace(0.2, 5.0, 12)
    result = NativeDiffuseFieldSolver()(_reference_model(), frequencies)

    assert np.all(np.isfinite(result.hvsr))
    assert np.all(result.hvsr > 0.0)


def test_dispersion_seeds_preserve_requested_frequency_order() -> None:
    frequencies = np.array([5.0, 0.2, 1.0, 2.5])
    seeds = phase_velocity_seeds_km_s(
        _reference_model(), frequencies, mode=0, wave="rayleigh"
    )

    assert seeds.shape == frequencies.shape
    assert np.all(np.isfinite(seeds))
    for frequency, seed in zip(frequencies, seeds, strict=True):
        single = phase_velocity_seeds_km_s(
            _reference_model(), [frequency], mode=0, wave="rayleigh"
        )
        np.testing.assert_allclose(seed, single[0], rtol=5.0e-3)
