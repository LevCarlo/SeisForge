from __future__ import annotations

from math import tau

import numpy as np
from scipy.fft import fft, fftfreq, ifft

from seisforge.ant.filters import (
    GaussianFTANFilter,
    PhaseMatchedFilter,
    analytic_spectrum,
    cosine_spectral_taper,
    gaussian_alpha_aftan_scaling,
    gaussian_alpha_from_distance,
    gaussian_alpha_pyftan_constant,
    gaussian_angular_response,
    gaussian_filter_bank_spectrum,
    gaussian_filter_trace,
    gaussian_time_window,
)


def test_gaussian_alpha_from_distance_uses_empirical_table():
    assert gaussian_alpha_from_distance(0.0) == 5.0
    assert gaussian_alpha_from_distance(250.0) == 12.0
    assert gaussian_alpha_from_distance(500.0) == 20.0
    assert gaussian_alpha_from_distance(1500.0) == 30.0
    assert gaussian_alpha_from_distance(50000.0) == 75.0


def test_gaussian_alpha_scaling_modes_are_explicit():
    assert gaussian_alpha_aftan_scaling(1000.0) == 20.0
    np.testing.assert_allclose(gaussian_alpha_aftan_scaling(500.0), 20.0 * np.sqrt(0.5))
    assert gaussian_alpha_pyftan_constant() == 20.0
    assert gaussian_alpha_pyftan_constant(1.5) == 30.0


def test_gaussian_angular_response_matches_aftan_formula():
    omega0 = tau / 5.0
    omega = np.array([omega0, omega0 * 1.1, omega0 * 1.5])
    response = gaussian_angular_response(omega, omega0, alpha=20.0, exponent_floor=None)

    expected = np.exp(-20.0 * ((omega - omega0) / omega0) ** 2)
    np.testing.assert_allclose(response, expected)
    assert response[0] == 1.0


def test_analytic_spectrum_suppresses_negative_frequencies():
    dt = 0.1
    time = np.arange(0.0, 20.0, dt)
    data = np.cos(2 * np.pi * 1.0 * time)
    freqs = fftfreq(data.size, d=dt)

    spectrum = analytic_spectrum(data)

    assert np.max(np.abs(spectrum[freqs < 0])) < 1e-9
    assert np.max(np.abs(spectrum[freqs > 0])) > 10.0


def test_gaussian_filter_trace_keeps_center_frequency():
    dt = 0.05
    time = np.arange(0.0, 100.0, dt)
    data = np.cos(2 * np.pi * time / 5.0) + 0.5 * np.cos(2 * np.pi * time / 2.0)

    filtered = gaussian_filter_trace(data, dt, center_period=5.0, alpha=40.0)
    spectrum = np.abs(fft(filtered))
    freqs = fftfreq(filtered.size, d=dt)
    peak_frequency = abs(freqs[int(np.argmax(spectrum))])

    assert peak_frequency == np.round(1.0 / 5.0, 12)


def test_gaussian_filter_bank_spectrum_returns_one_column_per_period():
    dt = 0.1
    time = np.arange(0.0, 40.0, dt)
    data = np.cos(2 * np.pi * time / 4.0)
    periods = np.array([3.0, 4.0, 5.0])

    bank = gaussian_filter_bank_spectrum(data, dt, periods, alpha=20.0)
    filtered = ifft(bank, axis=0).real
    energy = np.sum(filtered**2, axis=0)

    assert bank.shape == (data.size, periods.size)
    assert int(np.argmax(energy)) == 1


def test_gaussian_ftan_filter_is_public_filter_bank_interface():
    dt = 0.1
    time = np.arange(0.0, 40.0, dt)
    data = np.cos(2 * np.pi * time / 4.0)
    periods = np.array([3.0, 4.0, 5.0])

    filter_bank = GaussianFTANFilter(dt=dt, center_periods=periods, alpha=20.0)
    traces = filter_bank.traces(data)
    energy = np.sum(traces.real**2, axis=0)

    assert traces.shape == (data.size, periods.size)
    assert int(np.argmax(energy)) == 1


def test_cosine_spectral_taper_has_ramps_and_passband():
    omega = np.linspace(0.0, 10.0, 501)
    taper = cosine_spectral_taper(omega, omega_min=2.0, omega_max=6.0, alpha=20.0)

    assert taper[0] == 0.0
    assert taper[-1] == 0.0
    assert np.max(taper) == 1.0
    assert np.any((taper > 0.0) & (taper < 1.0))
    assert np.all(taper[(omega >= 2.4) & (omega <= 5.4)] == 1.0)


def test_gaussian_time_window_peaks_at_center():
    times = np.linspace(0.0, 10.0, 101)
    window = gaussian_time_window(times, center_time=4.0, half_width=1.0)

    assert int(np.argmax(window)) == 40
    assert window[40] == 1.0
    assert window[0] < window[20] < window[40]


def test_phase_matched_filter_returns_cleaned_trace_with_same_length():
    dt = 0.2
    time = np.arange(0.0, 120.0, dt)
    data = np.exp(-0.5 * ((time - 45.0) / 8.0) ** 2) * np.cos(2 * np.pi * time / 6.0)
    periods = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    group_velocity = np.full(periods.size, 3.0)

    pmf = PhaseMatchedFilter(
        dt=dt,
        distance_km=135.0,
        periods=periods,
        group_velocity=group_velocity,
        alpha=20.0,
    )
    cleaned = pmf.apply(data)

    assert cleaned.shape == data.shape
    assert np.all(np.isfinite(cleaned))
    assert np.max(np.abs(cleaned)) > 0.0
