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


def _symmetric_lobed_spectrum(dt: float, npts: int, peak_index: int) -> np.ndarray:
    """FFT of a complex signal whose envelope is exactly symmetric about ``peak_index``.

    Envelope: a main Gaussian lobe and weaker side lobes at +-15 s and +-30 s,
    so each side has an inner minimum (> 5 s from the peak, < 0.2 of it) and
    an outer minimum. The carrier phase is zero at the peak, so the real part
    is symmetric as well.
    """
    lag = (np.arange(npts) - peak_index) * dt
    envelope = np.exp(-0.5 * (lag / 2.0) ** 2)
    for offset_s in (15.0, 30.0):
        envelope += 0.1 * np.exp(-0.5 * ((np.abs(lag) - offset_s) / 2.0) ** 2)
    return fft(envelope * np.exp(1j * tau * lag / 4.0))


def _pmf_for_cleaning(dt: float, **kwargs) -> PhaseMatchedFilter:
    # Only dt and the cleaning parameters matter for clean_undispersed.
    return PhaseMatchedFilter(
        dt=dt,
        distance_km=100.0,
        periods=np.array([4.0, 8.0]),
        group_velocity=np.array([3.0, 3.0]),
        **kwargs,
    )


def test_pmf_cleaning_window_is_symmetric_for_symmetric_envelope():
    # Fortran AFTAN tgauss.f sets both outer limits from the symmetrised inner
    # limits: inner -/+ outer width. A symmetric envelope must give a
    # symmetric cleaned signal.
    dt, npts, peak_index = 0.5, 801, 400
    spectrum = _symmetric_lobed_spectrum(dt, npts, peak_index)
    signal_data = ifft(spectrum).real
    cleaned = _pmf_for_cleaning(dt).clean_undispersed(spectrum, npts)

    offsets = np.arange(1, peak_index + 1)
    # Same samples mirrored; only floating-point rounding of the FFT pair differs.
    np.testing.assert_allclose(
        cleaned[peak_index + offsets],
        cleaned[peak_index - offsets],
        rtol=0.0,
        atol=1e-12 * np.max(np.abs(signal_data)),
    )
    # The window really cuts something (side lobes are attenuated).
    assert np.max(np.abs(cleaned[peak_index + 70 :])) < 0.5 * np.max(np.abs(signal_data[peak_index + 70 :]))


def test_pmf_cleaning_window_pyaftan_mode_keeps_wider_right_taper():
    # clean_window="pyaftan" reproduces PyAFTAN (and SeisForge before the
    # fix): the right outer limit is the old right outer limit plus the outer
    # width, so the right half-Gaussian is wider than the left one.
    dt, npts, peak_index = 0.5, 801, 400
    spectrum = _symmetric_lobed_spectrum(dt, npts, peak_index)
    signal_data = ifft(spectrum).real
    cleaned = _pmf_for_cleaning(dt, clean_window="pyaftan").clean_undispersed(spectrum, npts)

    weights = np.divide(cleaned, signal_data, out=np.ones(npts), where=np.abs(signal_data) > 1e-3)
    offsets = np.arange(40, 120)
    right = weights[peak_index + offsets]
    left = weights[peak_index - offsets]
    keep = np.abs(signal_data[peak_index + offsets]) > 1e-3
    assert np.all(right[keep] >= left[keep] - 1e-12)
    assert np.any(right[keep] > left[keep] + 1e-3)


def test_pmf_rejects_unknown_clean_window():
    with np.testing.assert_raises(ValueError):
        _pmf_for_cleaning(0.5, clean_window="asymmetric")
