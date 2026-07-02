"""Frequency- and time-domain filters used by ANT dispersion workflows."""

from __future__ import annotations

from dataclasses import dataclass
from math import tau

import numpy as np
from scipy import interpolate, signal
from scipy.fft import fft, fftfreq, ifft


ALPHA_DISTANCE_KM = np.array([0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 20000.0])
ALPHA_VALUES = np.array([5.0, 8.0, 12.0, 20.0, 25.0, 35.0, 50.0, 75.0])


def gaussian_alpha_from_distance(
    distance_km: float,
    distance_nodes: np.ndarray | None = None,
    alpha_nodes: np.ndarray | None = None,
) -> float:
    """Interpolate the empirical Gaussian alpha from interstation distance."""
    distances = ALPHA_DISTANCE_KM if distance_nodes is None else np.asarray(distance_nodes, dtype=float)
    alphas = ALPHA_VALUES if alpha_nodes is None else np.asarray(alpha_nodes, dtype=float)
    if distances.ndim != 1 or alphas.ndim != 1 or distances.size != alphas.size:
        raise ValueError("distance_nodes and alpha_nodes must be 1-D arrays with equal length.")
    if np.any(np.diff(distances) <= 0):
        raise ValueError("distance_nodes must be strictly increasing.")
    return float(np.interp(float(distance_km), distances, alphas))


def gaussian_alpha_aftan_scaling(distance_km: float, factor: float = 1.0) -> float:
    """Original AFTAN automatic scaling: alpha = factor * 20 * sqrt(dist / 1000)."""
    if distance_km <= 0:
        raise ValueError("distance_km must be positive.")
    return float(factor) * 20.0 * np.sqrt(float(distance_km) / 1000.0)


def gaussian_alpha_pyftan_constant(factor: float = 1.0) -> float:
    """PyAFTAN/F77 modified scaling: alpha = factor * 20."""
    return float(factor) * 20.0


@dataclass(frozen=True)
class GaussianFTANFilter:
    """Gaussian narrow-band filter bank used by both basic FTAN and PMF FTAN."""

    dt: float
    center_periods: np.ndarray
    alpha: float | np.ndarray
    nfft: int | None = None

    def spectra(self, data: np.ndarray) -> np.ndarray:
        """Return one analytic filtered spectrum column per center period."""
        return gaussian_filter_bank_spectrum(
            data,
            self.dt,
            self.center_periods,
            self.alpha,
            nfft=self._nfft(data),
        )

    def traces(self, data: np.ndarray) -> np.ndarray:
        """Return complex filtered traces, one column per center period."""
        return ifft(self.spectra(data), axis=0)

    def _nfft(self, data: np.ndarray) -> int:
        return int(self.nfft) if self.nfft is not None else np.asarray(data).size


@dataclass(frozen=True)
class PhaseMatchedFilter:
    """Phase-matched filter stage between basic FTAN and cleaned FTAN."""

    dt: float
    distance_km: float
    periods: np.ndarray
    group_velocity: np.ndarray
    alpha: float = 20.0
    nalpha: float = 2.0
    min_half_length: float = 5.0
    amplitude_ratio: float = 0.2
    window_factor: float = 1.0
    nfft: int | None = None

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Return the redispersed PMF-cleaned signal."""
        data = np.asarray(data)
        nfft = self._nfft(data)
        freqs = fftfreq(nfft, d=self.dt)
        iphi, period_min, period_max = self.phase_correction(freqs)
        spectrum = analytic_spectrum(data, nfft)
        taper = self.spectral_taper(freqs, period_min, period_max)
        compressed = spectrum * np.exp(iphi) * taper
        cleaned = self.clean_undispersed(compressed, data.size)
        return self.redisperse(cleaned, iphi, data.size)

    def phase_correction(
        self,
        freqs: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Build the complex phase correction from a group-velocity branch."""
        periods = np.asarray(self.periods, dtype=float)
        group_velocity = np.asarray(self.group_velocity, dtype=float)
        valid = np.isfinite(periods) & np.isfinite(group_velocity)
        valid &= periods > 0
        valid &= group_velocity > 0
        periods = periods[valid]
        group_velocity = group_velocity[valid]
        if periods.size < 2:
            raise ValueError("PMF requires at least two valid period/velocity points.")

        order = np.argsort(periods)[::-1]
        periods = periods[order]
        group_velocity = group_velocity[order]
        omega = tau / periods
        keep = np.concatenate(([True], np.diff(omega) > 0))
        omega = omega[keep]
        group_velocity = group_velocity[keep]
        periods = periods[keep]
        if omega.size < 2:
            raise ValueError("PMF periods must define at least two unique frequencies.")

        travel_time = self.distance_km / group_velocity
        spline = interpolate.CubicSpline(
            omega,
            travel_time - travel_time[0],
            extrapolate=False,
        )
        phase = np.zeros(freqs.size)
        angular_freqs = tau * np.asarray(freqs, dtype=float)
        inside = (angular_freqs >= omega[0]) & (angular_freqs <= omega[-1])
        phase[inside] = [spline.integrate(omega[0], value) for value in angular_freqs[inside]]
        return 1j * phase, float(np.min(periods)), float(np.max(periods))

    def spectral_taper(
        self,
        freqs: np.ndarray,
        period_min: float,
        period_max: float,
    ) -> np.ndarray:
        """Cosine spectral taper used by the PMF correction stage."""
        frac = 1.0 / np.sqrt(float(self.nalpha) * float(self.alpha))
        f1 = (1.0 / period_max) * (1.0 - frac)
        f2 = (1.0 / period_max) * (1.0 + frac)
        f3 = 1.0 / period_min
        f4 = (1.0 / period_min) * (1.0 + frac)
        return _four_corner_frequency_taper(freqs, f1, f2, f3, f4)

    def clean_undispersed(self, spectrum: np.ndarray, npts: int) -> np.ndarray:
        """Clean the compressed PMF signal using local minima and Gaussian skirts."""
        analytic = ifft(spectrum)[:npts]
        signal_data = analytic.real
        envelope = np.abs(analytic)
        peak_index = int(np.argmax(envelope))
        peak_amp = float(envelope[peak_index])
        minima = _local_minima(envelope)
        left = minima[minima < peak_index]
        right = minima[minima > peak_index]

        left_cut = _closest_minimum(
            left[::-1],
            envelope[left][::-1],
            peak_index,
            peak_amp,
            self.dt,
            self.min_half_length,
            self.amplitude_ratio,
        )
        right_cut = _closest_minimum(
            right,
            envelope[right],
            peak_index,
            peak_amp,
            self.dt,
            self.min_half_length,
            self.amplitude_ratio,
        )

        window = np.ones(npts)
        if left_cut is not None:
            start = _scaled_index(left_cut, peak_index, self.window_factor, npts)
            width = max(1, peak_index - start)
            edge = np.arange(start, dtype=float)
            window[:start] = np.exp(-0.5 * ((edge - start) / width) ** 2)
        if right_cut is not None:
            stop = _scaled_index(right_cut, peak_index, self.window_factor, npts)
            width = max(1, stop - peak_index)
            edge = np.arange(stop, npts, dtype=float)
            window[stop:] = np.exp(-0.5 * ((edge - stop) / width) ** 2)
        return signal_data * window

    def redisperse(self, data: np.ndarray, iphi: np.ndarray, npts: int) -> np.ndarray:
        """Apply the inverse phase correction after PMF cleaning."""
        spectrum = analytic_spectrum(data, iphi.size)
        return ifft(spectrum * np.exp(-iphi)).real[:npts]

    def _nfft(self, data: np.ndarray) -> int:
        return int(self.nfft) if self.nfft is not None else np.asarray(data).size


def analytic_spectrum(data: np.ndarray, nfft: int | None = None) -> np.ndarray:
    """Return the one-sided analytic FFT spectrum used by FTAN filters."""
    data = np.asarray(data)
    if nfft is None:
        nfft = data.size
    spectrum = fft(data, n=nfft)
    hilbert_filter = np.zeros(nfft)
    if nfft % 2 == 0:
        hilbert_filter[0] = hilbert_filter[nfft // 2] = 1
        hilbert_filter[1 : nfft // 2] = 2
    else:
        hilbert_filter[0] = 1
        hilbert_filter[1 : (nfft + 1) // 2] = 2
    return spectrum * hilbert_filter


def gaussian_angular_response(
    omega: np.ndarray,
    omega0: float,
    alpha: float,
    exponent_floor: float | None = -40.0,
) -> np.ndarray:
    """AFTAN/Fortran Gaussian response: exp(-alpha * ((omega - omega0) / omega0)^2)."""
    omega = np.asarray(omega, dtype=float)
    if omega0 <= 0:
        raise ValueError("omega0 must be positive.")
    exponent = -float(alpha) * ((omega - float(omega0)) / float(omega0)) ** 2
    if exponent_floor is not None:
        response = np.zeros_like(exponent)
        keep = exponent >= exponent_floor
        response[keep] = np.exp(exponent[keep])
        return response
    return np.exp(exponent)


def gaussian_frequency_response(
    frequency: np.ndarray,
    center_frequency: float,
    alpha: float,
    exponent_floor: float | None = -40.0,
) -> np.ndarray:
    """Frequency-domain form of the Gaussian narrow-band filter."""
    return gaussian_angular_response(
        tau * np.asarray(frequency, dtype=float),
        tau * float(center_frequency),
        alpha,
        exponent_floor=exponent_floor,
    )


def gaussian_filter_trace(
    data: np.ndarray,
    dt: float,
    center_period: float,
    alpha: float,
    *,
    nfft: int | None = None,
    analytic: bool = True,
    real_output: bool = True,
) -> np.ndarray:
    """Filter a trace with one Gaussian narrow-band filter."""
    data = np.asarray(data)
    if center_period <= 0:
        raise ValueError("center_period must be positive.")
    if nfft is None:
        nfft = data.size
    freqs = fftfreq(nfft, d=dt)
    spectrum = analytic_spectrum(data, nfft) if analytic else fft(data, n=nfft)
    response = gaussian_frequency_response(freqs, 1.0 / center_period, alpha)
    filtered = ifft(spectrum * response)[: data.size]
    if real_output:
        return filtered.real
    return filtered


def gaussian_filter_bank_spectrum(
    data: np.ndarray,
    dt: float,
    center_periods: np.ndarray,
    alpha: float | np.ndarray,
    *,
    nfft: int | None = None,
) -> np.ndarray:
    """Return analytic FFT spectra filtered at many center periods."""
    data = np.asarray(data)
    center_periods = np.asarray(center_periods, dtype=float)
    if np.any(center_periods <= 0):
        raise ValueError("center_periods must be positive.")
    if nfft is None:
        nfft = data.size
    spectrum = analytic_spectrum(data, nfft)
    omega = tau * fftfreq(nfft, d=dt)
    omega0 = tau / center_periods
    alpha_array = np.asarray(alpha, dtype=float)
    if alpha_array.ndim == 0:
        alpha_array = np.full(center_periods.size, float(alpha_array))
    if alpha_array.shape != center_periods.shape:
        raise ValueError("alpha must be scalar or match center_periods.")
    responses = np.column_stack(
        [
            gaussian_angular_response(omega, one_omega0, one_alpha)
            for one_omega0, one_alpha in zip(omega0, alpha_array)
        ]
    )
    return spectrum[:, None] * responses


def cosine_spectral_taper(
    omega: np.ndarray,
    omega_min: float,
    omega_max: float,
    alpha: float,
    *,
    min_width_bins: int = 16,
    threshold: float = 0.5,
) -> np.ndarray:
    """Cosine spectral taper matching the intent of AFTAN ``tapers.f``."""
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1 or omega.size < 2:
        raise ValueError("omega must be a 1-D array with at least two samples.")
    if omega_min <= 0 or omega_max <= omega_min:
        raise ValueError("omega_min and omega_max must satisfy 0 < omega_min < omega_max.")
    dom = float(np.median(np.diff(omega)))
    if dom <= 0:
        raise ValueError("omega must be sorted in increasing order.")

    lower_width = max(float(min_width_bins) * dom, omega_min * np.sqrt(threshold / alpha))
    upper_width = max(float(min_width_bins) * dom, omega_max * np.sqrt(threshold / alpha))
    left0 = omega_min - lower_width / 2.0
    left1 = omega_min + lower_width / 2.0
    right0 = omega_max - upper_width / 2.0
    right1 = omega_max + upper_width / 2.0

    window = np.zeros_like(omega)
    lower = (omega >= left0) & (omega < left1)
    if left1 > left0:
        window[lower] = 0.5 * (1.0 - np.cos(np.pi * (omega[lower] - left0) / (left1 - left0)))
    middle = (omega >= left1) & (omega <= right0)
    window[middle] = 1.0
    upper = (omega > right0) & (omega <= right1)
    if right1 > right0:
        window[upper] = 0.5 * (1.0 + np.cos(np.pi * (omega[upper] - right0) / (right1 - right0)))
    return window


def gaussian_time_window(
    times: np.ndarray,
    center_time: float,
    half_width: float,
    *,
    exponent_floor: float = -24.0,
) -> np.ndarray:
    """Gaussian time-domain window used for phase-matched trace cleanup."""
    times = np.asarray(times, dtype=float)
    if half_width <= 0:
        raise ValueError("half_width must be positive.")
    exponent = -0.5 * ((times - float(center_time)) / float(half_width)) ** 2
    window = np.zeros_like(times)
    keep = exponent >= exponent_floor
    window[keep] = np.exp(exponent[keep])
    return window


def apply_gaussian_time_window(
    data: np.ndarray,
    times: np.ndarray,
    center_time: float,
    half_width: float,
    *,
    exponent_floor: float = -24.0,
) -> np.ndarray:
    """Apply a Gaussian time-domain window without changing input length."""
    return np.asarray(data) * gaussian_time_window(
        times,
        center_time,
        half_width,
        exponent_floor=exponent_floor,
    )


def _four_corner_frequency_taper(
    freqs: np.ndarray,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
) -> np.ndarray:
    if not (0 <= f1 < f2 < f3 < f4):
        raise ValueError("PMF taper corners must satisfy 0 <= f1 < f2 < f3 < f4.")
    freqs = np.asarray(freqs, dtype=float)
    taper = np.zeros(freqs.size)
    lower = (freqs >= f1) & (freqs < f2)
    taper[lower] = 0.5 * (1.0 - np.cos(np.pi * (freqs[lower] - f1) / (f2 - f1)))
    middle = (freqs >= f2) & (freqs <= f3)
    taper[middle] = 1.0
    upper = (freqs > f3) & (freqs <= f4)
    taper[upper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[upper] - f3) / (f4 - f3)))
    return taper


def _local_minima(data: np.ndarray) -> np.ndarray:
    return signal.find_peaks(-np.asarray(data))[0]


def _closest_minimum(
    indices: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    peak_amp: float,
    dt: float,
    min_half_length: float,
    amplitude_ratio: float,
) -> int | None:
    if indices.size == 0:
        return None
    far_enough = np.abs(indices - peak_index) * dt > min_half_length
    quiet_enough = values < amplitude_ratio * peak_amp
    candidates = indices[far_enough & quiet_enough]
    if candidates.size == 0:
        return None
    return int(candidates[0])


def _scaled_index(index: int, center: int, factor: float, npts: int) -> int:
    scaled = int(round((index - center) * factor + center))
    return min(max(scaled, 0), npts - 1)
