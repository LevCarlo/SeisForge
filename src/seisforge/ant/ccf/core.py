"""Pure-array core for ZZ ambient-noise cross-correlation (CC and PCC2).

Conventions
-----------
- **Component & lag sign** (shared with :mod:`seisforge.ant.rotation`,
  Lin et al. 2008): for virtual source station ``A`` and receiver station ``B``,

      C_AB(t) = sum_tau u_A(tau) * u_B(tau + t)

  Positive lag ``t > 0`` corresponds to wave propagation from ``A`` to ``B``.
  This matches ``np.correlate(u_B, u_A, "full")`` (note the argument order:
  reversing ``u_A`` and ``u_B`` reverses time).
- **Fourier convention**: ``scipy.fft.rfft`` uses ``exp(-i 2 pi f t)``, so
  the cross-spectrum is ``conj(U_A(f)) * U_B(f)``, i.e. ``S_A^H S_B``.
- **Reverse-pair reading (ZZ)**: ``C_BA(t) = C_AB(-t)``.
- **Linear correlation padding & guard band**: to avoid circular wrap-around
  anywhere inside ``[-max_lag_s - lag_guard_s, +max_lag_s + lag_guard_s]``,
  ``nfft`` must satisfy
  ``nfft >= window_samples + max_lag_samples + lag_guard_samples``.
  All ``nfft // 2 + 1`` one-sided ``rfft`` frequency bins are kept (no
  frequency cropping).
- **Continuous-waveform RAM**: running-absolute-mean temporal normalization
  (Bensen et al. 2007) runs on the continuous waveform before windowing to
  avoid per-window edge artefacts, and invalid-sample masks are dilated
  outward by the RAM half-width.
- **PCC2 (nu = 2)** (Schimmel 1999; Ventosa, Schimmel & Stutzmann 2019):
  from the analytic signal ``a = x + i H[x]`` and unit-modulus phase signal
  ``u = a / |a| = cos(phi) + i sin(phi)``,

      PCC2_AB(t) = (1 / L) * sum_tau Re[conj(u_A(tau)) * u_B(tau + t)]
                 = (1 / L) * [CC(cos phi_A, cos phi_B)(t)
                              + CC(sin phi_A, sin phi_B)(t)]

  Stacking ``cos(phi) / sqrt(L)`` and ``sin(phi) / sqrt(L)`` along the window
  axis reuses the exact same real ``rfft`` + blocked GEMM path without a
  separate complex FFT pipeline. Each valid time window still counts as 1 in
  ``n_windows``.
- **Units**: internal units are ``s`` and ``Hz``; suffixes appear in parameter
  names. No function mutates its inputs.

References
----------
- Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
  Moschetti, M. P., Shapiro, N. M., & Yang, Y. (2007). Processing seismic
  ambient noise data to obtain reliable broad-band surface wave dispersion
  measurements. Geophys. J. Int., 169(3), 1239-1260,
  doi:10.1111/j.1365-246X.2007.03374.x.
- Schimmel, M. (1999). Phase cross-correlations: Design, comparisons, and
  applications. Bull. Seismol. Soc. Am., 89(5), 1366-1378,
  doi:10.1785/BSSA0890051366.
- Ventosa, S., Schimmel, M., & Stutzmann, E. (2019). Towards the processing
  of large data volumes with phase cross-correlation. Seismol. Res. Lett.,
  90(4), 1663-1669, doi:10.1785/0220190022.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, signal
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

from .models import (
    CC_TYPES,
    WHITEN_MODES,
    BandpassSpec,
    DailyCCFResult,
    WhitenSpec,
)

__all__ = [
    "apply_bandpass",
    "apply_time_weights",
    "correlate_upper_triangle_blocked",
    "correlate_zz_day",
    "expand_mask_for_ram",
    "lag_axis_s",
    "minimum_nfft",
    "prepare_cc_spectra",
    "prepare_pcc2_spectra",
    "ram_weight_continuous",
    "reverse_ccf_zz",
    "slice_windows",
    "spectral_whiten_windows",
    "time_normalize_ram",
    "upper_triangle_pairs",
]


def minimum_nfft(
    window_samples: int,
    max_lag_samples: int,
    lag_guard_samples: int = 0,
    *,
    circular: bool = False,
    next_fast: bool = True,
) -> int:
    """Return the minimum FFT length free of circular wrap-around.

    For linear correlation, wrap-around at ``|lag| <= max_lag_samples +
    lag_guard_samples`` is avoided iff
    ``nfft >= window_samples + max_lag_samples + lag_guard_samples``.
    """
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2.")
    if max_lag_samples < 1:
        raise ValueError("max_lag_samples must be at least 1.")
    if lag_guard_samples < 0:
        raise ValueError("lag_guard_samples must be non-negative.")

    half_lag = max_lag_samples + lag_guard_samples
    if circular:
        if half_lag >= window_samples // 2:
            raise ValueError(
                "Circular correlation requires max_lag_samples + lag_guard_samples "
                "< window_samples // 2."
            )
        return int(window_samples)

    required = int(window_samples + half_lag)
    return int(next_fast_len(required, real=True)) if next_fast else required


def lag_axis_s(
    sampling_rate_hz: float,
    max_lag_s: float,
    lag_guard_s: float = 0.0,
) -> np.ndarray:
    """Return the symmetric lag axis in seconds including any guard band."""
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive.")
    if max_lag_s <= 0 or not np.isfinite(max_lag_s):
        raise ValueError("max_lag_s must be finite and positive.")
    if lag_guard_s < 0 or not np.isfinite(lag_guard_s):
        raise ValueError("lag_guard_s must be finite and non-negative.")

    half_lag_samples = int(round((max_lag_s + lag_guard_s) * sampling_rate_hz))
    if half_lag_samples < 1:
        raise ValueError("(max_lag_s + lag_guard_s) * sampling_rate_hz must be >= 1.")
    return np.arange(-half_lag_samples, half_lag_samples + 1, dtype=np.float64) / float(
        sampling_rate_hz
    )


def apply_bandpass(
    data: np.ndarray,
    sampling_rate_hz: float,
    spec: BandpassSpec,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter along the last axis (float64)."""
    arr = np.asarray(data, dtype=np.float64)
    nyquist_hz = 0.5 * float(sampling_rate_hz)
    if spec.freq_max_hz >= nyquist_hz:
        raise ValueError(
            f"Bandpass freq_max_hz ({spec.freq_max_hz}) must be below "
            f"Nyquist ({nyquist_hz} Hz)."
        )
    sos = signal.butter(
        spec.order,
        [spec.freq_min_hz / nyquist_hz, spec.freq_max_hz / nyquist_hz],
        btype="bandpass",
        output="sos",
    )
    if spec.zerophase:
        return signal.sosfiltfilt(sos, arr, axis=-1)
    return signal.sosfilt(sos, arr, axis=-1)


def expand_mask_for_ram(
    valid_mask: np.ndarray,
    half_width_samples: int,
) -> np.ndarray:
    """Shrink a boolean validity mask by ``half_width_samples`` around invalid samples.

    Samples within ``half_width_samples`` of any invalid sample (or NaN) would
    pull gap or filter-transient values into a centered running average of width
    ``2 * half_width_samples + 1`` and are therefore marked invalid.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    if half_width_samples < 0:
        raise ValueError("half_width_samples must be non-negative.")
    if half_width_samples == 0:
        return mask.copy()

    width = 2 * int(half_width_samples) + 1
    invalid = ~mask
    # mode="constant", cval=0.0 preserves boundary samples unless an actual gap lies nearby.
    dilated_invalid = ndimage.maximum_filter1d(
        invalid.astype(np.uint8),
        size=width,
        axis=-1,
        mode="constant",
        cval=0,
    ).astype(bool)
    return ~dilated_invalid


def ram_weight_continuous(
    waveforms: np.ndarray,
    valid_mask: np.ndarray,
    sampling_rate_hz: float,
    half_width_s: float,
    *,
    bandpass: BandpassSpec | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute running-absolute-mean (RAM) weights on continuous waveforms.

    Computing the denominator separately from :func:`apply_time_weights` lets
    single-channel ZZ and future multi-component group rules (``z|ne``, ``zne``)
    share the exact same normalization step.

    Parameters
    ----------
    waveforms : ndarray, shape (n_channels, n_samples)
        Continuous waveform array. Invalid samples may be NaN or 0.
    valid_mask : ndarray of bool, shape (n_channels, n_samples)
        True where samples are valid (neither gap nor filter transient).
    sampling_rate_hz : float
        Sampling rate in Hz.
    half_width_s : float
        Half-width of the running average window in seconds (full window is
        ``2 * round(half_width_s * sampling_rate_hz) + 1`` samples).
    bandpass : BandpassSpec, optional
        Optional bandpass applied only to the weight-estimation copy
        (Bensen et al. 2007).

    Returns
    -------
    weight : ndarray of float64, shape (n_channels, n_samples)
        Non-negative running-absolute-mean denominator.
    expanded_mask : ndarray of bool, shape (n_channels, n_samples)
        Validity mask shrunk by the RAM half-width around invalid samples.
    """
    data = np.asarray(waveforms, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError("waveforms and valid_mask must be 2-D arrays with matching shape.")
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive.")
    if half_width_s < 0 or not np.isfinite(half_width_s):
        raise ValueError("half_width_s must be finite and non-negative.")

    finite_mask = mask & np.isfinite(data)
    clean = np.where(finite_mask, data, 0.0)
    envelope_source = (
        apply_bandpass(clean, sampling_rate_hz, bandpass)
        if bandpass is not None
        else clean
    )

    half_width_samples = int(round(half_width_s * sampling_rate_hz))
    width = 2 * half_width_samples + 1
    # float64 uniform_filter1d avoids the ~1% cumsum drift seen in float32.
    weight = ndimage.uniform_filter1d(
        np.abs(envelope_source),
        size=width,
        axis=-1,
        mode="nearest",
    )
    expanded_mask = expand_mask_for_ram(finite_mask, half_width_samples)
    return weight, expanded_mask


def apply_time_weights(
    waveforms: np.ndarray,
    weight: np.ndarray,
    valid_mask: np.ndarray,
    *,
    water_level: float = 1e-12,
) -> np.ndarray:
    """Divide waveforms by non-negative time weights, zeroing invalid samples.

    ``water_level`` (assumed default: 1e-12) guards against division by zero
    on silent segments without mutating ``waveforms``.
    """
    data = np.asarray(waveforms, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if data.shape != w.shape or data.shape != mask.shape:
        raise ValueError("waveforms, weight and valid_mask must have identical shapes.")
    if water_level < 0 or not np.isfinite(water_level):
        raise ValueError("water_level must be finite and non-negative.")

    safe_mask = mask & np.isfinite(data) & np.isfinite(w) & (w > water_level)
    safe_w = np.where(safe_mask, w, 1.0)
    return np.where(safe_mask, data / safe_w, 0.0)


def time_normalize_ram(
    waveforms: np.ndarray,
    valid_mask: np.ndarray,
    sampling_rate_hz: float,
    half_width_s: float,
    *,
    bandpass: BandpassSpec | None = None,
    water_level: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply continuous-waveform RAM temporal normalization and expand the mask."""
    weight, expanded_mask = ram_weight_continuous(
        waveforms,
        valid_mask,
        sampling_rate_hz,
        half_width_s,
        bandpass=bandpass,
    )
    normalized = apply_time_weights(
        waveforms,
        weight,
        expanded_mask,
        water_level=water_level,
    )
    return normalized, expanded_mask


def slice_windows(
    waveforms: np.ndarray,
    valid_mask: np.ndarray,
    window_samples: int,
    step_samples: int,
    *,
    taper_fraction: float = 0.05,
    demean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice continuous ``(n_channels, n_samples)`` data into tapered windows.

    A window is marked valid iff every sample in that window is True in
    ``valid_mask`` and finite in ``waveforms``. Invalid windows are explicitly
    zeroed via ``np.where`` so ``NaN * 0`` never leaks into FFTs.

    Parameters
    ----------
    waveforms : ndarray, shape (n_channels, n_samples)
    valid_mask : ndarray of bool, shape (n_channels, n_samples)
    window_samples : int
        Number of samples per window (``L``).
    step_samples : int
        Stride between consecutive window starts (``step_samples == window_samples``
        for non-overlapping windows).
    taper_fraction : float, optional
        Fraction of the window tapered at *each* end by a Tukey cosine taper
        (``0 <= taper_fraction <= 0.5``; assumed default: 0.05).
    demean : bool, optional
        If True (default), subtract the mean of each valid window before tapering.

    Returns
    -------
    windows : ndarray of float64, shape (n_channels, n_windows, window_samples)
    window_valid : ndarray of bool, shape (n_channels, n_windows)
    """
    data = np.asarray(waveforms, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if data.ndim != 2 or mask.shape != data.shape:
        raise ValueError("waveforms and valid_mask must be 2-D arrays with matching shape.")
    if window_samples < 2:
        raise ValueError("window_samples must be at least 2.")
    if step_samples < 1:
        raise ValueError("step_samples must be at least 1.")
    if not (0.0 <= taper_fraction <= 0.5):
        raise ValueError("taper_fraction must lie in [0.0, 0.5].")

    n_channels, n_samples = data.shape
    if n_samples < window_samples:
        return (
            np.empty((n_channels, 0, window_samples), dtype=np.float64),
            np.empty((n_channels, 0), dtype=bool),
        )

    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)
    offsets = np.arange(window_samples, dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]  # (n_windows, window_samples)

    raw_windows = data[:, indices]  # copy of shape (n_channels, n_windows, window_samples)
    raw_mask = mask[:, indices]
    window_valid = np.all(raw_mask & np.isfinite(raw_windows), axis=-1)

    windows = np.where(window_valid[:, :, None], raw_windows, 0.0)
    if demean:
        means = np.mean(windows, axis=-1, keepdims=True)
        windows = np.where(window_valid[:, :, None], windows - means, 0.0)

    if taper_fraction > 0.0:
        taper = signal.windows.tukey(window_samples, alpha=2.0 * taper_fraction)
        windows = windows * taper[None, None, :]

    windows = np.where(window_valid[:, :, None], windows, 0.0)
    return windows, window_valid


def _four_corner_taper(freqs_hz: np.ndarray, spec: WhitenSpec) -> np.ndarray:
    """Cosine taper with 4 corners: [taper_min, freq_min, freq_max, taper_max]."""
    taper = np.zeros_like(freqs_hz, dtype=np.float64)
    if spec.freq_min_hz > spec.taper_min_hz:
        low = (freqs_hz >= spec.taper_min_hz) & (freqs_hz < spec.freq_min_hz)
        taper[low] = 0.5 * (
            1.0
            - np.cos(
                np.pi
                * (freqs_hz[low] - spec.taper_min_hz)
                / (spec.freq_min_hz - spec.taper_min_hz)
            )
        )
    mid = (freqs_hz >= spec.freq_min_hz) & (freqs_hz <= spec.freq_max_hz)
    taper[mid] = 1.0
    if spec.taper_max_hz > spec.freq_max_hz:
        high = (freqs_hz > spec.freq_max_hz) & (freqs_hz <= spec.taper_max_hz)
        taper[high] = 0.5 * (
            1.0
            + np.cos(
                np.pi
                * (freqs_hz[high] - spec.freq_max_hz)
                / (spec.taper_max_hz - spec.freq_max_hz)
            )
        )
    return taper


def spectral_whiten_windows(
    windows: np.ndarray,
    window_valid: np.ndarray,
    sampling_rate_hz: float,
    spec: WhitenSpec,
    *,
    mode: str = "vector_norm",
) -> np.ndarray:
    """Whiten L-sample windows in the frequency domain and return time-domain windows.

    Parameters
    ----------
    windows : ndarray, shape (n_channels, n_windows, window_samples)
    window_valid : ndarray of bool, shape (n_channels, n_windows)
    sampling_rate_hz : float
    spec : WhitenSpec
    mode : {"vector_norm", "classic"}, optional
        Denominator rule: ``"vector_norm"`` uses ``sqrt(smooth(|S|^2))``
        (default, consistent with rotation-invariant group whitening);
        ``"classic"`` uses ``smooth(|S|)``.
    """
    if mode not in WHITEN_MODES:
        raise ValueError(f"mode must be one of {WHITEN_MODES}, got {mode!r}.")
    win = np.asarray(windows, dtype=np.float64)
    valid = np.asarray(window_valid, dtype=bool)
    if win.ndim != 3 or valid.shape != win.shape[:2]:
        raise ValueError("windows must be 3-D and window_valid must match its first two axes.")

    window_samples = win.shape[-1]
    nyquist_hz = 0.5 * float(sampling_rate_hz)
    if spec.taper_max_hz > nyquist_hz + 1e-12:
        raise ValueError(
            f"Whitening taper_max_hz ({spec.taper_max_hz}) exceeds Nyquist ({nyquist_hz} Hz)."
        )

    if win.shape[1] == 0:
        return win.copy()

    freqs_hz = rfftfreq(window_samples, d=1.0 / float(sampling_rate_hz))
    df_hz = freqs_hz[1] - freqs_hz[0]
    smooth_half_bins = int(round(spec.smooth_half_width_hz / df_hz))
    smooth_width = 2 * smooth_half_bins + 1

    spectrum = rfft(win, n=window_samples, axis=-1)
    if mode == "vector_norm":
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        if smooth_width > 1:
            smoothed_power = ndimage.uniform_filter1d(
                power, size=smooth_width, axis=-1, mode="nearest"
            )
            denom = np.sqrt(np.maximum(smoothed_power, 0.0))
        else:
            denom = np.sqrt(power)
    else:
        amp = np.abs(spectrum)
        denom = (
            ndimage.uniform_filter1d(amp, size=smooth_width, axis=-1, mode="nearest")
            if smooth_width > 1
            else amp
        )

    max_denom = np.max(denom, axis=-1, keepdims=True)
    floor = spec.water_level * max_denom
    safe_denom = np.where(denom > floor, denom, np.where(max_denom > 0.0, floor, 1.0))
    taper = _four_corner_taper(freqs_hz, spec)
    whitened_spec = np.where(
        max_denom > 0.0,
        (spectrum / safe_denom) * taper[None, None, :],
        0.0,
    )
    whitened = irfft(whitened_spec, n=window_samples, axis=-1)
    return np.where(valid[:, :, None], whitened, 0.0)


def prepare_cc_spectra(
    windows: np.ndarray,
    window_valid: np.ndarray,
    sampling_rate_hz: float,
    nfft: int,
    *,
    whiten: WhitenSpec | None = None,
    whiten_mode: str = "vector_norm",
    circular: bool = False,
    dtype: np.dtype | type = np.complex64,
) -> np.ndarray:
    """Prepare one-sided ``rfft`` spectra for classical cross-correlation.

    By default (``circular=False``), spectral whitening is performed on the
    unpadded ``L``-sample window before zero-padding to ``nfft`` and taking the
    correlation FFT, preventing zero-padding spectral leakage. All
    ``nfft // 2 + 1`` bins are retained (no frequency cropping).
    """
    win = np.asarray(windows, dtype=np.float64)
    valid = np.asarray(window_valid, dtype=bool)
    if win.ndim != 3 or valid.shape != win.shape[:2]:
        raise ValueError("windows must be 3-D and window_valid must match its first two axes.")

    window_samples = win.shape[-1]
    if circular and nfft != window_samples:
        raise ValueError("circular=True requires nfft == window_samples.")
    if not circular and nfft < window_samples:
        raise ValueError("nfft must be at least window_samples.")

    prepared = (
        spectral_whiten_windows(win, valid, sampling_rate_hz, whiten, mode=whiten_mode)
        if whiten is not None
        else np.where(valid[:, :, None], win, 0.0)
    )
    spectra = rfft(prepared, n=nfft, axis=-1)
    spectra = np.where(valid[:, :, None], spectra, 0.0)
    return spectra.astype(dtype, copy=False)


def prepare_pcc2_spectra(
    windows: np.ndarray,
    window_valid: np.ndarray,
    sampling_rate_hz: float,
    nfft: int,
    *,
    bandpass: BandpassSpec | None = None,
    whiten: WhitenSpec | None = None,
    whiten_mode: str = "vector_norm",
    circular: bool = False,
    amplitude_water_level: float = 1e-12,
    dtype: np.dtype | type = np.complex64,
) -> np.ndarray:
    """Prepare two-real-channel ``rfft`` spectra for ZZ PCC2 (nu = 2).

    Uses the exact real-FFT decomposition of Schimmel (1999) for ``nu = 2``:

        PCC2_AB(t) = (1 / L) * [CC(cos phi_A, cos phi_B)(t)
                                + CC(sin phi_A, sin phi_B)(t)]

    Each valid window produces two rows (``cos(phi) / sqrt(L)`` and
    ``sin(phi) / sqrt(L)``) along the window axis, so the output shape is
    ``(n_channels, 2 * n_windows, nfft // 2 + 1)`` and feeds directly into
    :func:`correlate_upper_triangle_blocked`.

    Parameters
    ----------
    windows : ndarray, shape (n_channels, n_windows, window_samples)
    window_valid : ndarray of bool, shape (n_channels, n_windows)
    sampling_rate_hz : float
    nfft : int
    bandpass : BandpassSpec, optional
        Bandpass applied to windows before computing instantaneous phase.
    whiten : WhitenSpec, optional
        Spectral whitening applied on ``L``-sample windows before computing
        instantaneous phase.
    whiten_mode : {"vector_norm", "classic"}, optional
        Whitening denominator rule.
    circular : bool, optional
        If True, require ``nfft == window_samples``.
    amplitude_water_level : float, optional
        Relative envelope threshold ``amplitude_water_level * max(|a|)`` below
        which unit-modulus phase samples are zeroed (assumed default: 1e-12).
    dtype : complex dtype, optional
        Output spectrum dtype (default: ``np.complex64``).
    """
    win = np.asarray(windows, dtype=np.float64)
    valid = np.asarray(window_valid, dtype=bool)
    if win.ndim != 3 or valid.shape != win.shape[:2]:
        raise ValueError("windows must be 3-D and window_valid must match its first two axes.")
    if amplitude_water_level < 0 or not np.isfinite(amplitude_water_level):
        raise ValueError("amplitude_water_level must be finite and non-negative.")

    n_channels, n_windows, window_samples = win.shape
    if circular and nfft != window_samples:
        raise ValueError("circular=True requires nfft == window_samples.")
    if not circular and nfft < window_samples:
        raise ValueError("nfft must be at least window_samples.")

    n_freqs = nfft // 2 + 1
    if n_windows == 0:
        return np.empty((n_channels, 0, n_freqs), dtype=dtype)

    prepared = np.where(valid[:, :, None], win, 0.0)
    if bandpass is not None:
        prepared = apply_bandpass(prepared, sampling_rate_hz, bandpass)
        prepared = np.where(valid[:, :, None], prepared, 0.0)
    if whiten is not None:
        prepared = spectral_whiten_windows(
            prepared, valid, sampling_rate_hz, whiten, mode=whiten_mode
        )

    analytic = signal.hilbert(prepared, axis=-1)
    amp = np.abs(analytic)
    max_amp = np.max(amp, axis=-1, keepdims=True)
    threshold = amplitude_water_level * max_amp
    good_sample = valid[:, :, None] & (max_amp > 0.0) & (amp > threshold)
    safe_amp = np.where(good_sample, amp, 1.0)
    scale = 1.0 / np.sqrt(float(window_samples))
    cos_phi = np.where(good_sample, analytic.real / safe_amp * scale, 0.0)
    sin_phi = np.where(good_sample, analytic.imag / safe_amp * scale, 0.0)

    spec_cos = rfft(cos_phi, n=nfft, axis=-1)
    spec_sin = rfft(sin_phi, n=nfft, axis=-1)
    stacked = np.concatenate([spec_cos, spec_sin], axis=1)
    return stacked.astype(dtype, copy=False)


def upper_triangle_pairs(
    n_stations: int,
    *,
    include_autocorr: bool = True,
    pair_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return canonical upper-triangle station pairs ``(i, j)`` with ``i <= j``."""
    if n_stations < 1:
        raise ValueError("n_stations must be at least 1.")
    k = 0 if include_autocorr else 1
    ii, jj = np.triu_indices(n_stations, k=k)
    if pair_mask is not None:
        mask = np.asarray(pair_mask, dtype=bool)
        if mask.shape != (n_stations, n_stations):
            raise ValueError(
                f"pair_mask shape {mask.shape} does not match ({n_stations}, {n_stations})."
            )
        keep = mask[ii, jj]
        ii, jj = ii[keep], jj[keep]
    return np.column_stack([ii, jj]).astype(np.int64)


def correlate_upper_triangle_blocked(
    spectra: np.ndarray,
    window_valid: np.ndarray,
    nfft: int,
    max_lag_samples: int,
    *,
    lag_guard_samples: int = 0,
    block_size: int = 64,
    pair_indices: np.ndarray | None = None,
    include_autocorr: bool = True,
    backend: str = "numpy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute upper-triangle daily CCF sums via source-blocked GEMM.

    Parameters
    ----------
    spectra : ndarray of complex, shape (n_stations, n_spec_cols, n_freqs)
        One-sided ``rfft`` spectra with ``n_freqs == nfft // 2 + 1``.
        For ``cc``, ``n_spec_cols == n_windows``; for ``pcc2``,
        ``n_spec_cols == 2 * n_windows``.
    window_valid : ndarray of bool, shape (n_stations, n_windows)
        True where a station's time window is valid (used for exact integer
        window counts).
    nfft : int
        FFT length used to build ``spectra``.
    max_lag_samples : int
        Maximum lag in samples (excluding guard band).
    lag_guard_samples : int, optional
        Extra lag samples on each side (e.g. for Hilbert transform in PWS).
    block_size : int, optional
        Number of virtual-source stations per GEMM block (assumed default: 64).
    pair_indices : ndarray of int, shape (n_pairs, 2), optional
        Subset of upper-triangle pairs ``(i, j)`` with ``0 <= i <= j < n_stations``.
        If None, all upper-triangle pairs are computed.
    include_autocorr : bool, optional
        Used only when ``pair_indices`` is None.
    backend : {"numpy", "cupy"}, optional
        Array execution backend for the blocked cross-spectrum GEMM and ``irfft``
        (default: ``"numpy"``). ``"cupy"`` requires the optional ``cupy`` package.

    Returns
    -------
    pairs : ndarray of int64, shape (n_pairs, 2)
    ccf_sum : ndarray of float64, shape (n_pairs, 2 * half_lag + 1)
        Sum across valid windows; rows with ``n_windows == 0`` are ``0.0``.
    n_windows : ndarray of int64, shape (n_pairs,)
        Number of jointly valid time windows per pair.
    """
    if backend not in {"numpy", "cupy"}:
        raise ValueError(f"backend must be 'numpy' or 'cupy', got {backend!r}.")
    xp = np
    xp_irfft = irfft
    if backend == "cupy":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "backend='cupy' requires CuPy; install a CUDA-matching cupy package."
            ) from exc
        xp = cp
        xp_irfft = cp.fft.irfft

    spec = np.asarray(spectra)
    valid = np.asarray(window_valid, dtype=bool)
    if spec.ndim != 3:
        raise ValueError("spectra must have shape (n_stations, n_spec_cols, n_freqs).")
    n_stations, _, n_freqs = spec.shape
    if n_freqs != nfft // 2 + 1:
        raise ValueError(
            f"spectra last axis ({n_freqs}) does not match nfft // 2 + 1 ({nfft // 2 + 1})."
        )
    if valid.ndim != 2 or valid.shape[0] != n_stations:
        raise ValueError("window_valid must have shape (n_stations, n_windows).")
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if max_lag_samples < 1 or lag_guard_samples < 0:
        raise ValueError("max_lag_samples must be >= 1 and lag_guard_samples >= 0.")

    half_lag = int(max_lag_samples + lag_guard_samples)
    if half_lag >= nfft // 2 + (nfft % 2):
        raise ValueError(
            f"max_lag_samples + lag_guard_samples ({half_lag}) exceeds nfft // 2 ({nfft // 2})."
        )

    if pair_indices is None:
        pairs = upper_triangle_pairs(n_stations, include_autocorr=include_autocorr)
    else:
        pairs = np.asarray(pair_indices, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pair_indices must have shape (n_pairs, 2).")
        if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 1] >= n_stations) or np.any(
            pairs[:, 0] > pairs[:, 1]
        ):
            raise ValueError(
                "pair_indices must satisfy 0 <= i <= j < n_stations for every pair."
            )

    n_pairs = pairs.shape[0]
    n_lags = 2 * half_lag + 1
    ccf_sum = np.zeros((n_pairs, n_lags), dtype=np.float64)
    n_win_out = np.zeros(n_pairs, dtype=np.int64)

    if n_pairs == 0 or valid.shape[1] == 0:
        return pairs, ccf_sum, n_win_out

    # Transpose once to (n_freqs, n_stations, n_spec_cols) for batched GEMM.
    spec_f = xp.asarray(np.ascontiguousarray(np.transpose(spec, (2, 0, 1))))
    valid_int = valid.astype(np.int32)

    for start_i in range(0, n_stations, block_size):
        end_i = min(start_i + block_size, n_stations)
        in_block = (pairs[:, 0] >= start_i) & (pairs[:, 0] < end_i)
        if not np.any(in_block):
            continue

        block_pair_pos = np.flatnonzero(in_block)
        local_i = pairs[block_pair_pos, 0] - start_i
        local_j = pairs[block_pair_pos, 1] - start_i

        # Only receivers j >= start_i participate in the upper triangle.
        s_i_conj = xp.conj(spec_f[:, start_i:end_i, :])  # (F, B_i, W)
        s_j_t = xp.swapaxes(spec_f[:, start_i:, :], 1, 2)  # (F, W, B_j)
        cross_spec = xp.matmul(s_i_conj, s_j_t)  # (F, B_i, B_j)

        # Select only requested pairs before irfft.
        pair_spec = xp.transpose(
            cross_spec[:, xp.asarray(local_i), xp.asarray(local_j)], (1, 0)
        )  # (P_blk, F)
        corr_full = xp_irfft(pair_spec, n=nfft, axis=-1)  # (P_blk, nfft)
        corr_cut_xp = xp.concatenate(
            [corr_full[:, nfft - half_lag :], corr_full[:, : half_lag + 1]],
            axis=-1,
        )
        corr_cut = (
            corr_cut_xp.get()
            if backend == "cupy"
            else np.asarray(corr_cut_xp, dtype=np.float64)
        )

        counts_block = valid_int[start_i:end_i] @ valid_int[start_i:].T
        pair_counts = counts_block[local_i, local_j].astype(np.int64)

        # Enforce zero-count semantics: zero rows where n_windows == 0.
        corr_cut = np.where(pair_counts[:, None] > 0, corr_cut, 0.0)
        ccf_sum[block_pair_pos] = corr_cut
        n_win_out[block_pair_pos] = pair_counts

    return pairs, ccf_sum, n_win_out


def reverse_ccf_zz(ccf: np.ndarray) -> np.ndarray:
    """Reverse a ZZ cross-correlation from pair ``(A, B)`` to ``(B, A)``.

    For ZZ, swapping source and receiver is pure time reversal:
    ``C_ZZ^{BA}(t) = C_ZZ^{AB}(-t)``.
    """
    return np.asarray(ccf)[..., ::-1].copy()


def correlate_zz_day(
    waveforms: np.ndarray,
    valid_mask: np.ndarray,
    sampling_rate_hz: float,
    window_length_s: float,
    max_lag_s: float,
    *,
    lag_guard_s: float = 0.0,
    step_s: float | None = None,
    cc_type: str = "cc",
    ram_half_width_s: float | None = None,
    ram_bandpass: BandpassSpec | None = None,
    pcc2_bandpass: BandpassSpec | None = None,
    whiten: WhitenSpec | None = None,
    whiten_mode: str = "vector_norm",
    taper_fraction: float = 0.05,
    circular: bool = False,
    nfft: int | None = None,
    block_size: int = 64,
    pair_indices: np.ndarray | None = None,
    include_autocorr: bool = True,
    spectrum_dtype: np.dtype | type = np.complex64,
    backend: str = "numpy",
) -> DailyCCFResult:
    """Compute one day of ZZ cross-correlation sums from continuous arrays.

    Pure-array entry point (no file IO). For ``cc_type == "cc"``, optional
    continuous-waveform RAM runs before windowing and whitening; for
    ``cc_type == "pcc2"``, RAM is skipped (PCC2 is amplitude-invariant) and
    optional ``pcc2_bandpass`` / ``whiten`` run before unit-modulus analytic
    phase extraction.
    """
    if cc_type not in CC_TYPES:
        raise ValueError(f"cc_type must be one of {CC_TYPES}, got {cc_type!r}.")
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive.")
    if window_length_s <= 0 or not np.isfinite(window_length_s):
        raise ValueError("window_length_s must be finite and positive.")
    if max_lag_s <= 0 or not np.isfinite(max_lag_s):
        raise ValueError("max_lag_s must be finite and positive.")
    if lag_guard_s < 0 or not np.isfinite(lag_guard_s):
        raise ValueError("lag_guard_s must be finite and non-negative.")

    window_samples = int(round(window_length_s * sampling_rate_hz))
    step_samples = (
        window_samples if step_s is None else int(round(step_s * sampling_rate_hz))
    )
    max_lag_samples = int(round(max_lag_s * sampling_rate_hz))
    lag_guard_samples = int(round(lag_guard_s * sampling_rate_hz))

    min_nfft = minimum_nfft(
        window_samples,
        max_lag_samples,
        lag_guard_samples,
        circular=circular,
        next_fast=True,
    )
    if nfft is None:
        used_nfft = min_nfft
    else:
        min_required = minimum_nfft(
            window_samples,
            max_lag_samples,
            lag_guard_samples,
            circular=circular,
            next_fast=False,
        )
        if circular and nfft != window_samples:
            raise ValueError("circular=True requires nfft == window_samples.")
        if not circular and nfft < min_required:
            raise ValueError(
                f"nfft ({nfft}) must be at least "
                f"window_samples + max_lag_samples + lag_guard_samples ({min_required}) "
                "to prevent wrap-around."
            )
        used_nfft = int(nfft)

    data = np.asarray(waveforms, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if cc_type == "cc" and ram_half_width_s is not None:
        data, mask = time_normalize_ram(
            data,
            mask,
            sampling_rate_hz,
            ram_half_width_s,
            bandpass=ram_bandpass,
        )

    windows, window_valid = slice_windows(
        data,
        mask,
        window_samples,
        step_samples,
        taper_fraction=taper_fraction,
    )

    if cc_type == "cc":
        spectra = prepare_cc_spectra(
            windows,
            window_valid,
            sampling_rate_hz,
            used_nfft,
            whiten=whiten,
            whiten_mode=whiten_mode,
            circular=circular,
            dtype=spectrum_dtype,
        )
    else:
        spectra = prepare_pcc2_spectra(
            windows,
            window_valid,
            sampling_rate_hz,
            used_nfft,
            bandpass=pcc2_bandpass,
            whiten=whiten,
            whiten_mode=whiten_mode,
            circular=circular,
            dtype=spectrum_dtype,
        )

    pairs, ccf_sum, n_windows = correlate_upper_triangle_blocked(
        spectra,
        window_valid,
        used_nfft,
        max_lag_samples,
        lag_guard_samples=lag_guard_samples,
        block_size=block_size,
        pair_indices=pair_indices,
        include_autocorr=include_autocorr,
        backend=backend,
    )
    lags = lag_axis_s(sampling_rate_hz, max_lag_s, lag_guard_s)
    return DailyCCFResult(
        lag_s=lags,
        pair_indices=pairs,
        ccf_sum=ccf_sum,
        n_windows=n_windows,
        cc_type=cc_type,
        sampling_rate_hz=float(sampling_rate_hz),
        max_lag_s=float(max_lag_s),
        lag_guard_s=float(lag_guard_s),
    )
