"""Multi-day stacking for ZZ ambient-noise cross-correlation functions.

Conventions
-----------
- **Production linear stack**:

      S_lin(p, t) = sum_d (w_d * ccf_sum_d(p, t)) / sum_d (w_d * n_windows_d(p))

  where ``w_d`` is 1 by default (window-weighted average across all valid
  windows). Days or pairs with ``n_windows_d(p) == 0`` contribute nothing;
  pairs with zero total windows across all selected days return ``NaN``.
  Accumulation uses ``float64``.
- **Phase-Weighted Stacking (PWS, experimental)** (Schimmel & Paulssen 1997):
  using daily mean CCFs ``s_d(t) = ccf_sum_d(t) / n_windows_d`` as stacking
  units, compute the analytic signal ``a_d(t) = s_d(t) + i H[s_d](t)`` on the
  full lag axis (including ``lag_guard_s`` to absorb Hilbert edge truncation),
  accumulate the instantaneous phase coherence over valid days only (missing
  days with ``n_windows == 0`` are excluded, never treated as zero phase):

      c(t) = (1 / N_days) * sum_{d: valid} a_d(t) / |a_d(t)|
      S_pws(t) = |c(t)|**nu * S_lin(t)

  The returned PWS waveform is trimmed to ``[-max_lag_s, +max_lag_s]``,
  dropping the ``lag_guard_s`` margin.
- **Time-Frequency Phase-Weighted Stacking (tf-PWS, experimental)**
  (Stockwell et al. 1996; Schimmel & Gallart 2007): using the forward Stockwell
  transform ``S_d(tau, f)`` on each valid daily mean CCF ``s_d(t)``, accumulate
  the 2-D time-frequency phase coherence across valid days,

      c(tau, f) = (1 / N_days) * sum_{d: valid} S_d(tau, f) / |S_d(tau, f)|
      S_tf_pws(tau, f) = |c(tau, f)|**nu * S[S_lin](tau, f)

  and apply the exact Stockwell (1996) inverse transform (time summation per
  frequency bin followed by ``Re[IFFT]``), then trim to
  ``[-max_lag_s, +max_lag_s]``.
- **No mixing of ``cc_type``**: combining ``"cc"`` and ``"pcc2"`` daily results
  raises ``ValueError``.
- **Streaming equivalence**: :class:`LinearStackAccumulator`,
  :class:`PWSStackAccumulator`, and :class:`TFPWSStackAccumulator` produce the
  same results as :func:`linear_stack_daily`, :func:`pws_stack_daily`, and
  :func:`tf_pws_stack_daily` while holding only one day in memory at a time.

References
----------
- Schimmel, M., & Paulssen, H. (1997). Noise reduction and detection of weak,
  coherent signals through phase-weighted stacks. Geophys. J. Int., 130(2),
  497-505, doi:10.1111/j.1365-246X.1997.tb05664.x.
- Schimmel, M., & Gallart, J. (2007). Frequency-dependent phase coherence for
  noise suppression in seismic array data. J. Geophys. Res., 112, B04303,
  doi:10.1029/2006JB004680.
- Stockwell, R. G., Mansinha, L., & Lowe, R. P. (1996). Localization of the
  complex spectrum: the S transform. IEEE Trans. Signal Process., 44(4),
  998-1001, doi:10.1109/78.492555.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy import ndimage, signal
from scipy.fft import fft, fftfreq, ifft

from .models import DailyCCFResult, StackedCCFResult

__all__ = [
    "DAY_WEIGHTING_MODES",
    "LinearStackAccumulator",
    "PWSStackAccumulator",
    "TFPWSStackAccumulator",
    "inverse_stockwell_transform",
    "linear_stack_daily",
    "pws_stack_daily",
    "stockwell_transform",
    "tf_pws_stack_daily",
    "trim_lag_guard",
]

DAY_WEIGHTING_MODES = ("windows", "equal")


def trim_lag_guard(
    lag_s: np.ndarray,
    ccf: np.ndarray,
    max_lag_s: float,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim ``lag_guard_s`` margins from a lag axis and CCF array."""
    lags = np.asarray(lag_s, dtype=np.float64)
    arr = np.asarray(ccf)
    if lags.ndim != 1 or arr.shape[-1] != lags.size:
        raise ValueError("lag_s must be 1-D and match the last axis of ccf.")
    keep_half = int(round(max_lag_s * sampling_rate_hz))
    center = lags.size // 2
    if keep_half > center:
        raise ValueError(
            f"max_lag_s ({max_lag_s}) exceeds available lag half-width ({center / sampling_rate_hz})."
        )
    sl = slice(center - keep_half, center + keep_half + 1)
    return lags[sl].copy(), arr[..., sl].copy()


def _validate_compatible(template: DailyCCFResult, day: DailyCCFResult) -> None:
    if day.cc_type != template.cc_type:
        raise ValueError(
            f"Cannot mix cc_type={template.cc_type!r} and {day.cc_type!r} in one stack."
        )
    if not np.isclose(day.sampling_rate_hz, template.sampling_rate_hz, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Incompatible sampling_rate_hz: {day.sampling_rate_hz} vs {template.sampling_rate_hz}."
        )
    if not np.isclose(day.max_lag_s, template.max_lag_s, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Incompatible max_lag_s: {day.max_lag_s} vs {template.max_lag_s}."
        )
    if not np.isclose(day.lag_guard_s, template.lag_guard_s, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Incompatible lag_guard_s: {day.lag_guard_s} vs {template.lag_guard_s}."
        )
    if day.lag_s.shape != template.lag_s.shape or not np.allclose(
        day.lag_s, template.lag_s, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Daily CCF lag axes do not match.")
    if day.pair_indices.shape != template.pair_indices.shape or not np.array_equal(
        day.pair_indices, template.pair_indices
    ):
        raise ValueError("Daily CCF pair_indices do not match.")


def _normalize_pair_mask(
    pair_mask: np.ndarray | None,
    n_pairs: int,
) -> np.ndarray:
    if pair_mask is None:
        return np.ones(n_pairs, dtype=bool)
    mask = np.asarray(pair_mask, dtype=bool)
    if mask.shape != (n_pairs,):
        raise ValueError(f"pair_mask must have shape ({n_pairs},), got {mask.shape}.")
    return mask


class LinearStackAccumulator:
    """Streaming ``float64`` linear stacker over daily ZZ CCFs.

    Parameters
    ----------
    day_weighting : {"windows", "equal"}, optional
        ``"windows"`` (default) sums daily ``ccf_sum`` and divides by total
        ``n_windows`` (equivalent to weighting each valid window equally across
        the entire deployment). ``"equal"`` averages daily means with equal
        weight per valid day.
    """

    def __init__(self, *, day_weighting: str = "windows") -> None:
        if day_weighting not in DAY_WEIGHTING_MODES:
            raise ValueError(
                f"day_weighting must be one of {DAY_WEIGHTING_MODES}, got {day_weighting!r}."
            )
        self._day_weighting = day_weighting
        self._template: DailyCCFResult | None = None
        self._numerator: np.ndarray | None = None
        self._weight_sum: np.ndarray | None = None
        self._n_windows: np.ndarray | None = None
        self._n_days: np.ndarray | None = None

    def add_day(
        self,
        day: DailyCCFResult,
        *,
        pair_mask: np.ndarray | None = None,
        weight: float = 1.0,
    ) -> None:
        """Accumulate one day's ZZ CCF into the linear stack."""
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("weight must be finite and non-negative.")
        if self._template is None:
            self._template = day
            self._numerator = np.zeros(day.ccf_sum.shape, dtype=np.float64)
            self._weight_sum = np.zeros(day.n_windows.shape, dtype=np.float64)
            self._n_windows = np.zeros(day.n_windows.shape, dtype=np.int64)
            self._n_days = np.zeros(day.n_windows.shape, dtype=np.int64)
        else:
            _validate_compatible(self._template, day)

        if weight == 0.0:
            return

        keep = _normalize_pair_mask(pair_mask, day.n_windows.size) & (day.n_windows > 0)
        if not np.any(keep):
            return

        assert self._numerator is not None
        assert self._weight_sum is not None
        assert self._n_windows is not None
        assert self._n_days is not None

        if self._day_weighting == "windows":
            self._numerator[keep] += weight * day.ccf_sum[keep]
            self._weight_sum[keep] += weight * day.n_windows[keep]
        else:
            daily_mean = day.ccf_sum[keep] / day.n_windows[keep, None]
            self._numerator[keep] += weight * daily_mean
            self._weight_sum[keep] += weight

        self._n_windows[keep] += day.n_windows[keep]
        self._n_days[keep] += 1

    def finalize(self, *, trim_guard: bool = False) -> StackedCCFResult:
        """Return the normalized linear stack."""
        if self._template is None:
            raise ValueError("Cannot finalize an empty LinearStackAccumulator.")
        assert self._numerator is not None
        assert self._weight_sum is not None
        assert self._n_windows is not None
        assert self._n_days is not None

        ccf = np.full(self._numerator.shape, np.nan, dtype=np.float64)
        valid = self._weight_sum > 0.0
        if np.any(valid):
            ccf[valid] = self._numerator[valid] / self._weight_sum[valid, None]

        lag_s = self._template.lag_s
        lag_guard_s = self._template.lag_guard_s
        if trim_guard and lag_guard_s > 0.0:
            lag_s, ccf = trim_lag_guard(
                lag_s, ccf, self._template.max_lag_s, self._template.sampling_rate_hz
            )
            lag_guard_s = 0.0

        return StackedCCFResult(
            lag_s=lag_s,
            pair_indices=self._template.pair_indices,
            ccf=ccf,
            n_windows=self._n_windows,
            n_days=self._n_days,
            method="linear",
            cc_type=self._template.cc_type,
            sampling_rate_hz=self._template.sampling_rate_hz,
            max_lag_s=self._template.max_lag_s,
            lag_guard_s=lag_guard_s,
        )


class PWSStackAccumulator:
    """Streaming Phase-Weighted Stack (Schimmel & Paulssen 1997, experimental).

    Parameters
    ----------
    power : float, optional
        Phase-coherence exponent ``nu`` (default: 2.0, Schimmel & Paulssen 1997).
    day_weighting : {"windows", "equal"}, optional
        Weighting used for the linear stack factor ``S_lin``. Phase coherence
        weights each valid day equally.
    amplitude_water_level : float, optional
        Relative analytic-envelope floor ``amplitude_water_level * max(|a_d|)``
        below which a sample's unit-phase contribution is zeroed
        (assumed default: 1e-12).
    """

    def __init__(
        self,
        *,
        power: float = 2.0,
        day_weighting: str = "windows",
        amplitude_water_level: float = 1e-12,
    ) -> None:
        if power <= 0 or not np.isfinite(power):
            raise ValueError("power must be finite and positive.")
        if amplitude_water_level < 0 or not np.isfinite(amplitude_water_level):
            raise ValueError("amplitude_water_level must be finite and non-negative.")
        self._power = float(power)
        self._amplitude_water_level = float(amplitude_water_level)
        self._linear = LinearStackAccumulator(day_weighting=day_weighting)
        self._template: DailyCCFResult | None = None
        self._phase_sum: np.ndarray | None = None
        self._n_days_phase: np.ndarray | None = None

    def add_day(
        self,
        day: DailyCCFResult,
        *,
        pair_mask: np.ndarray | None = None,
        weight: float = 1.0,
    ) -> None:
        """Accumulate one day's ZZ CCF into the PWS stack."""
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("weight must be finite and non-negative.")
        if self._template is None:
            self._template = day
            self._phase_sum = np.zeros(day.ccf_sum.shape, dtype=np.complex128)
            self._n_days_phase = np.zeros(day.n_windows.shape, dtype=np.int64)
        else:
            _validate_compatible(self._template, day)

        self._linear.add_day(day, pair_mask=pair_mask, weight=weight)
        if weight == 0.0:
            return

        keep = _normalize_pair_mask(pair_mask, day.n_windows.size) & (day.n_windows > 0)
        if not np.any(keep):
            return

        assert self._phase_sum is not None
        assert self._n_days_phase is not None

        daily_mean = day.ccf_sum[keep] / day.n_windows[keep, None]
        analytic = signal.hilbert(daily_mean, axis=-1)
        amp = np.abs(analytic)
        max_amp = np.max(amp, axis=-1, keepdims=True)
        floor = self._amplitude_water_level * max_amp
        good = (max_amp > 0.0) & (amp > floor)
        safe_amp = np.where(good, amp, 1.0)
        unit_phase = np.where(good, analytic / safe_amp, 0.0 + 0.0j)

        self._phase_sum[keep] += unit_phase
        self._n_days_phase[keep] += 1

    def finalize(self) -> StackedCCFResult:
        """Return the phase-weighted stack trimmed to ``±max_lag_s``."""
        if self._template is None:
            raise ValueError("Cannot finalize an empty PWSStackAccumulator.")
        assert self._phase_sum is not None
        assert self._n_days_phase is not None

        linear_res = self._linear.finalize(trim_guard=False)
        ccf_pws = np.full(linear_res.ccf.shape, np.nan, dtype=np.float64)
        valid = self._n_days_phase > 0
        if np.any(valid):
            coherence = np.abs(
                self._phase_sum[valid] / self._n_days_phase[valid, None]
            )
            ccf_pws[valid] = (coherence**self._power) * linear_res.ccf[valid]

        lag_s, ccf_trimmed = trim_lag_guard(
            linear_res.lag_s,
            ccf_pws,
            linear_res.max_lag_s,
            linear_res.sampling_rate_hz,
        )
        return StackedCCFResult(
            lag_s=lag_s,
            pair_indices=linear_res.pair_indices,
            ccf=ccf_trimmed,
            n_windows=linear_res.n_windows,
            n_days=linear_res.n_days,
            method="pws",
            cc_type=linear_res.cc_type,
            sampling_rate_hz=linear_res.sampling_rate_hz,
            max_lag_s=linear_res.max_lag_s,
            lag_guard_s=0.0,
            pws_power=self._power,
        )


def linear_stack_daily(
    daily_results: Iterable[DailyCCFResult],
    *,
    day_weighting: str = "windows",
    pair_masks: Sequence[np.ndarray | None] | None = None,
    day_weights: Sequence[float] | None = None,
    trim_guard: bool = False,
) -> StackedCCFResult:
    """Linearly stack a sequence of daily ZZ CCFs."""
    days = list(daily_results)
    if not days:
        raise ValueError("daily_results must contain at least one DailyCCFResult.")
    if pair_masks is not None and len(pair_masks) != len(days):
        raise ValueError("pair_masks length must match daily_results.")
    if day_weights is not None and len(day_weights) != len(days):
        raise ValueError("day_weights length must match daily_results.")

    acc = LinearStackAccumulator(day_weighting=day_weighting)
    for idx, day in enumerate(days):
        pmask = None if pair_masks is None else pair_masks[idx]
        w = 1.0 if day_weights is None else float(day_weights[idx])
        acc.add_day(day, pair_mask=pmask, weight=w)
    return acc.finalize(trim_guard=trim_guard)


def pws_stack_daily(
    daily_results: Iterable[DailyCCFResult],
    *,
    power: float = 2.0,
    day_weighting: str = "windows",
    pair_masks: Sequence[np.ndarray | None] | None = None,
    day_weights: Sequence[float] | None = None,
    amplitude_water_level: float = 1e-12,
) -> StackedCCFResult:
    """Compute a Phase-Weighted Stack (Schimmel & Paulssen 1997, experimental)."""
    days = list(daily_results)
    if not days:
        raise ValueError("daily_results must contain at least one DailyCCFResult.")
    if pair_masks is not None and len(pair_masks) != len(days):
        raise ValueError("pair_masks length must match daily_results.")
    if day_weights is not None and len(day_weights) != len(days):
        raise ValueError("day_weights length must match daily_results.")

    acc = PWSStackAccumulator(
        power=power,
        day_weighting=day_weighting,
        amplitude_water_level=amplitude_water_level,
    )
    for idx, day in enumerate(days):
        pmask = None if pair_masks is None else pair_masks[idx]
        w = 1.0 if day_weights is None else float(day_weights[idx])
        acc.add_day(day, pair_mask=pmask, weight=w)
    return acc.finalize()


def _resolve_stockwell_grid(
    n_samples: int,
    sampling_rate_hz: float,
    *,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    nfft: int | None = None,
) -> tuple[int, int, int, np.ndarray]:
    """Validate and resolve Stockwell FFT length and one-sided frequency bins."""
    if n_samples < 2:
        raise ValueError("Input signal must have at least 2 samples.")
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive.")

    if nfft is None:
        resolved_nfft = 1 << (int(n_samples) - 1).bit_length()
    else:
        resolved_nfft = int(nfft)
        if resolved_nfft < n_samples:
            raise ValueError(f"nfft ({resolved_nfft}) must be >= n_samples ({n_samples}).")

    nq = resolved_nfft // 2
    df_hz = float(sampling_rate_hz) / float(resolved_nfft)
    nyquist_hz = 0.5 * float(sampling_rate_hz)

    if freq_min_hz is not None and (freq_min_hz < 0 or not np.isfinite(freq_min_hz)):
        raise ValueError("freq_min_hz must be finite and non-negative.")
    if freq_max_hz is not None and (freq_max_hz <= 0 or not np.isfinite(freq_max_hz)):
        raise ValueError("freq_max_hz must be finite and positive.")
    if (
        freq_min_hz is not None
        and freq_max_hz is not None
        and freq_max_hz <= freq_min_hz
    ):
        raise ValueError("freq_max_hz must be greater than freq_min_hz.")
    if freq_min_hz is not None and freq_min_hz > nyquist_hz:
        raise ValueError(
            f"freq_min_hz ({freq_min_hz}) exceeds Nyquist ({nyquist_hz} Hz)."
        )

    lo = 0 if freq_min_hz is None else max(0, int(np.floor(freq_min_hz / df_hz)))
    hi = nq if freq_max_hz is None else min(nq, int(np.ceil(freq_max_hz / df_hz)))
    if lo > hi:
        raise ValueError(
            f"Empty frequency bin range [{lo}, {hi}] for "
            f"[{freq_min_hz}, {freq_max_hz}] Hz."
        )
    freqs_hz = np.arange(lo, hi + 1, dtype=np.float64) * df_hz
    return resolved_nfft, lo, hi, freqs_hz


def _build_stockwell_gauss_table(
    resolved_nfft: int,
    lo: int,
    hi: int,
    sigma: float,
) -> np.ndarray:
    """Precompute Stockwell Gaussian frequency windows for bins ``[lo, hi]``."""
    m_bins = fftfreq(resolved_nfft) * float(resolved_nfft)
    m_sq = m_bins * m_bins
    gauss_scale = 2.0 * (np.pi * float(sigma)) ** 2
    n_bins = np.arange(lo, hi + 1, dtype=np.float64)
    safe_n = np.where(n_bins > 0.0, n_bins, 1.0)
    table = np.exp(-gauss_scale * m_sq[None, :] / (safe_n[:, None] ** 2))
    if lo == 0:
        table[0, :] = 0.0
    return table


def _stockwell_transform_core(
    traces: np.ndarray,
    resolved_nfft: int,
    lo: int,
    hi: int,
    gauss_table: np.ndarray,
    *,
    freq_block: int = 32,
) -> np.ndarray:
    """Batched forward Stockwell transform using contiguous spectrum slicing."""
    n_traces, n_samples = traces.shape
    padded = np.zeros((n_traces, resolved_nfft), dtype=np.float64)
    padded[:, :n_samples] = traces

    # Analytic signal spectrum: double strictly positive frequencies, zero negative.
    spectrum = fft(padded, axis=-1)
    analytic_spec = np.zeros_like(spectrum)
    analytic_spec[:, 0] = spectrum[:, 0]
    half_pos = (resolved_nfft + 1) // 2
    analytic_spec[:, 1:half_pos] = 2.0 * spectrum[:, 1:half_pos]
    if resolved_nfft % 2 == 0:
        analytic_spec[:, resolved_nfft // 2] = spectrum[:, resolved_nfft // 2]

    # Doubling along the frequency axis turns circular shifts into contiguous slices.
    analytic_doubled = np.concatenate([analytic_spec, analytic_spec], axis=-1)
    n_freq = hi - lo + 1
    st = np.empty((n_traces, n_freq, resolved_nfft), dtype=np.complex128)

    row_start = 0
    if lo == 0:
        mean_val = np.mean(padded, axis=-1, keepdims=True)
        st[:, 0, :] = mean_val.astype(np.complex128)
        row_start = 1

    # Batch IFFTs across frequency blocks to minimize Python/FFT dispatch overhead.
    for b_start in range(row_start, n_freq, freq_block):
        b_end = min(b_start + freq_block, n_freq)
        b_len = b_end - b_start
        shifted = np.empty((n_traces, b_len, resolved_nfft), dtype=np.complex128)
        for j, row in enumerate(range(b_start, b_end)):
            n_bin = lo + row
            shifted[:, j, :] = analytic_doubled[:, n_bin : n_bin + resolved_nfft]
        shifted *= gauss_table[None, b_start:b_end, :]
        st[:, b_start:b_end, :] = ifft(shifted, axis=-1)

    return st


def stockwell_transform(
    data: np.ndarray,
    sampling_rate_hz: float,
    *,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    sigma: float = 1.0,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the forward Stockwell transform (Stockwell et al. 1996).

    Parameters
    ----------
    data : ndarray, shape (n_samples,) or (n_traces, n_samples)
        Real-valued input waveform(s). Not mutated.
    sampling_rate_hz : float
        Sampling rate in Hz.
    freq_min_hz, freq_max_hz : float or None, optional
        Optional one-sided frequency bounds in Hz. Defaults to ``[0, Nyquist]``.
    sigma : float, optional
        Gaussian window width factor in the generalized Stockwell transform
        (default: 1.0, Stockwell et al. 1996). Larger values increase frequency
        resolution and decrease time resolution.
    nfft : int or None, optional
        FFT length (zero-padded). Defaults to the next power of two
        ``>= n_samples``.

    Returns
    -------
    freqs_hz : ndarray of float64, shape (n_freq,)
        Frequencies in Hz corresponding to the frequency axis of ``st``.
    st : ndarray of complex128, shape (..., n_freq, nfft)
        Complex Stockwell time-frequency spectrum (matching input leading
        dimensions: ``(n_freq, nfft)`` for 1-D input, ``(n_traces, n_freq, nfft)``
        for 2-D input).
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"data must be 1-D or 2-D, got {arr.ndim}-D.")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive.")

    was_1d = arr.ndim == 1
    traces = arr[None, :] if was_1d else arr
    _, n_samples = traces.shape

    resolved_nfft, lo, hi, freqs_hz = _resolve_stockwell_grid(
        n_samples,
        sampling_rate_hz,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        nfft=nfft,
    )
    gauss_table = _build_stockwell_gauss_table(resolved_nfft, lo, hi, sigma)
    st = _stockwell_transform_core(traces, resolved_nfft, lo, hi, gauss_table)

    if was_1d:
        return freqs_hz, st[0]
    return freqs_hz, st


def inverse_stockwell_transform(
    st: np.ndarray,
    freqs_hz: np.ndarray,
    sampling_rate_hz: float,
    *,
    n_samples: int | None = None,
) -> np.ndarray:
    """Compute the exact Stockwell (1996) inverse S-transform.

    Sums ``st`` along the time axis for each frequency bin to recover the
    analytic Fourier spectrum and returns ``Re[IFFT]``.

    Parameters
    ----------
    st : ndarray of complex, shape (n_freq, nfft) or (n_traces, n_freq, nfft)
        Time-frequency array returned by :func:`stockwell_transform` (possibly
        weighted in the time-frequency domain).
    freqs_hz : ndarray of float, shape (n_freq,)
        Frequency axis in Hz corresponding to ``st``.
    sampling_rate_hz : float
        Sampling rate in Hz.
    n_samples : int or None, optional
        If provided, truncate the reconstructed time series to the first
        ``n_samples`` samples.

    Returns
    -------
    reconstructed : ndarray of float64, shape (n_out,) or (n_traces, n_out)
        Reconstructed real time series.
    """
    arr = np.asarray(st, dtype=np.complex128)
    if arr.ndim not in (2, 3):
        raise ValueError(f"st must be 2-D or 3-D, got {arr.ndim}-D.")
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size != arr.shape[-2]:
        raise ValueError("freqs_hz must be 1-D and match the frequency axis of st.")
    if sampling_rate_hz <= 0 or not np.isfinite(sampling_rate_hz):
        raise ValueError("sampling_rate_hz must be finite and positive.")

    was_2d = arr.ndim == 2
    st_3d = arr[None, :, :] if was_2d else arr
    n_traces, _, nfft = st_3d.shape

    if n_samples is not None and not (1 <= n_samples <= nfft):
        raise ValueError(f"n_samples ({n_samples}) must lie in [1, {nfft}].")

    df_hz = float(sampling_rate_hz) / float(nfft)
    bins = np.rint(freqs / df_hz).astype(np.int64)
    if np.any(bins < 0) or np.any(bins > nfft // 2):
        raise ValueError("freqs_hz maps outside the one-sided FFT bin range [0, nfft // 2].")

    analytic_spec = np.zeros((n_traces, nfft), dtype=np.complex128)
    analytic_spec[:, bins] = np.sum(st_3d, axis=-1)
    reconstructed = np.real(ifft(analytic_spec, axis=-1))

    if n_samples is not None:
        reconstructed = reconstructed[:, :n_samples]
    reconstructed = np.ascontiguousarray(reconstructed, dtype=np.float64)
    return reconstructed[0] if was_2d else reconstructed


class TFPWSStackAccumulator:
    """Streaming Time-Frequency Phase-Weighted Stack (Schimmel & Gallart 2007).

    Uses the Stockwell transform (Stockwell et al. 1996) to compute
    frequency-dependent instantaneous phase coherence across valid daily CCFs,
    weights the S-transform of the linear stack by ``|c(tau, f)|**power``, and
    inverts via the Stockwell (1996) inverse transform.

    Note on defaults: ``power`` defaults to ``2.0`` for consistency with
    :class:`PWSStackAccumulator`; ANCC's ``TF_PWS`` utility defaults to
    ``power=1.0`` and ``day_weighting="equal"``.

    Parameters
    ----------
    power : float, optional
        Phase-coherence exponent ``nu`` (default: 2.0).
    day_weighting : {"windows", "equal"}, optional
        Weighting used for the linear stack factor ``S_lin``. Phase coherence
        weights each valid day equally.
    freq_min_hz, freq_max_hz : float or None, optional
        Optional frequency band in Hz for the Stockwell transform (defaults to
        ``[0, Nyquist]``).
    sigma : float, optional
        Gaussian window width factor in the Stockwell transform (default: 1.0).
    amplitude_water_level : float, optional
        Relative time-frequency amplitude floor ``amplitude_water_level *
        max(|S_d|)`` below which unit-phase contributions are zeroed
        (assumed default: 1e-12).
    smooth_samples : int, optional
        Optional half-width in samples for centered time-axis smoothing of the
        phase weight ``|c(tau, f)|**power`` (default: 0, disabled).
    normalize_weights : bool, optional
        If True, normalize each frequency row of the phase weight by its
        maximum over time (default: False, disabled).
    """

    def __init__(
        self,
        *,
        power: float = 2.0,
        day_weighting: str = "windows",
        freq_min_hz: float | None = None,
        freq_max_hz: float | None = None,
        sigma: float = 1.0,
        amplitude_water_level: float = 1e-12,
        smooth_samples: int = 0,
        normalize_weights: bool = False,
    ) -> None:
        if power <= 0 or not np.isfinite(power):
            raise ValueError("power must be finite and positive.")
        if sigma <= 0 or not np.isfinite(sigma):
            raise ValueError("sigma must be finite and positive.")
        if amplitude_water_level < 0 or not np.isfinite(amplitude_water_level):
            raise ValueError("amplitude_water_level must be finite and non-negative.")
        if smooth_samples < 0:
            raise ValueError("smooth_samples must be non-negative.")

        self._power = float(power)
        self._freq_min_hz = freq_min_hz
        self._freq_max_hz = freq_max_hz
        self._sigma = float(sigma)
        self._amplitude_water_level = float(amplitude_water_level)
        self._smooth_samples = int(smooth_samples)
        self._normalize_weights = bool(normalize_weights)

        self._linear = LinearStackAccumulator(day_weighting=day_weighting)
        self._template: DailyCCFResult | None = None
        self._freqs_hz: np.ndarray | None = None
        self._nfft: int | None = None
        self._lo: int = 0
        self._hi: int = 0
        self._gauss_table: np.ndarray | None = None
        self._tf_phase_sum: np.ndarray | None = None
        self._n_days_phase: np.ndarray | None = None

    def add_day(
        self,
        day: DailyCCFResult,
        *,
        pair_mask: np.ndarray | None = None,
        weight: float = 1.0,
    ) -> None:
        """Accumulate one day's ZZ CCF into the tf-PWS stack."""
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("weight must be finite and non-negative.")
        if self._template is None:
            nfft, lo, hi, freqs_hz = _resolve_stockwell_grid(
                day.lag_s.size,
                day.sampling_rate_hz,
                freq_min_hz=self._freq_min_hz,
                freq_max_hz=self._freq_max_hz,
            )
            self._template = day
            self._nfft = nfft
            self._lo = lo
            self._hi = hi
            self._freqs_hz = freqs_hz
            self._gauss_table = _build_stockwell_gauss_table(nfft, lo, hi, self._sigma)
            self._tf_phase_sum = np.zeros(
                (day.n_windows.size, freqs_hz.size, nfft),
                dtype=np.complex128,
            )
            self._n_days_phase = np.zeros(day.n_windows.shape, dtype=np.int64)
        else:
            _validate_compatible(self._template, day)

        self._linear.add_day(day, pair_mask=pair_mask, weight=weight)
        if weight == 0.0:
            return

        keep = _normalize_pair_mask(pair_mask, day.n_windows.size) & (day.n_windows > 0)
        if not np.any(keep):
            return

        assert self._tf_phase_sum is not None
        assert self._n_days_phase is not None
        assert self._nfft is not None
        assert self._gauss_table is not None

        daily_mean = day.ccf_sum[keep] / day.n_windows[keep, None]
        st = _stockwell_transform_core(
            daily_mean,
            self._nfft,
            self._lo,
            self._hi,
            self._gauss_table,
        )
        amp = np.abs(st)
        max_amp = np.max(amp, axis=(-2, -1), keepdims=True)
        floor = self._amplitude_water_level * max_amp
        good = (max_amp > 0.0) & (amp > floor)
        safe_amp = np.where(good, amp, 1.0)
        unit_phase = np.where(good, st / safe_amp, 0.0 + 0.0j)

        self._tf_phase_sum[keep] += unit_phase
        self._n_days_phase[keep] += 1

    def finalize(self) -> StackedCCFResult:
        """Return the time-frequency phase-weighted stack trimmed to ``±max_lag_s``."""
        if self._template is None:
            raise ValueError("Cannot finalize an empty TFPWSStackAccumulator.")
        assert self._tf_phase_sum is not None
        assert self._n_days_phase is not None
        assert self._freqs_hz is not None
        assert self._nfft is not None

        linear_res = self._linear.finalize(trim_guard=False)
        ccf_tf_pws = np.full(linear_res.ccf.shape, np.nan, dtype=np.float64)
        valid = self._n_days_phase > 0
        if np.any(valid):
            coherence = np.abs(
                self._tf_phase_sum[valid] / self._n_days_phase[valid, None, None]
            )
            tf_weight = coherence**self._power

            if self._normalize_weights:
                row_max = np.max(tf_weight, axis=-1, keepdims=True)
                tf_weight = np.where(row_max > 0.0, tf_weight / np.where(row_max > 0.0, row_max, 1.0), 0.0)

            if self._smooth_samples > 0:
                tf_weight = ndimage.uniform_filter1d(
                    tf_weight,
                    size=2 * self._smooth_samples + 1,
                    axis=-1,
                    mode="nearest",
                )

            assert self._gauss_table is not None
            st_lin = _stockwell_transform_core(
                linear_res.ccf[valid],
                self._nfft,
                self._lo,
                self._hi,
                self._gauss_table,
            )
            ccf_tf_pws[valid] = inverse_stockwell_transform(
                st_lin * tf_weight,
                self._freqs_hz,
                linear_res.sampling_rate_hz,
                n_samples=linear_res.lag_s.size,
            )

        lag_s, ccf_trimmed = trim_lag_guard(
            linear_res.lag_s,
            ccf_tf_pws,
            linear_res.max_lag_s,
            linear_res.sampling_rate_hz,
        )
        return StackedCCFResult(
            lag_s=lag_s,
            pair_indices=linear_res.pair_indices,
            ccf=ccf_trimmed,
            n_windows=linear_res.n_windows,
            n_days=linear_res.n_days,
            method="tf_pws",
            cc_type=linear_res.cc_type,
            sampling_rate_hz=linear_res.sampling_rate_hz,
            max_lag_s=linear_res.max_lag_s,
            lag_guard_s=0.0,
            pws_power=self._power,
        )


def tf_pws_stack_daily(
    daily_results: Iterable[DailyCCFResult],
    *,
    power: float = 2.0,
    day_weighting: str = "windows",
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
    sigma: float = 1.0,
    pair_masks: Sequence[np.ndarray | None] | None = None,
    day_weights: Sequence[float] | None = None,
    amplitude_water_level: float = 1e-12,
    smooth_samples: int = 0,
    normalize_weights: bool = False,
) -> StackedCCFResult:
    """Compute a Time-Frequency Phase-Weighted Stack (Schimmel & Gallart 2007)."""
    days = list(daily_results)
    if not days:
        raise ValueError("daily_results must contain at least one DailyCCFResult.")
    if pair_masks is not None and len(pair_masks) != len(days):
        raise ValueError("pair_masks length must match daily_results.")
    if day_weights is not None and len(day_weights) != len(days):
        raise ValueError("day_weights length must match daily_results.")

    acc = TFPWSStackAccumulator(
        power=power,
        day_weighting=day_weighting,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        sigma=sigma,
        amplitude_water_level=amplitude_water_level,
        smooth_samples=smooth_samples,
        normalize_weights=normalize_weights,
    )
    for idx, day in enumerate(days):
        pmask = None if pair_masks is None else pair_masks[idx]
        w = 1.0 if day_weights is None else float(day_weights[idx])
        acc.add_day(day, pair_mask=pmask, weight=w)
    return acc.finalize()
