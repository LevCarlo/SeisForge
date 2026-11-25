"""Validated immutable domain models for ambient-noise cross-correlation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CC_TYPES",
    "STACK_METHODS",
    "WHITEN_MODES",
    "BandpassSpec",
    "DailyCCFResult",
    "StackedCCFResult",
    "WhitenSpec",
]

CC_TYPES = ("cc", "pcc2")
WHITEN_MODES = ("vector_norm", "classic")
STACK_METHODS = ("linear", "pws", "tf_pws")


def _readonly_array(array: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    arr = np.array(array, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class BandpassSpec:
    """Zero-phase Butterworth bandpass parameters for preprocessing.

    Parameters
    ----------
    freq_min_hz : float
        Lower corner frequency in Hz (strictly positive).
    freq_max_hz : float
        Upper corner frequency in Hz (strictly greater than ``freq_min_hz``).
    order : int, optional
        Butterworth filter order (assumed default: 4).
    zerophase : bool, optional
        If True (default), apply forward-backward filtering with
        ``scipy.signal.sosfiltfilt``.
    """

    freq_min_hz: float
    freq_max_hz: float
    order: int = 4
    zerophase: bool = True

    def __post_init__(self) -> None:
        if self.freq_min_hz <= 0 or not np.isfinite(self.freq_min_hz):
            raise ValueError("freq_min_hz must be finite and positive.")
        if self.freq_max_hz <= self.freq_min_hz or not np.isfinite(self.freq_max_hz):
            raise ValueError("freq_max_hz must be finite and greater than freq_min_hz.")
        if self.order < 1:
            raise ValueError("order must be at least 1.")


@dataclass(frozen=True)
class WhitenSpec:
    """Frequency-domain whitening parameters on the L-sample window spectrum.

    The spectral taper is a four-corner cosine taper:
    ``0 <= taper_min_hz <= freq_min_hz < freq_max_hz <= taper_max_hz``.

    Parameters
    ----------
    freq_min_hz : float
        Lower flat-pass corner frequency in Hz.
    freq_max_hz : float
        Upper flat-pass corner frequency in Hz.
    taper_min_hz : float
        Lower zero-response corner frequency in Hz.
    taper_max_hz : float
        Upper zero-response corner frequency in Hz.
    smooth_half_width_hz : float
        Half-width of the running spectral smoothing window in Hz (0 disables
        smoothing and normalizes bin-by-bin).
    water_level : float, optional
        Relative water level added to the smoothed spectral denominator as
        ``water_level * max(denom)`` (assumed default: 1e-6).
    """

    freq_min_hz: float
    freq_max_hz: float
    taper_min_hz: float
    taper_max_hz: float
    smooth_half_width_hz: float = 0.02
    water_level: float = 1e-6

    def __post_init__(self) -> None:
        corners = (
            self.taper_min_hz,
            self.freq_min_hz,
            self.freq_max_hz,
            self.taper_max_hz,
        )
        if not all(np.isfinite(c) for c in corners):
            raise ValueError("All whitening corner frequencies must be finite.")
        if not (
            0.0
            <= self.taper_min_hz
            <= self.freq_min_hz
            < self.freq_max_hz
            <= self.taper_max_hz
        ):
            raise ValueError(
                "Whitening corners must satisfy "
                "0 <= taper_min_hz <= freq_min_hz < freq_max_hz <= taper_max_hz."
            )
        if self.smooth_half_width_hz < 0 or not np.isfinite(self.smooth_half_width_hz):
            raise ValueError("smooth_half_width_hz must be finite and non-negative.")
        if self.water_level < 0 or not np.isfinite(self.water_level):
            raise ValueError("water_level must be finite and non-negative.")


@dataclass(frozen=True)
class DailyCCFResult:
    """Daily ZZ cross-correlation sums and valid window counts.

    Zero-count semantics
    --------------------
    ``ccf_sum`` stores the unnormalized sum across valid windows of that day;
    pairs with ``n_windows == 0`` store ``0.0`` in ``ccf_sum`` so daily sums
    can be added directly without NaN propagation. Consumers convert zero-count
    pairs to ``NaN`` when forming normalized CCFs.
    """

    lag_s: np.ndarray
    pair_indices: np.ndarray
    ccf_sum: np.ndarray
    n_windows: np.ndarray
    cc_type: str
    sampling_rate_hz: float
    max_lag_s: float
    lag_guard_s: float = 0.0

    def __post_init__(self) -> None:
        if self.cc_type not in CC_TYPES:
            raise ValueError(f"cc_type must be one of {CC_TYPES}, got {self.cc_type!r}.")
        if self.sampling_rate_hz <= 0 or not np.isfinite(self.sampling_rate_hz):
            raise ValueError("sampling_rate_hz must be finite and positive.")
        if self.max_lag_s <= 0 or not np.isfinite(self.max_lag_s):
            raise ValueError("max_lag_s must be finite and positive.")
        if self.lag_guard_s < 0 or not np.isfinite(self.lag_guard_s):
            raise ValueError("lag_guard_s must be finite and non-negative.")

        lag_s = _readonly_array(self.lag_s, dtype=np.float64)
        pair_indices = _readonly_array(self.pair_indices, dtype=np.int64)
        ccf_sum = _readonly_array(self.ccf_sum, dtype=np.float64)
        n_windows = _readonly_array(self.n_windows, dtype=np.int64)

        if lag_s.ndim != 1 or lag_s.size == 0:
            raise ValueError("lag_s must be a non-empty 1-D array.")
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("pair_indices must have shape (n_pairs, 2).")
        n_pairs = pair_indices.shape[0]
        if ccf_sum.shape != (n_pairs, lag_s.size):
            raise ValueError(
                f"ccf_sum shape {ccf_sum.shape} does not match "
                f"(n_pairs={n_pairs}, n_lags={lag_s.size})."
            )
        if n_windows.shape != (n_pairs,):
            raise ValueError(
                f"n_windows shape {n_windows.shape} does not match (n_pairs={n_pairs},)."
            )
        if np.any(n_windows < 0):
            raise ValueError("n_windows must be non-negative.")

        object.__setattr__(self, "lag_s", lag_s)
        object.__setattr__(self, "pair_indices", pair_indices)
        object.__setattr__(self, "ccf_sum", ccf_sum)
        object.__setattr__(self, "n_windows", n_windows)

    def daily_mean(self) -> np.ndarray:
        """Return the normalized daily CCF, with ``NaN`` where ``n_windows == 0``."""
        out = np.full(self.ccf_sum.shape, np.nan, dtype=np.float64)
        valid = self.n_windows > 0
        if np.any(valid):
            out[valid] = self.ccf_sum[valid] / self.n_windows[valid, None]
        return out


@dataclass(frozen=True)
class StackedCCFResult:
    """Multi-day stacked ZZ cross-correlation functions.

    Pairs with ``n_windows == 0`` (or ``n_days == 0``) hold ``NaN`` in ``ccf``.
    """

    lag_s: np.ndarray
    pair_indices: np.ndarray
    ccf: np.ndarray
    n_windows: np.ndarray
    n_days: np.ndarray
    method: str
    cc_type: str
    sampling_rate_hz: float
    max_lag_s: float
    lag_guard_s: float = 0.0
    pws_power: float | None = None

    def __post_init__(self) -> None:
        if self.method not in STACK_METHODS:
            raise ValueError(f"method must be one of {STACK_METHODS}, got {self.method!r}.")
        if self.cc_type not in CC_TYPES:
            raise ValueError(f"cc_type must be one of {CC_TYPES}, got {self.cc_type!r}.")
        if self.sampling_rate_hz <= 0 or not np.isfinite(self.sampling_rate_hz):
            raise ValueError("sampling_rate_hz must be finite and positive.")
        if self.max_lag_s <= 0 or not np.isfinite(self.max_lag_s):
            raise ValueError("max_lag_s must be finite and positive.")
        if self.lag_guard_s < 0 or not np.isfinite(self.lag_guard_s):
            raise ValueError("lag_guard_s must be finite and non-negative.")
        if self.method in ("pws", "tf_pws") and (self.pws_power is None or self.pws_power <= 0):
            raise ValueError(f"pws_power must be positive when method={self.method!r}.")

        lag_s = _readonly_array(self.lag_s, dtype=np.float64)
        pair_indices = _readonly_array(self.pair_indices, dtype=np.int64)
        ccf = _readonly_array(self.ccf, dtype=np.float64)
        n_windows = _readonly_array(self.n_windows, dtype=np.int64)
        n_days = _readonly_array(self.n_days, dtype=np.int64)

        if lag_s.ndim != 1 or lag_s.size == 0:
            raise ValueError("lag_s must be a non-empty 1-D array.")
        if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("pair_indices must have shape (n_pairs, 2).")
        n_pairs = pair_indices.shape[0]
        if ccf.shape != (n_pairs, lag_s.size):
            raise ValueError(
                f"ccf shape {ccf.shape} does not match "
                f"(n_pairs={n_pairs}, n_lags={lag_s.size})."
            )
        if n_windows.shape != (n_pairs,) or n_days.shape != (n_pairs,):
            raise ValueError("n_windows and n_days must have shape (n_pairs,).")
        if np.any(n_windows < 0) or np.any(n_days < 0):
            raise ValueError("n_windows and n_days must be non-negative.")

        object.__setattr__(self, "lag_s", lag_s)
        object.__setattr__(self, "pair_indices", pair_indices)
        object.__setattr__(self, "ccf", ccf)
        object.__setattr__(self, "n_windows", n_windows)
        object.__setattr__(self, "n_days", n_days)
