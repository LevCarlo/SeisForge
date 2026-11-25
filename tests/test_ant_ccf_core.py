"""Independent verification tests for seisforge.ant.ccf (ZZ core, PCC2, stacking)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from seisforge.ant.ccf import (
    BandpassSpec,
    DailyCCFResult,
    LinearStackAccumulator,
    PWSStackAccumulator,
    TFPWSStackAccumulator,
    WhitenSpec,
    correlate_upper_triangle_blocked,
    correlate_zz_day,
    expand_mask_for_ram,
    inverse_stockwell_transform,
    lag_axis_s,
    linear_stack_daily,
    minimum_nfft,
    prepare_cc_spectra,
    prepare_pcc2_spectra,
    pws_stack_daily,
    reverse_ccf_zz,
    slice_windows,
    spectral_whiten_windows,
    stockwell_transform,
    tf_pws_stack_daily,
    time_normalize_ram,
)


def _brute_force_cc_from_windows(
    windows: np.ndarray,
    window_valid: np.ndarray,
    max_lag_samples: int,
    lag_guard_samples: int = 0,
    *,
    include_autocorr: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent float64 pair-by-pair, window-by-window reference using np.correlate."""
    n_stations, n_windows, n_samples = windows.shape
    half_lag = max_lag_samples + lag_guard_samples
    center = n_samples - 1
    sl = slice(center - half_lag, center + half_lag + 1)

    pairs = []
    sums = []
    counts = []
    for i in range(n_stations):
        j_start = i if include_autocorr else i + 1
        for j in range(j_start, n_stations):
            acc = np.zeros(2 * half_lag + 1, dtype=np.float64)
            cnt = 0
            for w in range(n_windows):
                if window_valid[i, w] and window_valid[j, w]:
                    # C_AB(t) = sum_tau u_A(tau) u_B(tau + t) == np.correlate(u_B, u_A, "full")
                    full = np.correlate(windows[j, w], windows[i, w], mode="full")
                    acc += full[sl]
                    cnt += 1
            pairs.append((i, j))
            sums.append(acc)
            counts.append(cnt)

    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def _brute_force_pcc2_from_windows(
    windows: np.ndarray,
    window_valid: np.ndarray,
    max_lag_samples: int,
    lag_guard_samples: int = 0,
    *,
    amplitude_water_level: float = 1e-12,
    include_autocorr: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent float64 complex analytic Schimmel (1999) nu=2 reference."""
    n_stations, n_windows, n_samples = windows.shape
    half_lag = max_lag_samples + lag_guard_samples
    center = n_samples - 1
    sl = slice(center - half_lag, center + half_lag + 1)

    unit = np.zeros((n_stations, n_windows, n_samples), dtype=np.complex128)
    for i in range(n_stations):
        for w in range(n_windows):
            if not window_valid[i, w]:
                continue
            analytic = signal.hilbert(windows[i, w])
            amp = np.abs(analytic)
            max_amp = float(np.max(amp))
            if max_amp > 0.0:
                good = amp > amplitude_water_level * max_amp
                unit[i, w, good] = analytic[good] / amp[good]

    pairs = []
    sums = []
    counts = []
    for i in range(n_stations):
        j_start = i if include_autocorr else i + 1
        for j in range(j_start, n_stations):
            acc = np.zeros(2 * half_lag + 1, dtype=np.float64)
            cnt = 0
            for w in range(n_windows):
                if window_valid[i, w] and window_valid[j, w]:
                    # np.correlate(a, v) computes sum_n a[n + k] * conj(v[n])
                    full = np.correlate(unit[j, w], unit[i, w], mode="full").real / float(
                        n_samples
                    )
                    acc += full[sl]
                    cnt += 1
            pairs.append((i, j))
            sums.append(acc)
            counts.append(cnt)

    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(sums, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def test_sign_and_known_delay_match_np_correlate():
    fs = 10.0
    n_samples = 400
    delay_samples = 23  # +2.3 s propagation from station 0 (A) to station 1 (B)
    rng = np.random.default_rng(42)

    wave_a = np.zeros(n_samples, dtype=np.float64)
    wave_b = np.zeros(n_samples, dtype=np.float64)
    pulse = signal.ricker(41, 4.0) if hasattr(signal, "ricker") else (
        np.exp(-0.5 * (np.arange(-20, 21) / 3.0) ** 2) * np.cos(0.6 * np.arange(-20, 21))
    )
    wave_a[120:161] = pulse
    wave_b[120 + delay_samples : 161 + delay_samples] = pulse
    # Add small uncorrelated background so demean doesn't distort peak location.
    wave_a += 1e-4 * rng.standard_normal(n_samples)
    wave_b += 1e-4 * rng.standard_normal(n_samples)

    waveforms = np.vstack([wave_a, wave_b])
    valid_mask = np.ones_like(waveforms, dtype=bool)

    res = correlate_zz_day(
        waveforms,
        valid_mask,
        sampling_rate_hz=fs,
        window_length_s=40.0,
        max_lag_s=5.0,
        taper_fraction=0.0,
        spectrum_dtype=np.complex128,
    )
    # Pair (0, 1) is index 1 in upper triangle [(0,0), (0,1), (1,1)].
    assert tuple(res.pair_indices[1]) == (0, 1)
    ccf_01 = res.daily_mean()[1]
    peak_lag_s = res.lag_s[int(np.argmax(ccf_01))]
    assert peak_lag_s == pytest.approx(delay_samples / fs, abs=1e-12)

    # Exact agreement with np.correlate(u_B, u_A, "full") on demeaned traces.
    a0 = wave_a - np.mean(wave_a)
    b0 = wave_b - np.mean(wave_b)
    expected = np.correlate(b0, a0, mode="full")[
        (n_samples - 1) - 50 : (n_samples - 1) + 51
    ]
    # Machine-precision tolerance for complex128 FFT vs direct time-domain correlation.
    np.testing.assert_allclose(ccf_01, expected, rtol=1e-12, atol=1e-12)


def test_boundary_impulses_have_no_wraparound_in_guard_band():
    fs = 20.0
    win_s = 10.0  # L = 200
    max_lag_s = 3.0  # M = 60
    guard_s = 1.0  # G = 20 -> half_lag = 80, min nfft = 280
    n_samples = int(win_s * fs)

    # Impulse at window start in A, near window end in B: true lag is +(L - 2) = +198 > half_lag.
    # If nfft were too small (< L + M + G), this impulse would wrap into negative lags inside guard!
    a = np.zeros(n_samples, dtype=np.float64)
    b = np.zeros(n_samples, dtype=np.float64)
    a[0] = 1.0
    b[n_samples - 2] = 1.0

    windows = np.stack([a, b], axis=0)[:, None, :]
    valid = np.ones((2, 1), dtype=bool)
    nfft = minimum_nfft(n_samples, 60, 20, next_fast=True)
    assert nfft >= n_samples + 60 + 20

    spectra = prepare_cc_spectra(windows, valid, fs, nfft, dtype=np.complex128)
    _, ccf_sum, _ = correlate_upper_triangle_blocked(
        spectra, valid, nfft, max_lag_samples=60, lag_guard_samples=20
    )
    # Pair (0, 1) must be identically zero everywhere in [-80, +80] (no wrap-around into guard).
    np.testing.assert_allclose(ccf_sum[1], 0.0, atol=1e-14)

    # Passing nfft below window_samples + max_lag_samples + lag_guard_samples must be rejected.
    with pytest.raises(ValueError, match="wrap-around"):
        correlate_zz_day(
            np.vstack([a, b]),
            np.ones((2, n_samples), dtype=bool),
            sampling_rate_hz=fs,
            window_length_s=win_s,
            max_lag_s=max_lag_s,
            lag_guard_s=guard_s,
            nfft=n_samples + 60,  # misses guard band!
        )


def test_spectral_whitening_flattens_passband_and_avoids_padding_leakage():
    fs = 20.0
    n_samples = 1000
    rng = np.random.default_rng(7)
    # Strongly coloured noise (1/f-like via cumulative sum + high-pass)
    raw = rng.standard_normal((2, 3, n_samples))
    sos = signal.butter(2, 0.5 / (0.5 * fs), btype="lowpass", output="sos")
    coloured = signal.sosfiltfilt(sos, raw, axis=-1) + 0.01 * raw
    valid = np.ones((2, 3), dtype=bool)

    spec = WhitenSpec(
        taper_min_hz=0.2,
        freq_min_hz=0.4,
        freq_max_hz=4.0,
        taper_max_hz=5.0,
        smooth_half_width_hz=0.1,
    )
    whitened = spectral_whiten_windows(coloured, valid, fs, spec, mode="vector_norm")
    assert whitened.shape == coloured.shape

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(whitened, axis=-1))
    in_band = (freqs >= 0.6) & (freqs <= 3.8)
    out_band = freqs >= 5.5
    # Whitened RMS amplitude inside passband should be ~1 with modest bin-to-bin scatter.
    mean_in = np.mean(amp[:, :, in_band])
    assert 0.7 < mean_in < 1.3
    # Outside taper corners spectrum must be zero to numerical precision.
    assert np.max(amp[:, :, out_band]) < 1e-12

    # Whiten-before-pad: preparing spectra on nfft > L must have zero energy in padded samples [L:nfft].
    nfft = 1400
    spectra = prepare_cc_spectra(
        coloured, valid, fs, nfft, whiten=spec, dtype=np.complex128
    )
    reconstructed = np.fft.irfft(spectra, n=nfft, axis=-1)
    np.testing.assert_allclose(reconstructed[:, :, n_samples:], 0.0, atol=1e-13)


def test_ram_continuous_suppresses_spike_and_expands_mask():
    fs = 10.0
    n_samples = 400
    rng = np.random.default_rng(123)
    wave = rng.standard_normal((2, n_samples))
    wave[0, 100] = 500.0  # large earthquake/transient spike
    wave[1, 250:255] = np.nan  # gap

    mask = np.ones((2, n_samples), dtype=bool)
    mask[1, 250:255] = False

    half_width_s = 1.0  # 10 samples
    norm, expanded = time_normalize_ram(wave, mask, fs, half_width_s=half_width_s)

    # Spike amplitude must be strongly compressed relative to raw 500.0.
    assert np.isfinite(norm).all()
    assert abs(norm[0, 100]) < 25.0

    # Mask around gap [250:255] must be shrunk outward by 10 samples on both sides -> [240:265].
    expected_mask = expand_mask_for_ram(mask, 10)
    np.testing.assert_array_equal(expanded, expected_mask)
    assert not np.any(expanded[1, 240:265])
    assert expanded[1, 239] and expanded[1, 265]
    # Zeroed in invalid region; no NaN anywhere.
    np.testing.assert_array_equal(norm[1, 240:265], 0.0)


@pytest.mark.parametrize("block_size", [1, 2, 3, 64])
def test_blocked_gemm_cc_matches_brute_force_reference(block_size):
    fs = 10.0
    n_stations = 5
    n_samples = 650  # 6 windows of 100 samples with step 100
    rng = np.random.default_rng(99)
    waveforms = rng.standard_normal((n_stations, n_samples))
    valid_mask = np.ones((n_stations, n_samples), dtype=bool)
    # Inject gaps on different stations/windows + make station 4 completely invalid
    waveforms[1, 120:130] = np.nan
    valid_mask[1, 120:130] = False
    waveforms[3, 350:360] = np.nan
    valid_mask[3, 350:360] = False
    valid_mask[4, :] = False

    whiten = WhitenSpec(
        taper_min_hz=0.2,
        freq_min_hz=0.4,
        freq_max_hz=3.5,
        taper_max_hz=4.0,
        smooth_half_width_hz=0.1,
    )
    norm, exp_mask = time_normalize_ram(waveforms, valid_mask, fs, half_width_s=0.5)
    windows, win_valid = slice_windows(norm, exp_mask, window_samples=100, step_samples=100)
    whitened_win = spectral_whiten_windows(windows, win_valid, fs, whiten)

    ref_pairs, ref_sum, ref_cnt = _brute_force_cc_from_windows(
        whitened_win, win_valid, max_lag_samples=20, lag_guard_samples=5
    )

    # complex128 check: matches float64 brute force to ~1e-11 (FFT rounding).
    res128 = correlate_zz_day(
        waveforms,
        valid_mask,
        sampling_rate_hz=fs,
        window_length_s=10.0,
        max_lag_s=2.0,
        lag_guard_s=0.5,
        ram_half_width_s=0.5,
        whiten=whiten,
        block_size=block_size,
        spectrum_dtype=np.complex128,
    )
    np.testing.assert_array_equal(res128.pair_indices, ref_pairs)
    np.testing.assert_array_equal(res128.n_windows, ref_cnt)
    np.testing.assert_allclose(res128.ccf_sum, ref_sum, rtol=1e-11, atol=1e-11)

    # complex64 check (production default): tolerance 1e-5 for float32 spectral rounding + BLAS order.
    res64 = correlate_zz_day(
        waveforms,
        valid_mask,
        sampling_rate_hz=fs,
        window_length_s=10.0,
        max_lag_s=2.0,
        lag_guard_s=0.5,
        ram_half_width_s=0.5,
        whiten=whiten,
        block_size=block_size,
        spectrum_dtype=np.complex64,
    )
    np.testing.assert_allclose(res64.ccf_sum, ref_sum, rtol=1e-5, atol=1e-5)

    # Zero-count pairs (involving station 4) store 0.0 in ccf_sum and NaN in daily_mean().
    zero_pairs = res64.n_windows == 0
    assert np.any(zero_pairs)
    np.testing.assert_array_equal(res64.ccf_sum[zero_pairs], 0.0)
    assert np.all(np.isnan(res64.daily_mean()[zero_pairs]))


def test_pcc2_two_real_channel_matches_schimmel_brute_force():
    fs = 10.0
    n_stations = 4
    n_samples = 500
    rng = np.random.default_rng(2026)
    waveforms = rng.standard_normal((n_stations, n_samples))
    # Inject a silent/zero window on station 2 to verify |a| ~ 0 water-level safety
    waveforms[2, 200:300] = 0.0
    valid_mask = np.ones((n_stations, n_samples), dtype=bool)
    valid_mask[1, 110:120] = False

    bp = BandpassSpec(freq_min_hz=0.3, freq_max_hz=3.5)
    windows, win_valid = slice_windows(waveforms, valid_mask, window_samples=100, step_samples=100)
    # Apply the same bandpass to build the independent brute-force input.
    from seisforge.ant.ccf.core import apply_bandpass

    bp_win = np.where(win_valid[:, :, None], apply_bandpass(windows, fs, bp), 0.0)
    ref_pairs, ref_sum, ref_cnt = _brute_force_pcc2_from_windows(
        bp_win, win_valid, max_lag_samples=20, lag_guard_samples=5
    )

    res128 = correlate_zz_day(
        waveforms,
        valid_mask,
        sampling_rate_hz=fs,
        window_length_s=10.0,
        max_lag_s=2.0,
        lag_guard_s=0.5,
        cc_type="pcc2",
        pcc2_bandpass=bp,
        block_size=2,
        spectrum_dtype=np.complex128,
    )
    np.testing.assert_array_equal(res128.pair_indices, ref_pairs)
    np.testing.assert_array_equal(res128.n_windows, ref_cnt)
    # Machine-precision agreement between two-real-channel rfft GEMM and complex Schimmel (1999).
    np.testing.assert_allclose(res128.ccf_sum, ref_sum, rtol=1e-11, atol=1e-11)

    # complex64 production default: 1e-5 tolerance.
    res64 = correlate_zz_day(
        waveforms,
        valid_mask,
        sampling_rate_hz=fs,
        window_length_s=10.0,
        max_lag_s=2.0,
        lag_guard_s=0.5,
        cc_type="pcc2",
        pcc2_bandpass=bp,
        block_size=2,
        spectrum_dtype=np.complex64,
    )
    np.testing.assert_allclose(res64.ccf_sum, ref_sum, rtol=1e-5, atol=1e-5)

    # Auto-correlation of a non-silent station at lag 0 must equal 1.0 in daily_mean().
    zero_lag_idx = len(res128.lag_s) // 2
    assert res128.daily_mean()[0, zero_lag_idx] == pytest.approx(1.0, abs=1e-12)


def test_reverse_ccf_zz_equals_swapped_stations():
    fs = 10.0
    rng = np.random.default_rng(55)
    waveforms = rng.standard_normal((2, 300))
    valid_mask = np.ones((2, 300), dtype=bool)

    res_ab = correlate_zz_day(
        waveforms, valid_mask, fs, 10.0, 2.0, spectrum_dtype=np.complex128
    )
    res_ba = correlate_zz_day(
        waveforms[::-1], valid_mask[::-1], fs, 10.0, 2.0, spectrum_dtype=np.complex128
    )
    np.testing.assert_allclose(
        reverse_ccf_zz(res_ab.daily_mean()[1]),
        res_ba.daily_mean()[1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_linear_stack_unequal_windows_zero_counts_and_streaming():
    fs = 10.0
    lags = lag_axis_s(fs, max_lag_s=2.0, lag_guard_s=0.5)
    pairs = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    n_lags = lags.size

    rng = np.random.default_rng(10)
    mean_d1 = rng.standard_normal((3, n_lags))
    mean_d2 = rng.standard_normal((3, n_lags))

    # Day 1: 172 windows for pair 0, 0 (missing) for pair 1, 0 for pair 2
    n1 = np.array([172, 0, 0], dtype=np.int64)
    sum1 = mean_d1 * n1[:, None]
    # Day 2: 173 windows for pair 0, 100 for pair 1, 0 (always missing) for pair 2
    n2 = np.array([173, 100, 0], dtype=np.int64)
    sum2 = mean_d2 * n2[:, None]

    day1 = DailyCCFResult(lags, pairs, sum1, n1, "cc", fs, 2.0, 0.5)
    day2 = DailyCCFResult(lags, pairs, sum2, n2, "cc", fs, 2.0, 0.5)

    stacked = linear_stack_daily([day1, day2])
    expected_p0 = (172.0 * mean_d1[0] + 173.0 * mean_d2[0]) / (172.0 + 173.0)
    np.testing.assert_allclose(stacked.ccf[0], expected_p0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(stacked.ccf[1], mean_d2[1], rtol=1e-14, atol=1e-14)
    assert np.all(np.isnan(stacked.ccf[2]))
    np.testing.assert_array_equal(stacked.n_windows, [345, 100, 0])
    np.testing.assert_array_equal(stacked.n_days, [2, 1, 0])

    # Streaming accumulator must match batch bit-for-bit.
    acc = LinearStackAccumulator()
    acc.add_day(day1)
    acc.add_day(day2)
    streamed = acc.finalize()
    np.testing.assert_array_equal(streamed.ccf[:2], stacked.ccf[:2])
    assert np.all(np.isnan(streamed.ccf[2]))

    # Equal-day weighting and guard trimming.
    stacked_eq = linear_stack_daily([day1, day2], day_weighting="equal", trim_guard=True)
    assert stacked_eq.lag_s.size == 41  # [-2.0, +2.0] at 10 Hz
    assert stacked_eq.lag_guard_s == 0.0
    sl = slice(5, 46)
    np.testing.assert_allclose(
        stacked_eq.ccf[0], 0.5 * (mean_d1[0, sl] + mean_d2[0, sl]), rtol=1e-14, atol=1e-14
    )


def test_pws_stack_streaming_matches_batch_and_excludes_missing_days():
    fs = 10.0
    lags = lag_axis_s(fs, max_lag_s=2.0, lag_guard_s=1.0)
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int64)
    rng = np.random.default_rng(77)

    # Coherent Gaussian-modulated pulse at t = +0.5 s across 12 days + incoherent noise.
    pulse = np.exp(-0.5 * ((lags - 0.5) / 0.15) ** 2) * np.cos(2.0 * np.pi * 1.5 * (lags - 0.5))
    days = []
    for d in range(12):
        noise = 0.25 * rng.standard_normal((2, lags.size))
        ccf_sum = np.zeros((2, lags.size), dtype=np.float64)
        n_win = np.array([100, 0 if d % 2 == 0 else 100], dtype=np.int64)
        ccf_sum[0] = (pulse + noise[0]) * n_win[0]
        if n_win[1] > 0:
            # Pure coherent pulse on valid days; missing on even days must NOT dilute coherence.
            ccf_sum[1] = pulse * n_win[1]
        days.append(DailyCCFResult(lags, pairs, ccf_sum, n_win, "cc", fs, 2.0, 1.0))

    pws_batch = pws_stack_daily(days, power=2.0)
    acc = PWSStackAccumulator(power=2.0)
    for day in days:
        acc.add_day(day)
    pws_stream = acc.finalize()

    # Streaming PWS matches batch PWS to machine precision; guard band trimmed.
    assert pws_batch.lag_s.size == 41
    assert pws_batch.lag_guard_s == 0.0
    np.testing.assert_allclose(pws_stream.ccf, pws_batch.ccf, rtol=1e-13, atol=1e-13)

    # Pair 1 has identical phase on all 6 valid days and missing on 6 days -> |c(t)| == 1.0.
    lin_trimmed = linear_stack_daily(days, trim_guard=True)
    np.testing.assert_allclose(pws_batch.ccf[1], lin_trimmed.ccf[1], rtol=1e-12, atol=1e-12)

    # Pair 0: PWS suppresses incoherent noise away from the pulse more than linear stack.
    far_noise = np.abs(pws_batch.lag_s - 0.5) > 0.8
    assert np.std(pws_batch.ccf[0, far_noise]) < 0.5 * np.std(lin_trimmed.ccf[0, far_noise])


def test_stacking_rejects_mixed_cc_types():
    fs = 10.0
    lags = lag_axis_s(fs, 2.0)
    pairs = np.array([[0, 1]], dtype=np.int64)
    d_cc = DailyCCFResult(lags, pairs, np.ones((1, lags.size)), np.array([10]), "cc", fs, 2.0)
    d_pcc = DailyCCFResult(
        lags, pairs, np.ones((1, lags.size)), np.array([10]), "pcc2", fs, 2.0
    )
    with pytest.raises(ValueError, match="Cannot mix cc_type"):
        linear_stack_daily([d_cc, d_pcc])
    with pytest.raises(ValueError, match="Cannot mix cc_type"):
        pws_stack_daily([d_cc, d_pcc])
    with pytest.raises(ValueError, match="Cannot mix cc_type"):
        tf_pws_stack_daily([d_cc, d_pcc])


def test_stockwell_transform_exact_invertibility_and_no_mutation():
    fs = 10.0
    rng = np.random.default_rng(123)
    t = np.arange(121, dtype=np.float64) / fs
    x1 = (
        0.3
        + np.exp(-0.5 * ((t - 4.0) / 0.8) ** 2) * np.cos(2.0 * np.pi * 1.2 * t)
        + 0.4 * np.sin(2.0 * np.pi * 3.1 * t)
    )
    x2 = rng.standard_normal(121)
    batch = np.vstack([x1, x2])
    batch_copy = batch.copy()

    freqs_1d, st_1d = stockwell_transform(x1, fs, sigma=1.0)
    rec_1d = inverse_stockwell_transform(st_1d, freqs_1d, fs, n_samples=x1.size)
    # Stockwell (1996) time-sum inverse is mathematically exact on full band.
    np.testing.assert_allclose(rec_1d, x1, rtol=1e-12, atol=1e-12)

    freqs_2d, st_2d = stockwell_transform(batch, fs, sigma=1.5)
    rec_2d = inverse_stockwell_transform(st_2d, freqs_2d, fs, n_samples=batch.shape[1])
    np.testing.assert_allclose(rec_2d, batch, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(batch, batch_copy)


def test_tf_pws_streaming_matches_batch_excludes_missing_and_suppresses_noise():
    fs = 10.0
    lags = lag_axis_s(fs, max_lag_s=2.0, lag_guard_s=1.0)
    pairs = np.array([[0, 1], [0, 2]], dtype=np.int64)
    rng = np.random.default_rng(88)

    pulse = np.exp(-0.5 * ((lags - 0.5) / 0.2) ** 2) * np.cos(2.0 * np.pi * 1.2 * (lags - 0.5))
    days = []
    for d in range(12):
        noise = 0.25 * rng.standard_normal((2, lags.size))
        ccf_sum = np.zeros((2, lags.size), dtype=np.float64)
        n_win = np.array([100, 0 if d % 2 == 0 else 100], dtype=np.int64)
        ccf_sum[0] = (pulse + noise[0]) * n_win[0]
        if n_win[1] > 0:
            ccf_sum[1] = pulse * n_win[1]
        days.append(DailyCCFResult(lags, pairs, ccf_sum, n_win, "cc", fs, 2.0, 1.0))

    tf_batch = tf_pws_stack_daily(days, power=1.0)
    acc = TFPWSStackAccumulator(power=1.0)
    for day in days:
        acc.add_day(day)
    tf_stream = acc.finalize()

    assert tf_batch.method == "tf_pws"
    assert tf_batch.lag_s.size == 41
    assert tf_batch.lag_guard_s == 0.0
    np.testing.assert_allclose(tf_stream.ccf, tf_batch.ccf, rtol=1e-13, atol=1e-13)

    # Pair 1 is identical on all 6 valid days and missing on 6 days -> |c(tau, f)| == 1.0,
    # so full-band tf-PWS equals the trimmed linear stack to machine precision.
    lin_trimmed = linear_stack_daily(days, trim_guard=True)
    np.testing.assert_allclose(tf_batch.ccf[1], lin_trimmed.ccf[1], rtol=1e-11, atol=1e-11)

    # Pair 0: tf-PWS suppresses incoherent noise away from the coherent pulse.
    far_noise = np.abs(tf_batch.lag_s - 0.5) > 0.8
    assert np.std(tf_batch.ccf[0, far_noise]) < 0.6 * np.std(lin_trimmed.ccf[0, far_noise])


def test_tf_pws_preserves_coherent_band_under_out_of_band_incoherent_noise():
    fs = 10.0
    lags = lag_axis_s(fs, max_lag_s=4.0, lag_guard_s=1.0)
    pairs = np.array([[0, 1]], dtype=np.int64)
    rng = np.random.default_rng(2026)

    # Coherent 0.6 Hz wavepacket overlapping in time with strong incoherent 3.5 Hz noise.
    sig = np.exp(-0.5 * ((lags - 0.0) / 0.8) ** 2) * np.cos(2.0 * np.pi * 0.6 * lags)
    days = []
    for _ in range(16):
        rand_phase = rng.uniform(0.0, 2.0 * np.pi)
        hf_incoherent = 1.5 * np.cos(2.0 * np.pi * 3.5 * lags + rand_phase)
        ccf_sum = ((sig + hf_incoherent) * 50.0)[None, :]
        days.append(DailyCCFResult(lags, pairs, ccf_sum, np.array([50]), "cc", fs, 4.0, 1.0))

    tf_res = tf_pws_stack_daily(days, power=2.0, freq_min_hz=0.2, freq_max_hz=1.5)
    pws_res = pws_stack_daily(days, power=2.0)
    _, sig_trimmed = (
        tf_res.lag_s,
        sig[10:-10],
    )
    # Time-domain PWS has its analytic phase corrupted by the stronger 3.5 Hz random-phase wave,
    # whereas tf-PWS isolates the coherent 0.6 Hz band.
    tf_err = np.linalg.norm(tf_res.ccf[0] - sig_trimmed) / np.linalg.norm(sig_trimmed)
    pws_err = np.linalg.norm(pws_res.ccf[0] - sig_trimmed) / np.linalg.norm(sig_trimmed)
    assert tf_err < 0.15
    assert tf_err < 0.3 * pws_err


def test_core_functions_do_not_mutate_inputs():
    fs = 10.0
    rng = np.random.default_rng(1)
    wave = rng.standard_normal((3, 300))
    wave[0, 50] = np.nan
    mask = np.isfinite(wave)
    wave_copy = wave.copy()
    mask_copy = mask.copy()

    res = correlate_zz_day(wave, mask, fs, 10.0, 2.0, ram_half_width_s=0.5)
    _ = prepare_pcc2_spectra(
        np.ones((2, 2, 100)), np.ones((2, 2), dtype=bool), fs, nfft=128
    )
    _ = linear_stack_daily([res])

    np.testing.assert_array_equal(wave, wave_copy)
    np.testing.assert_array_equal(mask, mask_copy)
