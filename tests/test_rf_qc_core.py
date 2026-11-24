"""Leave-one-out F test of RF QC: calibration under its null hypothesis."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace
from scipy import stats

from seisforge.rf.qc_core import effective_dof, leave_one_out_f_test
from seisforge.rf.traces import RFstream

DT_S = 0.1
TIME_S = np.arange(-5.0, 30.0 + DT_S / 2, DT_S)


def _noise(rng, n_traces, gauss_a=None):
    """Unit-variance white noise, optionally through exp(-omega^2 / 4 a^2)."""
    pad = 400
    noise = rng.normal(size=(n_traces, TIME_S.size + 2 * pad))
    if gauss_a is not None:
        omega = 2 * np.pi * np.fft.rfftfreq(noise.shape[1], DT_S)
        noise = np.fft.irfft(np.fft.rfft(noise) * np.exp(-omega**2 / (4 * gauss_a**2)),
                             noise.shape[1])
    noise = noise[:, pad:-pad]
    return noise / noise.std()


def _signal():
    return np.exp(-(2.5 * TIME_S) ** 2) + 0.3 * np.exp(-(2.5 * (TIME_S - 4.4)) ** 2)


@pytest.mark.parametrize("gauss_a", [None, 2.5, 5.0])
def test_p_values_are_uniform_under_the_null_hypothesis(gauss_a):
    # Common signal plus independent, equal-variance stationary noise: the
    # p-values must be U(0, 1). 150 sets of 20 traces = 3000 p-values; the
    # KS test at 1e-3 fails a correct implementation once in 1000 runs (the
    # seed is fixed, so the outcome is reproducible), and the rejection rate
    # at 0.05 has a binomial standard error of 0.004.
    rng = np.random.default_rng(7)
    p_values = np.concatenate([
        leave_one_out_f_test(_signal() + 0.05 * _noise(rng, 20, gauss_a))[0]
        for _ in range(150)
    ])
    assert stats.kstest(p_values, "uniform").pvalue > 1e-3
    assert abs(np.mean(p_values < 0.05) - 0.05) < 0.015


@pytest.mark.parametrize("gauss_a", [2.5, 5.0])
def test_effective_dof_matches_gaussian_filtered_white_noise(gauss_a):
    # White noise through exp(-omega^2 / 4a^2) has autocorrelation
    # exp(-a^2 tau^2 / 2), so sum rho^2 dt = sqrt(pi) / a and
    # nu = n dt a / sqrt(pi) (long-window limit; 5 % covers the estimate
    # from 40 traces and the window end effect).
    rng = np.random.default_rng(2)
    nu = effective_dof(_noise(rng, 40, gauss_a))
    expected = TIME_S.size * DT_S * gauss_a / np.sqrt(np.pi)
    assert nu == pytest.approx(expected, rel=0.05)


def test_effective_dof_of_white_noise_is_the_sample_count():
    nu = effective_dof(_noise(np.random.default_rng(3), 40))
    assert nu == pytest.approx(TIME_S.size, rel=0.05)


def test_outlier_trace_gets_a_small_p_value():
    rng = np.random.default_rng(4)
    data = _signal() + 0.05 * _noise(rng, 20, 2.5)
    data[3] = -data[3]
    p_value, _, _ = leave_one_out_f_test(data)
    assert p_value[3] < 1e-6
    assert np.all(p_value[np.arange(20) != 3] > 1e-3)


def test_f_test_select_metrics_are_probabilities():
    # Before 0.2.0 the metric was 1 - (F(f) - F(1/f)), above 1 whenever f < 1.
    rng = np.random.default_rng(5)
    data = _signal() + 0.05 * _noise(rng, 20, 2.5)
    traces = []
    for row in data:
        trace = Trace(row)
        trace.stats.delta = DT_S
        trace.stats.sac = {"b": TIME_S[0], "e": TIME_S[-1]}
        traces.append(trace)
    selected = RFstream(st=Stream(traces)).f_test_select(threshold=0.0, window=(-5, 30))
    metrics = np.array(selected.step_metrics)
    assert np.all((metrics >= 0) & (metrics <= 1))


def test_f_test_needs_three_traces():
    with pytest.raises(ValueError, match="three"):
        leave_one_out_f_test(np.zeros((2, 10)))
