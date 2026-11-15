"""Checks of `seisforge.ant.synthetic` (synthetic CCFs from a known dispersion curve)."""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace
from scipy.signal import hilbert

from seisforge.ant.synthetic import (
    group_velocity_from_phase,
    symmetric_ccf_time_axis,
    synthetic_ccf,
)


def test_group_velocity_is_exact_for_a_linear_phase_velocity():
    # c = a + b T: U = c / (1 + b T / c) = c**2 / (c + b T) exactly, and
    # central differences of a linear function are exact.
    period = np.linspace(2.0, 10.0, 17)
    c = 2.5 + 0.06 * period
    np.testing.assert_allclose(group_velocity_from_phase(period, c), c**2 / (c + 0.06 * period), rtol=1e-12)


def test_non_dispersive_ccf_is_a_pulse_centred_on_distance_over_velocity():
    # Analytic limit: constant c and phi0 = 0 (pi_over_4 = 0) gives a
    # zero-phase band-limited pulse, symmetric about t = r / c on each branch.
    delta = 0.05
    time = symmetric_ccf_time_axis(120.0, delta)
    period = np.array([0.5, 50.0])
    ccf = synthetic_ccf(time, 150.0, period, np.full(2, 3.0), min_period_s=2.0, max_period_s=10.0, pi_over_4=0.0)
    np.testing.assert_array_equal(ccf, ccf[::-1])  # two-sided and symmetric
    causal = time >= 0
    peak = time[causal][np.argmax(ccf[causal])]
    assert peak == pytest.approx(50.0, abs=delta)  # one sample
    centre = np.flatnonzero(np.isclose(time, 50.0))[0]
    np.testing.assert_allclose(ccf[centre - 40 : centre], ccf[centre + 40 : centre : -1], atol=1e-6 * ccf.max())


def test_envelope_peak_of_a_narrow_band_travels_at_the_group_velocity():
    # Independent of the phase constant: the envelope of a narrow band
    # arrives at r / U, with U from the stated c(T) (not from the synthetic).
    period = np.linspace(1.0, 40.0, 400)
    c = 2.8 + 0.04 * period
    delta = 0.1
    time = symmetric_ccf_time_axis(300.0, delta)
    distance = 600.0
    ccf = synthetic_ccf(time, distance, period, c, min_period_s=9.5, max_period_s=10.5)
    causal = time > 0
    envelope = np.abs(hilbert(ccf[causal]))
    u = group_velocity_from_phase(period, c)[np.argmin(np.abs(period - 10.0))]
    # Tolerance: the band and its octave ramps average U over about 5-20 s;
    # U varies by 1.4 % over that range here.
    assert time[causal][np.argmax(envelope)] == pytest.approx(distance / u, rel=0.015)


def test_aftan_measures_the_input_phase_and_group_velocity(tmp_path):
    from seisforge.ant.aftan import run_aftan_config

    period = np.linspace(1.0, 40.0, 400)
    c = 2.8 + 0.04 * period
    delta = 0.25
    time = symmetric_ccf_time_axis(300.0, delta)
    ccf = synthetic_ccf(time, 200.0, period, c, min_period_s=4.0, max_period_s=16.0, pi_over_4=-1.0)
    trace = Trace(data=ccf.astype(np.float32))
    trace.stats.delta = delta
    trace.stats.sac = {"b": time[0], "dist": 200.0}
    (tmp_path / "ccf").mkdir()
    trace.write(str(tmp_path / "ccf" / "XX.A_XX.B_ZZ.SAC"), format="SAC")
    # A rough constant reference (within about 5 % on average) picks the 2 pi cycle.
    np.savetxt(tmp_path / "reference.dat", np.column_stack([period, np.full(period.size, 3.2)]))
    (tmp_path / "aftan.yml").write_text(
        "station: XX.A\nio: {input_dir: ccf, output_dir: ftan, sac_pattern: '*.SAC'}\n"
        "aftan:\n  branch: positive\n  min_period: 5.0\n  max_period: 14.0\n"
        "  period_sampling: {mode: uniform, uniform: {step: 1.0}}\n"
        "  velocity_window: {min: 2.0, max: 4.5}\n"
        "  basic: {alpha: {mode: constant, constant: {value: 20.0}}}\n"
        "  prediction_file: reference.dat\n"
    )
    run_aftan_config(tmp_path / "aftan.yml")
    table = np.loadtxt(next((tmp_path / "ftan").glob("*.dat")))
    instant_period, group, phase = table[:, 1], table[:, 2], table[:, 3]
    # Same phase convention as AFTAN's ZZ pi_over_4 = -1 (module docstring).
    # Tolerances: AFTAN's Gaussian filter and instantaneous-period estimate,
    # about 0.5 % here; the distance is 3.4-11 wavelengths.
    np.testing.assert_allclose(phase, np.interp(instant_period, period, c), rtol=0.005)
    u = group_velocity_from_phase(period, c)
    np.testing.assert_allclose(group, np.interp(instant_period, period, u), rtol=0.01)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"min_period_s": 0.5}, "inside the dispersion curve"),
        ({"max_period_s": 60.0}, "inside the dispersion curve"),
    ],
)
def test_band_outside_the_curve_is_rejected(kwargs, match):
    time = symmetric_ccf_time_axis(50.0, 0.1)
    arguments = {"min_period_s": 2.0, "max_period_s": 10.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        synthetic_ccf(time, 50.0, np.array([1.0, 40.0]), np.array([3.0, 3.5]), **arguments)


def _bessel_ccf(time, distance, period, c, order_weights, band):
    """Independent closed form: irfft of W(f) * sum_n a_n J_n(k r) (module docstring)."""
    from scipy.special import jv

    from seisforge.ant.synthetic import _band_weights

    delta = time[1] - time[0]
    n_half = time.size // 2
    n_fft = 4 * (n_half + 1)
    f = np.fft.rfftfreq(n_fft, delta)
    w = _band_weights(f, *band)
    used = w > 0
    k = 2 * np.pi * f[used] / np.interp(1 / f[used], period, c)
    spectrum = np.zeros(f.size, dtype=complex)
    spectrum[used] = w[used] * sum(a * jv(n, k * distance) for n, a in order_weights.items())
    full = np.fft.irfft(spectrum, n_fft)
    return np.concatenate([full[n_fft - n_half :], full[: n_half + 1]])


def test_plane_wave_rayleigh_ccfs_match_the_bessel_closed_forms_after_rotation():
    # Independent check of the azimuth average and of its pairing with the
    # package's ZNE -> ZRT rotation: ZZ = J0, ZR = -e J1, RZ = +e J1,
    # RR = e^2 (J0 - J2) / 2, TT = e^2 J1 / (k r) (module docstring), computed with scipy.special.
    from seisforge.ant.rotation import rotate_zne_ccf_to_zrt
    from seisforge.ant.synthetic import synthetic_rayleigh_ccf_zne

    period = np.linspace(1.0, 40.0, 400)
    c = 2.8 + 0.04 * period
    time = symmetric_ccf_time_axis(150.0, 0.25)
    distance, azimuth, e = 120.0, 37.0, 0.7
    band = (4.0, 16.0)
    zne = synthetic_rayleigh_ccf_zne(
        time, distance, azimuth, period, c, min_period_s=band[0], max_period_s=band[1], ellipticity=e
    )
    zrt = rotate_zne_ccf_to_zrt(zne, azimuth, azimuth + 180.0)
    expected = {
        "ZZ": {0: 1.0},
        "ZR": {1: -e},
        "RZ": {1: e},
        "RR": {0: e**2 / 2, 2: -(e**2) / 2},
        # J1(x) / x = (J0(x) + J2(x)) / 2: Rayleigh near field only, no Love waves.
        "TT": {0: e**2 / 2, 2: e**2 / 2},
    }
    scale = np.max(np.abs(zrt["ZZ"]))
    for name, weights in expected.items():
        # Tolerance: the default azimuth count makes the discretisation error
        # of the direction average exponentially small (~1e-12 here).
        np.testing.assert_allclose(
            zrt[name], _bessel_ccf(time, distance, period, c, weights, band), atol=1e-9 * scale, err_msg=name
        )
    for name in ("ZT", "TZ", "RT", "TR"):
        assert np.max(np.abs(zrt[name])) < 1e-9 * scale
