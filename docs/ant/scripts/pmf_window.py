"""Figure pmf_window.svg (and -dark.svg): the PMF cleaning window, built from the rule on the page.

The compressed (phase-corrected) signal is that of `pmf_comparison.py` (the
noisy trace of `_ccf_case`). The window is constructed here directly from the
rule written in docs/ant/dispersion.md (section 5, step 2) and compared with
the window `PhaseMatchedFilter.clean_undispersed` applies; the two must agree
sample by sample, so the page describes what the code does.
"""

import numpy as np
from numpy.fft import ifft
from scipy.signal import argrelextrema

import _ccf_case as case
import _style
from _style import plt
from seisforge.ant.filters import PhaseMatchedFilter, analytic_spectrum

time, zz = case.zz_ccf()
_, noisy = case.noisy_trace(time, zz)
config = case.aftan_config(prediction=case.prediction_file(), pmf=True)
meas = case.measurement(noisy, config)
basic = meas.measure()
pmf = config.pmf
flt = PhaseMatchedFilter(
    dt=case.DELTA_S, distance_km=case.DISTANCE_KM, periods=basic["period"],
    group_velocity=basic["group_velocity"], alpha=meas.pmf_alpha, nalpha=pmf.nalpha,
    min_half_length=pmf.min_half_length, amplitude_ratio=pmf.amplitude_ratio,
    window_factor=pmf.window_factor, nfft=meas.nfft,
)
data = meas.trace.data
freqs = np.fft.fftfreq(meas.nfft, d=case.DELTA_S)
iphi, period_min, period_max = flt.phase_correction(freqs)
spectrum = analytic_spectrum(data, meas.nfft) * np.exp(iphi) * flt.spectral_taper(freqs, period_min, period_max)
compressed = ifft(spectrum)[: data.size]
envelope = np.abs(compressed)
n = np.arange(data.size)

# --- The rule of the page ---------------------------------------------------
p = int(np.argmax(envelope))
minima = argrelextrema(envelope, np.less)[0]
qualifies = (np.abs(minima - p) * case.DELTA_S > pmf.min_half_length) & (
    envelope[minima] < pmf.amplitude_ratio * envelope[p])
left = minima[qualifies & (minima < p)][::-1]  # nearest first
right = minima[qualifies & (minima > p)]
l1, l2 = left[0], left[1]
r1, r2 = right[0], right[1]
h = max(p - l1, r1 - p)  # inner half-width (samples)
w = max(l1 - l2, r2 - r1)  # taper width (samples)
f = pmf.window_factor
inner = f * h
width = f * w + 1
window = np.where(np.abs(n - p) <= inner, 1.0, np.exp(-((np.abs(n - p) - inner) ** 2) / (2 * width**2)))

package = flt.clean_undispersed(spectrum, data.size)
assert np.allclose(window * compressed.real, package, rtol=0, atol=1e-12 * np.abs(package).max())
print(f"window from the page rule equals the package window; h = {h * case.DELTA_S:.1f} s, "
      f"w = {w * case.DELTA_S:.1f} s")

# --- Figure -----------------------------------------------------------------
lag = case.DELTA_S * n
peak = envelope[p]


def draw():
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(lag, compressed.real / peak, color=_style.MUTED, lw=0.6, label="compressed signal (real part)")
    ax.plot(lag, envelope / peak, color=_style.MEASURED, lw=1.1, label="envelope")
    ax.plot(lag, window, color=_style.SECOND, lw=1.8, label="window $W$")
    ax.axhline(pmf.amplitude_ratio, color=_style.ALTERNATIVE, ls=":", lw=0.8)
    ax.text(lag[p] - 58, pmf.amplitude_ratio + 0.03, "amplitude ratio", color=_style.MUTED, fontsize=8)
    for idx, name in ((l2, "$n_{l2}$"), (l1, "$n_{l1}$"), (r1, "$n_{r1}$"), (r2, "$n_{r2}$")):
        ax.plot(lag[idx], envelope[idx] / peak, "v", color=_style.HIGHLIGHT, ms=5)
        ax.annotate(name, (lag[idx], envelope[idx] / peak), xytext=(0, -14), textcoords="offset points",
                    ha="center", color=_style.HIGHLIGHT, fontsize=8)
    ax.annotate("", (lag[p] - inner * case.DELTA_S, 1.08), (lag[p] + inner * case.DELTA_S, 1.08),
                arrowprops={"arrowstyle": "<->", "color": _style.INK, "lw": 0.8})
    ax.text(lag[p], 1.12, r"$W = 1$ on $|n - n_p| \leq h$", ha="center", fontsize=8)
    ax.set_xlim(lag[p] - 60, lag[p] + 60)
    ax.set_ylim(-1.05, 1.25)
    ax.set_xlabel("lag time (s)")
    ax.set_ylabel("normalised amplitude")
    ax.legend(loc="lower left", ncol=3)
    fig.tight_layout()
    return fig


_style.save_themed(draw, __file__)
