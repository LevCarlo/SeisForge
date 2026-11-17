"""Figure pmf_comparison.png: phase-matched filtering, step by step.

Input: the synthetic ZZ CCF of `_ccf_case` plus a faster interfering
arrival and noise (`_ccf_case.noisy_trace`). The steps are those of
`seisforge.ant.filters.PhaseMatchedFilter.apply`, called one by one:
phase correction from the basic-FTAN group velocity, compression, cleaning
window, redispersion. (d) compares the group velocity of basic FTAN on the
noisy trace and of the PMF stage (`AFTANMeasurement.measure_pmf`) with the
truth.
"""

import numpy as np
from matplotlib.ticker import NullFormatter
from numpy.fft import ifft

import _ccf_case as case
import _style
from _style import plt
from seisforge.ant.filters import PhaseMatchedFilter, analytic_spectrum

time, zz = case.zz_ccf()
clean, noisy = case.noisy_trace(time, zz)
scale = np.abs(clean.data).max()

config = case.aftan_config(prediction=case.prediction_file(), pmf=True)
meas = case.measurement(noisy, config)
basic = meas.measure()
pmf = meas.measure_pmf(basic)
clean_meas = case.measurement(clean, config)
reference = clean_meas.measure()
reference_pmf = clean_meas.measure_pmf(reference)

# The PMF steps one by one, with the same parameters as measure_pmf uses.
flt = PhaseMatchedFilter(
    dt=case.DELTA_S, distance_km=case.DISTANCE_KM, periods=basic["period"],
    group_velocity=basic["group_velocity"], alpha=meas.pmf_alpha, nalpha=config.pmf.nalpha,
    min_half_length=config.pmf.min_half_length, amplitude_ratio=config.pmf.amplitude_ratio,
    window_factor=config.pmf.window_factor, nfft=meas.nfft,
)
data = meas.trace.data  # tapered input, as in measure_pmf
freqs = np.fft.fftfreq(meas.nfft, d=case.DELTA_S)
iphi, period_min, period_max = flt.phase_correction(freqs)
compressed_spectrum = analytic_spectrum(data, meas.nfft) * np.exp(iphi) * flt.spectral_taper(
    freqs, period_min, period_max)
compressed = ifft(compressed_spectrum)[: data.size]
windowed = flt.clean_undispersed(compressed_spectrum, data.size)
window = np.divide(windowed, compressed.real, out=np.zeros_like(windowed), where=np.abs(compressed.real) > 1e-12 * scale)
cleaned = flt.redisperse(windowed, iphi, data.size)
assert np.allclose(cleaned, pmf["signal"])  # same as the package's one-call path

lag = case.DELTA_S * np.arange(data.size)
fig = plt.figure(figsize=(10.5, 7.2))
grid = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
ax_in = fig.add_subplot(grid[0, 0])
ax_cmp = fig.add_subplot(grid[1, 0])
ax_out = fig.add_subplot(grid[2, 0])
ax_u = fig.add_subplot(grid[:, 1])

ax_in.plot(lag, noisy.data / scale, color="k", lw=0.7, label="noisy input")
ax_in.plot(lag, clean.data / scale, color=_style.HIGHLIGHT, lw=0.7, alpha=0.7, label="noise-free CCF")
ax_in.set_title("(1) input: CCF + faster interfering arrival + noise")
ax_in.legend(loc="upper right", ncol=2)

peak = np.abs(compressed).max()
ax_cmp.plot(lag, compressed.real / peak, color="k", lw=0.7, label=r"Re $X\,e^{i\Phi}$ (compressed)")
ax_cmp.plot(lag, np.abs(compressed) / peak, color=_style.MEASURED, lw=1, label="envelope")
ax_cmp.plot(lag, np.clip(window, 0, 1.05), color=_style.SECOND, lw=1.5, label="cleaning window")
ax_cmp.axvline(case.DISTANCE_KM / basic["group_velocity"][-1], color=_style.ALTERNATIVE, ls=":")
ax_cmp.set_title(r"(2)-(3) phase-corrected (compressed) signal and window; dotted: $r/U(\omega_a)$")
ax_cmp.legend(loc="upper right", ncol=3)

ax_out.plot(lag, cleaned / scale, color="k", lw=0.8, label="PMF output (redispersed)")
ax_out.plot(lag, clean.data / scale, color=_style.HIGHLIGHT, lw=0.7, alpha=0.7, label="noise-free CCF")
ax_out.set_title("(4) cleaned, redispersed signal: input to the second FTAN pass")
ax_out.legend(loc="upper right", ncol=2)
for ax in (ax_in, ax_cmp, ax_out):
    ax.set_xlim(0, 300)
    ax.set_ylabel("normalised amplitude")
ax_out.set_xlabel("lag time (s)")

periods = np.geomspace(case.MIN_PERIOD_S, case.MAX_PERIOD_S, 200)
ax_u.plot(periods, case.true_group_velocity(periods), color=_style.TRUTH, lw=1.5, label="true")
ax_u.plot(basic["period"], basic["group_velocity"], "o", ms=3, color=_style.MEASURED, label="basic FTAN, noisy")
ax_u.plot(pmf["period"], pmf["group_velocity"], "s", ms=3, mfc="none", color=_style.SECOND, label="PMF, noisy")
ax_u.plot(reference["period"], reference["group_velocity"], "-", lw=0.8, color=_style.ALTERNATIVE,
          label="basic FTAN, noise-free")
ax_u.plot(reference_pmf["period"], reference_pmf["group_velocity"], "--", lw=0.8, color=_style.SECOND,
          label="PMF, noise-free")
ax_u.set_xscale("log")
ax_u.set_xticks([6, 10, 20, 40], labels=["6", "10", "20", "40"])
ax_u.xaxis.set_minor_formatter(NullFormatter())
ax_u.set_xlabel("period (s)")
ax_u.set_ylabel("group velocity (km/s)")
ax_u.set_title("group velocity")
ax_u.legend(loc="upper left", bbox_to_anchor=(0.06, 1.0))

for ax, label in zip((ax_in, ax_cmp, ax_out, ax_u), "abcd"):
    _style.panel_label(ax, label)
fig.tight_layout()
_style.save(fig, __file__)


def rms_error(result):
    return float(np.sqrt(np.mean((result["group_velocity"] - case.true_group_velocity(result["period"])) ** 2)))


print(f"group-velocity RMS error: basic noisy {rms_error(basic):.3f}, PMF noisy {rms_error(pmf):.3f}, "
      f"basic noise-free {rms_error(reference):.3f}, PMF noise-free {rms_error(reference_pmf):.3f} km/s")
edge = np.argmax(reference_pmf["period"])
print(f"PMF noise-free at its longest period {reference_pmf['period'][edge]:.1f} s: U error "
      f"{reference_pmf['group_velocity'][edge] - case.true_group_velocity(reference_pmf['period'][edge]):+.3f} km/s")
