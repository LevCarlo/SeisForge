"""Figure synthetic_wraparound.svg (and -dark.svg): why the FFT is padded.

A 20 s receiver function (-5 to 15 s) of the 35 km crust of `_rf.CRUST`,
p = 0.06 s/km, a = 2.5, computed three ways with
`seisforge.rf.synthetic.receiver_function`:

- ``n_fft`` equal to the window: the Moho multiples (15-19 s) and later
  reverberations wrap into the start of the window;
- the default (padding plus the decay check);
- ``n_fft = 131072`` (6554 s), the reference: nothing arriving within that
  time can wrap.
"""

import numpy as np

import _rf
from _rf import _style, plt
from seisforge.rf.synthetic import receiver_function

DT_S, BEGIN_S, N_SAMPLES, GAUSS_A, P_S_PER_KM = 0.05, -5.0, 400, 2.5, 0.06
model = [_rf.CRUST[key] for key in ("thickness_km", "vp_km_s", "vs_km_s", "rho_g_cm3")]


def run(n_fft):
    return receiver_function(*model, P_S_PER_KM, dt_s=DT_S, n_samples=N_SAMPLES,
                             gauss_a=GAUSS_A, time_shift_s=-BEGIN_S, n_fft=n_fft)


time_s, reference = run(131072)
_, wrapped = run(N_SAMPLES)
_, default = run(None)
peak = np.abs(reference).max()
error_wrapped = np.max(np.abs(wrapped - reference)) / peak
error_default = np.max(np.abs(default - reference)) / peak
assert error_default < 1e-3 < error_wrapped, (error_default, error_wrapped)


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    ax.plot(time_s, reference / peak, color=_style.TRUTH, lw=1.6, label="reference")
    ax.plot(time_s, wrapped / peak, color=_style.HIGHLIGHT, lw=0.9, label="n_fft = window")
    ax.plot(time_s, default / peak, color=_style.MEASURED, lw=0.9, ls="--", label="default")
    ax.set_xlabel("time after direct P (s)")
    ax.set_ylabel("RF / peak")
    ax.set_title("20 s window")
    ax.legend(loc="upper right")
    ax = axes[1]
    floor = 1e-12
    ax.semilogy(time_s, np.abs(wrapped - reference) / peak + floor, color=_style.HIGHLIGHT,
                lw=0.9, label=f"n_fft = window (max {error_wrapped:.0e})")
    ax.semilogy(time_s, np.abs(default - reference) / peak + floor, color=_style.MEASURED,
                lw=0.9, label=f"default (max {error_default:.0e})")
    ax.set_ylim(1e-8, 1)
    ax.set_xlabel("time after direct P (s)")
    ax.set_ylabel("|difference| / peak")
    ax.set_title("error against the reference")
    ax.legend(loc="lower right")
    for axis, label in zip(axes, "ab"):
        _style.panel_label(axis, label)
    fig.tight_layout()
    return fig


_rf.save(draw, __file__)
print(f"largest error / peak: n_fft = window {error_wrapped:.1e}, default {error_default:.1e}")
