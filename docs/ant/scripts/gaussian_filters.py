"""Figure gaussian_filters.svg (and -dark.svg): the AFTAN Gaussian filter for several alpha.

(a) frequency response ``exp(-alpha ((w - w0) / w0)^2)`` from
`seisforge.ant.filters.gaussian_frequency_response`; (b) the filtered
analytic signal of a unit impulse at t = 0 from
`seisforge.ant.filters.gaussian_filter_trace`, with the closed-form envelope
``exp(-w0^2 t^2 / (4 alpha))`` (dashed) as the independent check; (c) the
resolution trade-off sigma_w / w0 = 1 / sqrt(2 alpha) and sigma_t / T0 =
sqrt(2 alpha) / (2 pi).
"""

import numpy as np

import _style
from _style import plt
from seisforge.ant.filters import gaussian_filter_trace, gaussian_frequency_response

T0_S = 10.0
DT_S = 0.05
ALPHAS = (5.0, 20.0, 80.0)


def colors():
    """One colour per alpha, from the current palette."""
    return (_style.EXTRA2, _style.MEASURED, _style.EXTRA)


n = 8001
time = (np.arange(n) - n // 2) * DT_S
impulse = np.zeros(n)
impulse[n // 2] = 1.0 / DT_S  # unit-area impulse at t = 0
worst = 0.0
omega0 = 2 * np.pi / T0_S
responses = []
for alpha in ALPHAS:
    # Put the impulse at the start so the FFT window holds both tails, then
    # shift back: the filter is zero-phase, the analytic output is centred on 0.
    analytic = gaussian_filter_trace(np.roll(impulse, -(n // 2)), DT_S, T0_S, alpha,
                                     real_output=False)
    analytic = np.roll(analytic, n // 2)
    envelope = np.abs(analytic)
    closed = envelope.max() * np.exp(-(omega0 * time) ** 2 / (4 * alpha))
    worst = max(worst, np.max(np.abs(envelope - closed)) / envelope.max())
    responses.append((analytic, envelope, closed))


def draw():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1.2, 1]})

    frequency = np.linspace(0.0, 0.25, 2000)
    for alpha, color in zip(ALPHAS, colors()):
        axes[0].plot(frequency, gaussian_frequency_response(frequency, 1 / T0_S, alpha),
                     color=color, label=rf"$\alpha$ = {alpha:g}")
    axes[0].axvline(1 / T0_S, color=_style.INK, lw=0.8, ls=":")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("filter gain")
    axes[0].set_title(rf"frequency response, $T_0$ = {T0_S:g} s")
    axes[0].legend()

    for color, (analytic, envelope, closed) in zip(colors(), responses):
        axes[1].plot(time, analytic.real / envelope.max(), color=color, lw=0.8)
        axes[1].plot(time, envelope / envelope.max(), color=color, lw=1.8)
        axes[1].plot(time, closed / envelope.max(), color=_style.INK, lw=0.8, ls="--")
    axes[1].set_xlim(-45, 45)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("normalised amplitude")
    axes[1].set_title("impulse response: real part and envelope\n(dashed: closed-form envelope)")

    alpha_axis = np.geomspace(1, 300, 200)
    axes[2].loglog(alpha_axis, 1 / np.sqrt(2 * alpha_axis), color=_style.MEASURED,
                   label=r"$\sigma_\omega/\omega_0 = 1/\sqrt{2\alpha}$")
    axes[2].loglog(alpha_axis, np.sqrt(2 * alpha_axis) / (2 * np.pi), color=_style.SECOND,
                   label=r"$\sigma_t/T_0 = \sqrt{2\alpha}/2\pi$")
    for alpha, color in zip(ALPHAS, colors()):
        axes[2].axvline(alpha, color=color, lw=0.8, ls=":")
    axes[2].set_xlabel(r"$\alpha$")
    axes[2].set_ylabel("relative width")
    axes[2].set_title("resolution trade-off")
    axes[2].legend(loc="lower left")

    for ax, label in zip(axes, "abc"):
        _style.panel_label(ax, label)
    fig.tight_layout()
    return fig


_style.save_themed(draw, __file__)
print(f"largest envelope misfit to the closed form: {worst:.1e} (relative)")
# The response is exactly Gaussian except where exp(...) < e^-40 is set to zero
# and for the one-sided (analytic) cut at w = 0, which matters only for small alpha.
assert worst < 1e-2
