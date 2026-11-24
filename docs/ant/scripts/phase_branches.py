"""Figure phase_branches.svg (and -dark.svg): the 2 pi branch family of phase velocity.

(a) The AFTAN phase measurement of the synthetic ZZ CCF (`_ccf_case`) fixes
phase slowness only up to N T / r. The branches N = -2..2 are drawn from the
package's uncorrected curve with `apply_phase_cycle_shift`; the true curve
sits on N = 0. (b) The relative velocity spacing between neighbouring
branches, lambda / r = c T / r, for several distances: a reference curve must
be closer than about half of it. (c) The package's branch score
(`phase_cycle_reference_misfit`) for references that are uniformly too fast
by eps, compared with the threshold derived on the page, eps* = mean(lambda/r)/2.
"""

import numpy as np

import _ccf_case as case
import _style
from _style import plt
from seisforge.ant.aftan.core import apply_phase_cycle_shift, phase_cycle_reference_misfit

time, zz = case.zz_ccf()
trace = case.positive_trace(time, zz)
basic = case.measurement(trace, case.aftan_config(prediction=case.prediction_file())).measure()
period = basic["period"]
uncorrected = basic["phase_velocity_uncorrected"]
r = case.DISTANCE_KM

eps_values = np.linspace(-0.25, 0.25, 101)
true_c = case.true_phase_velocity(period)
chosen = []
shifts = np.arange(-2, 3)
for eps in eps_values:
    misfits = [phase_cycle_reference_misfit(period, uncorrected, true_c * (1 + eps), r, k) for k in shifts]
    chosen.append(shifts[int(np.nanargmin(misfits))])
threshold = 0.5 * np.mean(true_c * period / r)


def draw():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    ax = axes[0]
    for n in (-2, -1, 1, 2):
        ax.plot(period, apply_phase_cycle_shift(period, uncorrected, r, n), color=_style.ALTERNATIVE, lw=1)
        end = apply_phase_cycle_shift(period, uncorrected, r, n)[-1]
        if end < 6.4:
            ax.text(period[-1] * 1.02, end, f"N = {n:+d}", fontsize=7, va="center", color=_style.MUTED)
    ax.plot(period, uncorrected, color=_style.MEASURED, lw=2, label="N = 0 (measured)")
    ax.plot(period, case.true_phase_velocity(period), color=_style.TRUTH, ls="--", lw=1, label="true")
    ax.set_xlabel("period (s)")
    ax.set_ylabel("phase velocity (km/s)")
    ax.set_ylim(1.5, 6.5)
    ax.set_xlim(period.min(), period.max() * 1.18)
    ax.set_title(f"branches $1/c_N = 1/c_0 + N\\,T/r$, r = {r:g} km")
    ax.legend(loc="upper center")

    ax = axes[1]
    periods = np.geomspace(2, 100, 100)
    c_ref = case.true_phase_velocity(periods)
    for distance, style in ((100, ":"), (200, "-."), (400, "-"), (1000, "--")):
        ax.loglog(periods, c_ref * periods / distance, color=_style.INK, ls=style, label=f"r = {distance} km")
    ax.axhline(0.05, color=_style.HIGHLIGHT, lw=1)
    ax.text(2.2, 0.055, "5 %", color=_style.HIGHLIGHT, fontsize=8)
    ax.set_xlabel("period (s)")
    ax.set_ylabel(r"$\Delta c / c \approx \lambda / r$")
    ax.set_title("relative distance to the next branch")
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.step(100 * eps_values, chosen, where="mid", color=_style.MEASURED)
    for sign in (-1, 1):
        ax.axvline(100 * sign * threshold, color=_style.HIGHLIGHT, ls="--", lw=1)
    ax.text(100 * threshold + 0.5, 1.6, r"$\varepsilon^* = \langle\lambda/r\rangle/2$", color=_style.HIGHLIGHT,
            fontsize=8)
    ax.set_xlabel(r"reference error $\varepsilon$ (%), $c_{ref} = (1+\varepsilon)\,c_{true}$")
    ax.set_ylabel("selected shift k")
    ax.set_yticks(shifts)
    ax.set_title("branch chosen by the package score")

    for ax, label in zip(axes, "abc"):
        _style.panel_label(ax, label)
    fig.tight_layout()
    return fig


_style.save_themed(draw, __file__)

wrong = np.asarray(chosen) != 0
first_wrong = np.min(np.abs(eps_values[wrong])) if np.any(wrong) else np.nan
print(f"derived threshold eps* = {threshold:.3f}; smallest |eps| with a wrong branch = {first_wrong:.3f}")
