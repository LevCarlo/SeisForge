"""Figure ftan_synthetic.svg (and -dark.svg): FTAN of a synthetic ZZ CCF with known dispersion.

(a) the positive-lag CCF; (b) the FTAN envelope |S(t, w0)| mapped to group
velocity r / t (each period column normalised), with the basic-FTAN group
velocity pick (at the instantaneous period) and the true group velocity;
(c) misfits of group and phase velocity to the truth. The true curve is given
as prediction file only to fix the 2 pi cycle (see phase_branches.py).
"""

import numpy as np
from matplotlib.ticker import NullFormatter

import _ccf_case as case
import _style
from _style import plt
from seisforge.ant.filters import GaussianFTANFilter

time, zz = case.zz_ccf()
trace = case.positive_trace(time, zz)
meas = case.measurement(trace, case.aftan_config(prediction=case.prediction_file()))
basic = meas.measure()

lag = case.DELTA_S * np.arange(trace.stats.npts)
periods = np.geomspace(case.MIN_PERIOD_S, case.MAX_PERIOD_S, 120)
envelope = np.abs(GaussianFTANFilter(case.DELTA_S, periods, meas.basic_alpha, nfft=meas.nfft).traces(
    meas.trace.data))[: trace.stats.npts]
velocity_axis = np.linspace(2.0, 4.5, 300)
with np.errstate(divide="ignore"):
    velocity_of_lag = np.where(lag > 0, case.DISTANCE_KM / lag, np.inf)
energy = np.empty((velocity_axis.size, periods.size))
for j in range(periods.size):
    column = envelope[:, j] / envelope[:, j].max()
    energy[:, j] = np.interp(velocity_axis, velocity_of_lag[::-1], column[::-1])

du = basic["group_velocity"] - case.true_group_velocity(basic["period"])
dc = basic["phase_velocity"] - case.true_phase_velocity(basic["period"])


def draw():
    fig = plt.figure(figsize=(10.5, 6.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.6, 1], width_ratios=[1.5, 1])
    ax_trace = fig.add_subplot(grid[0, :])
    ax_map = fig.add_subplot(grid[1, 0])
    ax_err = fig.add_subplot(grid[1, 1])

    ax_trace.plot(lag, trace.data / np.abs(trace.data).max(), color=_style.INK, lw=0.7)
    for u in (4.5, 2.0):
        ax_trace.axvline(case.DISTANCE_KM / u, color=_style.ALTERNATIVE, ls=":")
    ax_trace.set_xlim(0, 300)
    ax_trace.set_xlabel("lag time (s)")
    ax_trace.set_ylabel("normalised CCF")
    ax_trace.set_title(f"synthetic ZZ CCF, positive lag, r = {case.DISTANCE_KM:g} km "
                       "(dotted: group-velocity window 2.0-4.5 km/s)")

    ax_map.pcolormesh(periods, velocity_axis, energy, shading="auto", cmap=_style.GREY_CMAP, rasterized=True)
    truth_periods = np.geomspace(case.MIN_PERIOD_S, case.MAX_PERIOD_S, 200)
    ax_map.plot(truth_periods, case.true_group_velocity(truth_periods), color=_style.HIGHLIGHT, lw=1.5,
                label="true group velocity")
    ax_map.plot(truth_periods, case.true_phase_velocity(truth_periods), color=_style.HIGHLIGHT, lw=1.5,
                ls="--", label="true phase velocity")
    ax_map.plot(basic["period"], basic["group_velocity"], "o", ms=3, color=_style.MEASURED,
                label="group-velocity pick")
    ax_map.plot(basic["period"], basic["phase_velocity"], "s", ms=3, mfc="none", color=_style.SECOND,
                label="phase velocity")
    ax_map.set_xscale("log")
    ax_map.set_xticks([6, 10, 20, 40], labels=["6", "10", "20", "40"])
    ax_map.xaxis.set_minor_formatter(NullFormatter())
    ax_map.set_xlabel("period (s)")
    ax_map.set_ylabel("velocity (km/s)")
    ax_map.set_title(rf"FTAN envelope, $\alpha$ = {meas.basic_alpha:g} (columns normalised)")
    ax_map.legend(loc="lower right")

    ax_err.plot(basic["period"], du, "o-", ms=3, color=_style.MEASURED, label="group velocity")
    ax_err.plot(basic["period"], dc, "s-", ms=3, mfc="none", color=_style.SECOND, label="phase velocity")
    ax_err.axhline(0, color=_style.INK, lw=0.8)
    ax_err.set_xscale("log")
    ax_err.set_xticks([6, 10, 20, 40], labels=["6", "10", "20", "40"])
    ax_err.xaxis.set_minor_formatter(NullFormatter())
    ax_err.set_xlabel("instantaneous period (s)")
    ax_err.set_ylabel("measured - true (km/s)")
    ax_err.set_title("misfit to the true curves")
    ax_err.legend()

    for ax, label in zip((ax_trace, ax_map, ax_err), "abc"):
        _style.panel_label(ax, label)
    fig.tight_layout()
    return fig


_style.save_themed(draw, __file__)
print(f"max |dU| = {np.max(np.abs(du)):.3f} km/s, max |dc| = {np.max(np.abs(dc)):.3f} km/s, "
      f"periods {basic['period'].min():.1f}-{basic['period'].max():.1f} s")
