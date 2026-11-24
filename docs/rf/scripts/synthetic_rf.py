"""Figure synthetic_rf.svg (and -dark.svg): a synthetic P receiver function and two checks.

(a) `seisforge.rf.synthetic.receiver_function` for the crust of `_rf.CRUST`
(H = 35 km, Vp/Vs = 1.75), p = 0.06 s/km, a = 2.5, unit-peak amplitude.
Ticks: Ps, PpPs and PpSs+PsPs times from `seisforge.rf.hk_core.hk_phase_delays_s`
(ray theory, an independent formula); each pulse peak is checked to lie
within one sample of its tick, with the expected sign.
(b) Direct-P amplitude of a half-space against the closed form
2 beta^2 p q_beta / (1 - 2 beta^2 p^2) (docs/rf/scripts/synthetic_derivation.py).
"""

import numpy as np

import _rf
from _rf import _style, plt
from seisforge.rf.hk_core import hk_phase_delays_s

P_S_PER_KM, GAUSS_A, DT_S = 0.06, 2.5, 0.05

time_s, rf = _rf.synthetic_rf(p=P_S_PER_KM, gauss_a=GAUSS_A, dt_s=DT_S, begin_s=-5, end_s=25)
vp, vs = _rf.CRUST["vp_km_s"][0], _rf.CRUST["vs_km_s"][0]
arrivals = {"P": 0.0}
for name, delay in zip(("Ps", "PpPs", "PpSs+PsPs"),
                       hk_phase_delays_s(np.array([_rf.H_TRUE_KM]), np.array([vp]),
                                         np.array([vs]), P_S_PER_KM)):
    arrivals[name] = float(delay[0, 0])
signs = {"P": 1, "Ps": 1, "PpPs": 1, "PpSs+PsPs": -1}
for name, t_arrival in arrivals.items():
    window = np.abs(time_s - t_arrival) < 1.0
    i_peak = np.argmax(np.abs(rf[window]))  # largest pulse near the tick
    assert abs(time_s[window][i_peak] - t_arrival) <= DT_S, (name, t_arrival)
    assert np.sign(rf[window][i_peak]) == signs[name], name

half_vp, half_vs = 6.3, 3.6
half_space = {"thickness_km": np.array([0.0]), "vp_km_s": np.array([half_vp]),
              "vs_km_s": np.array([half_vs]), "rho_g_cm3": np.array([2.8])}
p_axis = np.linspace(0.0, 0.08, 9)
package = np.array([
    _rf.synthetic_rf(half_space, p, gauss_a=GAUSS_A, dt_s=0.01, begin_s=-5, end_s=10)[1].max()
    for p in p_axis
])
p_fine = np.linspace(0.0, 0.08, 200)
q_b = np.sqrt(1 / half_vs**2 - p_fine**2)
closed = 2 * half_vs**2 * p_fine * q_b / (1 - 2 * half_vs**2 * p_fine**2)
q_b_axis = np.sqrt(1 / half_vs**2 - p_axis**2)
misfit = np.max(np.abs(package - 2 * half_vs**2 * p_axis * q_b_axis / (1 - 2 * half_vs**2 * p_axis**2)))
assert misfit < 1e-3, misfit


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.axhline(0, color=_style.MUTED, lw=0.6)
    ax.plot(time_s, rf, color=_style.MEASURED, lw=1.2)
    for name, t_arrival in arrivals.items():
        ax.axvline(t_arrival, color=_style.INK, lw=0.7, ls=":")
        ax.text(t_arrival, 0.98, name, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=8, color=_style.INK,
                bbox={"facecolor": _style.BACKGROUND, "edgecolor": "none", "pad": 1})
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_xlabel("time after direct P (s)")
    ax.set_ylabel("radial RF (unit-peak Gaussian)")
    ax.set_title(f"H = {_rf.H_TRUE_KM:g} km, Vp/Vs = {_rf.K_TRUE:g}, p = {P_S_PER_KM} s/km, a = {GAUSS_A}")

    ax = axes[1]
    ax.plot(p_fine, closed, color=_style.TRUTH, lw=1.2, label="closed form")
    ax.plot(p_axis, package, "o", color=_style.MEASURED, ms=4, label="package")
    ax.set_xlabel("ray parameter p (s/km)")
    ax.set_ylabel("direct-P amplitude R/Z")
    ax.set_title("half-space (Vp 6.3, Vs 3.6 km/s)")
    ax.legend(loc="lower right")
    for axis, label in zip(axes, "ab"):
        _style.panel_label(axis, label)
    fig.tight_layout()
    return fig


_rf.save(draw, __file__)
print("arrival times (s): " + ", ".join(f"{k} {v:.2f}" for k, v in arrivals.items()))
print(f"half-space amplitude, package vs closed form: {misfit:.1e}")
