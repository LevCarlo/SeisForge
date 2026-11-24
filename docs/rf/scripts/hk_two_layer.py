"""Figure hk_two_layer.svg (and -dark.svg): sediment first, then the crust beneath it.

Noise-free synthetic RFs (`seisforge.rf.synthetic.receiver_function`,
p = 0.06 s/km, dt = 0.1 s as in `examples/04_rf_qc_hk`) of two three-layer
models, both with the Moho at 35 km (crust Vp 6.3 km/s, Vp/Vs 1.75; mantle
as in `_rf.CRUST`) under a 2 km top layer of density 2.2 g/cm^3 (assumed
values):

- "slow sediment": Vs 1.8 km/s, Vp/Vs 2.2;
- "weak contrast": Vs 2.5 km/s, Vp/Vs 2.0.

Top row: the a = 5 RFs and their one-layer H-k stacks over H 0-5 km
(`HkStack`, Vs fixed at the true value, as the reference config fixes the
sediment Vs, and `min_ps_delay_s` = 0.3 s as in the reference config). Grid
points whose Ps delay is shorter than 0.3 s are hatched: there the stack
samples direct P. Bottom row, slow sediment only: the a = 2.5 RF stacked for
the crust with one layer, and with two layers using the sediment pick of the
top row (`HkStack(mode=2)`), as `seisforge rf hkseq --mode both` does.
Known answers: both sediments are found to within one H step and 0.06 in
Vp/Vs (the a = 5 pulse sampled at 0.1 s), neither stack is dominated by
direct P, and the two-layer crust is within 0.5 km of the Moho.
"""

import numpy as np

import _rf
from _rf import _style, plt
from seisforge.rf.hk_workflow import HkParam, HkStack

P_S_PER_KM = 0.06
MODELS = {
    "slow sediment": (1.8, 2.2),
    "weak contrast": (2.5, 2.0),
}
MIN_PS_SED, MIN_PS_CRUST = 0.3, 0.6
H_SED = HkParam("H", 0.0, 5.0, 0.05)
K_SED = HkParam("k", 1.5, 3.0, 0.01)
H_CRUST = HkParam("H", 25.0, 45.0, 0.1)
K_CRUST = HkParam("k", 1.6, 1.9, 0.005)


def model(vs_sed, k_sed, h_sed=2.0):
    return {
        "thickness_km": np.array([h_sed, _rf.H_TRUE_KM - h_sed, 0.0]),
        "vp_km_s": np.array([vs_sed * k_sed, _rf.VP_CRUST_KM_S, 8.0]),
        "vs_km_s": np.array([vs_sed, _rf.VP_CRUST_KM_S / _rf.K_TRUE, 4.5]),
        "rho_g_cm3": np.array([2.2, 2.8, 3.3]),
    }


sediment = {}
for name, (vs_sed, k_sed) in MODELS.items():
    time_s, data = _rf.synthetic_rf(model(vs_sed, k_sed), P_S_PER_KM, gauss_a=5.0, dt_s=0.1,
                                    end_s=30)
    result = HkStack(_rf.rf_trace(time_s, data, P_S_PER_KM), H_SED, K_SED, Vp=None, Vs=vs_sed,
                     min_ps_delay_s=MIN_PS_SED).stack()
    sediment[name] = (time_s, data, result)

h0, k0 = (sediment["slow sediment"][2].attrs[key] for key in ("H_best", "k_best"))
vs0 = MODELS["slow sediment"][0]
time_lo, data_lo = _rf.synthetic_rf(model(*MODELS["slow sediment"]), P_S_PER_KM, gauss_a=2.5, dt_s=0.1)
trace_lo = _rf.rf_trace(time_lo, data_lo, P_S_PER_KM)
one_layer = HkStack(trace_lo, H_CRUST, K_CRUST, Vp=_rf.VP_CRUST_KM_S,
                    min_ps_delay_s=MIN_PS_CRUST).stack()
two_layer = HkStack(trace_lo, H_CRUST, K_CRUST, Vp=_rf.VP_CRUST_KM_S, mode=2,
                    h0=h0, vp0=None, vs0=vs0, k0=k0, min_ps_delay_s=MIN_PS_CRUST).stack()
# Known answers (see the module docstring).
for name, (vs_sed, k_sed) in MODELS.items():
    attrs = sediment[name][2].attrs
    assert abs(attrs["H_best"] - 2.0) <= 0.05 + 1e-9 and abs(attrs["k_best"] - k_sed) <= 0.06, name
    assert not attrs["direct_p_dominates"], name
assert abs(two_layer.attrs["H_best"] - _rf.H_TRUE_KM) <= 0.5
assert abs(one_layer.attrs["H_best"] - _rf.H_TRUE_KM) > 1.0


def stack_panel(ax, result, h_grid, k_grid, truth, title):
    amp = result.amp_stack.values
    ax.pcolormesh(h_grid.values, k_grid.values, amp / np.abs(amp).max(), cmap="RdBu_r",
                  vmin=-1, vmax=1, shading="auto", rasterized=True)
    if result.excluded_direct_p.values.any():
        ax.contourf(h_grid.values, k_grid.values, result.excluded_direct_p.values.astype(float),
                    levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.plot(*truth, "o", mfc="none", mec="black", ms=9, mew=1.2)
    ax.plot(result.attrs["H_best"], result.attrs["k_best"], "+", color="black", ms=10, mew=1.5)
    ax.set_title(title + f"\npick ({result.attrs['H_best']:.2f} km, {result.attrs['k_best']:.2f})")
    ax.set_xlabel("H (km)")
    ax.set_ylabel(r"$\kappa$")
    ax.grid(False)


def draw():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    ax = axes[0, 0]
    for (name, (time_s, data, _)), color in zip(sediment.items(), (_style.MEASURED, _style.SECOND)):
        ax.plot(time_s, data, color=color, lw=1.0, label=name)
    ax.set_xlim(-1, 4)
    ax.set_xlabel("time after direct P (s)")
    ax.set_ylabel("RF, a = 5")
    ax.set_title("top 2 km: sediment Ps and multiples")
    ax.legend(fontsize=7)
    for ax, name in zip(axes[0, 1:], MODELS):
        vs_sed, k_sed = MODELS[name]
        stack_panel(ax, sediment[name][2], H_SED, K_SED, (2.0, k_sed), f"sediment stack, {name}")

    ax = axes[1, 0]
    ax.plot(time_lo, data_lo, color=_style.MEASURED, lw=1.0)
    ax.set_xlim(-2, 25)
    ax.set_xlabel("time after direct P (s)")
    ax.set_ylabel("RF, a = 2.5")
    ax.set_title("slow sediment over the crust")
    stack_panel(axes[1, 1], one_layer, H_CRUST, K_CRUST, (_rf.H_TRUE_KM, _rf.K_TRUE),
                "crust, one layer")
    stack_panel(axes[1, 2], two_layer, H_CRUST, K_CRUST, (_rf.H_TRUE_KM, _rf.K_TRUE),
                f"crust under the picked sediment\n(h0 = {h0:.2f} km, k0 = {k0:.2f})")
    for axis, label in zip(axes.ravel(), "abcdef"):
        _style.panel_label(axis, label)
    fig.tight_layout()
    return fig


_rf.save(draw, __file__)
for name, (_, _, result) in sediment.items():
    print(f"sediment {name}: truth (2.0, {MODELS[name][1]}), pick ({result.attrs['H_best']:.2f}, "
          f"{result.attrs['k_best']:.2f})")
for name, result in (("one layer", one_layer), ("two layers", two_layer)):
    print(f"crust {name}: pick ({result.attrs['H_best']:.2f}, {result.attrs['k_best']:.3f}), "
          f"truth ({_rf.H_TRUE_KM}, {_rf.K_TRUE})")
