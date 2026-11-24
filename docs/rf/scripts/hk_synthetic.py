"""Figure hk_synthetic.svg (and -dark.svg): H-k stack of one noise-free synthetic RF.

The RF is `seisforge.rf.synthetic.receiver_function` for the crust of
`_rf.CRUST` (truth H = 35 km, Vp/Vs = 1.75), p = 0.06 s/km, a = 2.5. The
stack is `seisforge.rf.hk_workflow.HkStack` with Vp = 6.3 km/s (the true
value) and the default weights (0.5, 0.2, 0.3). Panels (a)-(c) show the
corrected amplitude at each phase delay; the dashed lines are the tangents
dH/dk = -H tau'(k) / tau(k) of docs/rf/scripts/hk_derivation.py through the
truth. (d) the weighted stack; the maximum must lie within one grid step of
the truth.
"""

import numpy as np

import _rf
from _rf import _style, plt
from seisforge.rf.hk_workflow import HkParam, HkStack

P_S_PER_KM, GAUSS_A = 0.06, 2.5
H_GRID = HkParam("H", 25.0, 45.0, 0.1)
K_GRID = HkParam("k", 1.60, 1.90, 0.005)

time_s, data = _rf.synthetic_rf(p=P_S_PER_KM, gauss_a=GAUSS_A)
result = HkStack(_rf.rf_trace(time_s, data, P_S_PER_KM), H_GRID, K_GRID,
                 Vp=_rf.VP_CRUST_KM_S, min_ps_delay_s=0.6).stack()
h_best, k_best = result.attrs["H_best"], result.attrs["k_best"]
assert abs(h_best - _rf.H_TRUE_KM) <= 0.1 + 1e-9 and abs(k_best - _rf.K_TRUE) <= 0.005 + 1e-9, (h_best, k_best)
assert all(result.attrs[f"n_invalid_{name}"] == 0 for name in ("Ps", "PpPs", "PsPs"))


def slope(kind, h=_rf.H_TRUE_KM, k=_rf.K_TRUE, vp=_rf.VP_CRUST_KM_S, p=P_S_PER_KM):
    """dH/dk along a constant-delay curve (hk_derivation.py), Vs = Vp / k."""
    eta_s = np.sqrt(k**2 / vp**2 - p**2)
    eta_p = np.sqrt(1 / vp**2 - p**2)
    d_eta_s = k / (vp**2 * eta_s)
    tau, d_tau = {"Ps": (eta_s - eta_p, d_eta_s), "PpPs": (eta_s + eta_p, d_eta_s),
                  "PsPs": (2 * eta_s, 2 * d_eta_s)}[kind]
    return -h * d_tau / tau


PANELS = [
    ("a", "Ps", r"$A(t_{\mathrm{Ps}})$", result.amp_Ps.values),
    ("b", "PpPs", r"$A(t_{\mathrm{PpPs}})$", result.amp_PpPs.values),
    ("c", "PsPs", r"$-A(t_{\mathrm{PsPs}})$", -result.amp_PsPs.values),
    ("d", None, r"stack $0.5\,$Ps$ + 0.2\,$PpPs$ - 0.3\,$PsPs", result.amp_stack.values),
]


def draw():
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0), sharey=True)
    for ax, (label, kind, title, amp) in zip(axes, PANELS):
        ax.pcolormesh(H_GRID.values, K_GRID.values, amp / np.abs(amp).max(),
                      cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
        kinds = [kind] if kind else ["Ps", "PpPs", "PsPs"]
        for name in kinds:
            k_line = np.array([K_GRID.vmin, K_GRID.vmax])
            ax.plot(_rf.H_TRUE_KM + slope(name) * (k_line - _rf.K_TRUE), k_line,
                    color="black", lw=0.8, ls="--")
        ax.plot(_rf.H_TRUE_KM, _rf.K_TRUE, "o", mfc="none", mec="black", ms=9, mew=1.2)
        if kind is None:
            ax.plot(h_best, k_best, "+", color="black", ms=9, mew=1.4)
        ax.set_xlim(H_GRID.vmin, H_GRID.vmax)
        ax.set_ylim(K_GRID.vmin, K_GRID.vmax)
        ax.set_title(title)
        ax.set_xlabel("H (km)")
        ax.grid(False)
        _style.panel_label(ax, label)
    axes[0].set_ylabel(r"$\kappa = V_P/V_S$")
    fig.tight_layout()
    return fig


_rf.save(draw, __file__)
print(f"best H = {h_best:.2f} km, k = {k_best:.3f} (truth {_rf.H_TRUE_KM}, {_rf.K_TRUE}); "
      "tangent slopes dH/dk: " + ", ".join(f"{n} {slope(n):.1f}" for n in ("Ps", "PpPs", "PsPs")))
