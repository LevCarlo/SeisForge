"""Derive the H-k delays, their trade-offs, and the QC retained fractions with SymPy.

Writes the results shown on docs/rf/hk.md into docs/rf/generated/ and checks
them against the package:

1. Delays from ray geometry. A plane P wave with horizontal slowness p crosses
   the interface at depth H at time t0 + p x. A ray leg of velocity v and
   angle theta (sin theta = p v) through the layer takes H / (v cos theta) and
   moves H tan theta sideways. Summing the legs of a phase and subtracting
   the direct P gives its delay; compared with
   `seisforge.rf.hk_core.hk_phase_delays_s` (to 1e-12 s).
2. Trade-off slopes dH/dk along a curve of constant delay (Vp fixed,
   Vs = Vp / k), and the sensitivity d ln H / d ln Vp at fixed k; compared
   with finite differences of the package delays.
3. Fraction of a bivariate Gaussian with Mahalanobis distance below r.

    PYTHONPATH=<worktree>/src python docs/rf/scripts/hk_derivation.py
"""

from __future__ import annotations

import numpy as np
import sympy as sp

import _rf
from seisforge.rf.hk_core import hk_phase_delays_s, velocities_from_vpvs_ratio

H, p, alpha, beta, k, r, rho_ = sp.symbols("H p alpha beta kappa r rho", positive=True)
q = {"P": sp.sqrt(1 / alpha**2 - p**2), "S": sp.sqrt(1 / beta**2 - p**2)}
speed = {"P": alpha, "S": beta}


def leg(kind):
    """Travel time minus p * (sideways offset) of one crossing of the layer."""
    v = speed[kind]
    sin = p * v
    cos = sp.sqrt(1 - sin**2)
    return H / (v * cos) - p * H * sin / cos


# Legs in order of travel, upward legs first; lower case marks reflections at
# the free surface. PpSs and PsPs have the same legs and arrive together.
PHASES = {"Ps": ["S"], "PpPs": ["P", "P", "S"], "PpSs+PsPs": ["P", "S", "S"]}
delays = {
    name: sp.simplify(sum(leg(kind) for kind in legs) - leg("P"))
    for name, legs in PHASES.items()
}
expected = {"Ps": H * (q["S"] - q["P"]), "PpPs": H * (q["S"] + q["P"]), "PpSs+PsPs": 2 * H * q["S"]}
for name in PHASES:
    assert sp.simplify(delays[name] - expected[name]) == 0, name

# Trade-off at fixed Vp: Vs = Vp / k. Along t = H tau(k) = const,
# dH/dk = -H tau'(k) / tau(k).
tau = {name: sp.simplify((expected[name] / H).subs(beta, alpha / k)) for name in PHASES}
slope = {name: sp.simplify(-H * sp.diff(tau[name], k) / tau[name]) for name in PHASES}
slope_vertical = {name: sp.simplify(slope[name].subs(p, 0)) for name in PHASES}
assert slope_vertical == {"Ps": -H / (k - 1), "PpPs": -H / (k + 1), "PpSs+PsPs": -H / k}
# Fixed k: H = t / tau(Vp); d ln H / d ln Vp = -Vp d ln tau / d Vp, = 1 at p = 0.
vp_sensitivity = {name: sp.simplify(-alpha * sp.diff(sp.log(tau[name]), alpha)) for name in PHASES}
for name in PHASES:
    assert sp.simplify(vp_sensitivity[name].subs(p, 0)) == 1

# Mahalanobis distance of a bivariate Gaussian: d^2 is chi-square with two
# degrees of freedom, P(d < r) = int_0^r rho exp(-rho^2/2) d rho.
fraction = sp.integrate(rho_ * sp.exp(-rho_**2 / 2), (rho_, 0, r))
assert sp.simplify(fraction - (1 - sp.exp(-r**2 / 2))) == 0

# ---- LaTeX snippets ----
eta_s, eta_p = sp.symbols(r"\eta_s \eta_p", positive=True)
to_eta = {q["S"]: eta_s, q["P"]: eta_p}
labels = {"Ps": r"t_{\mathrm{Ps}}", "PpPs": r"t_{\mathrm{PpPs}}",
          "PpSs+PsPs": r"t_{\mathrm{PpSs}} = t_{\mathrm{PsPs}}"}
rows = [f"{labels[name]} &= {sp.latex(expected[name].subs(to_eta), order='old')}" for name in PHASES]
_rf.GENERATED_DIR.mkdir(exist_ok=True)
(_rf.GENERATED_DIR / "hk_delays.tex").write_text(
    "$$\n\\boxed{\\begin{aligned}\n" + "\\\\\n".join(rows)
    + "\n\\end{aligned}}\n\\label{eq:delays}\n$$\n",
    encoding="utf-8",
)
(_rf.GENERATED_DIR / "hk_tradeoff.tex").write_text(
    "$$\n\\left.\\frac{dH}{d\\kappa}\\right|_{t,\\;p=0} = "
    + sp.latex(slope_vertical["Ps"]) + r"\ (\mathrm{Ps}),\qquad "
    + sp.latex(slope_vertical["PpPs"]) + r"\ (\mathrm{PpPs}),\qquad "
    + sp.latex(slope_vertical["PpSs+PsPs"]) + r"\ (\mathrm{PpSs{+}PsPs})"
    + "\n\\label{eq:tradeoff}\n$$\n",
    encoding="utf-8",
)

# ---- numerical checks against the package ----
p0, vp0, h0 = 0.06, _rf.VP_CRUST_KM_S, _rf.H_TRUE_KM
kgrid = np.array([1.6, _rf.K_TRUE, 1.9])
vp_k, vs_k = velocities_from_vpvs_ratio(kgrid, vp_km_s=vp0)
package = hk_phase_delays_s(np.array([h0]), vp_k, vs_k, p0)
worst_delay = 0.0
for name, values in zip(PHASES, package):
    f = sp.lambdify((H, p, alpha, beta), delays[name], "numpy")
    worst_delay = max(worst_delay, float(np.max(np.abs(values[:, 0] - f(h0, p0, vp_k, vs_k)))))
assert worst_delay < 1e-12, worst_delay


def package_delay(i, thickness, kappa, vp=vp0):
    v_p, v_s = velocities_from_vpvs_ratio(np.array([kappa]), vp_km_s=vp)
    return float(hk_phase_delays_s(np.array([thickness]), v_p, v_s, p0)[i][0, 0])


# Central differences; step 1e-5 gives about 1e-8 relative error, 1e-6 is safe.
step = 1e-5
worst_slope = worst_vp = 0.0
for i, name in enumerate(PHASES):
    dt_dk = (package_delay(i, h0, _rf.K_TRUE + step) - package_delay(i, h0, _rf.K_TRUE - step)) / (2 * step)
    dt_dh = (package_delay(i, h0 + step, _rf.K_TRUE) - package_delay(i, h0 - step, _rf.K_TRUE)) / (2 * step)
    numeric = -dt_dk / dt_dh
    exact = float(slope[name].subs({H: h0, p: p0, k: _rf.K_TRUE, alpha: vp0}))
    worst_slope = max(worst_slope, abs(numeric / exact - 1))
    dt_dvp = (package_delay(i, h0, _rf.K_TRUE, vp0 * (1 + step))
              - package_delay(i, h0, _rf.K_TRUE, vp0 * (1 - step))) / (2 * step)
    numeric_vp = -dt_dvp / (h0 * dt_dh)  # d ln H / d ln Vp at fixed t and k
    exact_vp = float(vp_sensitivity[name].subs({p: p0, k: _rf.K_TRUE, alpha: vp0}))
    worst_vp = max(worst_vp, abs(numeric_vp / exact_vp - 1))
    print(f"{name:10s} dH/dk = {exact:7.2f} km (p = {p0}), "
          f"{float(slope_vertical[name].subs({H: h0, k: _rf.K_TRUE})):7.2f} km (p = 0); "
          f"d ln H / d ln Vp = {exact_vp:.3f}")
assert worst_slope < 1e-6 and worst_vp < 1e-6, (worst_slope, worst_vp)
print(f"wrote {_rf.GENERATED_DIR}; delays vs package {worst_delay:.1e} s, "
      f"slopes {worst_slope:.1e}, Vp sensitivity {worst_vp:.1e} (relative)")
