"""Derive the half-space receiver function and the Gaussian pulse with SymPy.

Writes the two results shown on docs/rf/synthetic.md into docs/rf/generated/
(the page includes them), then checks them against the package:

- ``R/Z`` of a P wave incident on the free surface of a half-space, from the
  two traction-free conditions, against
  `seisforge.rf.synthetic.p_wave_surface_response` (to 1e-12);
- the inverse transform of ``exp(-w^2 / (4 a^2))``, whose peak ``a / sqrt(pi)``
  is the ``impulse`` amplitude of a spike, against
  `seisforge.rf.synthetic.receiver_function` on a half-space (to 1e-3).

    PYTHONPATH=<worktree>/src python docs/rf/scripts/synthetic_derivation.py

Conventions as in `seisforge.rf.synthetic`: z positive down, plane waves
exp(i w (p x + q z - t)), q > 0 down-going; P motion along the slowness
vector, SV motion along (q, -p); traction scaled by i w; R = u_x, Z = -u_z.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

import _rf
from seisforge.rf.synthetic import p_wave_surface_response

p, alpha, beta, rho, a, t, w = sp.symbols("p alpha beta rho a t omega", positive=True)
A_P, B_S = sp.symbols("A_P B_S")  # reflected P and SV amplitudes
q_a = sp.sqrt(1 / alpha**2 - p**2)
q_b = sp.sqrt(1 / beta**2 - p**2)
mu = rho * beta**2
lam = rho * alpha**2 - 2 * mu


def wave(kind, q):
    """Displacement (u_x, u_z) and scaled tractions of a unit plane wave."""
    if kind == "P":
        ux, uz = alpha * p, alpha * q
    else:  # SV, perpendicular to the slowness (p, q)
        ux, uz = beta * q, -beta * p
    t_xz = mu * (q * ux + p * uz)
    t_zz = lam * (p * ux + q * uz) + 2 * mu * q * uz
    return sp.Matrix([ux, uz, t_xz, t_zz])


incident = wave("P", -q_a)  # up-going P, unit amplitude
total = incident + A_P * wave("P", q_a) + B_S * wave("S", q_b)
solution = sp.solve([total[2], total[3]], [A_P, B_S], dict=True)[0]
u_x, u_z = (sp.simplify(total[i].subs(solution)) for i in (0, 1))
ratio = sp.simplify(u_x / (-u_z))
expected = 2 * p * q_b / (1 / beta**2 - 2 * p**2)
assert sp.simplify(ratio - expected) == 0

pulse = sp.simplify(
    sp.integrate(sp.exp(-w**2 / (4 * a**2)) * sp.cos(w * t), (w, -sp.oo, sp.oo)) / (2 * sp.pi)
)
assert sp.simplify(pulse - a / sp.sqrt(sp.pi) * sp.exp(-a**2 * t**2)) == 0

# Printed form, checked against the derived ratio (SymPy's own printing puts
# the minus sign in front).
q_b_sym = sp.Symbol(r"q_\beta", positive=True)
printed = 2 * beta**2 * p * q_b_sym / (1 - 2 * beta**2 * p**2)
assert sp.simplify(printed.subs(q_b_sym, q_b) - ratio) == 0
_rf.GENERATED_DIR.mkdir(exist_ok=True)
(_rf.GENERATED_DIR / "halfspace_ratio.tex").write_text(
    "$$\n\\boxed{\\;\\frac{R}{Z} = "
    + r"\frac{2\beta^{2} p\, q_\beta}{1 - 2\beta^{2} p^{2}}"
    + ",\\qquad q_\\beta = \\sqrt{\\beta^{-2} - p^2}\\;}\n\\label{eq:halfspace}\n$$\n",
    encoding="utf-8",
)
(_rf.GENERATED_DIR / "gaussian_pulse.tex").write_text(
    "$$\n\\frac{1}{2\\pi}\\int_{-\\infty}^{\\infty} e^{-\\omega^2/4a^2}\\,e^{i\\omega t}\\,d\\omega = "
    + sp.latex(pulse)
    + "\n\\label{eq:pulse}\n$$\n",
    encoding="utf-8",
)

# Package against the symbolic half-space ratio (exact for the half-space:
# only rounding, so 1e-12).
evaluate = sp.lambdify((p, alpha, beta), ratio, "numpy")
worst = 0.0
for vp, vs in ((6.3, 3.6), (8.0, 4.5), (3.0, 1.2)):
    for ray_p in np.linspace(0.0, 0.95 / vp, 7):
        radial, vertical = p_wave_surface_response(
            np.array([0.0]), np.array([vp]), np.array([vs]), np.array([2.7]), ray_p,
            np.array([0.5, 5.0, 30.0]),
        )
        worst = max(worst, float(np.max(np.abs(radial / vertical - evaluate(ray_p, vp, vs)))))
assert worst < 1e-12, worst

# A half-space RF is the spike R/Z at 0 s; with "impulse" amplitude it peaks
# at (R/Z) a / sqrt(pi). The peak falls on a sample (t = 0), so only the
# FFT truncation of the Gaussian matters; 1e-3 is generous.
half_space = {k: v[1:] for k, v in _rf.CRUST.items()}
half_space["thickness_km"] = np.array([0.0])
_, impulse = _rf.synthetic_rf(half_space, 0.06, gauss_a=2.5, dt_s=0.01, amplitude="impulse")
peak_error = abs(impulse.max() / (evaluate(0.06, 8.0, 4.5) * 2.5 / np.sqrt(np.pi)) - 1)
assert peak_error < 1e-3, peak_error
print(f"wrote {_rf.GENERATED_DIR}; |R/Z| symbolic vs package {worst:.1e}, "
      f"impulse peak vs a/sqrt(pi) {peak_error:.1e} (relative)")
