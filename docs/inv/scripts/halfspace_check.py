"""Check (no figure): the forward wrappers against the Rayleigh wave of a half-space.

For a homogeneous half-space the Rayleigh phase velocity c and the surface
ellipticity are period-independent. With gamma = c^2 / Vs^2,
q = sqrt(1 - c^2 / Vp^2) and s = sqrt(1 - gamma), c solves
(2 - gamma)^2 = 4 q s and H/V = |2 - gamma - 2 q s| / (q gamma)
(derivation on the forward page, Appendix B). The half-space is passed to
`predict_dispersion` and `predict_rayleigh_hv` as two identical layers.
"""

import numpy as np
from scipy.optimize import brentq

from seisforge.inv import LayeredModel, predict_dispersion, predict_rayleigh_hv
from seisforge.inv.forward import DispersionRequest, RayleighHVRequest

VS_KM_S, RHO = 3.0, 2.7
PERIODS_S = np.array([0.5, 2.0, 8.0])

for vp_vs in (np.sqrt(3.0), 2.0, 3.0):
    vp = vp_vs * VS_KM_S

    def terms(c):
        gamma = (c / VS_KM_S) ** 2
        return gamma, np.sqrt(1 - (c / vp) ** 2), np.sqrt(1 - gamma)

    def secular(c):
        gamma, q, s = terms(c)
        return (2 - gamma) ** 2 - 4 * q * s

    c = brentq(secular, 0.5 * VS_KM_S, 0.999999 * VS_KM_S)
    gamma, q, s = terms(c)
    hv = abs(2 - gamma - 2 * q * s) / (q * gamma)
    model = LayeredModel(thickness=[50.0, 0.0], vp=[vp, vp], vs=[VS_KM_S] * 2, rho=[RHO] * 2)
    c_pkg = predict_dispersion(model, DispersionRequest(periods=PERIODS_S)).velocity
    hv_pkg = predict_rayleigh_hv(model, RayleighHVRequest(periods=PERIODS_S)).hv
    print(f"Vp/Vs {vp_vs:.3f}: c/Vs {c / VS_KM_S:.5f} (package max dev {np.max(np.abs(c_pkg - c)) / c:.1e}), "
          f"H/V {hv:.5f} (package max dev {np.max(np.abs(hv_pkg - hv)):.1e})")
