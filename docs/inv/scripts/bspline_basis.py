"""Figure bspline_basis.svg (and -dark.svg): the B-spline Vs parameterization.

The crust segment of examples/06_inversion: 1-15 km, 8 coefficients,
degree 3, geometric knots with alpha = 1.6. (a) Basis functions from
`seisforge.inv.bspline_basis` and their sum. (b) The truth profile of the
example and the same profile with one interior coefficient raised by
0.15 km/s. Checks printed: partition of unity, endpoint interpolation, and
that the change equals 0.15 B_3(z) exactly (the profile is linear in c).
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import BSplineVs, bspline_basis, bspline_knots

TOP_KM, BOTTOM_KM = 1.0, 15.0
ALPHA = 1.6
COEFFICIENTS = np.array([2.65, 2.85, 3.05, 3.20, 3.30, 3.45, 3.60, 3.70])  # example truth
PERTURBED, DELTA = 3, 0.15

z_km = np.linspace(TOP_KM, BOTTOM_KM, 1401)
basis, knots = bspline_basis(z_km, n_coefficients=8, domain=(TOP_KM, BOTTOM_KM), alpha=ALPHA)
knots_km = TOP_KM + (BOTTOM_KM - TOP_KM) * np.unique(bspline_knots(8, alpha=ALPHA))
profile = BSplineVs(COEFFICIENTS, knot_alpha=ALPHA).evaluate(z_km - TOP_KM, thickness=BOTTOM_KM - TOP_KM)
raised = COEFFICIENTS.copy()
raised[PERTURBED] += DELTA
profile_raised = BSplineVs(raised, knot_alpha=ALPHA).evaluate(z_km - TOP_KM, thickness=BOTTOM_KM - TOP_KM)

print("max |sum B - 1|:", np.max(np.abs(basis.sum(axis=1) - 1)))
print("endpoints - (c0, c7):", profile[0] - COEFFICIENTS[0], profile[-1] - COEFFICIENTS[-1])
print("max |change - 0.15 B_3|:", np.max(np.abs(profile_raised - profile - DELTA * basis[:, PERTURBED])))
print("interior knots (km):", np.round(knots_km[1:-1], 3))


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)
    colors = [_style.MEASURED, _style.SECOND, _style.EXTRA, _style.HIGHLIGHT, _style.EXTRA2,
              _style.MUTED, _style.MEASURED, _style.SECOND]
    for i in range(basis.shape[1]):
        axes[0].plot(basis[:, i], z_km, color=colors[i], lw=1.6 if i == PERTURBED else 1.0,
                     ls="-" if i < 6 else "--", label=f"$B_{i}$")
    axes[0].plot(basis.sum(axis=1), z_km, color=_style.INK, lw=1.0, ls=":", label=r"$\sum B_i$")
    for knot in knots_km:
        axes[0].axhline(knot, color=_style.ALTERNATIVE, lw=0.6)
    axes[0].set_xlabel("basis value")
    axes[0].set_ylabel("depth (km)")
    axes[0].set_title(rf"cubic basis, geometric knots $\alpha$ = {ALPHA:g}")
    axes[0].legend(ncol=3, fontsize=7, loc="lower right")
    _style.panel_label(axes[0], "(a)")

    axes[1].plot(profile, z_km, color=_style.INK, label="coefficients of the example truth")
    axes[1].plot(profile_raised, z_km, color=_style.HIGHLIGHT, label=rf"$c_{PERTURBED}$ + {DELTA:g} km/s")
    axes[1].fill_betweenx(z_km, profile, profile_raised, color=_style.HIGHLIGHT, alpha=0.2, lw=0)
    for knot in knots_km:
        axes[1].axhline(knot, color=_style.ALTERNATIVE, lw=0.6)
    axes[1].set_xlabel("Vs (km/s)")
    axes[1].set_title(r"$c_3$ moves Vs where $B_3 > 0$")
    axes[1].legend(fontsize=7, loc="lower left")
    axes[1].invert_yaxis()
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
