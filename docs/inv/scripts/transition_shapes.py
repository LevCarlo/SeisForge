"""Figure transition_shapes.svg (and -dark.svg): weights of a gradational interface.

`seisforge.inv.transition_weight` for an interface at 10 km with 10-90 %
width w = 2 km. (a) S(z) for both shapes, with the 10/50/90 % points and the
compact supports. (b) w dS/dz from finite differences against the analytic
maxima: logistic k/4 = ln 9 / 2 = 1.099 (the rescaling after truncation
adds 3e-4), and smootherstep 30/16 / (2 h/w) = 0.950 with h the half-support.
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import transition_half_support_km, transition_weight

DEPTH_KM, WIDTH_KM = 10.0, 2.0
z_km = np.linspace(4.0, 16.0, 24001)
shapes = ("sigmoid", "smootherstep")
weights = {s: transition_weight(z_km, DEPTH_KM, WIDTH_KM, s) for s in shapes}
slopes = {s: WIDTH_KM * np.gradient(weights[s], z_km) for s in shapes}
half = {s: transition_half_support_km(WIDTH_KM, s) for s in shapes}
analytic = {
    "sigmoid": np.log(9.0) / 2.0,
    "smootherstep": (30.0 / 16.0) / (2.0 * half["smootherstep"] / WIDTH_KM),
}
for s in shapes:
    at = transition_weight(np.array([DEPTH_KM - WIDTH_KM / 2, DEPTH_KM, DEPTH_KM + WIDTH_KM / 2]),
                           DEPTH_KM, WIDTH_KM, s)
    print(f"{s}: S(z_i - w/2, z_i, z_i + w/2) = {np.round(at, 6)}; half-support {half[s] / WIDTH_KM:.4f} w; "
          f"max w dS/dz {slopes[s].max():.4f} (analytic {analytic[s]:.4f})")


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    colors = {"sigmoid": _style.MEASURED, "smootherstep": _style.SECOND}
    for s in shapes:
        axes[0].plot(weights[s], z_km, color=colors[s], label=s)
        axes[0].axhspan(DEPTH_KM - half[s], DEPTH_KM + half[s], color=colors[s], alpha=0.08, lw=0)
        axes[1].plot(slopes[s], z_km, color=colors[s], label=s)
        axes[1].axvline(analytic[s], color=colors[s], ls="--", lw=0.8)
    for level, depth in ((0.1, DEPTH_KM - WIDTH_KM / 2), (0.5, DEPTH_KM), (0.9, DEPTH_KM + WIDTH_KM / 2)):
        axes[0].plot(level, depth, "o", color=_style.INK, ms=4)
    axes[0].set_xlabel("weight of the lower segment $S(z)$")
    axes[0].set_ylabel("depth (km)")
    axes[0].set_title(rf"$z_i$ = {DEPTH_KM:g} km, $w$ = {WIDTH_KM:g} km (shaded: support)")
    axes[0].legend(loc="lower left")
    _style.panel_label(axes[0], "(a)")
    axes[1].set_xlabel(r"$w\,dS/dz$  (dashed: analytic maximum)")
    axes[1].set_title(r"steepest gradient $\approx$ (1.10 or 0.95) $\Delta v / w$")
    axes[1].invert_yaxis()
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
