"""Figure rotation_geometry.png: station geometry and the R/T axes of the package (Lin et al. 2008).

The R and T unit vectors are not typed in by hand: they are read off
`seisforge.ant.rotation.rotate_zne_ccf_to_zrt` by rotating unit CCFs (for
example ``RZ`` of a CCF whose only non-zero entry is ``NZ = 1`` is the north
component of the source radial axis). Flat geometry, map view (x east,
y north).
"""

import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

import _style
from _style import plt
from seisforge.ant.rotation import rotate_zne_ccf_to_zrt

AZIMUTH_DEG = 60.0  # A -> B, clockwise from north (illustration)
BACK_AZIMUTH_DEG = AZIMUTH_DEG + 180.0  # flat Earth
DISTANCE = 4.0  # plot units


def axis_from_code(component: str) -> np.ndarray:
    """(east, north) unit vector of one rotated axis, read from the package."""
    vec = {}
    for horizontal in "NE":
        ccf = {c: 0.0 for c in ("ZZ", "ZN", "ZE", "NZ", "NN", "NE", "EZ", "EN", "EE")}
        if component[0] == "Z":  # receiver axis: put the unit on Z(source) x H(receiver)
            ccf["Z" + horizontal] = 1.0
        else:  # source axis: H(source) x Z(receiver)
            ccf[horizontal + "Z"] = 1.0
        vec[horizontal] = rotate_zne_ccf_to_zrt(ccf, AZIMUTH_DEG, BACK_AZIMUTH_DEG)[component]
    return np.array([vec["E"], vec["N"]])


R_A, T_A = axis_from_code("RZ"), axis_from_code("TZ")
R_B, T_B = axis_from_code("ZR"), axis_from_code("ZT")

A = np.array([0.0, 0.0])
B = DISTANCE * np.array([np.sin(np.deg2rad(AZIMUTH_DEG)), np.cos(np.deg2rad(AZIMUTH_DEG))])

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.grid(False)
ax.plot([A[0], B[0]], [A[1], B[1]], color=_style.ALTERNATIVE, lw=1, ls="--", zorder=0)
x_hat = (B - A) / DISTANCE
offset = 0.45 * np.array([x_hat[1], -x_hat[0]])  # to the right of the path
ax.add_patch(FancyArrowPatch(A + 1.2 * x_hat + offset, A + 2.8 * x_hat + offset,
                             arrowstyle="-|>", mutation_scale=14, color="0.45", lw=1.5))
ax.text(*(A + 2.0 * x_hat + 1.6 * offset), "propagation of the\npositive-lag arrival", color="0.4",
        fontsize=8, ha="left", va="top")

for name, p, dxy in (("A: virtual source\n(1st letter)", A, (0.2, -0.15)),
                     ("B: receiver\n(2nd letter)", B, (-2.0, 0.9))):
    ax.plot(*p, marker="^", ms=11, color="k")
    ax.text(p[0] + dxy[0], p[1] + dxy[1], name, ha="left", va="top", fontsize=8)
    ax.add_patch(FancyArrowPatch(p, p + [0, 1.2], arrowstyle="-|>", mutation_scale=10, color="0.3"))
    ax.text(p[0], p[1] + 1.3, "N", ha="center", va="bottom", fontsize=8)

scale = 1.5
for p, r, t, tag in ((A, R_A, T_A, "A"), (B, R_B, T_B, "B")):
    for vec, label, color in ((r, rf"$\hat R_{tag}$", _style.MEASURED), (t, rf"$\hat T_{tag}$", _style.SECOND)):
        ax.add_patch(FancyArrowPatch(p, p + scale * vec, arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=2))
        ax.text(*(p + (scale + 0.25) * vec), label, color=color, ha="center", va="center", fontsize=11)

# Angles are clockwise from north; matplotlib arcs run counter-clockwise from east.
ax.add_patch(Arc(A, 1.6, 1.6, theta1=90 - AZIMUTH_DEG, theta2=90, color="k"))
ax.text(*(A + 0.95 * np.array([np.sin(np.deg2rad(AZIMUTH_DEG / 2)), np.cos(np.deg2rad(AZIMUTH_DEG / 2))])),
        r"$\theta$", fontsize=11, ha="left", va="bottom")
ax.add_patch(Arc(B, 1.2, 1.2, theta1=90 - BACK_AZIMUTH_DEG, theta2=90, color="k"))
half = np.deg2rad(BACK_AZIMUTH_DEG / 2)
ax.text(*(B + 0.75 * np.array([np.sin(half), np.cos(half)])), r"$\psi$", fontsize=11, ha="left", va="center")

ax.set_aspect("equal")
ax.set_xlim(-2.6, 6.2)
ax.set_ylim(-2.2, 4.6)
ax.set_xlabel("east (arbitrary units)")
ax.set_ylabel("north (arbitrary units)")
ax.set_title(r"rotate_zne_ccf_to_zrt: $\theta$ = azimuth at A, $\psi$ = back azimuth at B"
             "\nboth radial axes point from A towards B (Lin et al. 2008, Fig. 2);"
             "\n$\\hat T = \\hat R$ turned 90° clockwise")
_style.save(fig, __file__)

# Consistency with seisforge.ant.rotation (flat Earth: R_A = R_B = x_hat, T_A = T_B).
assert np.allclose(R_A, R_B) and np.allclose(T_A, T_B)
assert np.allclose(R_A, x_hat)
# Clockwise by 90 deg in map view (x east, y north): (x, y) -> (y, -x).
assert np.allclose(T_A, [x_hat[1], -x_hat[0]])
