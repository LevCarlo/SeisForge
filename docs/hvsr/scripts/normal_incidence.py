"""Normal-incidence surface compliance of a layer over a half-space.

At k = 0 the P-SV and SH systems decouple into 1-D problems with the closed
form (derived on the Green-function page, Appendix)

    -Im C * omega * Z_N = Z_N^2 / (Z_N^2 cos^2(theta) + Z_1^2 sin^2(theta)),
    theta = omega H / v_1,  Z = rho v,

with v = Vs for the SH and SV terms and v = Vp for the P (vertical) term.
The script checks the package compliance against it and plots both.
"""

from _case import DENSITY_G_CM3, MODEL, THICKNESS_KM, VP_KM_S, VS_KM_S, _style, np, save

from seisforge.hvsr.propagator import psv_surface_compliance, sh_surface_compliance

FREQUENCY_HZ = np.linspace(0.05, 8.0, 800)


def closed_form(velocity):
    z1, zn = DENSITY_G_CM3[0] * velocity[0], DENSITY_G_CM3[1] * velocity[1]
    theta = 2 * np.pi * FREQUENCY_HZ * THICKNESS_KM / velocity[0]
    return zn**2 / (zn**2 * np.cos(theta) ** 2 + z1**2 * np.sin(theta) ** 2)


def package():
    omega = 2 * np.pi * FREQUENCY_HZ
    sh, sv, p = [], [], []
    for f in FREQUENCY_HZ:
        c = psv_surface_compliance(MODEL, f, 0.0)
        sv.append(c[0, 0])
        p.append(c[1, 1])
        sh.append(sh_surface_compliance(MODEL, f, 0.0))
    zs, zp = DENSITY_G_CM3[1] * VS_KM_S[1], DENSITY_G_CM3[1] * VP_KM_S[1]
    return (-np.imag(sh) * omega * zs, -np.imag(sv) * omega * zs, -np.imag(p) * omega * zp)


SH, SV, P = package()
S_EXACT, P_EXACT = closed_form(VS_KM_S), closed_form(VP_KM_S)
for name, got, want in (("SH", SH, S_EXACT), ("SV", SV, S_EXACT), ("P", P, P_EXACT)):
    error = np.max(np.abs(got / want - 1))
    print(f"{name}: max relative difference to closed form {error:.1e}")
    assert error < 1e-10


def draw():
    fig, ax = _style.plt.subplots(figsize=(6.4, 2.8))
    ax.semilogy(FREQUENCY_HZ, S_EXACT, color=_style.TRUTH, lw=2.4, alpha=0.35, label="closed form")
    ax.semilogy(FREQUENCY_HZ, P_EXACT, color=_style.TRUTH, lw=2.4, alpha=0.35)
    ax.semilogy(FREQUENCY_HZ, SH, color=_style.MEASURED, lw=1, label="SH, package")
    ax.semilogy(FREQUENCY_HZ, SV, color=_style.SECOND, lw=1, ls="--", label="SV, package")
    ax.semilogy(FREQUENCY_HZ, P, color=_style.EXTRA, lw=1, label="P, package")
    for n in range(4):
        ax.axvline((2 * n + 1) * VS_KM_S[0] / (4 * THICKNESS_KM), color=_style.MUTED, lw=0.6, ls=":")
    ax.axvline(VP_KM_S[0] / (4 * THICKNESS_KM), color=_style.EXTRA, lw=0.6, ls=":")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"$-\omega Z_N\,\mathrm{Im}\,C(k{=}0)$")
    ax.set_xlim(FREQUENCY_HZ[0], FREQUENCY_HZ[-1])
    ax.legend(loc="upper right", ncol=2)
    return fig


if __name__ == "__main__":
    save(draw, __file__)
