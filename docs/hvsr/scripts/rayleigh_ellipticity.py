"""Rayleigh-only H/V equals the Rayleigh ellipticity.

The residue of the surface compliance at a mode is rank one, so one Rayleigh
mode alone gives H/V = |u_x / u_z| (Diffuse-field H/V, section 3). Checked
against two independent results:

- the layered model: `disba.Ellipticity` (a different code, CPS-based);
- a homogeneous Poisson half-space: the ellipticity from the Rayleigh
  equation solved here with potentials (0.6812...).
"""

from _case import F0_HZ, MODEL, _style, np, save
from disba import Ellipticity
from scipy.optimize import brentq

from seisforge.hvsr import DiffuseFieldSettings, ElasticLayerModel, NativeDiffuseFieldSolver

RAYLEIGH_0 = DiffuseFieldSettings(max_rayleigh_modes=1, max_love_modes=0, include_body_waves=False)
FREQUENCY_HZ = np.geomspace(0.3, 5.0, 60)

solver = NativeDiffuseFieldSolver()
layered = solver(MODEL, FREQUENCY_HZ, RAYLEIGH_0).hvsr

periods = np.sort(1.0 / FREQUENCY_HZ)
ell = Ellipticity(np.array([MODEL.thickness_km[0], 0.0]), MODEL.vp_km_s, MODEL.vs_km_s,
                  MODEL.density_g_cm3)(periods, mode=0)
reference = np.interp(1.0 / FREQUENCY_HZ, ell.period, np.abs(ell.ellipticity))
relative = np.abs(layered / reference - 1)
regular = reference < 100.0  # excludes the u_z = 0 singularity next to f0
print(f"layered: max relative difference to disba {relative[regular].max():.1e} where H/V < 100; "
      f"at the singularity {layered[~regular]} vs {reference[~regular]}")
assert relative[regular].max() < 1e-4


def halfspace_ellipticity(vp, vs):
    """|u_x/u_z| at the surface from P/SV potentials (z down, exp(i(kx - wt)))."""
    def rayleigh(xi):  # (2 - xi^2)^2 - 4 sqrt(1 - xi^2 vs^2/vp^2) sqrt(1 - xi^2), xi = c / vs
        return (2 - xi**2) ** 2 - 4 * np.sqrt(1 - xi**2 * vs**2 / vp**2) * np.sqrt(1 - xi**2)
    xi = brentq(rayleigh, 0.5, 0.999999)
    na, nb = np.sqrt(1 - xi**2 * vs**2 / vp**2), np.sqrt(1 - xi**2)   # nu / k
    b = -2j * na / (1 + nb**2)                                         # B / A from sigma_xz = 0
    return abs((1j + nb * b) / (-na + 1j * b))


VP, VS = np.sqrt(3.0), 1.0
halfspace_exact = halfspace_ellipticity(VP, VS)
halfspace = solver(ElasticLayerModel([], [VP], [VS], [2.0]), [0.5, 1.0, 2.0], RAYLEIGH_0).hvsr
print(f"Poisson half-space: package {halfspace}, potentials {halfspace_exact:.6f}")
assert np.allclose(halfspace, halfspace_exact, rtol=1e-6)


def draw():
    fig, ax = _style.plt.subplots(figsize=(6.4, 2.8))
    ax.semilogy(FREQUENCY_HZ, reference, color=_style.TRUTH, lw=2.6, alpha=0.35, label="|ellipticity|, disba")
    ax.semilogy(FREQUENCY_HZ, layered, color=_style.MEASURED, lw=1, label="H/V, Rayleigh mode 0 only")
    ax.axhline(halfspace_exact, color=_style.SECOND, lw=1, ls="--",
               label=f"Poisson half-space, {halfspace_exact:.4f}")
    ax.axvline(F0_HZ, color=_style.MUTED, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("H/V")
    ax.legend(loc="upper right")
    return fig


if __name__ == "__main__":
    save(draw, __file__)
