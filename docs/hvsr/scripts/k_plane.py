"""The surface compliance along the horizontal wavenumber at 1 Hz.

Shows what the two parts of the DFA integral see: branch points at
k = omega / v_j, the body-wave interval 0 < k < omega / Vs_N, and the
surface-wave poles beyond it. The poles are found with the package root
finder from disba seeds; the script checks that each lies beyond
omega / Vs_N and that the compliance is real there (no radiation).
"""

from _case import MODEL, _style, np, save

from seisforge.hvsr.dispersion import phase_velocity_seeds_km_s, refine_surface_wavenumber_per_km
from seisforge.hvsr.propagator import psv_surface_compliance, sh_surface_compliance

FREQUENCY_HZ = 1.0
OMEGA = 2 * np.pi * FREQUENCY_HZ
SLOWNESS_S_PER_KM = np.linspace(0.002, 2.2, 3000)
SHIFT = 1e-6  # relative, keeps the curves finite at the poles

values = {"xx": [], "zz": [], "sh": []}
for p in SLOWNESS_S_PER_KM:
    k = OMEGA * p * (1 - 1j * SHIFT)
    c = psv_surface_compliance(MODEL, FREQUENCY_HZ, k)
    values["xx"].append(abs(c[0, 0]))
    values["zz"].append(abs(c[1, 1]))
    values["sh"].append(abs(sh_surface_compliance(MODEL, FREQUENCY_HZ, k)))

poles = {}
for wave in ("rayleigh", "love"):
    seed = phase_velocity_seeds_km_s(MODEL, [FREQUENCY_HZ], 0, wave)[0]
    k = refine_surface_wavenumber_per_km(MODEL, FREQUENCY_HZ, seed, wave)
    poles[wave] = k / OMEGA
    assert k > OMEGA / MODEL.vs_km_s[-1]
print("pole slowness (s/km):", {w: round(p, 4) for w, p in poles.items()})

# Beyond omega / Vs_N nothing radiates: C_xx, C_zz, C_yy are real (C_xz imaginary).
worst = 0.0
for p in SLOWNESS_S_PER_KM[SLOWNESS_S_PER_KM > 1.01 / MODEL.vs_km_s[-1]]:
    c = psv_surface_compliance(MODEL, FREQUENCY_HZ, OMEGA * p)
    sh = sh_surface_compliance(MODEL, FREQUENCY_HZ, OMEGA * p)
    for z in (c[0, 0], c[1, 1], sh):
        worst = max(worst, abs(z.imag) / abs(z))
print(f"max |Im C| / |C| beyond omega/Vs_N: {worst:.1e}")
assert worst < 1e-10


def draw():
    fig, ax = _style.plt.subplots(figsize=(6.4, 3.0))
    ax.axvspan(0, 1 / MODEL.vs_km_s[-1], color=_style.ALTERNATIVE, alpha=0.18, lw=0)
    ax.semilogy(SLOWNESS_S_PER_KM, values["xx"], color=_style.SECOND, lw=1, label=r"$|C_{xx}|$ (P-SV)")
    ax.semilogy(SLOWNESS_S_PER_KM, values["zz"], color=_style.EXTRA, lw=1, label=r"$|C_{zz}|$ (P-SV)")
    ax.semilogy(SLOWNESS_S_PER_KM, values["sh"], color=_style.MEASURED, lw=1, label=r"$|C_{yy}|$ (SH)")
    marks = [(1 / MODEL.vp_km_s[1], r"$1/\alpha_N$"), (1 / MODEL.vs_km_s[1], r"$1/\beta_N$"),
             (1 / MODEL.vp_km_s[0], r"$1/\alpha_1$")]
    for x, label in marks:
        ax.axvline(x, color=_style.MUTED, lw=0.7, ls=":")
        ax.text(x, 1.02, label, transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=_style.MUTED)
    for wave, x in poles.items():
        ax.axvline(x, color=_style.HIGHLIGHT, lw=0.7, ls="--")
        ax.text(x + 0.015, 0.45, f"{wave} 0", transform=ax.get_xaxis_transform(), rotation=90,
                fontsize=7, color=_style.HIGHLIGHT)
    ax.text(0.5, 0.92, "body-wave integral", transform=ax.get_xaxis_transform(), ha="center",
            fontsize=8, color=_style.MUTED)
    ax.set_xlim(0, SLOWNESS_S_PER_KM[-1])
    ax.set_xlabel(r"horizontal slowness $k/\omega$ (s/km)")
    ax.set_ylabel(r"$|C|$ (km per unit stress, $\omega$ = 2$\pi$ rad/s)")
    leaky = SLOWNESS_S_PER_KM[(SLOWNESS_S_PER_KM > 0.6) & (SLOWNESS_S_PER_KM < 0.95)]
    peak = leaky[np.argmax(np.array(values["xx"])[(SLOWNESS_S_PER_KM > 0.6) & (SLOWNESS_S_PER_KM < 0.95)])]
    ax.annotate("near-real pole", (peak, 80), (0.62, 900), fontsize=7, color=_style.MUTED,
                arrowprops={"arrowstyle": "-", "color": _style.MUTED, "lw": 0.6})
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


if __name__ == "__main__":
    save(draw, __file__)
