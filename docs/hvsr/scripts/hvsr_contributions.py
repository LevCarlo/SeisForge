"""Diffuse-field H/V of a layer over a half-space and its six terms.

Known answers checked here:

- the peak lies within 5 % of f0 = Vs1 / (4 H) = 1 Hz;
- scaling: H/V of a model with all thicknesses times 2 at f / 2 equals H/V
  of the original model at f (the equations contain H only as omega H / v);
- `horizontal_definition="mean"` is the sum divided by sqrt(2).
"""

from _case import F0_HZ, MODEL, _style, np, save

from seisforge.hvsr import DiffuseFieldSettings, ElasticLayerModel, NativeDiffuseFieldSolver

FREQUENCY_HZ = np.geomspace(0.2, 10.0, 81)
solver = NativeDiffuseFieldSolver()
result = solver(MODEL, FREQUENCY_HZ)
terms = result.contributions

peak_hz = FREQUENCY_HZ[np.nanargmax(result.hvsr)]
print(f"peak {np.nanmax(result.hvsr):.2f} at {peak_hz:.3f} Hz (f0 = {F0_HZ:g} Hz)")
assert abs(peak_hz / F0_HZ - 1) < 0.05

check_hz = np.array([0.3, 2.0, 5.0])  # scaling check at three frequencies across the band
double = ElasticLayerModel(2 * MODEL.thickness_km, MODEL.vp_km_s, MODEL.vs_km_s, MODEL.density_g_cm3)
scaling = np.max(np.abs(solver(double, check_hz / 2).hvsr / solver(MODEL, check_hz).hvsr - 1))
print(f"scaling H -> 2H, f -> f/2: max relative difference {scaling:.1e}")
assert scaling < 1e-6

mean = solver(MODEL, check_hz, DiffuseFieldSettings(horizontal_definition="mean")).hvsr
ratio = np.max(np.abs(mean * np.sqrt(2) / solver(MODEL, check_hz).hvsr - 1))
assert ratio < 1e-12


def draw():
    fig, (top, bottom) = _style.plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True,
                                             gridspec_kw={"height_ratios": [1, 1.3]})
    top.loglog(FREQUENCY_HZ, result.hvsr, color=_style.INK, lw=1.2)
    top.set_ylabel("H/V")
    for ax in (top, bottom):
        ax.axvline(F0_HZ, color=_style.MUTED, lw=0.6, ls=":")
    top.text(F0_HZ * 1.08, 0.88, r"$f_0=\beta_1/4H$", transform=top.get_xaxis_transform(), color=_style.MUTED, fontsize=8)
    _style.panel_label(top, "a")

    rows = [
        (terms.love_horizontal, "Love, horizontal", _style.MEASURED, "-"),
        (terms.rayleigh_horizontal, "Rayleigh, horizontal", _style.SECOND, "-"),
        (terms.psv_horizontal, "P-SV body, horizontal", _style.EXTRA, "-"),
        (terms.sh_horizontal, "SH body, horizontal", _style.EXTRA2, "-"),
        (terms.rayleigh_vertical, "Rayleigh, vertical", _style.SECOND, "--"),
        (terms.psv_vertical, "P-SV body, vertical", _style.EXTRA, "--"),
    ]
    for values, label, color, style in rows:
        bottom.loglog(FREQUENCY_HZ, np.where(values > 0, values, np.nan), color=color, ls=style,
                      lw=1, label=label)
    bottom.set_ylim(1e-3, None)
    bottom.set_xlabel("frequency (Hz)")
    bottom.set_ylabel(r"$E$ (common factor dropped)")
    bottom.legend(loc="lower right", ncol=2)
    _style.panel_label(bottom, "b")
    return fig


if __name__ == "__main__":
    save(draw, __file__)
