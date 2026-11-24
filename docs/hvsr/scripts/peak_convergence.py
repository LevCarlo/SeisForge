"""Why the H/V peak height does not converge (T21).

(a) H/V near f0 for several body-wave efforts and contour shifts.
(b) The P-SV horizontal body-wave integrand at 0.955 Hz: a peak far narrower
    than the spacing of the Gauss-Legendre nodes the native solver uses with
    default settings (50 per segment). The script asserts both facts.
"""

from _case import F0_HZ, MODEL, _style, np, save

from seisforge.hvsr import DiffuseFieldSettings, NativeDiffuseFieldSolver
from seisforge.hvsr.propagator import psv_surface_compliance

FREQUENCY_HZ = np.linspace(0.9, 1.1, 41)
EFFORTS = (100, 200, 400)  # body_wave_wavenumbers -> 50, 100, 200 nodes per segment
curves = {n: NativeDiffuseFieldSolver()(MODEL, FREQUENCY_HZ, DiffuseFieldSettings(body_wave_wavenumbers=n)).hvsr
          for n in EFFORTS}
smooth = NativeDiffuseFieldSolver(contour_shift_fraction=1e-2)(MODEL, FREQUENCY_HZ).hvsr

stack = np.array(list(curves.values()))
spread = (stack.max(0) - stack.min(0)) / stack.min(0)
above = FREQUENCY_HZ >= 1.015
print(f"spread over efforts: max {spread.max():.2f} (at {FREQUENCY_HZ[spread.argmax()]:.3f} Hz), "
      f"max {spread[above].max():.1e} at f >= 1.015 Hz")
assert spread.max() > 0.5 and spread[above].max() < 1e-3

SPIKE_HZ = 0.955
omega = 2 * np.pi * SPIKE_HZ
k_limit = omega / MODEL.vs_km_s[-1]
CENTRE = 0.744775  # k / k_limit of the peak (located on a finer grid)
unit = np.unique(np.concatenate((np.linspace(0.725, 0.765, 401),
                                 CENTRE + np.linspace(-2e-4, 2e-4, 801))))


def integrand(u):
    k = k_limit * u * (1 - 1e-8j)
    return np.array([-np.imag(kk * psv_surface_compliance(MODEL, SPIKE_HZ, kk)[0, 0]) for kk in k])


values = integrand(unit)
half = unit[values > 0.5 * values.max()]
width = half.max() - half.min() + 5e-7  # + fine-grid step
# Native default nodes: Gauss-Legendre, 50 per segment, segments [0, Vs_N/Vp_N] and [Vs_N/Vp_N, 1].
nodes, _ = np.polynomial.legendre.leggauss(50)
split = MODEL.vs_km_s[-1] / MODEL.vp_km_s[-1]
segment_nodes = 0.5 * ((1 - split) * nodes + 1 + split)
spacing = np.min(np.abs(np.diff(segment_nodes[np.argsort(np.abs(segment_nodes - unit[values.argmax()]))[:2]])))
print(f"peak at k/k_limit = {unit[values.argmax()]:.5f} (c = {1 / unit[values.argmax()]:.3f} km/s), "
      f"half width <= {width:.1e}, node spacing there {spacing:.1e}")
assert width < 0.01 * spacing


def draw():
    fig, (left, right) = _style.plt.subplots(1, 2, figsize=(7.2, 2.9))
    colors = (_style.MEASURED, _style.SECOND, _style.EXTRA)
    for (n, hv), color in zip(curves.items(), colors, strict=True):
        left.plot(FREQUENCY_HZ, hv, color=color, lw=1, marker=".", ms=3, label=f"body_wave_wavenumbers={n}")
    left.plot(FREQUENCY_HZ, smooth, color=_style.INK, lw=1, ls="--", label="contour shift $10^{-2}$")
    left.axvline(F0_HZ, color=_style.MUTED, lw=0.6, ls=":")
    left.set_xlabel("frequency (Hz)")
    left.set_ylabel("H/V")
    left.legend(loc="upper right", fontsize=6.5)
    _style.panel_label(left, "a")

    right.semilogy(unit, np.abs(values), color=_style.INK, lw=1)
    inside = segment_nodes[(segment_nodes > unit[0]) & (segment_nodes < unit[-1])]
    right.plot(inside, np.abs(integrand(inside)), "o", color=_style.HIGHLIGHT, ms=4, label="Gauss nodes (default)")
    right.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.06, 1.0))
    right.set_xlim(unit[0], unit[-1])
    right.set_xlabel(r"$k\,/\,(\omega/\beta_N)$ at 0.955 Hz")
    right.set_ylabel(r"$|\mathrm{Im}\,[k\,C_{xx}]|$")
    right.set_xticks([0.73, 0.745, 0.76])
    _style.panel_label(right, "b")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    save(draw, __file__)
