"""Why the H/V peak height did not converge (T21), and that it does now.

(a) H/V near f0: a fixed Gauss-Legendre rule of 50, 100 and 200 nodes per
    segment (what SeisForge used before 0.2.0, rebuilt here) against the
    adaptive Gauss-Kronrod quadrature at two tolerances.
(b) The P-SV horizontal body-wave integrand at 0.955 Hz: a peak far narrower
    than the spacing of the old 50 nodes; the adaptive rule bisects down to it.

The script also measures the half width of that peak against frequency and
checks that it vanishes like (f - f*)^2 (a leaky pole that touches the real
axis), and asserts the convergence of the adaptive result.
"""

from _case import F0_HZ, MODEL, _style, np, save
from scipy.optimize import brentq, minimize_scalar

from seisforge.hvsr import NativeDiffuseFieldSolver, assemble_diffuse_field_prediction
from seisforge.hvsr.body_waves import _checked_body_contributions, body_wave_segments
from seisforge.hvsr.modes import SurfaceWaveSettings, surface_wave_green_contributions
from seisforge.hvsr.propagator import psv_surface_compliance, sh_surface_compliance

FREQUENCY_HZ = np.linspace(0.9, 1.1, 41)
SHIFT = 1.0 - 1e-8j
SEGMENTS = body_wave_segments(MODEL)


def terms(frequency_hz, unit):
    """k C dk/du for (C_xx, C_zz, C_yy) at one u = k Vs_N / omega."""
    k_limit = 2 * np.pi * frequency_hz / MODEL.vs_km_s[-1]
    k = k_limit * unit * SHIFT
    psv = psv_surface_compliance(MODEL, frequency_hz, k)
    return np.array([psv[0, 0], psv[1, 1], sh_surface_compliance(MODEL, frequency_hz, k)]) * k * k_limit * SHIFT


def fixed_gauss_hvsr(order):
    """The pre-0.2.0 rule: `order` Gauss-Legendre nodes on each segment."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    body = []
    for f in FREQUENCY_HZ:
        total = np.zeros(3, complex)
        for left, right in zip(SEGMENTS[:-1], SEGMENTS[1:], strict=True):
            half = 0.5 * (right - left)
            total += half * sum(w * terms(f, left + half * (x + 1)) for x, w in zip(nodes, weights, strict=True))
        body.append(-total.imag * np.array([0.5, 1.0, 0.5]))
    body = np.array(body)
    contributions = _checked_body_contributions(body[:, 0], body[:, 1], body[:, 2], ())
    return assemble_diffuse_field_prediction(
        FREQUENCY_HZ, surface=SURFACE, body=contributions, horizontal_definition="sum", backend="script"
    ).hvsr


SURFACE = surface_wave_green_contributions(MODEL, FREQUENCY_HZ, settings=SurfaceWaveSettings())
fixed = {n: fixed_gauss_hvsr(n) for n in (50, 100, 200)}
adaptive = NativeDiffuseFieldSolver()(MODEL, FREQUENCY_HZ)
tight = NativeDiffuseFieldSolver(body_quadrature_rtol=1e-9)(MODEL, FREQUENCY_HZ)

stack = np.array(list(fixed.values()))
spread = (stack.max(0) - stack.min(0)) / stack.min(0)
change = np.max(np.abs(tight.hvsr / adaptive.hvsr - 1))
peak = int(np.argmax(adaptive.hvsr))
print(f"fixed rules: spread up to {spread.max():.2f} at {FREQUENCY_HZ[spread.argmax()]:.3f} Hz; "
      f"adaptive rtol 1e-6 vs 1e-9: max change {change:.1e}; "
      f"peak {adaptive.hvsr[peak]:.2f} at {FREQUENCY_HZ[peak]:.3f} Hz")
assert spread.max() > 0.5 and change < 1e-6
assert all(d.status.value == "ok" for d in adaptive.diagnostics)


def horizontal_integrand(frequency_hz, unit):
    return -terms(frequency_hz, unit)[0].imag


def half_width(frequency_hz):
    """Location and full width at half maximum of the narrow peak in u."""
    grid = np.linspace(0.6, 0.9, 6001)
    values = np.abs([horizontal_integrand(frequency_hz, u) for u in grid])
    i = int(np.argmax(values))
    best = minimize_scalar(lambda u: -abs(horizontal_integrand(frequency_hz, u)),
                           bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-14})
    centre, top = best.x, abs(horizontal_integrand(frequency_hz, best.x))
    level = lambda u: abs(horizontal_integrand(frequency_hz, u)) - top / 2  # noqa: E731
    return centre, brentq(level, centre, centre + 0.01) - brentq(level, centre - 0.01, centre)


offsets = np.array([0.946, 0.948, 0.950, 0.956, 0.958, 0.960])
widths = np.array([half_width(f)[1] for f in offsets])
# Fit width = a (f - f*)^2: the square root is linear in f on each side.
left, right = offsets < 0.952, offsets > 0.952
f_left = np.polyfit(offsets[left], np.sqrt(widths[left]), 1)
f_right = np.polyfit(offsets[right], -np.sqrt(widths[right]), 1)
zero_left, zero_right = -f_left[1] / f_left[0], -f_right[1] / f_right[0]
print(f"half width vs f: sqrt(width) is linear in f and vanishes at f* = {zero_left:.4f} Hz "
      f"(from below) and {zero_right:.4f} Hz (from above)")
assert abs(zero_left - zero_right) < 1e-3

SPIKE_HZ = 0.955
centre, width = half_width(SPIKE_HZ)
unit = np.unique(np.concatenate((np.linspace(0.725, 0.765, 401), centre + np.linspace(-2e-4, 2e-4, 801))))
values = np.array([horizontal_integrand(SPIKE_HZ, u) for u in unit])
nodes, _ = np.polynomial.legendre.leggauss(50)
split = SEGMENTS[1]
old_nodes = 0.5 * ((1 - split) * nodes + 1 + split)
ordered = np.sort(old_nodes)
spacing = np.diff(ordered)[np.searchsorted(ordered, centre) - 1]  # gap of the two nodes around the peak
print(f"peak at u = {centre:.5f} (c = {1 / centre:.3f} km/s), half width {width:.1e}, "
      f"old node spacing there ~{spacing:.1e}")
assert width < 0.01 * spacing


def draw():
    fig, (ax_a, ax_b) = _style.plt.subplots(1, 2, figsize=(7.2, 2.9))
    for (n, hv), color in zip(fixed.items(), (_style.MUTED, _style.SECOND, _style.EXTRA), strict=True):
        ax_a.plot(FREQUENCY_HZ, hv, color=color, lw=0.8, marker=".", ms=2.5, label=f"fixed Gauss, {n} nodes")
    ax_a.plot(FREQUENCY_HZ, adaptive.hvsr, color=_style.MEASURED, lw=1.6, label="adaptive (default)")
    ax_a.axvline(F0_HZ, color=_style.MUTED, lw=0.6, ls=":")
    ax_a.set_xlabel("frequency (Hz)")
    ax_a.set_ylabel("H/V")
    ax_a.legend(loc="upper right", fontsize=6.5)
    _style.panel_label(ax_a, "a")

    ax_b.semilogy(unit, np.abs(values), color=_style.INK, lw=1)
    inside = old_nodes[(old_nodes > unit[0]) & (old_nodes < unit[-1])]
    ax_b.plot(inside, np.abs([horizontal_integrand(SPIKE_HZ, u) for u in inside]), "o",
              color=_style.HIGHLIGHT, ms=4, label="old nodes (50)")
    ax_b.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.06, 1.0))
    ax_b.set_xlim(unit[0], unit[-1])
    ax_b.set_xlabel(r"$k\,/\,(\omega/\beta_N)$ at 0.955 Hz")
    ax_b.set_ylabel(r"$|\mathrm{Im}\,[k\,C_{xx}]|$")
    ax_b.set_xticks([0.73, 0.745, 0.76])
    _style.panel_label(ax_b, "b")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    save(draw, __file__)
