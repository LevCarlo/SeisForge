"""Figure layering_convergence.svg (and -dark.svg): layering error in units of the data sigma.

Truth model of examples/06_inversion (sediment gradient 1.70-2.60 km/s over
0-1 km, B-spline crust to 15 km), its 14 periods (0.8-7 s) and sigmas
(0.03 km/s, 0.02). (a) Layer tops of uniform dz = 0.25 km and of the graded
layering of the example (dz = 0.0625 km, r = 1.1, dz_max = 1 km), with the
target thickness h(z) = min(dz_max, dz + (r - 1) z). (b) Largest change of
the predicted phase velocity and Rayleigh ellipticity against a uniform
1/64 km reference, divided by sigma, for uniform and graded layerings of
decreasing dz. The independent check is convergence: the uniform errors
fall as n^-2 (layer thickness squared, the order of midpoint sampling); the
graded phase-velocity error levels off where the 1 km cap at depth dominates.
"""

from dataclasses import replace

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import (
    DiscretizationConfig,
    build_inversion_setup,
    load_yaml,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.forward import DispersionRequest, RayleighHVRequest
from seisforge.inv.layering import layered_model_from_parameterized, target_layer_thickness_km

TRUTH = np.array([1.70, 2.60, 2.65, 2.85, 3.05, 3.20, 3.30, 3.45, 3.60, 3.70])
REFERENCE_DZ_KM = 1.0 / 64

setup = build_inversion_setup(load_yaml(_inv.REPO / "examples/06_inversion/inv.yaml"))
model = setup.parameterization.vector_to_model(TRUTH)
data = _inv.REPO / "examples/06_inversion/data"
period_s, _, sigma_c = np.loadtxt(data / "vph_disp.dat", unpack=True)
_, _, sigma_hv = np.loadtxt(data / "hv.dat", unpack=True)
dispersion_request = DispersionRequest(periods=period_s)
hv_request = RayleighHVRequest(periods=period_s)


def predict(config):
    layered = layered_model_from_parameterized(model, config)
    return (layered.n_layers - 1,
            predict_dispersion(layered, dispersion_request).velocity,
            predict_rayleigh_hv(layered, hv_request).hv)


graded_example = DiscretizationConfig(dz=0.0625, dz_growth_ratio=1.1, dz_max=1.0, zmax=15.0)
_, c_ref, hv_ref = predict(DiscretizationConfig(dz=REFERENCE_DZ_KM, zmax=15.0))
series = {}
for label, configs in (
    ("uniform", [DiscretizationConfig(dz=dz, zmax=15.0) for dz in (0.5, 0.25, 0.125, 0.0625)]),
    ("graded, r = 1.1", [replace(graded_example, dz=dz) for dz in (0.25, 0.125, 0.0625, 0.03125)]),
):
    rows = []
    for config in configs:
        n, c, hv = predict(config)
        rows.append((n, np.max(np.abs(c - c_ref) / sigma_c), np.max(np.abs(hv - hv_ref) / sigma_hv)))
        print(f"{label:16s} dz {config.dz:.5f}: {n:4d} layers, max |dc|/sigma {rows[-1][1]:.4f}, "
              f"max |dHV|/sigma {rows[-1][2]:.4f}")
    series[label] = np.array(rows)

tops_uniform = layered_model_from_parameterized(model, DiscretizationConfig(dz=0.25, zmax=15.0)).layer_tops
tops_graded = layered_model_from_parameterized(model, graded_example).layer_tops
z_km = np.linspace(0.0, 15.0, 301)
h_km = target_layer_thickness_km(z_km, dz_km=0.0625, growth_ratio=1.1, max_thickness_km=1.0)
print("graded layer thickness <= h at its bottom:",
      bool(np.all(np.diff(tops_graded) <= target_layer_thickness_km(
          tops_graded[1:], dz_km=0.0625, growth_ratio=1.1, max_thickness_km=1.0) + 1e-12)))


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), gridspec_kw={"width_ratios": [0.9, 1.1]})
    for x, tops, color in ((0.0, tops_uniform, _style.ALTERNATIVE), (1.0, tops_graded, _style.MEASURED)):
        axes[0].hlines(tops, x, x + 0.8, color=color, lw=0.7)
    axes[0].plot(2.0 + h_km, z_km, color=_style.INK)
    axes[0].set_xticks([0.4, 1.4, 2.5])
    axes[0].set_xticklabels([f"uniform\n{tops_uniform.size - 1} layers", f"graded\n{tops_graded.size - 1} layers",
                             "h(z), km\n(offset 2)"])
    axes[0].set_ylabel("depth (km)")
    axes[0].set_ylim(15.0, 0.0)
    axes[0].set_title("layer tops")
    _style.panel_label(axes[0], "(a)")

    styles = {"uniform": (_style.ALTERNATIVE, "s"), "graded, r = 1.1": (_style.MEASURED, "o")}
    for label, rows in series.items():
        color, marker = styles[label]
        axes[1].loglog(rows[:, 0], rows[:, 1], color=color, marker=marker, ls="-", label=f"{label}: phase velocity")
        axes[1].loglog(rows[:, 0], rows[:, 2], color=color, marker=marker, ls="--", mfc="none",
                       label=f"{label}: ellipticity")
    n = np.array([20.0, 300.0])
    axes[1].loglog(n, 2.0 * (n / 20.0) ** -2, color=_style.INK, lw=0.8, ls=":", label=r"$\propto n^{-2}$")
    axes[1].axhline(0.2, color=_style.HIGHLIGHT, lw=0.8)
    axes[1].set_xlabel("number of finite layers $n$")
    axes[1].set_ylabel(r"max |prediction $-$ reference| / $\sigma$")
    axes[1].set_title("layering error, 0.8-7 s")
    axes[1].legend(fontsize=6.5, loc="lower left")
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
