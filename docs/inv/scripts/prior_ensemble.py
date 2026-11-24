"""Figure prior_ensemble.svg (and -dark.svg): the prior that the sampler actually sees.

Model space of examples/06_inversion/inv.yaml (10 parameters, uniform
bounds, six hard constraints). Draws uniform in the parameter box; the ones
that `GeophysicalPrior.evaluate` accepts are exact samples of the
constrained prior (rejection sampling). (a) 5-95 % bands of Vs(z) for the
box alone and for box + constraints. (b) Vs at 6 km under the box alone:
histogram of the draws against the exact density, the convolution of the
uniform densities of B_i(6 km) c_i, computed independently on a grid.
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import build_inversion_setup, load_yaml

N_DRAWS = 10000
SEED = 1
PROBE_KM = 6.0

setup = build_inversion_setup(load_yaml(_inv.REPO / "examples/06_inversion/inv.yaml"))
par = setup.parameterization
lower, upper = par.bounds()
rng = np.random.default_rng(SEED)
thetas = rng.uniform(lower, upper, size=(N_DRAWS, lower.size))
z_km = np.linspace(0.0, 15.0, 301)
box = np.array([par.vector_to_model(t).evaluate(z_km) for t in thetas])
accepted = np.array([setup.prior.evaluate(t).success for t in thetas])
print(f"accepted by the hard constraints: {accepted.sum()} of {N_DRAWS} ({accepted.mean():.2%})")

# Exact density of Vs(PROBE_KM) = sum_i B_i c_i, c_i ~ U(l_i, u_i), box only.
# The probe lies in the crust segment; its value is linear in the coefficients.
eye = np.eye(lower.size)
base = par.vector_to_model(lower).evaluate(np.array([PROBE_KM]))[0]
weight = np.array([par.vector_to_model(lower + eye[i] * (upper[i] - lower[i])).evaluate(np.array([PROBE_KM]))[0]
                   - base for i in range(lower.size)])  # B_i (u_i - l_i)
grid = np.linspace(0.0, weight.sum(), 4001)
step = grid[1] - grid[0]
density = np.zeros_like(grid)
density[0] = 1.0 / step
for w in weight[weight > 1e-12]:
    kernel = np.ones(max(1, int(round(w / step)))) / max(1, int(round(w / step)))
    density = np.convolve(density, kernel)[: grid.size]
density /= density.sum() * step
probe = box[:, np.argmin(np.abs(z_km - PROBE_KM))]
print(f"Vs({PROBE_KM:g} km): sample mean {probe.mean():.4f}, exact {base + np.sum(grid * density) * step:.4f}; "
      f"sample sd {probe.std():.4f}, exact {np.sqrt(np.sum(weight**2) / 12):.4f}")


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    for values, color, label in ((box, _style.ALTERNATIVE, "parameter box only"),
                                 (box[accepted], _style.MEASURED, "box + hard constraints")):
        p05, p50, p95 = np.percentile(values, [5, 50, 95], axis=0)
        axes[0].fill_betweenx(z_km, p05, p95, color=color, alpha=0.35, lw=0, label=f"{label}, 5-95 %")
        axes[0].plot(p50, z_km, color=color, lw=1.0)
    axes[0].axhline(PROBE_KM, color=_style.INK, ls=":", lw=0.8)
    axes[0].set_xlabel("Vs (km/s)")
    axes[0].set_ylabel("depth (km)")
    axes[0].set_title(f"prior Vs(z), {accepted.sum()} of {N_DRAWS} box draws pass")
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", fontsize=7)
    _style.panel_label(axes[0], "(a)")

    axes[1].hist(probe, bins=50, density=True, color=_style.ALTERNATIVE, alpha=0.6, label="box draws")
    axes[1].plot(base + grid, density, color=_style.INK, label=r"exact: convolution of uniforms")
    axes[1].set_xlabel(f"Vs at {PROBE_KM:g} km (km/s)")
    axes[1].set_ylabel("density")
    axes[1].set_title("uniform coefficients, non-uniform Vs")
    axes[1].legend(fontsize=7)
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
