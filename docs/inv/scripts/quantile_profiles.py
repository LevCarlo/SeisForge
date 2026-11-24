"""Figure quantile_profiles.svg (and -dark.svg): depth-wise statistics of a Vs ensemble.

Ensemble with a known answer: Vs = 2.0 km/s above an interface and 3.0 km/s
below it, the interface depth d uniform on [4, 6] km (2000 draws through
`seisforge.inv.extract_vs_ensemble` and an interface-layout
parameterization). At depth z the probability of the lower velocity is
F(z) = (z - 4) / 2, so the exact mean is 2 + F(z) (a linear ramp that no
member has) and the exact q-quantile is a step at 6 - 2q.
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import extract_vs_ensemble, parameterization_from_config

config = {
    "inversion": {"parameters": [
        {"name": "d", "initial": 5.0, "prior": {"bounds": [4.0, 6.0]}},
    ]},
    "model": {
        "zmax_km": 10.0,
        "interfaces": [{"depth_km": {"parameter": "d"}}],
        "segments": [{"profile": {"type": "constant", "value": 2.0}},
                     {"profile": {"type": "constant", "value": 3.0}}],
    },
}
parameterization = parameterization_from_config(config)
rng = np.random.default_rng(4)
depths = rng.uniform(4.0, 6.0, size=(2000, 1))
z_km = np.linspace(2.0, 8.0, 601)
ensemble = extract_vs_ensemble(depths, parameterization=parameterization, z=z_km)
cdf = np.clip((z_km - 4.0) / 2.0, 0.0, 1.0)
exact_mean = 2.0 + cdf
print("max |mean - exact|:", np.max(np.abs(ensemble.mean - exact_mean)))
for q, values in zip((0.05, 0.5, 0.95), ensemble.quantile([0.05, 0.5, 0.95])):
    step = z_km[np.argmax(values > 2.5)]
    print(f"q = {q}: step of the quantile profile at {step:.3f} km (exact {6 - 2 * q:.3f})")


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for profile in ensemble.vs[:25]:
        axes[0].plot(profile, z_km, color=_style.ALTERNATIVE, lw=0.6)
    axes[0].set_title("25 of 2000 members")
    axes[0].set_xlabel("Vs (km/s)")
    axes[0].set_ylabel("depth (km)")
    _style.panel_label(axes[0], "(a)")
    axes[1].fill_betweenx(z_km, ensemble.p05, ensemble.p95, color=_style.MEASURED, alpha=0.25, lw=0,
                          label="5-95 %")
    axes[1].plot(ensemble.median, z_km, color=_style.MEASURED, label="median")
    axes[1].plot(ensemble.mean, z_km, color=_style.SECOND, label="mean")
    axes[1].plot(exact_mean, z_km, color=_style.INK, ls=":", label="exact mean")
    axes[1].set_title("depth-wise statistics")
    axes[1].set_xlabel("Vs (km/s)")
    axes[1].legend(loc="lower left", fontsize=7)
    axes[1].invert_yaxis()
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
