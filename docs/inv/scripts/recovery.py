"""Figure recovery.svg (and -dark.svg): a whole inversion on synthetic data with a known truth.

The case of tests/test_inv_recovery.py: Vs at 0, 1 and 3 km (linear in
between, half-space below 3 km), uniform priors, monotonic Vs; truth
(1.4, 2.4, 3.2) km/s. Phase velocity and Rayleigh ellipticity at 0.5-5 s are
forwarded with the package's disba wrappers in the inversion's own layering
(uniform 0.25 km), Gaussian noise added (0.03 km/s, 0.02; seed 0). Three
adaptive chains of 1200 steps (burn-in 500). (a) Prior (exact, by rejection
from the parameter box) and posterior 5-95 % bands of Vs(z) and the truth.
(b, c) Predictions of the stored posterior draws against the data.
Because the truth uses the inversion's own parameterization and layering,
this tests the chain parameterization -> forward -> likelihood -> sampler,
not the representation of real structure.
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import (
    build_inversion_setup,
    convergence_diagnostics,
    extract_prediction_ensemble_from_result,
    extract_vs_ensemble,
    extract_vs_ensemble_from_result,
    predict_dispersion,
    predict_rayleigh_hv,
)
from seisforge.inv.forward import DispersionRequest, RayleighHVRequest

TRUTH = np.array([1.4, 2.4, 3.2])
PERIODS_S = np.array([0.5, 1.0, 2.0, 3.5, 5.0])
SIGMA_C, SIGMA_HV = 0.03, 0.02
z_km = np.arange(0.0, 3.0 + 1e-9, 0.05)


def parameter(name, initial, bounds):
    return {"name": name, "initial": initial, "prior": {"bounds": bounds}, "proposal": {"sigma": 0.03}}


config = {
    "inversion": {"parameters": [parameter("vs0", 1.5, [1.0, 2.0]), parameter("vs1", 2.5, [2.0, 3.0]),
                                 parameter("vs2", 3.2, [2.8, 3.6])]},
    "model": {"segments": [
        {"top_km": 0.0, "bottom_km": 1.0,
         "profile": {"type": "gradient", "top": {"parameter": "vs0"}, "bottom": {"parameter": "vs1"}}},
        {"top_km": 1.0, "bottom_km": 3.0,
         "profile": {"type": "gradient", "top": {"parameter": "vs1"}, "bottom": {"parameter": "vs2"}}},
    ]},
    "discretization": {"dz": 0.25},
    "constraints": {"constraints": [{"type": "positive_velocity"}, {"type": "monotonic_vs", "dz": 0.5}]},
    "sampler": {"n_steps": 1200, "burn_in": 500, "thin": 5, "n_chains": 3, "seed": 11,
                "initial_strategy": "prior_uniform",
                "adaptation": {"initial_diagonal_steps": 200, "update_interval_steps": 50,
                               "discard_early_history_fraction": 0.5}},
}
truth_layers = build_inversion_setup(config).prior.evaluate(TRUTH).layered_model
c_true = predict_dispersion(truth_layers, DispersionRequest(periods=PERIODS_S)).velocity
hv_true = predict_rayleigh_hv(truth_layers, RayleighHVRequest(periods=PERIODS_S)).hv
rng = np.random.default_rng(0)
config["observations"] = {
    "dispersion": {"periods": PERIODS_S.tolist(), "sigma": [SIGMA_C] * 5,
                   "velocity": (c_true + rng.normal(0.0, SIGMA_C, 5)).tolist()},
    "hv": {"periods": PERIODS_S.tolist(), "sigma": [SIGMA_HV] * 5,
           "hv": (hv_true + rng.normal(0.0, SIGMA_HV, 5)).tolist()},
}
setup = build_inversion_setup(config)
result = setup.run_posterior()
posterior = extract_vs_ensemble_from_result(result, parameterization=setup.parameterization, z=z_km)
predictions = extract_prediction_ensemble_from_result(result, target=setup.target)
truth = extract_vs_ensemble(TRUTH[None], parameterization=setup.parameterization, z=z_km).vs[0]

lower, upper = setup.parameterization.bounds()
box = np.random.default_rng(1).uniform(lower, upper, size=(4000, 3))
prior_draws = box[[setup.prior.evaluate(t).success for t in box]]
prior = extract_vs_ensemble(prior_draws, parameterization=setup.parameterization, z=z_km)

diagnostics = convergence_diagnostics(result.samples, result.log_prob)
covered = np.mean((truth >= posterior.p05) & (truth <= posterior.p95))
print(f"prior draws kept: {len(prior_draws)} of 4000; truth inside the posterior 5-95 % band at "
      f"{covered:.0%} of depths")
print("split R-hat", np.round(diagnostics.rhat, 3).tolist(), "ESS", np.round(diagnostics.ess).tolist())
print("band width / prior band width, mean over depth:",
      round(float(np.mean((posterior.p95 - posterior.p05) / (prior.p95 - prior.p05))), 3))


def draw():
    fig = plt.figure(figsize=(8.6, 3.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
    ax = fig.add_subplot(grid[:, 0])
    ax.fill_betweenx(z_km, prior.p05, prior.p95, color=_style.ALTERNATIVE, alpha=0.35, lw=0, label="prior 5-95 %")
    ax.fill_betweenx(z_km, posterior.p05, posterior.p95, color=_style.MEASURED, alpha=0.5, lw=0,
                     label="posterior 5-95 %")
    ax.plot(truth, z_km, color=_style.INK, label="truth")
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("depth (km)")
    ax.invert_yaxis()
    ax.legend(loc="lower left", fontsize=7)
    _style.panel_label(ax, "(a)")
    for row, (item, label, letter) in enumerate(((predictions.dispersion, "phase velocity (km/s)", "(b)"),
                                                 (predictions.hv, "ellipticity H/V", "(c)"))):
        axp = fig.add_subplot(grid[row, 1])
        q05, q95 = item.quantile([0.05, 0.95])
        axp.fill_between(item.periods, q05, q95, color=_style.MEASURED, alpha=0.4, lw=0, label="predicted, 5-95 %")
        axp.errorbar(item.periods, item.observed, item.sigma, fmt="o", ms=3, color=_style.INK, label="data")
        axp.set_xscale("log")
        axp.set_ylabel(label)
        _style.panel_label(axp, letter)
        if row == 0:
            axp.legend(fontsize=7, loc="lower right")
        else:
            axp.set_xlabel("period (s)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
