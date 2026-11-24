"""Figure adaptive_metropolis.svg (and -dark.svg): adaptive proposals on a tilted ridge.

Target: 2-D Gaussian, unit variances, correlation 0.98 (a stand-in for two
neighbouring B-spline coefficients). Four chains of `seisforge.inv.run_metropolis`,
6000 steps, burn-in 3000, start at (3, -3) far off the ridge, diagonal
proposal sigma 0.25; once fixed, once with adaptation (first update after 500
steps, then every 100, earliest half of the history left out). (a) Stored
draws and the target's 95 % ellipse; the frozen proposal of the adaptive
chain against (2.38^2 / d) Sigma. (b) First coordinate of chain 0 after
burn-in. Package ESS and split R-hat are printed and shown.
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import AdaptationConfig, MetropolisConfig, effective_sample_size, run_metropolis, split_rhat

RHO = 0.98
COVARIANCE = np.array([[1.0, RHO], [RHO, 1.0]])
PRECISION = np.linalg.inv(COVARIANCE)
START = np.array([3.0, -3.0])
BOUNDS = ([-10.0, -10.0], [10.0, 10.0])


def log_target(x):
    return -0.5 * x @ PRECISION @ x


common = dict(n_steps=6000, burn_in=3000, proposal_sigma=0.25, bounds=BOUNDS)
adaptation = AdaptationConfig(initial_diagonal_steps=500, update_interval_steps=100,
                              discard_early_history_fraction=0.5)
runs = {}
for label, config in (("fixed diagonal", MetropolisConfig(**common)),
                      ("adaptive", MetropolisConfig(**common, adaptation=adaptation))):
    result = run_metropolis(log_target, np.tile(START, (4, 1)), config, seeds=7)
    runs[label] = (result, effective_sample_size(result.samples), split_rhat(result.samples))
    print(f"{label:15s}: ESS {np.round(runs[label][1])}, split R-hat {np.round(runs[label][2], 3)}, "
          f"post-burn-in acceptance {np.round(result.post_burn_in_acceptance_rates, 2)}")
frozen = runs["adaptive"][0].chains[0].proposal_covariance[0]
optimal = 2.38**2 / 2 * COVARIANCE
print("frozen proposal covariance (chain 0):", np.round(frozen, 3).tolist())
print("(2.38^2/d) Sigma:", np.round(optimal, 3).tolist())


def ellipse(cov, scale=np.sqrt(5.991)):
    angle = np.linspace(0, 2 * np.pi, 200)
    return scale * np.linalg.cholesky(cov) @ np.vstack([np.cos(angle), np.sin(angle)])


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), gridspec_kw={"width_ratios": [1, 1.5]})
    colors = {"fixed diagonal": _style.ALTERNATIVE, "adaptive": _style.MEASURED}
    for label, (result, ess, rhat) in runs.items():
        draws = result.samples.reshape(-1, 2)
        axes[0].plot(draws[::4, 0], draws[::4, 1], ".", ms=1.5, color=colors[label], alpha=0.5,
                     label=f"{label}: ESS {ess.min():.0f}, R-hat {rhat.max():.2f}")
    axes[0].plot(*ellipse(COVARIANCE), color=_style.INK, lw=1.0, label="target, 95 %")
    axes[0].plot(*ellipse(frozen, 1.0), color=_style.HIGHLIGHT, lw=1.0, label="frozen proposal, 1 sd")
    axes[0].plot(*ellipse(optimal, 1.0), color=_style.HIGHLIGHT, lw=0.8, ls=":", label=r"$(2.38^2/d)\,\Sigma$, 1 sd")
    axes[0].plot(*START, "x", color=_style.INK)
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_xlabel(r"$\theta_1$")
    axes[0].set_ylabel(r"$\theta_2$")
    axes[0].legend(fontsize=6, loc="upper left")
    _style.panel_label(axes[0], "(a)")
    for label in runs:
        stored = runs[label][0].samples[0, :, 0]
        axes[1].plot(3000 + np.arange(stored.size), stored, color=colors[label], lw=0.6, label=label)
    axes[1].set_xlabel("step (stored draws, after burn-in)")
    axes[1].set_ylabel(r"$\theta_1$, chain 0")
    axes[1].legend(fontsize=7, loc="lower right")
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
