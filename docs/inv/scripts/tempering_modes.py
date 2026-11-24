"""Figure tempering_modes.svg (and -dark.svg): parallel tempering between separated modes.

Target on [-8, 8]: uniform prior times a likelihood with two narrow modes,
weights 0.3 at -4 and 0.7 at +4, width 0.25 (the case of
tests/test_inv_tempering.py); ln L drops by about 128 between them. Four
chains of `seisforge.inv.run_metropolis`, 8000 steps, burn-in 1000, proposal
sigma 0.3, all started at -4: (a) plain random walk, (b) geometric ladder of
6 temperatures up to 200. Histograms of the stored T = 1 draws against the
exact density; the fraction of draws in the right mode is printed (exact 0.7).
"""

import math

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import MetropolisConfig, geometric_temperatures, run_metropolis

WEIGHTS, MEANS, WIDTH = (0.3, 0.7), (-4.0, 4.0), 0.25


def log_likelihood(x):
    terms = [math.log(w) - 0.5 * ((x[0] - m) / WIDTH) ** 2 for w, m in zip(WEIGHTS, MEANS)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def split_target(x):
    return 0.0, log_likelihood(x)


common = dict(n_steps=8000, proposal_sigma=0.3, burn_in=1000, bounds=([-8.0], [8.0]))
ladder = geometric_temperatures(6, 200.0)
plain = run_metropolis(lambda x: log_likelihood(x), np.full((4, 1), -4.0), MetropolisConfig(**common), seeds=5)
tempered = run_metropolis(split_target, np.full((4, 1), -4.0),
                          MetropolisConfig(**common, temperatures=ladder), seeds=5)
x = np.linspace(-6, 6, 1201)
exact = sum(w * np.exp(-0.5 * ((x - m) / WIDTH) ** 2) / (WIDTH * np.sqrt(2 * np.pi)) for w, m in zip(WEIGHTS, MEANS))
fractions = {label: float(np.mean(result.samples[..., 0] > 0)) for label, result in
             (("plain", plain), ("tempered", tempered))}
print("ladder:", np.round(ladder, 2).tolist())
print("fraction of stored draws in the right mode:", fractions, "(exact 0.7)")
print("swap acceptance, chain 0:", np.round(tempered.chains[0].swap_acceptance, 2).tolist())


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, result, label, letter in ((axes[0], plain, "plain random walk", "(a)"),
                                      (axes[1], tempered, f"tempering, {len(ladder)} temperatures to 200", "(b)")):
        ax.hist(result.samples[..., 0].ravel(), bins=np.linspace(-6, 6, 121), density=True,
                color=_style.MEASURED, alpha=0.6, label="stored draws")
        ax.plot(x, exact, color=_style.INK, lw=1.0, label="exact")
        ax.set_title(f"{label}: right mode {fractions['plain' if letter == '(a)' else 'tempered']:.2f} (exact 0.70)")
        ax.set_xlabel(r"$\theta$")
        _style.panel_label(ax, letter)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper center")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
