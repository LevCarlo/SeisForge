"""Figure convergence_diagnostics.svg (and -dark.svg): ESS, split R-hat and lagging chains.

(a) Four AR(1) chains x_t = phi x_(t-1) + sqrt(1 - phi^2) e_t, 4000 draws
each: `seisforge.inv.effective_sample_size` against the exact large-n value
m n (1 - phi) / (1 + phi). (b) Four chains of independent N(0, 1) draws,
one of them shifted by delta standard deviations: `split_rhat` is 1 at
delta = 0 and rises with delta, and `lagging_chains`, fed the log density of
the N(0, 1) target, flags the shifted chain once the 5-95 % ranges of the log
densities no longer overlap (filled markers).
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import effective_sample_size, lagging_chains, split_rhat

M, N = 4, 4000
rng = np.random.default_rng(2)
phis = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98])
ess, exact = [], M * N * (1 - phis) / (1 + phis)
for phi in phis:
    x = np.empty((M, N))
    x[:, 0] = rng.standard_normal(M)
    noise = rng.standard_normal((M, N)) * np.sqrt(1 - phi**2)
    for t in range(1, N):
        x[:, t] = phi * x[:, t - 1] + noise[:, t]
    ess.append(effective_sample_size(x))
ess = np.array(ess)
for phi, e, a in zip(phis, ess, exact):
    print(f"phi {phi:4.2f}: ESS {e:7.0f}, exact {a:7.0f}, ratio {e / a:.3f}")

deltas = np.linspace(0.0, 6.0, 25)
rhat, flagged = [], []
base = rng.standard_normal((M, 1000))
for delta in deltas:
    x = base.copy()
    x[3] += delta
    rhat.append(split_rhat(x))
    flagged.append(3 in lagging_chains(-0.5 * x**2))
rhat, flagged = np.array(rhat), np.array(flagged)
print("R-hat at delta = 0, 0.5, 1:", np.round(np.interp([0, 0.5, 1.0], deltas, rhat), 3).tolist())
print("shifted chain first flagged at delta =", deltas[np.argmax(flagged)])


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(phis, exact, color=_style.INK, label=r"exact $mn(1-\phi)/(1+\phi)$")
    axes[0].plot(phis, ess, "o", color=_style.MEASURED, label="package")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"lag-1 autocorrelation $\phi$")
    axes[0].set_ylabel("effective sample size")
    axes[0].set_title(f"{M} chains x {N} draws")
    axes[0].legend()
    _style.panel_label(axes[0], "(a)")
    axes[1].plot(deltas, rhat, color=_style.MEASURED, lw=1.0)
    axes[1].plot(deltas[~flagged], rhat[~flagged], "o", mfc="none", color=_style.MEASURED, label="not flagged")
    axes[1].plot(deltas[flagged], rhat[flagged], "o", color=_style.HIGHLIGHT, label="flagged as lagging")
    for level in (1.01, 1.1):
        axes[1].axhline(level, color=_style.ALTERNATIVE, lw=0.8, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"shift of one chain $\delta$ (sd)")
    axes[1].set_ylabel("split R-hat")
    axes[1].set_title("one of four chains in another place")
    axes[1].legend()
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
