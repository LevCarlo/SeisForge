"""Figure noise_scale.svg (and -dark.svg): the hierarchical noise scale h.

Residuals r_i = 1.5 sigma_i e_i (e_i standard normal, fixed seed), so the
file's sigmas are 1.5 times too small. (a) log L(h) from
`seisforge.inv.gaussian_observable_fit` (dots) against the closed form
-chi2 / (2 h^2) - N ln h (line), for N = 14 and N = 140; the maximum is at
h^2 = chi2 / N. (b) Posterior of log10 h for a uniform prior on log10 h,
p proportional to L, which narrows as N grows: the sampled h measures the
overall misfit level, as the closed form says.
"""

import numpy as np

import _inv
from _inv import _style
from _style import plt
from seisforge.inv import gaussian_observable_fit

TRUE_FACTOR = 1.5
SEED = 0
log10_h = np.linspace(-0.3, 0.6, 181)
h = 10.0**log10_h
cases = {}
rng = np.random.default_rng(SEED)
for n in (14, 140):
    sigma = np.full(n, 0.03)
    residual = TRUE_FACTOR * sigma * rng.standard_normal(n)
    chi2 = np.sum((residual / sigma) ** 2)
    package = np.array([
        gaussian_observable_fit(periods=np.arange(1.0, n + 1), observed=residual, predicted=np.zeros(n),
                                sigma=sigma, noise_scale=value).loglike
        for value in h
    ])
    closed = -chi2 / (2 * h**2) - n * np.log(h)
    posterior = np.exp(closed - closed.max())
    posterior /= np.trapezoid(posterior, log10_h)
    cases[n] = (package, closed, posterior, np.sqrt(chi2 / n))
    print(f"N = {n}: max |package - closed form| = {np.max(np.abs(package - closed)):.2e}; "
          f"argmax h = {h[np.argmax(package)]:.3f}, sqrt(chi2/N) = {np.sqrt(chi2 / n):.3f}")


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    colors = {14: _style.MEASURED, 140: _style.SECOND}
    for n, (package, closed, posterior, best) in cases.items():
        shift = closed.max()
        axes[0].plot(h, closed - shift, color=colors[n], label=f"N = {n}")
        axes[0].plot(h[::12], package[::12] - shift, "o", color=colors[n], ms=3)
        axes[0].axvline(best, color=colors[n], ls=":", lw=0.8)
        axes[1].plot(log10_h, posterior, color=colors[n], label=f"N = {n}")
    axes[0].set_ylim(-12, 1)
    axes[0].set_xlabel("noise scale $h$")
    axes[0].set_ylabel(r"$\log L - \max \log L$")
    axes[0].set_title(r"line: closed form; dots: package; dotted: $\sqrt{\chi^2/N}$")
    axes[0].legend()
    _style.panel_label(axes[0], "(a)")
    axes[1].axvline(np.log10(TRUE_FACTOR), color=_style.INK, lw=0.8, label=f"true factor {TRUE_FACTOR:g}")
    axes[1].set_xlabel(r"$\log_{10} h$")
    axes[1].set_ylabel("posterior density")
    axes[1].set_title(r"uniform prior on $\log_{10} h$")
    axes[1].legend()
    _style.panel_label(axes[1], "(b)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    _inv.save(draw, __file__)
