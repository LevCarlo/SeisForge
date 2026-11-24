"""Figure hk_cluster.svg (and -dark.svg): from per-RF H-k picks to a station estimate.

(a) 60 noisy synthetic RFs of the crust of `_rf.CRUST` (truth H = 35 km,
Vp/Vs = 1.75): ray parameters uniform in 0.045-0.078 s/km, a = 2.5, white
Gaussian noise of 15 % of each trace's peak (assumed). Each RF is stacked
with `seisforge.rf.hk_workflow.HkStack` (Vp = 6.3 km/s, default weights);
the picks are screened with `HkQController` (energy 0.6, elliptical MAD
sigma 2, Mahalanobis distance 1, the reference-config values) and reduced
with the rule of `hk_workflow._estimate_hk_summary`, as `seisforge rf hkseq`
does.
(b) Fraction of a bivariate Gaussian kept by the two cluster tests against
the threshold, Monte Carlo through the package functions (4000 points, equal
energies) versus 1 - exp(-r^2/2) (docs/rf/scripts/hk_derivation.py), for
uncorrelated scatter and for a correlation of 0.8, as along an H-k
trade-off.
"""

import numpy as np
import xarray as xr

import _rf
from _rf import _style, plt
from seisforge.rf.hk_workflow import (
    HkParam,
    HkQController,
    HkStack,
    _estimate_hk_summary,
    elliptical_mad_filter,
)

QC = {"energy_threshold": 0.6, "ell_mad_sigma": 2.0, "mahalanobis_dist_threshold": 1.0}
H_GRID = HkParam("H", 25.0, 45.0, 0.1)
K_GRID = HkParam("k", 1.60, 1.90, 0.005)
N_RF, NOISE_FRACTION, SEED = 60, 0.15, 7

rng = np.random.default_rng(SEED)
results = []
for i, p in enumerate(rng.uniform(0.045, 0.078, N_RF)):
    time_s, data = _rf.synthetic_rf(p=p, gauss_a=2.5, dt_s=0.1)
    data = data + rng.normal(0.0, NOISE_FRACTION * np.abs(data).max(), data.size)
    trace = _rf.rf_trace(time_s, data, p, name=f"EV{i:03d}")
    results.append(HkStack(trace, H_GRID, K_GRID, Vp=_rf.VP_CRUST_KM_S).stack())

qc = HkQController(results, QC)
energy_mask, _ = qc.hk_energy_select()
ellipse_mask, _ = qc.elliptical_mad_select()
mahalanobis_mask, _ = qc.mahalanobis_dist_select()
h_picks = np.array([res.attrs["H_best"] for res in results])
k_picks = np.array([res.attrs["k_best"] for res in results])
h_est, k_est, h_spread, k_spread = _estimate_hk_summary(results, mahalanobis_mask)
assert abs(h_est - _rf.H_TRUE_KM) < 1.0 and abs(k_est - _rf.K_TRUE) < 0.03, (h_est, k_est)


def fake_results(points):
    """Minimal H-k datasets (equal energies) for the QC functions."""
    return [xr.Dataset(attrs={"H_best": h, "k_best": k, "amp_max": 1.0}) for h, k in points]


thresholds = np.linspace(0.25, 3.0, 12)
mc = {}
mc_rng = np.random.default_rng(1)
for rho in (0.0, 0.8):
    points = mc_rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 4000)
    datasets = fake_results(points)
    mc[("mahalanobis", rho)] = [
        HkQController(datasets, {"mahalanobis_dist_threshold": r}).mahalanobis_dist_select()[0].mean()
        for r in thresholds
    ]
    mc[("ellipse", rho)] = [elliptical_mad_filter(points[:, 0], points[:, 1], sigma=r).mean()
                            for r in thresholds]
# Mahalanobis uses the sample covariance, so it follows the closed form for any
# correlation; 4000 points give a binomial error of about 0.008, so 0.03 is safe.
for rho in (0.0, 0.8):
    assert np.max(np.abs(np.array(mc[("mahalanobis", rho)]) - (1 - np.exp(-thresholds**2 / 2)))) < 0.03
assert np.max(np.abs(np.array(mc[("ellipse", 0.0)]) - (1 - np.exp(-thresholds**2 / 2)))) < 0.03


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.plot(h_picks, k_picks, "o", mfc="none", mec=_style.ALTERNATIVE, ms=6, label="all picks")
    ax.plot(h_picks[ellipse_mask & energy_mask], k_picks[ellipse_mask & energy_mask], "x",
            color=_style.SECOND, ms=5, label="energy & elliptical MAD")
    ax.plot(h_picks[mahalanobis_mask], k_picks[mahalanobis_mask], "o", color=_style.MEASURED,
            ms=4, label="Mahalanobis < 1 (used)")
    ax.errorbar(h_est, k_est, xerr=h_spread, yerr=k_spread, fmt="s", color=_style.HIGHLIGHT,
                ms=6, capsize=3, label="estimate ± spread")
    ax.plot(_rf.H_TRUE_KM, _rf.K_TRUE, "*", color=_style.TRUTH, ms=12, label="truth")
    ax.set_xlabel("H (km)")
    ax.set_ylabel(r"$\kappa$")
    ax.set_title(f"{N_RF} noisy synthetic RFs, one pick each")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    r_fine = np.linspace(0, 3.0, 200)
    ax.plot(r_fine, 1 - np.exp(-r_fine**2 / 2), color=_style.TRUTH, lw=1.2,
            label=r"$1 - e^{-r^2/2}$")
    styles = {("mahalanobis", 0.0): ("o", _style.MEASURED), ("mahalanobis", 0.8): ("s", _style.MEASURED),
              ("ellipse", 0.0): ("o", _style.SECOND), ("ellipse", 0.8): ("s", _style.SECOND)}
    for (test, rho), (marker, color) in styles.items():
        name = "Mahalanobis" if test == "mahalanobis" else "elliptical MAD"
        ax.plot(thresholds, mc[(test, rho)], marker, mfc="none" if rho else color, mec=color,
                ls="none", ms=5, label=rf"{name}, $\rho$ = {rho:g}")
    ax.axvline(1.0, color=_style.MUTED, lw=0.7, ls=":")
    ax.axvline(2.0, color=_style.MUTED, lw=0.7, ls=":")
    ax.set_xlabel(r"threshold $r$ (Mahalanobis) or $\sigma$ (elliptical MAD)")
    ax.set_ylabel("fraction kept")
    ax.set_title("bivariate Gaussian, Monte Carlo")
    ax.legend(loc="lower right", fontsize=7)
    for axis, label in zip(axes, "ab"):
        _style.panel_label(axis, label)
    fig.tight_layout()
    return fig


_rf.save(draw, __file__)
print(f"kept: energy {energy_mask.sum()}, elliptical MAD {ellipse_mask.sum()}, "
      f"both {(energy_mask & ellipse_mask).sum()}, Mahalanobis {mahalanobis_mask.sum()} of {N_RF}")
print(f"estimate H = {h_est:.2f} ± {h_spread:.2f} km, k = {k_est:.3f} ± {k_spread:.3f}")
for rho in (0.0, 0.8):
    i = int(np.argmin(np.abs(thresholds - 2.0)))
    print(f"rho {rho}: at threshold {thresholds[i]:.2f}, ellipse keeps {mc[('ellipse', rho)][i]:.3f}, "
          f"Mahalanobis {mc[('mahalanobis', rho)][i]:.3f}")
