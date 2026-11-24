"""Plot the inversion example result: prior and posterior Vs(z) with the truth, and the data fit.

Run from the example folder after `run.sh` (which calls it):

    python plot_result.py [--prior output/prior] [--posterior output/posterior] \
        [--truth data/true_model.dat] [-o result.png]

It also prints the convergence numbers that README.md asks you to check.
`expected_result.png` is drawn with this script. Only plotting and printing are done here; the
quantiles and diagnostics come from the run outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def truth_stairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Depth and Vs of the layered truth (`true_model.dat`) as a staircase to the model bottom."""
    top, thickness, _vp, vs, _rho = np.loadtxt(path, comments="#", unpack=True)
    bottom = top + thickness
    bottom[-1] = top[-1]  # the half-space starts at the model bottom; draw no further
    depth = np.column_stack([top, bottom]).ravel()
    return depth[:-1], np.repeat(vs, 2)[:-1]


def plot(prior_dir: Path, posterior_dir: Path, truth: Path | None, output: Path) -> None:
    prior = xr.open_dataset(prior_dir / "disp_hv_prior_vs.nc")
    post = xr.open_dataset(posterior_dir / "disp_hv_vs.nc")
    pred = xr.open_dataset(posterior_dir / "disp_hv_predictive.nc")

    fig, (ax_vs, ax_c, ax_hv) = plt.subplots(1, 3, figsize=(11, 4))
    for ds, color, label in ((prior, "0.6", "prior"), (post, "C3", "posterior")):
        # ds["median"], not ds.median: the latter is the xarray method.
        ax_vs.fill_betweenx(ds["depth"], ds["p05"], ds["p95"], color=color, alpha=0.3, lw=0, label=f"{label} 5-95%")
        ax_vs.plot(ds["median"], ds["depth"], color=color, lw=1)
    if truth is not None:
        depth, vs = truth_stairs(truth)
        ax_vs.plot(vs, depth, color="k", lw=1.2, label="truth")
    ax_vs.set(xlabel="Vs (km/s)", ylabel="Depth (km)", ylim=(float(post["depth"].max()), 0))
    ax_vs.legend(loc="lower left")

    for ax, name, obs, unit in (
        (ax_c, "dispersion", "observed_velocity", "phase velocity (km/s)"),
        (ax_hv, "hv", "observed_hv", "Rayleigh H/V"),
    ):
        period = pred[f"{name}_period"]
        ax.fill_between(
            period, pred[f"{name}_p05"], pred[f"{name}_p95"], color="C3", alpha=0.3, lw=0,
            label="posterior predictive 5-95%",
        )
        ax.errorbar(period, pred[f"{name}_{obs}"], pred[f"{name}_sigma"], fmt="k.", label="observed")
        ax.set(xlabel="Period (s)", ylabel=unit)
    ax_c.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"wrote {output}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prior", type=Path, default=Path("output/prior"))
    parser.add_argument("--posterior", type=Path, default=Path("output/posterior"))
    parser.add_argument("--truth", type=Path, default=Path("data/true_model.dat"), help="omit the truth if missing")
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)
    truth = args.truth if args.truth.exists() else None
    plot(args.prior, args.posterior, truth, args.output)
    print_summary(args.posterior, truth)


def print_summary(posterior_dir: Path, truth: Path | None) -> None:
    """Print the numbers README.md lists under "How to tell the result is right"."""
    mcmc = xr.open_dataset(posterior_dir / "disp_hv_mcmc.nc")
    vs = xr.open_dataset(posterior_dir / "disp_hv_vs.nc")
    print(f"max split R-hat: {float(mcmc.split_rhat.max()):.3f}")
    print(f"min effective sample size: {float(mcmc.effective_sample_size.min()):.0f}")
    print(f"lagging chains: {mcmc.attrs.get('lagging_chains', 'none') or 'none'}")
    if truth is not None:
        top, _thickness, _vp, vs_true, _rho = np.loadtxt(truth, comments="#", unpack=True)
        depth = vs["depth"].values
        layer = np.clip(np.searchsorted(top, depth, side="right") - 1, 0, top.size - 1)
        inside = (vs_true[layer] >= vs["p05"].values) & (vs_true[layer] <= vs["p95"].values)
        print(f"truth inside the posterior 5-95 % band at {inside.sum()} of {inside.size} depths")


if __name__ == "__main__":
    main()
