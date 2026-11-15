"""Plot the STA001 example result: prior and posterior Vs(z) with the truth, and the data fit.

Run from the example folder after the runs of `docs/inv_getting_started.md`:

    python /path/to/SeisForge/tools/plot_sta001_result.py [--prior run_prior] \
        [--posterior run_disp_hv] [--truth true_model.dat] [-o sta001_result.png]

The figure in the docs, `docs/img/sta001_result.png`, is drawn with this
script. Only plotting is done here; the quantiles come from the run outputs.
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
    parser.add_argument("--prior", type=Path, default=Path("run_prior"))
    parser.add_argument("--posterior", type=Path, default=Path("run_disp_hv"))
    parser.add_argument("--truth", type=Path, default=Path("true_model.dat"), help="omit the truth if missing")
    parser.add_argument("-o", "--output", type=Path, default=Path("sta001_result.png"))
    args = parser.parse_args(argv)
    truth = args.truth if args.truth.exists() else None
    plot(args.prior, args.posterior, truth, args.output)


if __name__ == "__main__":
    main()
