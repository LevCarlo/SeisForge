"""Plot the end-to-end example: station curve against the truth, and Vs(z) against the truth.

Run from the example folder after run.sh's commands (run.sh calls it):

    python plot_result.py [--station XX.A22] [-o result.png]

Prints the station curve's difference from the truth and the inversion's
convergence numbers, which README.md asks you to check. Plotting and printing only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--station", default="XX.A22")
    parser.add_argument("--curve", type=Path, default=Path("output/vph_disp.dat"))
    parser.add_argument("--inversion", type=Path, default=Path("output/inversion"))
    parser.add_argument("--true-curve", type=Path, default=Path("data/true_phase_velocity.dat"))
    parser.add_argument("--true-model", type=Path, default=Path("data/true_model.dat"))
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)

    curve = np.loadtxt(args.curve)
    truth = np.loadtxt(args.true_curve)
    difference = curve[:, 1] - np.interp(curve[:, 0], truth[:, 0], truth[:, 1])
    print(f"station curve - truth (km/s) at {', '.join(f'{p:g}' for p in curve[:, 0])} s: "
          f"{', '.join(f'{d:+.3f}' for d in difference)}")
    mcmc = xr.open_dataset(args.inversion / "disp_hv_mcmc.nc")
    vs = xr.open_dataset(args.inversion / "disp_hv_vs.nc")
    print(f"max split R-hat: {float(mcmc.split_rhat.max()):.3f}")
    print(f"min effective sample size: {float(mcmc.effective_sample_size.min()):.0f}")
    top, _h, _vp, vs_true, _rho = np.loadtxt(args.true_model, unpack=True)
    depth = vs["depth"].values
    layer = np.clip(np.searchsorted(top, depth, side="right") - 1, 0, top.size - 1)
    inside = (vs_true[layer] >= vs["p05"].values) & (vs_true[layer] <= vs["p95"].values)
    print(f"truth inside the posterior 5-95 % band at {inside.sum()} of {inside.size} depths")
    print(f"max |posterior median - truth|: {np.max(np.abs(vs['median'].values - vs_true[layer])):.3f} km/s")

    fig, (ax_c, ax_vs) = plt.subplots(1, 2, figsize=(9, 4.5))
    ax_c.plot(truth[:, 0], truth[:, 1], "k-", label="truth")
    ax_c.errorbar(curve[:, 0], curve[:, 1], curve[:, 2], fmt="o", color="C3", label=f"{args.station} from Eikonal")
    ax_c.set(xlabel="Period (s)", ylabel="Rayleigh phase velocity (km/s)")
    ax_c.legend()
    ax_vs.fill_betweenx(depth, vs["p05"], vs["p95"], color="C3", alpha=0.3, lw=0, label="posterior 5-95%")
    ax_vs.plot(vs["median"], depth, color="C3", lw=1, label="posterior median")
    ax_vs.plot(vs_true[layer], depth, "k", lw=1.2, label="truth")
    ax_vs.set(xlabel="Vs (km/s)", ylabel="Depth (km)", ylim=(depth.max(), 0))
    ax_vs.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
