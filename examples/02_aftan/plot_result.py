"""Plot the AFTAN example: the CCFs, and measured phase and group velocity against the truth.

Run from the example folder after `seisforge ant aftan -c config.yml` (run.sh does both):

    python plot_result.py [--output-dir output] [--truth data/true_dispersion.dat] [-o result.png]

It also prints, for every measured file, the largest difference from the
truth, which README.md asks you to check. Plotting and printing only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from obspy import read  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ccf-dir", type=Path, default=Path("data/ccf"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--truth", type=Path, default=Path("data/true_dispersion.dat"), help="omitted if missing")
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)

    fig, (ax_w, ax_c) = plt.subplots(1, 2, figsize=(10, 4))
    for path in sorted(args.ccf_dir.glob("*.SAC")):
        trace = read(str(path))[0]
        sac = trace.stats.sac
        time = sac.b + np.arange(trace.stats.npts) * trace.stats.delta
        data = trace.data / np.abs(trace.data).max() * 8.0
        ax_w.plot(time, sac.dist + data, color="k", lw=0.6)
    ax_w.set(xlabel="Lag time (s)", ylabel="Distance (km)", xlim=(-80, 80), title="Input CCFs")

    truth = np.loadtxt(args.truth) if args.truth.exists() else None
    if truth is not None:
        ax_c.plot(truth[:, 0], truth[:, 1], "k-", lw=1.5, label="true phase velocity")
        ax_c.plot(truth[:, 0], truth[:, 2], "k--", lw=1.5, label="true group velocity")
    for index, path in enumerate(sorted(args.output_dir.glob("*.dat"))):
        table = np.loadtxt(path)
        period, group, phase = table[:, 1], table[:, 2], table[:, 3]
        color = f"C{index}"
        ax_c.plot(period, phase, "o", ms=3, color=color, label=path.stem)
        ax_c.plot(period, group, "x", ms=4, color=color)
        if truth is not None:
            d_phase = np.max(np.abs(phase - np.interp(period, truth[:, 0], truth[:, 1])))
            d_group = np.max(np.abs(group - np.interp(period, truth[:, 0], truth[:, 2])))
            print(f"{path.name}: max |phase - truth| {d_phase:.3f} km/s, max |group - truth| {d_group:.3f} km/s")
    ax_c.set(xlabel="Period (s)", ylabel="Velocity (km/s)", title="o phase, x group")
    ax_c.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
