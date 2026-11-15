"""Plot the rotation example: the nine ZRT CCFs after rotation, with the truth.

Run from the example folder after `seisforge ant rotate-ccf -c config.yml` (run.sh does both):

    python plot_result.py [--output-dir output/XX.A/XX.A_XX.B] [--truth data/true_zrt.npz] [-o result.png]

It prints, per component, the largest difference from the noise-free truth
relative to the largest ZZ amplitude, which README.md asks you to check.
Plotting and printing only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from obspy import read  # noqa: E402

COMPONENTS = ("ZZ", "ZR", "ZT", "RZ", "RR", "RT", "TZ", "TR", "TT")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("output/XX.A/XX.A_XX.B"))
    parser.add_argument("--truth", type=Path, default=Path("data/true_zrt.npz"), help="omitted if missing")
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)

    truth = np.load(args.truth) if args.truth.exists() else None
    fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)
    scale = None
    for ax, comp in zip(axes.ravel(), COMPONENTS, strict=True):
        trace = read(str(next(args.output_dir.glob(f"*_{comp}.SAC"))))[0]
        time = trace.stats.sac.b + np.arange(trace.stats.npts) * trace.stats.delta
        if scale is None:
            scale = np.abs(trace.data).max()  # ZZ first
        ax.plot(time, trace.data / scale, color="C3", lw=0.8, label="rotated")
        if truth is not None:
            ax.plot(truth["time_s"], truth[comp] / scale, "k--", lw=0.6, label="truth")
            print(f"{comp}: max |rotated - truth| = {np.max(np.abs(trace.data - truth[comp])) / scale:.3f} of max |ZZ|")
        ax.set_title(comp, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Lag time (s)")
    axes[0, 0].set_xlim(-60, 60)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
