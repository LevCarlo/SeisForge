"""Plot the H/V forward example: the model and its H/V curve.

    python plot_result.py [--hvsr output/hvsr.dat] [--model data/model.txt] [-o result.png]

Plotting only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hvsr", type=Path, default=Path("output/hvsr.dat"))
    parser.add_argument("--model", type=Path, default=Path("data/model.txt"))
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)

    model = np.atleast_2d(np.loadtxt(args.model))
    frequency, hvsr = np.loadtxt(args.hvsr, unpack=True)
    fig, (ax_m, ax_h) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 2]})
    tops = np.concatenate([[0.0], np.cumsum(model[:-1, 0])])
    bottom = tops[-1] + max(tops[-1], 0.05)
    depth = np.column_stack([tops, np.append(tops[1:], bottom)]).ravel() * 1000.0
    ax_m.plot(np.repeat(model[:, 2], 2), depth, "k", label="Vs")
    ax_m.plot(np.repeat(model[:, 1], 2), depth, "k--", label="Vp")
    ax_m.set(xlabel="Velocity (km/s)", ylabel="Depth (m)", ylim=(depth[-1], 0))
    ax_m.legend()
    ax_h.loglog(frequency, hvsr, "C3")
    ax_h.set(xlabel="Frequency (Hz)", ylabel="H/V (diffuse field)")
    ax_h.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
