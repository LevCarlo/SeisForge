"""Plot the RF example: RFs kept by QC with the H-k phase times, and the per-RF H-k estimates.

Run from the example folder after `seisforge rf qc` and `seisforge rf hkseq --mode low` (run.sh does all):

    python plot_result.py [--rf-dir output/Gauss_2.5_CLEAN] [--hk output/hkSeq/XX.RF01_LFreq_Hk_QC.dat] [-o result.png]

It prints the number of RFs kept and the median H and k with their median
absolute deviation over the RFs kept by the H-k QC (the summary hkseq logs
when 15 or more are kept). Phase times come from
`seisforge.rf.hk_core.hk_phase_delays_s`; plotting and printing only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from obspy import read  # noqa: E402

from seisforge.rf.hk_core import hk_phase_delays_s  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/Gauss_2.5"))
    parser.add_argument("--rf-dir", type=Path, default=Path("output/Gauss_2.5_CLEAN"))
    parser.add_argument("--hk", type=Path, default=Path("output/hkSeq/XX.RF01_LFreq_Hk_QC.dat"))
    parser.add_argument("--vp", type=float, default=6.3, help="crustal Vp of hkseq.yml (km/s)")
    parser.add_argument("--truth", type=float, nargs=2, default=(35.0, 1.75), metavar=("H_KM", "K"))
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)

    traces = sorted((read(str(p))[0] for p in args.rf_dir.glob("*.sac")), key=lambda tr: tr.stats.sac.user0)
    n_raw = len(list(args.raw_dir.glob("*.sac")))
    names, h_km, k = np.genfromtxt(args.hk, dtype=None, encoding="utf-8", usecols=(0, 1, 2), unpack=True)
    h_med, k_med = np.median(h_km), np.median(k)
    print(f"RFs kept by QC: {len(traces)} of {n_raw}")
    print(f"RFs kept by H-k QC: {h_km.size}")
    print(f"H = {h_med:.2f} +- {np.median(np.abs(h_km - h_med)):.2f} km, "
          f"k = {k_med:.3f} +- {np.median(np.abs(k - k_med)):.3f}")

    fig, (ax_rf, ax_hk) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    for index, trace in enumerate(traces):
        sac = trace.stats.sac
        time = sac.b + np.arange(trace.stats.npts) * trace.stats.delta
        ax_rf.plot(time, index + 2.0 * trace.data / np.abs(trace.data).max(), "k", lw=0.5)
        delays = hk_phase_delays_s(
            np.array([h_med]), np.array([args.vp]), np.array([args.vp / k_med]), sac.user0
        )
        for delay, color in zip(delays, ("C0", "C1", "C2"), strict=False):
            ax_rf.plot(np.ravel(delay), [index], "|", color=color, ms=8)
    ax_rf.set(xlabel="Time after P (s)", ylabel="RF (sorted by ray parameter)", xlim=(-5, 25),
              title="kept RFs; ticks: Ps, PpPs, PsPs for the H-k result")
    ax_hk.plot(h_km, k, "o", color="C3", label="per-RF H-k (kept)")
    ax_hk.plot(*args.truth, "k*", ms=14, label="truth")
    ax_hk.plot(h_med, k_med, "x", color="k", ms=10, label="median")
    ax_hk.set(xlabel="Crustal thickness H (km)", ylabel="Vp/Vs k")
    ax_hk.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
