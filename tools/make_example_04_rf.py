"""Write the synthetic receiver functions of the RF QC + H-k example (examples/04_rf_qc_hk).

Run from the repository root:

    PYTHONPATH=src python tools/make_example_04_rf.py [--overwrite]

Truth (assumed, not a real station): one crust over a mantle half-space,

    crust:  H = 35 km, Vp 6.3 km/s, Vp/Vs 1.75 (Vs 3.6 km/s), density 2.8 g/cm^3
    mantle: Vp 8.0 km/s, Vs 4.5 km/s, density 3.3 g/cm^3

Radial P receiver functions from `seisforge.rf.synthetic.receiver_function`
(amplitude "gaussian_peak": direct P peaks at its transmission amplitude)
at `N_GOOD` = 30 events with ray parameters uniform in 0.045-0.078 s/km and
back azimuths uniform in 0-360 deg, for the two Gaussian widths a = 5.0 and
2.5 (folders `Gauss_5.0`, `Gauss_2.5`), dt 0.1 s, -10 to 60 s. Each trace
gets independent Gaussian noise with standard deviation `NOISE_FRACTION`
= 3 % of its direct-P peak.

Bad traces the quality control must reject (assumed failure types, 3 each):
`lowsnr` (SNR header 2, below the threshold 3), `farslow` (ray parameter
0.09 s/km, outside 0.04-0.08), `flipped` (polarity reversed: negative
direct P) and `noisy` (noise 60 % of the P peak). Event names say which
is which (`EV001`-`EV030` good, `BAD_<kind>_<n>` bad).

SAC headers: `b` (-10 s), `delta`, `user0` = ray parameter (s/km), `user2` =
SNR (good traces: uniform 8-30, assumed), `baz`, `kstnm`/`knetwk` XX.RF01,
`kcmpnm` R. Random numbers: `numpy.random.default_rng(SEED)`.

Outputs (in `--out`, default `examples/04_rf_qc_hk/data`). Existing files
are kept unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from obspy import Trace

from example_truth import ROOT
from seisforge.rf.synthetic import receiver_function

EXAMPLE = ROOT / "examples" / "04_rf_qc_hk"
THICKNESS_KM = np.array([35.0, 0.0])
VP_KM_S = np.array([6.3, 8.0])
VS_KM_S = np.array([6.3 / 1.75, 4.5])
RHO_G_CM3 = np.array([2.8, 3.3])
GAUSS = {"Gauss_5.0": 5.0, "Gauss_2.5": 2.5}
DT_S, TIME_SHIFT_S, N_SAMPLES = 0.1, 10.0, 701
N_GOOD = 30
NOISE_FRACTION = 0.03
SEED = 20261115


def events(rng):
    """(name, ray parameter s/km, back azimuth deg, SNR, polarity, noise fraction)."""
    good = [
        (f"EV{i + 1:03d}", rng.uniform(0.045, 0.078), rng.uniform(0, 360), rng.uniform(8, 30), 1.0, NOISE_FRACTION)
        for i in range(N_GOOD)
    ]
    bad = []
    for n in range(3):
        bad.append((f"BAD_lowsnr_{n}", rng.uniform(0.05, 0.07), rng.uniform(0, 360), 2.0, 1.0, NOISE_FRACTION))
        bad.append((f"BAD_farslow_{n}", 0.09, rng.uniform(0, 360), 15.0, 1.0, NOISE_FRACTION))
        bad.append((f"BAD_flipped_{n}", rng.uniform(0.05, 0.07), rng.uniform(0, 360), 15.0, -1.0, NOISE_FRACTION))
        bad.append((f"BAD_noisy_{n}", rng.uniform(0.05, 0.07), rng.uniform(0, 360), 15.0, 1.0, 0.6))
    return good + bad


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(SEED)
    traces = {}
    for name, ray_parameter, baz, snr, polarity, noise in events(rng):
        for folder, gauss_a in GAUSS.items():
            time_s, rf = receiver_function(
                THICKNESS_KM, VP_KM_S, VS_KM_S, RHO_G_CM3, ray_parameter,
                dt_s=DT_S, n_samples=N_SAMPLES, gauss_a=gauss_a,
                time_shift_s=TIME_SHIFT_S, amplitude="gaussian_peak",
            )
            peak = np.abs(rf).max()
            data = polarity * rf + rng.normal(0.0, noise * peak, rf.size)
            trace = Trace(data=data.astype(np.float32))
            trace.stats.delta = DT_S
            trace.stats.sac = {
                "b": float(time_s[0]), "user0": float(ray_parameter), "user2": float(snr),
                "baz": float(baz), "kstnm": "RF01", "knetwk": "XX", "kcmpnm": "R", "kevnm": name,
            }
            traces[args.out / folder / f"{name}.XX.RF01.R.sac"] = trace
    existing = [str(p) for p in traces if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {len(existing)} files in {args.out}; pass --overwrite.")
    for path, trace in traces.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        trace.write(str(path), format="SAC")
    print(f"wrote {len(traces)} receiver functions to {args.out}")


if __name__ == "__main__":
    main()
