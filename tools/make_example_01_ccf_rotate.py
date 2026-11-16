"""Write the synthetic nine-component ZNE cross-correlations of the rotation example (examples/01_ccf_rotate).

Run from the repository root:

    PYTHONPATH=src python tools/make_example_01_ccf_rotate.py [--overwrite]

Data (all values assumed): one station pair, virtual source XX.A at
(110.0 E, 30.0 N) and receiver XX.B at (110.6 E, 30.5 N), about 80 km apart
at azimuth about 46 deg, so that N and E both carry radial and transverse
energy. A known ZRT wavefield is set first:

- ZZ, RR, ZR, RZ: Rayleigh waves with the phase velocity of the shared truth
  model (`tools/example_truth.py`), built with `seisforge.ant.synthetic` and
  the far-field phase constant of each component in the default ``lin2008``
  rotation convention (AFTAN pi_over_4 -1 for ZZ and RR, -3 for ZR, +1 for
  RZ; `seisforge.ant.aftan.phase`); relative amplitudes 1, 0.6, 0.5, 0.5
  (assumed);
- TT: a Love-like wave, the same curve times `LOVE_VELOCITY_FACTOR` = 1.1
  (assumed), amplitude 0.8;
- ZT, TZ, RT, TR: zero (no Rayleigh-Love coupling in a laterally
  homogeneous, isotropic medium).

It is rotated to ZNE with `seisforge.ant.rotation.rotate_zrt_ccf_to_zne`
(default ``lin2008`` R/T axes) using the geodesic azimuth and back azimuth of the pair (obspy), and
independent Gaussian noise of `NOISE_FRACTION` = 1 % of the largest ZZ
amplitude is added to every ZNE component (`numpy.random.default_rng(SEED)`).
Rotating back must therefore return the ZRT wavefield, with the four cross
terms at the noise level.

Outputs (in `--out`, default `examples/01_ccf_rotate/data`):
`ccf_zne/XX.A/XX.A_XX.B/XX.A_XX.B_<comp>.SAC` (comp = ZZ, ZN, ZE, NZ, NN,
NE, EZ, EN, EE; SAC headers b, delta, dist, evlo/evla, stlo/stla, kcmpnm)
and `true_zrt.npz` (the noise-free ZRT wavefield, for checking). Existing
files are kept unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from obspy import Trace
from obspy.geodetics.base import gps2dist_azimuth

from example_truth import CURVE_PERIODS_S, ROOT, truth_phase_velocity
from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zrt_ccf_to_zne
from seisforge.ant.synthetic import symmetric_ccf_time_axis, synthetic_ccf

EXAMPLE = ROOT / "examples" / "01_ccf_rotate"
SOURCE = ("XX.A", 110.0, 30.0)
RECEIVER = ("XX.B", 110.6, 30.5)
DELTA_S = 0.2
MAX_LAG_S = 100.0
MIN_PERIOD_S, MAX_PERIOD_S = 2.0, 8.0
LOVE_VELOCITY_FACTOR = 1.1
# component: (pi_over_4 of the AFTAN convention, relative amplitude), assumed amplitudes
RAYLEIGH = {"ZZ": (-1.0, 1.0), "RR": (-1.0, 0.6), "ZR": (-3.0, 0.5), "RZ": (1.0, 0.5)}
LOVE_AMPLITUDE = 0.8
NOISE_FRACTION = 0.01
SEED = 20261115


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)

    distance_m, azimuth, back_azimuth = gps2dist_azimuth(SOURCE[2], SOURCE[1], RECEIVER[2], RECEIVER[1])
    distance_km = distance_m / 1000.0
    time_s = symmetric_ccf_time_axis(MAX_LAG_S, DELTA_S)
    velocity = truth_phase_velocity(CURVE_PERIODS_S)
    band = {"min_period_s": MIN_PERIOD_S, "max_period_s": MAX_PERIOD_S}
    zrt = {comp: np.zeros(time_s.size) for comp in ZRT_COMPONENTS}
    for comp, (pi_over_4, amplitude) in RAYLEIGH.items():
        zrt[comp] = amplitude * synthetic_ccf(time_s, distance_km, CURVE_PERIODS_S, velocity, pi_over_4=pi_over_4, **band)
    zrt["TT"] = LOVE_AMPLITUDE * synthetic_ccf(
        time_s, distance_km, CURVE_PERIODS_S, LOVE_VELOCITY_FACTOR * velocity, pi_over_4=-1.0, **band
    )
    zne = rotate_zrt_ccf_to_zne(zrt, azimuth, back_azimuth)
    rng = np.random.default_rng(SEED)
    noise_std = NOISE_FRACTION * np.abs(zrt["ZZ"]).max()

    pair = f"{SOURCE[0]}_{RECEIVER[0]}"
    folder = args.out / "ccf_zne" / SOURCE[0] / pair
    files = {comp: folder / f"{pair}_{comp}.SAC" for comp in zne}
    truth = args.out / "true_zrt.npz"
    existing = [str(p) for p in [*files.values(), truth] if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {', '.join(existing)}; pass --overwrite.")
    folder.mkdir(parents=True, exist_ok=True)
    for comp, path in files.items():
        trace = Trace(data=(zne[comp] + rng.normal(0.0, noise_std, time_s.size)).astype(np.float32))
        trace.stats.delta = DELTA_S
        trace.stats.sac = {
            "b": float(time_s[0]), "dist": distance_km,
            "evlo": SOURCE[1], "evla": SOURCE[2], "stlo": RECEIVER[1], "stla": RECEIVER[2],
            "kevnm": SOURCE[0], "kstnm": "B", "knetwk": "XX", "kcmpnm": comp,
        }
        trace.write(str(path), format="SAC")
        print(f"wrote {path}")
    np.savez(truth, time_s=time_s, azimuth_deg=azimuth, back_azimuth_deg=back_azimuth,
             noise_std=noise_std, **zrt)
    print(f"wrote {truth} (azimuth {azimuth:.2f}, back azimuth {back_azimuth:.2f}, {distance_km:.1f} km)")


if __name__ == "__main__":
    main()
