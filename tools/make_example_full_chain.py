"""Write the synthetic data of the end-to-end example (examples/full_chain).

Run from the repository root:

    PYTHONPATH=src python tools/make_example_full_chain.py [--overwrite]

Truth: the model of `examples/06_inversion` (`tools/example_truth.py`), the
same at every location (laterally homogeneous), so Vs(z) under every station
is that model.

Array (assumed, not a real network): 5 x 5 stations XX.Aij, longitude
110.0-111.6 deg in steps of 0.4 deg, latitude 30.0-31.44 deg in steps of
0.36 deg (about 38-40 km apart). The spacing is set by the Eikonal quality
control: cells closer than about 70 km to a virtual source are rejected
(travel-time curvature above 0.005 s/km^2, sources within 3 wavelengths), so
the inner stations need sources 80 km and more away; a 20 km spacing left
the centre station without any valid period. For every station pair (the alphabetically
first station is the virtual source) a symmetric ZZ CCF from
`seisforge.ant.synthetic` with the truth's Rayleigh phase velocity, band
2-8 s, sampled every `DELTA_S` = 0.4 s over +-`MAX_LAG_S` = 150 s, plus
independent Gaussian noise of standard deviation `NOISE_FRACTION` = 2 % of
the CCF peak (`numpy.random.default_rng(SEED)`).

Rayleigh H/V: SeisForge does not measure H/V from data yet, so the H/V the
inversion uses is synthetic: the truth's ellipticity at `HV_PERIODS_S`
(0.8-8 s) plus Gaussian noise of standard deviation `HV_SIGMA` = 0.02
(same generator, after the CCFs).

Outputs (in `--out`, default `examples/full_chain/data`): `ccf/<src>_<rcv>_ZZ.SAC`
(headers b, delta, dist, evlo/evla, stlo/stla, kcmpnm), `stations.txt`
(`station longitude_deg latitude_deg`), `hv.dat` (`period hv sigma`),
`reference_phase_velocity.dat` (constant 2.9 km/s, the rough AFTAN reference
of examples/02_aftan), `true_phase_velocity.dat` and `true_model.dat` (for
checking). Existing files are kept unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

import numpy as np
from obspy import Trace
from obspy.geodetics.base import gps2dist_azimuth

from example_truth import CURVE_PERIODS_S, INVERSION_EXAMPLE, ROOT, truth_hv, truth_phase_velocity
from seisforge.ant.synthetic import symmetric_ccf_time_axis, synthetic_ccf

EXAMPLE = ROOT / "examples" / "full_chain"
LONGITUDES_DEG = np.round(110.0 + 0.4 * np.arange(5), 3)
LATITUDES_DEG = np.round(30.0 + 0.36 * np.arange(5), 3)
MIN_PERIOD_S, MAX_PERIOD_S = 2.0, 8.0
DELTA_S = 0.4
MAX_LAG_S = 150.0
NOISE_FRACTION = 0.02
HV_PERIODS_S = np.array([0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0])
HV_SIGMA = 0.02
REFERENCE_VELOCITY_KM_S = 2.9
SEED = 20261115


def stations():
    return [
        (f"XX.A{i}{j}", float(lon), float(lat))
        for j, lat in enumerate(LATITUDES_DEG)
        for i, lon in enumerate(LONGITUDES_DEG)
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)
    if (args.out / "stations.txt").exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite the data in {args.out}; pass --overwrite.")
    (args.out / "ccf").mkdir(parents=True, exist_ok=True)

    velocity = truth_phase_velocity(CURVE_PERIODS_S)
    time_s = symmetric_ccf_time_axis(MAX_LAG_S, DELTA_S)
    rng = np.random.default_rng(SEED)
    for source, receiver in itertools.combinations(stations(), 2):
        distance_km = gps2dist_azimuth(source[2], source[1], receiver[2], receiver[1])[0] / 1000.0
        ccf = synthetic_ccf(
            time_s, distance_km, CURVE_PERIODS_S, velocity,
            min_period_s=MIN_PERIOD_S, max_period_s=MAX_PERIOD_S, pi_over_4=-1.0,
        )
        ccf = ccf + rng.normal(0.0, NOISE_FRACTION * np.abs(ccf).max(), ccf.size)
        trace = Trace(data=ccf.astype(np.float32))
        trace.stats.delta = DELTA_S
        trace.stats.sac = {
            "b": float(time_s[0]), "dist": distance_km,
            "evlo": source[1], "evla": source[2], "stlo": receiver[1], "stla": receiver[2],
            "kcmpnm": "ZZ",
        }
        trace.write(str(args.out / "ccf" / f"{source[0]}_{receiver[0]}_ZZ.SAC"), format="SAC")

    with (args.out / "stations.txt").open("w", encoding="utf-8") as stream:
        stream.write("# station longitude_deg latitude_deg (synthetic array, tools/make_example_full_chain.py)\n")
        for name, lon, lat in stations():
            stream.write(f"{name} {lon:.3f} {lat:.3f}\n")

    hv = truth_hv(HV_PERIODS_S) + rng.normal(0.0, HV_SIGMA, HV_PERIODS_S.size)
    np.savetxt(
        args.out / "hv.dat",
        np.column_stack([HV_PERIODS_S, hv, np.full(HV_PERIODS_S.size, HV_SIGMA)]),
        fmt=["%.2f", "%.4f", "%.3f"],
        header="Synthetic Rayleigh H/V of the truth model (tools/make_example_full_chain.py);\n"
        f"Gaussian noise sigma {HV_SIGMA}, numpy default_rng({SEED}) after the CCF noise.\n"
        "period_s  rayleigh_hv  sigma",
    )
    reference_periods_s = np.arange(1.0, 10.0 + 1e-9, 1.0)
    np.savetxt(
        args.out / "reference_phase_velocity.dat",
        np.column_stack([reference_periods_s, np.full(reference_periods_s.size, REFERENCE_VELOCITY_KM_S)]),
        fmt=["%.1f", "%.3f"],
        header="Rough reference phase velocity for AFTAN cycle selection (assumed constant)\n"
        "period_s  phase_velocity_km_s",
    )
    periods = np.arange(2.0, 9.0)
    np.savetxt(
        args.out / "true_phase_velocity.dat",
        np.column_stack([periods, truth_phase_velocity(periods)]),
        fmt=["%d", "%.4f"],
        header="Truth (tools/example_truth.py)\nperiod_s  phase_velocity_km_s",
    )
    shutil.copy(INVERSION_EXAMPLE / "data" / "true_model.dat", args.out / "true_model.dat")
    print(f"wrote {len(list((args.out / 'ccf').glob('*.SAC')))} CCFs and the tables in {args.out}")


if __name__ == "__main__":
    main()
