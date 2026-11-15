"""Write the synthetic cross-correlations of the AFTAN example (examples/02_aftan).

Run from the repository root:

    PYTHONPATH=src python tools/make_example_02_aftan.py [--overwrite]

Data (all values assumed, not from a real array): one virtual-source station
XX.S00 and three receivers XX.R01-XX.R03 due east of it at about 40, 80 and
120 km. Each CCF is the symmetric ZZ synthetic of `seisforge.ant.synthetic`
for the Rayleigh phase velocity of the shared truth model
(`tools/example_truth.py`), band 2-8 s, sampled every `DELTA_S` = 0.2 s
over +-`MAX_LAG_S` = 200 s, plus independent Gaussian noise of standard
deviation `NOISE_FRACTION` = 2 % of the CCF's peak amplitude
(`numpy.random.default_rng(SEED)`, SEED = 20261115). SAC headers: `b`,
`delta`, `dist` (km), `evlo/evla` (virtual source), `stlo/stla` (receiver),
`kcmpnm` = ZZ, `kevnm`/`kstnm` = source/receiver station.

Outputs (in `--out`, default `examples/02_aftan/data`): `ccf/*.SAC`,
`true_dispersion.dat` (period, phase velocity, group velocity of the truth;
the group velocity from `seisforge.ant.synthetic.group_velocity_from_phase`)
and `reference_phase_velocity.dat`, the rough reference curve AFTAN uses to
pick the 2 pi cycle of the phase: a constant `REFERENCE_VELOCITY_KM_S` =
2.9 km/s (assumed, a single regional average; the truth is 2.54-3.03 km/s
at 2-8 s). On these data constants of 2.8-3.0 km/s pick the right cycle,
2.6 and 3.2 km/s do not at 80-120 km.
Existing files are kept unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from obspy import Trace
from obspy.geodetics.base import gps2dist_azimuth

from example_truth import CURVE_PERIODS_S, ROOT, truth_phase_velocity
from seisforge.ant.synthetic import group_velocity_from_phase, symmetric_ccf_time_axis, synthetic_ccf

EXAMPLE = ROOT / "examples" / "02_aftan"
SOURCE = ("XX.S00", 110.0, 30.0)  # name, longitude, latitude (assumed)
# Receivers due east, about 40, 80 and 120 km away (assumed).
RECEIVERS = (("XX.R01", 110.415, 30.0), ("XX.R02", 110.83, 30.0), ("XX.R03", 111.245, 30.0))
MIN_PERIOD_S, MAX_PERIOD_S = 2.0, 8.0
DELTA_S = 0.2
MAX_LAG_S = 200.0
NOISE_FRACTION = 0.02
SEED = 20261115
REFERENCE_VELOCITY_KM_S = 2.9


def write_ccf(path: Path, source, receiver, time_s, data, distance_km) -> None:
    """Write one CCF as SAC with the header fields the workflows read."""
    trace = Trace(data=data.astype(np.float32))
    trace.stats.delta = DELTA_S
    trace.stats.sac = {
        "b": float(time_s[0]),
        "dist": float(distance_km),
        "evlo": source[1],
        "evla": source[2],
        "stlo": receiver[1],
        "stla": receiver[2],
        "kevnm": source[0],
        "kstnm": receiver[0].split(".")[1],
        "knetwk": receiver[0].split(".")[0],
        "kcmpnm": "ZZ",
    }
    trace.write(str(path), format="SAC")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)

    velocity = truth_phase_velocity(CURVE_PERIODS_S)
    time_s = symmetric_ccf_time_axis(MAX_LAG_S, DELTA_S)
    rng = np.random.default_rng(SEED)
    files = {}
    for receiver in RECEIVERS:
        distance_km = gps2dist_azimuth(SOURCE[2], SOURCE[1], receiver[2], receiver[1])[0] / 1000.0
        ccf = synthetic_ccf(
            time_s, distance_km, CURVE_PERIODS_S, velocity,
            min_period_s=MIN_PERIOD_S, max_period_s=MAX_PERIOD_S, pi_over_4=-1.0,
        )
        ccf = ccf + rng.normal(0.0, NOISE_FRACTION * np.abs(ccf).max(), ccf.size)
        files[args.out / "ccf" / f"{SOURCE[0]}_{receiver[0]}_ZZ.SAC"] = (receiver, ccf, distance_km)
    truth_file = args.out / "true_dispersion.dat"
    reference_file = args.out / "reference_phase_velocity.dat"
    existing = [str(p) for p in [*files, truth_file, reference_file] if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {', '.join(existing)}; pass --overwrite.")
    (args.out / "ccf").mkdir(parents=True, exist_ok=True)
    for path, (receiver, ccf, distance_km) in files.items():
        write_ccf(path, SOURCE, receiver, time_s, ccf, distance_km)
        print(f"wrote {path} ({distance_km:.1f} km)")

    band = (CURVE_PERIODS_S >= 1.5) & (CURVE_PERIODS_S <= 10.0)
    np.savetxt(
        truth_file,
        np.column_stack(
            [CURVE_PERIODS_S, velocity, group_velocity_from_phase(CURVE_PERIODS_S, velocity)]
        )[band],
        fmt=["%.2f", "%.4f", "%.4f"],
        header="\n".join(
            [
                "Truth of the synthetic CCFs (tools/make_example_02_aftan.py, tools/example_truth.py):",
                "fundamental-mode Rayleigh velocities of the examples/06_inversion truth model.",
                "period_s  phase_velocity_km_s  group_velocity_km_s",
            ]
        ),
    )
    print(f"wrote {truth_file}")
    reference_periods_s = np.arange(1.0, 10.0 + 1e-9, 1.0)
    np.savetxt(
        reference_file,
        np.column_stack([reference_periods_s, np.full(reference_periods_s.size, REFERENCE_VELOCITY_KM_S)]),
        fmt=["%.1f", "%.3f"],
        header="Rough reference phase velocity (assumed constant, tools/make_example_02_aftan.py)\n"
        "period_s  phase_velocity_km_s",
    )
    print(f"wrote {reference_file}")


if __name__ == "__main__":
    main()
