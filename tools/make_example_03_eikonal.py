"""Write the synthetic travel-time data of the Eikonal example (examples/03_eikonal).

Run from the repository root:

    PYTHONPATH=src python tools/make_example_03_eikonal.py [--overwrite]

Data (all values assumed, not from a real array): 7 x 7 stations XX.Sij on a
regular grid, longitude 110.0-111.2 deg in steps of 0.2 deg and latitude
30.0-31.08 deg in steps of 0.18 deg (about 19-20 km apart). Every directed
station pair gets the phase velocity of the shared truth model
(`tools/example_truth.py`) at the integer periods 2-8 s; the medium is
laterally homogeneous, so the exact travel time along any path is
distance / c(T) and Eikonal tomography should return c(T) everywhere inside
the array. No noise is added: the example checks the tomography alone
(`full_chain` adds the AFTAN measurement).

Outputs (in `--out`, default `examples/03_eikonal/data`):
`measurements.nc`, the Eikonal input dataset in the same format
`seisforge ant collect-dispersion` writes (built with
`seisforge.ant.glue.io.measurement_dataset`), and `true_phase_velocity.dat`.
Existing files are kept unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from example_truth import ROOT, truth_phase_velocity
from seisforge.ant.glue.io import measurement_dataset, write_measurement_dataset

EXAMPLE = ROOT / "examples" / "03_eikonal"
PERIODS_S = (2, 3, 4, 5, 6, 7, 8)
LONGITUDES_DEG = np.round(110.0 + 0.2 * np.arange(7), 3)
LATITUDES_DEG = np.round(30.0 + 0.18 * np.arange(7), 3)


def stations():
    """(name, longitude, latitude) of the 7 x 7 grid."""
    return [
        (f"XX.S{i}{j}", float(lon), float(lat))
        for j, lat in enumerate(LATITUDES_DEG)
        for i, lon in enumerate(LONGITUDES_DEG)
    ]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args(argv)

    velocity = truth_phase_velocity(np.asarray(PERIODS_S, dtype=float))
    paths = [
        {
            "source_station": src[0],
            "receiver_station": rcv[0],
            "source_longitude_deg": src[1],
            "source_latitude_deg": src[2],
            "receiver_longitude_deg": rcv[1],
            "receiver_latitude_deg": rcv[2],
            "phase_velocity_km_s": velocity,
            "aftan_file": "synthetic",
            "aftan_branch": "synthetic",
        }
        for src in stations()
        for rcv in stations()
        if src[0] != rcv[0]
    ]
    dataset = measurement_dataset(
        periods_s=PERIODS_S,
        paths=paths,
        attrs={
            "schema_version": 1,
            "component": "ZZ",
            "longitude_convention": "-180_180",
            "source": "tools/make_example_03_eikonal.py: exact travel times in a laterally homogeneous medium",
        },
        include_snr=False,
        include_amplitude=False,
    )
    truth_file = args.out / "true_phase_velocity.dat"
    if truth_file.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {truth_file}; pass --overwrite.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"wrote {write_measurement_dataset(dataset, args.out / 'measurements.nc', overwrite=args.overwrite)}")
    np.savetxt(
        truth_file,
        np.column_stack([PERIODS_S, velocity]),
        fmt=["%d", "%.4f"],
        header="Truth (tools/make_example_03_eikonal.py, tools/example_truth.py)\nperiod_s  phase_velocity_km_s",
    )
    print(f"wrote {truth_file}")


if __name__ == "__main__":
    main()
