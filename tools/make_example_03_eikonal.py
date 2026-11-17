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
`measurements.nc`, the Eikonal input dataset (format:
docs/ant/tomography.md; written with `seisforge.ant.eikonal.io.write_netcdf`),
and `true_phase_velocity.dat`.
Existing files are kept unless `--overwrite` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from example_truth import ROOT, truth_phase_velocity
from seisforge.ant.eikonal.io import write_netcdf
from seisforge.provenance import software_provenance

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
    pairs = [(src, rcv) for src in stations() for rcv in stations() if src[0] != rcv[0]]
    dataset = xr.Dataset(
        {"phase_velocity_km_s": (("path", "period"), np.tile(velocity, (len(pairs), 1)))},
        coords={
            "path": np.arange(len(pairs)),
            "period": np.arange(len(PERIODS_S)),
            "period_s": ("period", np.asarray(PERIODS_S, dtype=np.int32)),
            "source_station": ("path", np.array([src[0] for src, _ in pairs])),
            "receiver_station": ("path", np.array([rcv[0] for _, rcv in pairs])),
            "source_longitude_deg": ("path", np.array([src[1] for src, _ in pairs])),
            "source_latitude_deg": ("path", np.array([src[2] for src, _ in pairs])),
            "receiver_longitude_deg": ("path", np.array([rcv[1] for _, rcv in pairs])),
            "receiver_latitude_deg": ("path", np.array([rcv[2] for _, rcv in pairs])),
        },
        attrs={
            "schema_version": 1,
            "component": "ZZ",
            "longitude_convention": "-180_180",
            "source": "tools/make_example_03_eikonal.py: exact travel times in a laterally homogeneous medium",
            **software_provenance(),
        },
    )
    dataset["phase_velocity_km_s"].attrs["units"] = "km s-1"
    truth_file = args.out / "true_phase_velocity.dat"
    if truth_file.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {truth_file}; pass --overwrite.")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"wrote {write_netcdf(dataset, args.out / 'measurements.nc', overwrite=args.overwrite, compression_level=4)}")
    np.savetxt(
        truth_file,
        np.column_stack([PERIODS_S, velocity]),
        fmt=["%d", "%.4f"],
        header="Truth (tools/make_example_03_eikonal.py, tools/example_truth.py)\nperiod_s  phase_velocity_km_s",
    )
    print(f"wrote {truth_file}")


if __name__ == "__main__":
    main()
