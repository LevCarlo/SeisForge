"""Glue for this example: AFTAN .dat -> Eikonal input, Eikonal stack -> station curve.

This script belongs to the example, not to SeisForge. It holds only what this
example needs; adapt it to your own file names, periods and uncertainty
choices. run.sh calls it twice:

    python glue.py measurements              # output/aftan/*.dat -> output/measurements.nc
    python glue.py station-curve XX.A22      # output/eikonal/eikonal_stack.nc -> output/vph_disp.dat

Units: periods in s, velocities in km/s, coordinates in degrees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

STATIONS = Path("data/stations.txt")        # station longitude_deg latitude_deg
AFTAN_DIR = Path("output/aftan")             # written by `seisforge ant aftan`
MEASUREMENTS = Path("output/measurements.nc")  # read by `seisforge ant eikonal` (eikonal.yml)
STACK = Path("output/eikonal/eikonal_stack.nc")
STATION_CURVE = Path("output/vph_disp.dat")  # read by `seisforge inv` (obs.yaml)
PERIODS_S = [4, 5, 6, 7, 8]                  # must match eikonal.yml input.periods_s
# Assumed value: lower bound of the station-curve sigma. The Eikonal standard
# error of the mean treats the single-source estimates as independent, which
# they are not, so it is too small; 0.05 km/s is a guess at the map error of
# this sparse (40 km) array (README).
SIGMA_FLOOR_KM_S = 0.05


def read_stations() -> dict[str, tuple[float, float]]:
    rows = (line.split("#", 1)[0].split() for line in STATIONS.read_text().splitlines())
    return {row[0]: (float(row[1]), float(row[2])) for row in rows if row}


def read_aftan_dat(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """`# key: value` header lines, then a `# <column names>` line, then the table."""
    header, names = {}, None
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            continue
        text = line[1:].strip()
        if text.startswith("target_period_s"):
            names = text.split()
        elif ":" in text:
            key, value = text.split(":", 1)
            header[key.strip()] = value.strip()
    table = np.loadtxt(path, comments="#", ndmin=2)
    return header, {name: table[:, i] for i, name in enumerate(names)}


def to_periods(period_s: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Linear interpolation onto PERIODS_S; NaN outside the measured range (no extrapolation)."""
    keep = np.isfinite(period_s) & np.isfinite(value)
    period_s, value = period_s[keep], value[keep]
    target = np.asarray(PERIODS_S, dtype=float)
    out = np.full(target.shape, np.nan)
    if period_s.size == 0:
        return out
    if np.any(np.diff(period_s) <= 0):
        raise ValueError("AFTAN instantaneous periods are not increasing (bad pick).")
    inside = (target >= period_s[0]) & (target <= period_s[-1])
    out[inside] = np.interp(target[inside], period_s, value)
    return out


def measurements() -> None:
    """One directed path per AFTAN branch file, phase velocity at PERIODS_S."""
    stations = read_stations()
    rows = []
    for path in sorted(AFTAN_DIR.glob("*.dat")):
        header, columns = read_aftan_dat(path)
        # File names are <source>_<receiver>_<component>.SAC (README). The
        # negative lag travels from the receiver to the source: reverse the pair.
        source, receiver = header["input_file"].split("_")[:2]
        if header["selected_branch"] == "negative":
            source, receiver = receiver, source
        # Resample on the instantaneous period, the period the pick belongs to.
        velocity = to_periods(columns["instant_period_s"], columns["phase_velocity_km_s"])
        if np.any(np.isfinite(velocity)):
            rows.append((source, receiver, velocity))

    def coordinate(index: int, which: int) -> np.ndarray:
        values = np.array([stations[row[index]][which] for row in rows])
        # Longitudes in [-180, 180), the convention declared below and used by eikonal.yml.
        return np.mod(values + 180.0, 360.0) - 180.0 if which == 0 else values

    dataset = xr.Dataset(
        {"phase_velocity_km_s": (("path", "period"), np.array([row[2] for row in rows]))},
        coords={
            "period_s": ("period", np.array(PERIODS_S, dtype=np.int32)),
            "source_station": ("path", [row[0] for row in rows]),
            "receiver_station": ("path", [row[1] for row in rows]),
            "source_longitude_deg": ("path", coordinate(0, 0)),
            "source_latitude_deg": ("path", coordinate(0, 1)),
            "receiver_longitude_deg": ("path", coordinate(1, 0)),
            "receiver_latitude_deg": ("path", coordinate(1, 1)),
        },
        # The attributes the Eikonal reader requires (docs/ant/tomography.md, section 7).
        attrs={"schema_version": 1, "component": "ZZ", "longitude_convention": "-180_180"},
    )
    dataset.to_netcdf(MEASUREMENTS, engine="h5netcdf")
    print(f"wrote {MEASUREMENTS}: {len(rows)} directed paths, periods {PERIODS_S} s")


def station_curve(station: str) -> None:
    """Phase velocity and sigma of the Eikonal stack at one station."""
    longitude, latitude = read_stations()[station]
    with xr.open_dataset(STACK, engine="h5netcdf") as stack:
        stack = stack.load()
    grid_lon, grid_lat = stack["longitude"].values, stack["latitude"].values
    mask = stack["mask"].transpose("period", "latitude", "longitude").values.astype(bool)

    def bilinear(name: str) -> np.ndarray:
        """Linear in longitude and latitude between the 4 surrounding nodes; NaN unless all 4 are valid."""
        values = stack[name].transpose("period", "latitude", "longitude").values
        valid = ~mask & np.isfinite(values)
        col = int(np.clip(np.searchsorted(grid_lon, longitude) - 1, 0, grid_lon.size - 2))
        row = int(np.clip(np.searchsorted(grid_lat, latitude) - 1, 0, grid_lat.size - 2))
        tx = (longitude - grid_lon[col]) / (grid_lon[col + 1] - grid_lon[col])
        ty = (latitude - grid_lat[row]) / (grid_lat[row + 1] - grid_lat[row])
        total, usable = np.zeros(values.shape[0]), np.ones(values.shape[0], dtype=bool)
        for r, c, weight in ((row, col, (1 - tx) * (1 - ty)), (row, col + 1, tx * (1 - ty)),
                             (row + 1, col, (1 - tx) * ty), (row + 1, col + 1, tx * ty)):
            usable &= valid[:, r, c]
            total = total + weight * np.where(valid[:, r, c], values[:, r, c], 0.0)
        return np.where(usable, total, np.nan)

    velocity = bilinear("phase_velocity_km_s")
    # sigma: standard error of the stacked mean (stack variable velocity_sem_km_s),
    # raised to SIGMA_FLOOR_KM_S where it is smaller.
    sem = bilinear("velocity_sem_km_s")
    sigma = np.where(np.isfinite(sem) & (sem < SIGMA_FLOOR_KM_S), SIGMA_FLOOR_KM_S, sem)
    period_s = stack["period_s"].values.astype(int)
    keep = np.isfinite(velocity) & np.isfinite(sigma) & (sigma > 0)
    header = [
        f"phase velocity at {station} ({longitude}, {latitude}) from {STACK}, bilinear",
        f"sigma = max(velocity_sem_km_s, {SIGMA_FLOOR_KM_S}) km/s (floor assumed, glue.py)",
        f"dropped periods (masked or no sigma): {period_s[~keep].tolist()}",
        "period_s phase_velocity_km_s sigma_km_s",
    ]
    np.savetxt(STATION_CURVE, np.column_stack([period_s[keep], velocity[keep], sigma[keep]]),
               fmt=("%g", "%.5f", "%.5f"), header="\n".join(header))
    print(f"wrote {STATION_CURVE}: {int(keep.sum())} periods; dropped {period_s[~keep].tolist()}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["measurements"]:
        measurements()
    elif sys.argv[1:2] == ["station-curve"] and len(sys.argv) == 3:
        station_curve(sys.argv[2])
    else:
        sys.exit(__doc__)
