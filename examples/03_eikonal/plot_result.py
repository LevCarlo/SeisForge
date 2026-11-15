"""Plot the Eikonal example: one phase-velocity map, and the map values against the truth.

Run from the example folder after `seisforge ant eikonal -c config.yml` (run.sh does both):

    python plot_result.py [--stack output/eikonal_stack.nc] [--period 5] [-o result.png]

It prints, per period, the largest difference from the truth inside the
array (at least one station spacing from its edge, the box --interior),
which README.md asks you to check. Plotting and printing only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stack", type=Path, default=Path("output/eikonal_stack.nc"))
    parser.add_argument("--measurements", type=Path, default=Path("data/measurements.nc"))
    parser.add_argument("--truth", type=Path, default=Path("data/true_phase_velocity.dat"), help="omitted if missing")
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument(
        "--interior", type=float, nargs=4, default=(110.2, 111.0, 30.18, 30.9),
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("result.png"))
    args = parser.parse_args(argv)

    stack = xr.open_dataset(args.stack, engine="h5netcdf")
    data = xr.open_dataset(args.measurements, engine="h5netcdf")
    truth = np.loadtxt(args.truth) if args.truth.exists() else None
    lon0, lon1, lat0, lat1 = args.interior
    interior = stack.sel(longitude=slice(lon0, lon1), latitude=slice(lat0, lat1))

    fig, (ax_map, ax_c) = plt.subplots(1, 2, figsize=(10, 4.2))
    velocity = stack["phase_velocity_km_s"].sel(period=args.period)
    mesh = ax_map.pcolormesh(stack["longitude"], stack["latitude"], velocity, shading="nearest", cmap="RdBu")
    fig.colorbar(mesh, ax=ax_map, label="phase velocity (km/s)")
    ax_map.plot(data["source_longitude_deg"], data["source_latitude_deg"], "k^", ms=4)
    ax_map.add_patch(plt.Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, ls="--"))
    ax_map.set(xlabel="Longitude", ylabel="Latitude", title=f"{args.period} s (dashed: interior)")

    periods = stack["period"].values
    values = interior["phase_velocity_km_s"].values.reshape(periods.size, -1)
    ax_c.plot(periods, np.nanmedian(values, axis=1), "o", color="C3", label="map median (interior)")
    ax_c.fill_between(periods, np.nanmin(values, axis=1), np.nanmax(values, axis=1), color="C3", alpha=0.3,
                      label="map min-max (interior)")
    if truth is not None:
        ax_c.plot(truth[:, 0], truth[:, 1], "k-", label="truth")
        for period, row in zip(periods, values, strict=True):
            c = np.interp(period, truth[:, 0], truth[:, 1])
            print(f"{period} s: {np.isfinite(row).sum()} of {row.size} interior cells, "
                  f"max |map - truth| {np.nanmax(np.abs(row - c)):.4f} km/s")
    ax_c.set(xlabel="Period (s)", ylabel="Phase velocity (km/s)")
    ax_c.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
