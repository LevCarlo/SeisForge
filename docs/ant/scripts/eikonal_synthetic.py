"""Figure eikonal_synthetic.png: Eikonal tomography of a medium with a known answer.

Medium (assumed, illustration): phase velocity increasing linearly to the
north, v(y) = V0 + G (y - y0). For a linear velocity profile the Eikonal
equation has the closed-form solution (the rays are circular arcs)

    t(p1, p2) = arccosh(1 + G^2 |p1 - p2|^2 / (2 v(p1) v(p2))) / |G|,

which gives exact phase travel times between every pair of stations, with
no ray tracing. Coordinates are the grid's own arc lengths (x along the
parallel, y along the meridian) of a near-equatorial grid treated as flat;
the 0.1 % metric differences this ignores are far below the errors shown.

Each station in turn is the virtual source. Its travel times to all other
stations are turned into the path-average velocities the package expects
(geodesic distance / t) and passed to `compute_event_field` (interpolation,
gradient, QC with package defaults), then all sources are combined with
`stack_event_fields`. (a) one source's travel-time surface; (b) its apparent
velocity |grad t|^-1 with the near-source mask; (c) the stacked velocity;
(d) stacked minus true velocity.
"""

import numpy as np
import xarray as xr
from pyproj import Geod

import _style
from _style import plt
from seisforge.ant.eikonal import compute_event_field, stack_event_fields
from seisforge.ant.eikonal.grid import build_geographic_grid
from seisforge.ant.eikonal.models import (
    CycleSkipConfig,
    EikonalMethodConfig,
    EikonalQCConfig,
    GridConfig,
    InterpolationConfig,
    StackingConfig,
)

PERIOD_S = 20
V0_KM_S, GRADIENT_PER_S = 3.4, 0.0015  # v = V0 + G (y - y_mid); assumed values
LON0, LAT0, SIZE_DEG = 100.0, -2.5, 5.0
STATION_COUNT = 140

geod = Geod(ellps="WGS84")
grid = build_geographic_grid(GridConfig(
    longitude_convention="-180_180",
    min_longitude_deg=LON0, max_longitude_deg=LON0 + SIZE_DEG,
    min_latitude_deg=LAT0, max_latitude_deg=LAT0 + SIZE_DEG,
    longitude_step_deg=0.2, latitude_step_deg=0.2,
))
y_mid_km = grid.y_km[-1] / 2


def flat_xy(longitude, latitude):
    """Arc length east along the parallel and north along the meridian (km)."""
    longitude, latitude = np.broadcast_arrays(np.asarray(longitude, float), np.asarray(latitude, float))
    _, _, east_m = geod.inv(np.full(longitude.shape, LON0), latitude, longitude, latitude)
    _, _, north_m = geod.inv(longitude, np.full(latitude.shape, LAT0), longitude, latitude)
    return east_m / 1000.0, north_m / 1000.0


def velocity(y_km):
    return V0_KM_S + GRADIENT_PER_S * (y_km - y_mid_km)


def travel_time(x1, y1, x2, y2):
    v1, v2 = velocity(y1), velocity(y2)
    squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
    return np.arccosh(1 + GRADIENT_PER_S**2 * squared / (2 * v1 * v2)) / GRADIENT_PER_S


rng = np.random.default_rng(3)
station_lon = LON0 + 0.2 + (SIZE_DEG - 0.4) * rng.random(STATION_COUNT)
station_lat = LAT0 + 0.2 + (SIZE_DEG - 0.4) * rng.random(STATION_COUNT)
station_x, station_y = flat_xy(station_lon, station_lat)

configs = dict(
    grid=grid,
    cycle_skip=CycleSkipConfig(enabled=False),  # exact times: nothing to unwrap
    interpolation=InterpolationConfig(),
    method=EikonalMethodConfig(),
    qc=EikonalQCConfig(),
)
fields = []
for source in range(STATION_COUNT):
    others = np.arange(STATION_COUNT) != source
    t = travel_time(station_x[source], station_y[source], station_x[others], station_y[others])
    _, _, distance_m = geod.inv(np.full(others.sum(), station_lon[source]), np.full(others.sum(), station_lat[source]),
                                station_lon[others], station_lat[others])
    fields.append(compute_event_field(
        source_station=f"S{source:03d}",
        source_longitude_deg=station_lon[source], source_latitude_deg=station_lat[source],
        receiver_longitude_deg=station_lon[others], receiver_latitude_deg=station_lat[others],
        phase_velocity_km_s=distance_m / 1000.0 / t, period_s=PERIOD_S, **configs,
    ))

dims = ("source", "latitude", "longitude")
event_fields = xr.Dataset(
    {
        "slowness_s_per_km": (dims, np.stack([f.slowness_s_per_km for f in fields])),
        "great_circle_azimuth_deg": (dims, np.stack([f.great_circle_azimuth_deg for f in fields])),
        "rejection_reason": (dims, np.stack([f.rejection_reason for f in fields])),
        "source_accepted": (("source",), np.array([f.source_accepted for f in fields])),
    },
    coords={"latitude": grid.latitude_deg, "longitude": grid.longitude_deg},
)
stack = stack_event_fields(event_fields, StackingConfig(), include_helmholtz=False)
grid_x, grid_y = flat_xy(grid.longitude_2d_deg, grid.latitude_2d_deg)
true_velocity = velocity(grid_y)
stacked = stack["phase_velocity_km_s"].values
error = stacked - true_velocity

example = int(np.argmin((station_lon - (LON0 + 1.2)) ** 2 + (station_lat - (LAT0 + 1.5)) ** 2))
field = fields[example]
accepted = field.rejection_reason == 0

fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2), sharex=True, sharey=True)
extent = dict(shading="auto")
lon, lat = grid.longitude_deg, grid.latitude_deg

ax = axes[0, 0]
contours = ax.contour(lon, lat, field.travel_time_s, levels=np.arange(0, 400, 20), colors="k", linewidths=0.7)
ax.clabel(contours, fontsize=7, fmt="%d s")
ax.plot(station_lon, station_lat, "^", ms=4, color=_style.ALTERNATIVE)
ax.plot(station_lon[example], station_lat[example], "*", ms=14, color=_style.HIGHLIGHT)
ax.set_title(f"(a) travel-time surface of one source, T = {PERIOD_S} s")

ax = axes[0, 1]
masked = np.where(accepted, field.apparent_velocity_km_s, np.nan)
mesh = ax.pcolormesh(lon, lat, masked, vmin=3.0, vmax=3.8, **extent)
ax.contour(lon, lat, field.travel_time_s, levels=np.arange(0, 400, 20), colors="w", linewidths=0.4)
ax.plot(station_lon[example], station_lat[example], "*", ms=14, color=_style.HIGHLIGHT)
fig.colorbar(mesh, ax=ax, label="km/s")
ax.set_title("(b) $1/|\\nabla\\tau|$ of this source, QC-accepted cells")

ax = axes[1, 0]
mesh = ax.pcolormesh(lon, lat, stacked, vmin=3.0, vmax=3.8, **extent)
ax.contour(lon, lat, true_velocity, levels=np.arange(3.0, 3.85, 0.1), colors="k", linewidths=0.6,
           linestyles="--")
fig.colorbar(mesh, ax=ax, label="km/s")
accepted_sources = int(sum(f.source_accepted for f in fields))
ax.set_title(f"(c) stack of {accepted_sources} sources; dashed: true velocity")

ax = axes[1, 1]
mesh = ax.pcolormesh(lon, lat, error, vmin=-0.08, vmax=0.08, cmap="RdBu_r", **extent)
fig.colorbar(mesh, ax=ax, label="km/s")
ax.set_title("(d) stacked minus true (colour scale saturates)")

for ax in axes.flat:
    ax.set_aspect("equal")
    ax.grid(False)
for ax in axes[1]:
    ax.set_xlabel("longitude (°)")
for ax in axes[:, 0]:
    ax.set_ylabel("latitude (°)")
fig.tight_layout()
_style.save(fig, __file__)

finite = np.isfinite(error)
interior = finite & (grid_x > 100) & (grid_x < grid_x.max() - 100) & (grid_y > 100) & (grid_y < grid_y.max() - 100)
print(f"stacked cells: {finite.sum()} of {finite.size}; error median {np.nanmedian(error):+.4f}, "
      f"RMS {np.sqrt(np.nanmean(error[finite] ** 2)):.4f}, interior RMS "
      f"{np.sqrt(np.nanmean(error[interior] ** 2)):.4f} km/s")
