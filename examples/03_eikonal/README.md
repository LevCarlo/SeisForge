# 03 Eikonal: phase-velocity maps from station-pair measurements

Eikonal tomography turns the phase velocities measured between every pair of
stations into a phase-velocity map for each period. Here the measurements
are synthetic: 49 stations, a medium that is the same everywhere, so the map
should show the same known velocity everywhere inside the array
(`tools/make_example_03_eikonal.py`). Every config key:
`examples/reference_configs/eikonal.yml`; method: `docs/eikonal_workflow.md`.

## Input

| file | what it is |
|---|---|
| `data/measurements.nc` | phase velocity of every directed station pair at integer periods |
| `data/true_phase_velocity.dat` | the truth (only for checking) |
| `config.yml` | settings |

`measurements.nc` is the file `seisforge ant collect-dispersion` writes from
AFTAN results (see `full_chain`). It has dimensions `path` (one per directed
pair) and `period`, the variable `phase_velocity_km_s(path, period)`, the
integer `period_s(period)`, and per path `source_station`,
`receiver_station`, `source_longitude_deg`, `source_latitude_deg`,
`receiver_longitude_deg`, `receiver_latitude_deg`. Attributes
`schema_version: 1`, `component`, `longitude_convention`. Path 0:

```text
source_station XX.S00 (110.0, 30.0) -> receiver_station XX.S10 (110.2, 30.0)
period_s             2      3      4      5      6      7      8
phase_velocity_km_s  2.538  2.686  2.780  2.855  2.921  2.979  3.029
```

## Run

```bash
bash run.sh
```

About 4 s (4 processes). Delete `output/` before running again.

## Output

| file | what it is |
|---|---|
| `output/eikonal_stack.nc` | the maps: `phase_velocity_km_s(period, latitude, longitude)`, its standard error `velocity_sem_km_s`, measurement counts, `mask` |
| `output/event_fields/<period>s.nc` | travel-time and velocity field of every virtual source |
| `result.png` | the 5 s map, and map values against the truth |

## How to tell the result is right

`result.png` should look like `expected_result.png`. `run.sh` prints the
largest map error inside the array (dashed box, at least one station spacing
from the edge); on this data it is 0.013-0.019 km/s at every period
(acceptable: below 0.03 km/s, i.e. 1 %), and the interior median is within
0.002 km/s of the truth.

Common wrong-looking results:

- **Wrong values at the edge of the array and outside it** (up to 0.2 km/s
  here): the travel-time field is extrapolated there. Use only cells well
  inside the stations.
- **Empty cells (NaN)**, here a few in the middle and more at 2 and 8 s:
  quality control removed them (too few measurements, a source closer than
  3 wavelengths, poor azimuthal coverage). Normal; more stations help.
- **The whole map empty at a period**: `stacking.min_raw_measurements_per_cell`
  is larger than the number of stations, or the period is missing from
  `input.periods_s`/the dataset.
- **Stripes or rings around stations**: real-data measurement errors
  (cycle skips in AFTAN); check the AFTAN phase velocities first.

## Using your own data

1. `input.dataset`: your `measurements.nc` (from `seisforge ant collect-dispersion`).
2. `input.periods_s`: integer periods in your dataset.
3. `grid.*`: a box around your stations, and the cell size (about a
   quarter to a half of the station spacing).
4. `stacking.min_raw_measurements_per_cell`: below your number of stations
   (default 50).
5. `compute.workers`: number of processes.

If your dataset has an `snr` variable, set `input.min_snr` to drop weak
measurements. Other quality-control settings (`qc.*`) have defaults suited
to regional arrays; see the reference config.
