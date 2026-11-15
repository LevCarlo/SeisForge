# Surface-Wave Eikonal and Helmholtz Tomography

!!! tip "Runnable example"
    `examples/03_eikonal` runs this workflow on synthetic data with a known answer
    (`bash run.sh`); see the [examples overview](examples.md). The commands below
    use the full reference config in `examples/reference_configs/`.


SeisForge builds local phase-slowness maps from directed virtual-source travel
time fields and stacks them with azimuth-balanced event weights. The workflow
implements isotropic ambient-noise Eikonal tomography and optional Helmholtz
amplitude correction. Azimuthal anisotropy is not yet implemented.

## Input Dataset

The Eikonal workflow starts from an already collected and integer-period
resampled Xarray Dataset. `seisforge ant collect-dispersion` builds it from
AFTAN output (see [Collect AFTAN measurements](#collect-aftan-measurements));
any other tool may write it directly if it follows this contract.

The NetCDF dataset must have `path` and `period` dimensions and these values:

| Name | Dimensions | Description |
| --- | --- | --- |
| `period_s` | `period` | Unique, increasing integer periods in seconds |
| `source_station` | `path` | Physical virtual-source identifier |
| `receiver_station` | `path` | Physical receiver identifier |
| `source_longitude_deg` | `path` | Source longitude |
| `source_latitude_deg` | `path` | Source latitude |
| `receiver_longitude_deg` | `path` | Receiver longitude |
| `receiver_latitude_deg` | `path` | Receiver latitude |
| `phase_velocity_km_s` | `path, period` | Resampled path-average phase velocity |

Optional variables are `snr(path, period)`, `measurement_valid(path, period)`,
`amplitude(path, period)`, and `distance_km(path)`. `amplitude` becomes required
when `helmholtz.enabled` is true and its finite values must be positive. Its
absolute normalization is arbitrary because the correction uses
`Laplacian(A)/A`. Missing period measurements
are represented by NaN. If `distance_km` is supplied, SeisForge checks it
against the WGS84 coordinates to within 0.5 km.

Required global attributes are:

```text
schema_version: 1
component: ZZ
longitude_convention: -180_180
```

When `input.min_snr` is not null, the dataset must also define `snr` and the
global attribute `snr_scale` as either `linear` or `db`. The threshold is
interpreted in that declared scale.

Each directed source/receiver pair may occur only once. Coordinates associated
with a station identifier must be consistent across every path. Input and grid
longitude conventions must match explicitly.

## Collect AFTAN Measurements

```bash
seisforge ant collect-dispersion 'FTAN/**/*.dat' \
    --periods 8:40:2 --stations stations.txt -o measurements.nc
```

Quote the glob so the command, not the shell, expands it (`**` recurses).
`--periods` takes `start:stop:step` (stop included) or `8,10,12`. The output
then goes into `input.dataset` of the Eikonal config. What the command does:

- **Which files.** Each `.dat` header names its AFTAN stage; only
  `--aftan-stage` files are used (`basic`, default, or `pmf`), so the
  `*.pmf.dat` and basic files can sit in the same folders.
- **Which station is the source.** Station names and the component come from
  the SAC file name recorded in the `.dat` header (`input_file`), matched
  against `--name-pattern` (default
  `{source_station}_{receiver_station}_{component}*`; use `--component ZZ`
  when the name has no component). The stored pair is source -> receiver for
  positive lags. A `negative`-branch file is the reversed path
  (receiver -> source) with the reciprocal component, so measuring both
  branches gives both directed paths. A symmetric `stack`-branch file is one
  path; `--mirror-stack` also uses it for the reversed path. A directed path
  found twice is an error.
- **Coordinates.** Either `--stations FILE` with lines
  `station longitude_deg latitude_deg`, or `--sac-dir DIR`, which finds each
  `input_file` below `DIR` and reads `evlo/evla` (stored source) and
  `stlo/stla` (stored receiver). Longitudes are written in
  `--longitude-convention` (default `-180_180`; must match the grid).
- **Resampling.** Phase velocity and SNR are interpolated linearly in period
  onto the integer periods, between neighbouring AFTAN samples, without
  extrapolation (NaN outside the measured range). The period axis is the
  instantaneous period by default (`--period-axis instant`), the period the
  AFTAN phase velocity belongs to; `--period-axis target` uses the filter
  centre periods. A curve whose period axis is not increasing, or that does
  not reach any requested period, is skipped and reported.
- **SNR.** `snr(path, period)` is always written; `snr_scale` is `linear` or
  `db` from the AFTAN column name (`snr` or `snr_db`; mixing them is an
  error), so `input.min_snr` works directly.
- **Amplitude.** `--amplitude` also writes the AFTAN amplitude for the
  Helmholtz correction. It is only meaningful if all CCFs of a virtual source
  share one amplitude normalization.

The file records these choices in its attributes, and each path keeps the
`aftan_file` and `aftan_branch` it came from. The command validates the
dataset with the Eikonal reader before writing it, and refuses to replace an
existing file without `--overwrite`.

## Run The Workflow

Start from the example configuration:

```bash
seisforge ant eikonal -c examples/reference_configs/eikonal.yml
```

The two computational stages can be run independently:

```bash
# Interpolate travel-time fields, apply single-source QC, and take gradients.
seisforge ant eikonal -c eikonal.yml --stage fields

# Re-stack existing event-field files after changing stacking parameters.
seisforge ant eikonal -c eikonal.yml --stage stack
```

Relative paths in YAML are resolved relative to the configuration file.
Existing outputs raise an error unless `output.overwrite` is true.

## Single-Source Fields

For each period, SeisForge writes `event_fields/<period>s.nc` with dimensions
`source, latitude, longitude`. The main variables are travel time, travel-time
Laplacian, slowness, apparent velocity, propagation azimuth, great-circle
azimuth, azimuth deflection, and `rejection_reason`.

The rejection flags are:

| Value | Meaning |
| ---: | --- |
| 0 | accepted |
| 1 | interpolation unstable under the configured RBF smoothing comparison |
| 2 | insufficient nearby/quadrant station coverage |
| 3 | apparent velocity outside configured bounds |
| 4 | non-finite field or adjacent cell |
| 5 | source distance shorter than the far-field wavelength buffer |
| 6 | travel-time Laplacian exceeds the curvature threshold |
| 7 | configured grid boundary mask |
| 8 | insufficient source measurements or interpolation failure |

The `source_accepted` variable applies the configured valid-grid fraction and
count to the whole virtual-source field. Rejected fields remain in the file for
diagnostics but do not enter stacking.

## Helmholtz Amplitude Correction

Enable the correction for datasets containing amplitude measurements:

```yaml
helmholtz:
  enabled: true
  min_period_s: 0.0
  min_amplitude: 0.0
  max_relative_interpolation_difference: 0.01
  curvature_reference_velocity_km_s: 4.0
```

The amplitude observations for each virtual source are interpolated on the same
grid and with the same backend as travel time. Unlike travel time, no synthetic
value is inserted at the source. When interpolation stability checking is
enabled, the primary and comparison amplitude fields are compared using the
Helmholtz-specific relative threshold above.

For angular frequency `omega = 2*pi/period`, the corrected phase slowness is

```text
s_H^2 = |gradient(travel_time)|^2 - Laplacian(amplitude)/(amplitude*omega^2)
c_H = 1/s_H
```

Derivatives are taken on the regular longitude/latitude grid with metric
spacing: WGS84 geodesic arc length along each latitude row (`x`, east) and along
the central meridian (`y`, north). In these coordinates the surface Laplacian
is

```text
Laplacian(f) = f_xx + f_yy - (tan(lat) / N) f_y
```

where `N` is the WGS84 prime-vertical radius of curvature. The last term comes
from parallels converging toward the pole: the parallel radius `r = N cos(lat)`
changes along `y` as `dr/dy = -sin(lat)`. It is applied to both the
travel-time Laplacian (curvature QC) and the amplitude Laplacian (Helmholtz
correction).

```yaml
eikonal:
  laplacian_metric: ellipsoidal   # default; locally_flat = compatibility only
```

`locally_flat` drops the metric term and reproduces earlier SeisForge results.
Against the analytic Laplacian of epicentral distance, `1/(R tan(D/R))`, the
ellipsoidal form is accurate to 0.12% at 0-64 deg latitude. The flat form is off
by a median 6.5% at 35-39 deg N and up to 25% at 60-64 deg N for a source
300-700 km due south. The practical effect is small: the travel-time Laplacian
stays far below the default curvature threshold (`5e-3 s/km^2`), and Helmholtz
velocities for geometric-spreading amplitudes change by less than 0.5 m/s at
20-50 s. Event-field files record `laplacian_metric` and `laplacian_formula`.

The curvature QC follows surfpy's criterion
`abs(Laplacian(A)/(A*omega^2)) <= 1/c_ref^2`, where `c_ref` is
`curvature_reference_velocity_km_s`. The event-field files retain amplitude,
its Laplacian, the correction term, corrected slowness and velocity, and an
independent `helmholtz_rejection_reason`:

| Value | Meaning |
| ---: | --- |
| 0 | accepted |
| 1 | correction not computed for this source/period |
| 2 | corresponding Eikonal value rejected |
| 3 | amplitude interpolation unstable |
| 4 | amplitude non-positive/non-finite or adjacent to an invalid cell |
| 5 | amplitude curvature correction exceeds the configured limit |
| 6 | corrected squared slowness is non-positive |

The stack retains the ordinary variables and, for periods where correction was
applied, adds variables prefixed with `helmholtz_`. Thus the corrected and
uncorrected maps can be compared without rerunning event-field generation.

## Isotropic Stack

The final `eikonal_stack.nc` contains period-dependent phase velocity,
slowness, slowness standard deviation, velocity standard error, raw and
quality-controlled measurement counts, effective measurement count,
azimuthal coverage, and mask.

Weights are inverse to the number of valid measurements within the configured
azimuth window. With `require_peer: true`, an isolated propagation direction
is excluded. The implementation counts only valid neighboring events, fixing
the ambiguous indexing in the reference surfpy helper. Stacking is performed
in slowness, followed by the configured iterative standard-deviation outlier
rejection.

## Extract A Station Curve For The Inversion

```bash
seisforge ant extract-station-curve output/eikonal_stack.nc \
    --station STA001 --stations stations.txt -o vph_disp.dat
```

The command samples the stack at a station (or at `--longitude/--latitude`)
and writes the three-column `period_s phase_velocity_km_s sigma_km_s` file
that `observations.dispersion.file` of `seisforge inv` reads. Sampling is
bilinear between the four surrounding grid nodes (`--interpolation nearest`
takes the nearest node); a period is dropped when a needed node is masked or
has no uncertainty, and the dropped periods are listed in the header.
`--field helmholtz` uses the Helmholtz-corrected stack.

**Which sigma.** The stack holds two spreads. `slowness_std_s_per_km` is the
weighted standard deviation of the single-source slowness values: how much
the individual fields scatter, which includes azimuthal anisotropy and
source-specific errors and does not shrink with more sources.
`velocity_sem_km_s` is the standard error of the weighted mean velocity,
standard deviation / sqrt(n_eff) with the effective count
n_eff = 1 / sum(w^2). The inversion fits the stacked mean, so the default is
`--sigma sem`, the uncertainty of that mean, as Lin et al. (2009) report it.
It assumes independent single-source fields; they share stations and
interpolation, so it underestimates the error, most where many sources
overlap. Two things compensate:

- `--sigma-floor-km-s` (default 0.01 km/s, an assumed value, not from a
  source) raises smaller values to the floor; the periods affected are listed
  in the header.
- A sampled noise scale in the inversion (`noise_parameter`, see
  [Likelihood](inv_usage.md#likelihood)) lets the data rescale all sigmas;
  the standard error then sets only their relative size.

`--sigma std` uses the single-source spread instead, converted to velocity
as std_s / s^2 (first order in std_s / s), a more conservative choice when
anisotropy or bad sources dominate.

The header records the stack file, location, field, interpolation, sigma
source and definition, floor, and software provenance. An existing output is
replaced only with `--overwrite`.

## Numerical Backends

The default backend is SciPy `RBFInterpolator` with coordinates projected to
local kilometres. `comparison_smoothing` is RBF regularization; it is not GMT
surface tension.

An optional PyGMT backend reproduces surfpy's block-median plus GMT `surface`
sequence. PyGMT and the GMT shared library must both be installed:

```bash
python -m pip install -e ".[gmt]"
```

Then select:

```yaml
interpolation:
  method: gmt_surface
  block_reduce: median  # median or none
  tension: 0.0
  stability_check:
    enabled: true
    comparison_tension: 0.2
    max_travel_time_difference_s: 2.0
```

The primary and comparison GMT fields use the same observations and grid but
different tension values. Their absolute travel-time difference drives
rejection reason 1. Backend-specific parameters remain separate so RBF
smoothing is never interpreted as GMT tension.

Every NetCDF output stores the resolved YAML configuration, input dataset path,
project name, stage, units, and schema version. The output directory also
contains `eikonal.log` with stage-level source counts and paths.
