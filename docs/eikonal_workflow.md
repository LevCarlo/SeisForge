# Surface-Wave Eikonal and Helmholtz Tomography

SeisForge builds local phase-slowness maps from directed virtual-source travel
time fields and stacks them with azimuth-balanced event weights. The workflow
implements isotropic ambient-noise Eikonal tomography and optional Helmholtz
amplitude correction. Azimuthal anisotropy is not yet implemented.

## Input Dataset

The Eikonal workflow starts from an already collected and integer-period
resampled Xarray Dataset. Parsing AFTAN text files, resolving physical CCF
branches, and attaching station coordinates belong in the project-specific
upstream glue script.

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

## Run The Workflow

Start from the example configuration:

```bash
seisforge ant eikonal -c examples/configs/eikonal.yml
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
