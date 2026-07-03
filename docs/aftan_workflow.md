# AFTAN Dispersion Measurement

Use one station-level config per run:

```bash
seisforge ant aftan -c examples/configs/aftan_station.yml
```

The config is intentionally local and explicit: one station name, one input
directory, one SAC glob pattern, one output directory, and a compact set of
AFTAN parameters. Each matched SAC file writes:

- `<input_stem>.npz` with arrays `period`, `group_velocity`, `phase_velocity`,
  `amplitude`, `snr`, and metadata.
- `<input_stem>.dat` with columns `target_period_s instant_period_s
  group_velocity_km_s phase_velocity_km_s amplitude snr`. Amplitude is written
  in scientific notation to avoid losing very small values.
- When `pmf.enabled: true`, `<input_stem>.pmf.npz` and `<input_stem>.pmf.dat`
  are also written for the phase-matched second FTAN pass.
- `aftan.log` in the output directory.

The current config uses explicit branch and alpha choices:

```yaml
aftan:
  debug: false
  branch: positive  # positive, negative, both, stack
  period_sampling:
    mode: uniform
    step: 0.1
  basic:
    alpha:
      mode: constant
      value: 20.0
    trig_threshold: 50.0
    jump_points: 3
  pmf:
    enabled: false
    alpha:
      mode: constant
      value: 20.0
    trig_threshold: 20.0
    jump_points: 3
    period_bounds:
      mode: raw  # raw or step
      # step: 0.1
      min_method: floor
      max_method: ceil
```

`branch: both` writes one result for each positive and negative lag. `branch:
stack` averages the positive lag and the time-reversed negative lag. If a
two-sided branch is requested for a one-sided trace, SeisForge falls back to the
positive branch and records a warning in `aftan.log`.

`period_sampling` defines the target filter periods. `uniform` uses a fixed
period step or a fixed count; `list` uses the periods supplied in the config;
`geomspace` uses the PyAFTAN-style logarithmic grid:
`count = max(min_count, int(log(max_period / min_period) / dfreq))`, followed by
`geomspace(min_period, max_period, count)`. Despite the historical `dfreq` name,
this is a logarithmic period/frequency sampling density, not a linear Hz step.

When `aftan.debug: true`, the `.dat`, `.npz`, and optional xarray energy map
include phase-diagnostic outputs. The debug `.dat` appends
`phase_derivative_rad_s hilbert_phase_derivative_rad_s
hilbert_instant_period_s instant_period_delta_s` before `snr`. The `hilbert_*`
values estimate the instantaneous phase derivative from a Hilbert transform of
the real filtered trace at the same envelope peak used by the normal FTAN
picker. `instant_period_delta_s` is
`hilbert_instant_period_s - instant_period_s`.

Optional QC outputs can be enabled in config or from the CLI:

```bash
seisforge ant aftan -c examples/configs/aftan_station.yml --write-energy-map --plot-energy-map
```

When enabled, each trace writes `<input_stem>.basic_ftan.nc` as an xarray
dataset and, optionally, `<input_stem>.basic_ftan.png`. The PNG is a compact
summary with shared period axis: group-velocity FTAN energy, phase-velocity
cycle candidates when enabled, and the picked SNR curve. The group-energy
matrix, picked group-velocity curve, picked amplitude, and picked SNR are stored
in the xarray dataset.

For PMF outputs, `<input_stem>.pmf_ftan.nc` stores the same energy matrix after
phase-matched cleanup. Its `amplitude` variable is computed from the PMF-clean
waveform, and the dataset attribute `signal_source` is `pmf_clean_waveform`.

Set `energy_map.phase_velocity: true` to also write
`<input_stem>.phase_velocity.nc`. This is a diagnostic phase-cycle candidate map
rather than an independent phase-energy measurement: for each picked group
arrival, SeisForge expands nearby `2*pi` phase-cycle aliases into candidate
phase velocities. When plotting is enabled, this phase panel is included in the
same `<input_stem>.basic_ftan.png` summary figure rather than written as a
separate PNG.

This first SeisForge implementation ports the pure-Python FTAN path from
PyAFTAN and removes its global `PARAM` dependency. The Fortran path, large
dataset formatters, and tomography writers are intentionally left out for now.

The filter workflow is organized around two public concepts in
`seisforge.ant.filters`:

- `GaussianFTANFilter` performs the basic AFTAN Gaussian filter bank.
- `PhaseMatchedFilter` performs phase correction, spectral tapering,
  time-domain cleanup, and redispersion before the second Gaussian FTAN pass.

The alpha helpers keep the three common choices explicit:

- `gaussian_alpha_from_distance`: empirical distance-alpha table.
- `gaussian_alpha_aftan_scaling`: original AFTAN `20 * sqrt(dist / 1000)`.
- `gaussian_alpha_pyftan_constant`: PyAFTAN-style `20 * factor`.

At runtime the log records the requested branch, effective branch, alpha mode,
resolved alpha value, and basic QC metrics including valid period count,
instant/target period mismatch, group-velocity range, and maximum picked
amplitude.

### Short-Distance Period Guard

`max_period_nwl` is a SeisForge engineering guard, not an original AFTAN or
PyAFTAN parameter. It prevents very long periods from entering the FTAN grid
when the interstation distance is too short for the reference travel time to
contain enough wave cycles. The effective maximum period is

```text
effective_max_period =
    min(max_period, distance_km / (reference_velocity * max_period_nwl))
```

Equivalently, a period is allowed only when

```text
distance_km / reference_velocity >= max_period_nwl * period
```

The default `max_period_nwl: 0.5` is deliberately permissive. For example, with
`distance_km = 50` and `reference_velocity = 4`, the guard gives
`effective_max_period = 25 s`, so it will not limit a `0.1-10 s` run. Increase
the value toward `1-2` for stricter short-path QC when batch processing; decrease
it to relax the guard and let AFTAN's trigger/longest-branch logic decide. The
log reports `effective_period_range` for each trace so this guard is visible
during testing.

## Method Notes

### Gaussian FTAN

The first AFTAN stage applies a bank of narrow-band Gaussian filters to the
analytic spectrum of the input waveform. For each target angular frequency
`omega0`, the filter response is

```text
G(omega, omega0) = exp[-alpha * ((omega - omega0) / omega0)^2].
```

The filtered trace at each `omega0` forms one column of the frequency-time
amplitude map. The group arrival is picked from the envelope maximum in that
column, and the group velocity is computed as

```text
U(T) = distance / t_peak(T).
```

The measured period at the picked peak is not necessarily equal to the target
period of the Gaussian filter. AFTAN estimates the instantaneous frequency from
the local phase derivative near the envelope maximum. For later QC and HV work,
both values should be retained:

- `target_period`: the nominal period requested by the filter bank.
- `instant_period`: the observed period inferred from the local phase behavior.

The parameter `alpha` controls the frequency-time resolution tradeoff. A larger
`alpha` gives a narrower frequency band but a broader time-domain wave packet.
A smaller `alpha` gives better time localization but increases leakage from
neighboring periods or modes. Distance-based alpha rules are useful starting
points, but the usable value also depends on the target period band: short,
narrow bands can show large target/instant period mismatch when alpha is too
small, while overly large alpha can make the picked branch jumpier.

### Phase-Matched Filter

Phase-matched filtering is a second-stage cleanup procedure. It is not a
replacement for Gaussian FTAN; it sits between two Gaussian FTAN passes:

```text
raw waveform
  -> basic Gaussian FTAN
  -> preliminary group/phase branch
  -> phase-matched filter
  -> cleaned waveform
  -> second Gaussian FTAN
```

The basic idea is to use the preliminary dispersion curve to remove the
predicted dispersive phase from the waveform spectrum. After this correction,
energy that follows the selected surface-wave branch is compressed into a more
compact pulse in the time domain, while off-branch energy remains less focused.
The algorithm then tapers the usable spectrum, cleans the compressed waveform
around the dominant pulse, and restores the predicted phase before running FTAN
again.

In implementation terms, PMF has four conceptual steps:

1. **Phase correction**: build a phase operator from the preliminary group
   velocity branch. The correction uses the predicted travel time
   `t(omega) = distance / U(omega)` and integrates it over angular frequency to
   form a phase term.
2. **Spectral taper**: restrict the corrected spectrum to the reliable period
   interval of the preliminary branch. This avoids unstable edge behavior and
   limits leakage from poorly constrained frequencies.
3. **Time-domain cleanup**: transform the phase-corrected spectrum back to the
   time domain, identify the focused pulse, and suppress energy outside the
   pulse using Gaussian skirts or local-minimum based windows.
4. **Redispersion**: transform the cleaned pulse back to the frequency domain
   and apply the inverse phase correction, producing a waveform whose selected
   branch is cleaner but still has the physical dispersive phase.

The second Gaussian FTAN pass is then applied to this cleaned waveform. Its
group-velocity picks are usually more stable when the first-pass branch was
reasonable. Phase velocity can also improve, but only if the phase correction,
cycle unwrapping, and prediction curve are consistent.
SeisForge uses the same `basic.alpha` value for the basic pass, PMF spectral
taper setup, and the second Gaussian FTAN pass. The `pmf` section only controls
phase-matched cleanup and second-pass picking parameters.

SeisForge follows the AFTAN/PyAFTAN period strategy for PMF. The PMF correction
curve is built from the first-pass apparent/instant period and group velocity,
not from the nominal target periods. The cleaned second FTAN pass reuses the
configured sampling style, but its period range is clipped to the first-pass
apparent-period branch.

By default, `pmf.period_bounds.mode: raw` uses the exact apparent-period bounds
from the first pass. For short-period work this may produce non-round PMF target
grids such as `0.7947, 0.8947, ...`. Set `mode: step` to snap the PMF range to a
more readable decimal grid:

```yaml
pmf:
  enabled: true
  period_bounds:
    mode: step
    step: 0.1
    min_method: floor
    max_method: ceil
```

With this setting, a first-pass apparent-period range of `0.794-7.414 s` becomes
`0.7-7.5 s` before the configured `period_sampling` grid is generated. The raw
and snapped bounds are both stored in the PMF `.npz` output as
`pmf_raw_period_min/max` and `pmf_period_min/max`.

### Alpha Choices

Three alpha choices are kept separate because they encode different assumptions:

1. **Empirical distance table**: an externally calibrated relation between
   interstation distance and Gaussian width.
2. **Original AFTAN scaling**:

   ```text
   alpha = ffact * 20 * sqrt(distance_km / 1000).
   ```

   This makes the filter narrower for longer paths and broader for shorter
   paths.
3. **PyAFTAN-style constant factor**:

   ```text
   alpha = ffact * 20.
   ```

   The Python PyAFTAN path effectively uses configured alpha values directly,
   while its modified F77 source removes the original distance scaling.

### Measurement Implications

`alpha`, spectral tapering, and PMF cleanup all affect the final dispersion
measurement indirectly through the filtered waveform:

- Group velocity depends on the envelope maximum. Any filter choice that shifts
  or broadens the envelope can shift `t_peak`.
- Phase velocity depends on the phase sampled near the selected envelope peak
  and on resolving the `2*pi` ambiguity. Leakage, unstable PMF correction, or a
  poor prediction curve can move the solution by one or more cycles.
- Amplitude should be retained per period because later HV measurements need the
  period-dependent amplitude after filtering and branch selection.
- QC should compare `target_period`, `instant_period`, group velocity,
  amplitude, SNR, and branch continuity rather than relying on one column alone.

### SNR Windows

SNR is measured on the filtered trace for each picked period. The signal window
is period-dependent and centered on the picked group arrival
`t_pick = distance / group_velocity`:

```text
signal_window = [t_pick - signal_half_width_factor * T,
                 t_pick + signal_half_width_factor * T]
```

```yaml
snr:
  definition: local  # local, pyftan, or aftan
  output_db: false
  noise_mode: tail  # tail or complement
  signal_half_width_factor: 1.0
  noise_guard_factor: 1.0
```

`definition: local` uses the period-scaled picked-arrival window above.
`noise_mode: tail` keeps a trailing noise window: after the signal window,
SeisForge waits `dsn` seconds and measures `nlen` seconds of noise.
`noise_mode: complement` treats all samples outside the picked signal window
plus `noise_guard_factor * T` as noise on the already selected branch trace.
This is useful for phase-weighted stacks where the far tail can be close to zero
and therefore make tail-window RMS unrealistically small.

`definition: pyftan` follows the PyAFTAN Python SNR convention: a broad
Bensen-style signal window from `distance / vmax - bfact * Tmax` to
`distance / vmin + efact * Tmax`, followed by a tail-noise window controlled by
`dsn` and `nlen`. `definition: aftan` follows the Fortran AFTAN diagram
definition, using the selected FTAN peak divided by the geometric mean of its
left and right local minima. Set `output_db: true` to write `20*log10(SNR)` and
label the `.dat` column as `snr_db`; the default is the linear ratio.

Useful alpha diagnostics are:

- The relative mismatch `abs(instant_period - target_period) / target_period`.
  Large values usually mean the Gaussian window is too broad in frequency for
  that band, or the selected envelope peak is contaminated by nearby energy.
- The group-energy map. Too-small alpha often smears energy across periods,
  while too-large alpha broadens wave packets in time and can make the picked
  branch jump between local maxima.
- The phase-velocity candidate map. If neighboring `2*pi` aliases have similar
  strength, phase picks are not robust enough for PMF tuning.
- A small alpha sweep on representative traces. For short periods, inspect a
  grid such as `alpha = 20, 40, 60, 80` and prefer the smallest value that keeps
  `instant_period` close to `target_period` without creating obvious branch
  jumps.

### Retained Branch Fraction

AFTAN's Fortran implementation has a `perc` check: after trigger correction, if
the final continuous branch is shorter than a configured percentage of the
initial filter grid, the final result is marked invalid. SeisForge keeps the
same information as QC but defaults to a gentler policy:

```yaml
qc:
  min_valid_fraction: 0.0
  fail_on_short_branch: false
```

`min_valid_fraction` compares the number of retained final periods with the
number of initially attempted target periods. The default `0.0` disables the
warning. Set `0.5` to mimic the common AFTAN `perc = 50%` completeness check. By
default SeisForge only logs a warning and writes `qc_retained_fraction` to the
`.npz`; set `fail_on_short_branch: true` only when you want strict automated
rejection.

This QC does not replace downstream interpolation. A typical workflow can still
measure on a dense FTAN grid, then interpolate accepted curves onto integer or
regularly spaced periods during post-processing.

## Alignment With AFTAN And PyAFTAN

SeisForge currently aligns with the reference implementations in these places:

- The Gaussian FTAN kernel uses the same response form as AFTAN:
  `exp[-alpha * ((omega - omega0) / omega0)^2]`.
- Alpha selection exposes the three useful conventions: original AFTAN distance
  scaling, PyAFTAN's modified constant scaling, and a user empirical distance
  table.
- Basic and PMF passes have separate Gaussian alpha and jump-correction
  parameters, matching PyAFTAN's `bas`/`pmf` split. This is useful because PMF
  often benefits from a different second-pass trigger threshold or filter width.
- PMF uses the first-pass apparent/instant period branch to build the phase
  correction and then runs a second Gaussian FTAN pass on the PMF-clean waveform.
- Jump correction follows the AFTAN trigger idea with independent
  `basic.trig_threshold`/`basic.jump_points` and
  `pmf.trig_threshold`/`pmf.jump_points`: detect excessive curvature in the
  dispersion branch, repair short jump segments, and keep the longest stable
  branch.
- `pi_over_4` enters the picked phase as a constant phase shift and therefore
  affects phase velocity, not group velocity.

Current differences and simplifications:

- The Python PMF is a compact implementation. It uses cubic-spline group-delay
  integration, a four-corner spectral taper, and local-minimum time cleanup, but
  it is not a line-by-line port of `tgauss`/Fortran PMF internals.
- Phase velocity is prediction-sensitive. Without a reliable `prediction_file`,
  the cycle choice falls back to `reference_velocity`; this should be treated as
  diagnostic for now.
- The phase-velocity xarray map is a cycle-candidate diagnostic map, not an
  independent phase-energy transform.
- SNR is configurable. `definition: aftan` is closest to the FTAN diagram
  contrast, `definition: pyftan` follows broad PyAFTAN-style windowing, and
  `definition: local` is the current SeisForge default for short CCF tests.
- `qc.period_rel_warning` only logs target/instant period mismatch. It does not
  reject, trim, or smooth the picked branch.
