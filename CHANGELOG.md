# Changelog

Changes that matter for users, newest first. **Changed results** lists changes
that alter numbers produced by an existing configuration; each names the
switch that restores the old behaviour where one exists.

## Unreleased (branch `dev`, version 0.2.0)

### Changed results

- **AFTAN time taper.** The 5 % edge taper on zero-lag CCF branches was
  replaced by a taper on the group-velocity window `[dist/vmax, dist/vmin]`
  (`aftan.time_taper.mode: velocity_window`, default). The old taper
  suppressed the first 5 % of the trace and biased short-distance group
  velocities (synthetic 100 km pair at 5 s: 3.00 -> 2.61 km/s). SNR is now
  measured on the untapered waveform. Old behaviour:
  `time_taper.mode: legacy_edge_fraction` (bit-for-bit).
- **AFTAN phase-cycle selection.** The global 2-pi cycle shift now minimises
  the mean relative phase-slowness misfit to the reference curve, and runs
  only when a `prediction_file` is given; otherwise the Fortran-AFTAN anchor
  is kept (shift 0). The previous score could drive long-period phase
  velocity far from Fortran AFTAN (9-16 km/s instead of 3.8 km/s on the
  example pair). Diagnostic variables `*_reference_score_cycles` were renamed
  `*_reference_misfit`.
- **Eikonal/Helmholtz Laplacian.** The ellipsoidal metric term
  `-(tan(lat)/N) df/dy` is now included (`eikonal.laplacian_metric:
  ellipsoidal`, default). Helmholtz velocities change by less than 0.5 m/s in
  practice. Old behaviour: `laplacian_metric: locally_flat`.
- **H-k stacking.** Delays outside the RF trace or post-critical rays now
  contribute zero instead of the first/last sample; results inside the trace
  are unchanged. Counts are recorded as `n_invalid_Ps`, `n_invalid_PpPs`,
  `n_invalid_PsPs`.

### Added

- **Parallel tempering** (`Sampler.tempering`): each chain becomes a
  ladder of tempered replicas with state swaps between neighbours; only the
  `T = 1` replica is stored. Swap acceptance is written to `*_mcmc.nc`.
  Without `tempering` chains draw exactly the same random numbers as before.
- **Sampled noise scales** (hierarchical noise): an observation may name a
  parameter holding `log10 h`, which scales its uncertainties to `h * sigma`
  in the likelihood (`-chi2 / (2 h^2) - N ln h`). Without it the likelihood
  is unchanged.
- **MCMC convergence diagnostics** (`seisforge.inv.diagnostics`): split
  R-hat, effective sample size, and lagging (stuck) chains, written to
  `*_mcmc.nc` and the run log, with warnings for clearly unconverged runs.
- **Inversion interfaces:** `Interface(depth_km, width_km, shape)` with
  sampled absolute depths and gradational (sigmoid or smootherstep)
  transitions; per-segment Vp/rho scaling; Gardner density; normal and
  half-normal priors; interface-aware layering with `layers_per_segment`,
  `transition_layers`, `layer_value`, and the `layering_report` diagnostic.
  Existing configurations reproduce their layered models and log posteriors
  bit for bit.
- **Synthetic P receiver functions** (`seisforge.rf.synthetic`,
  `seisforge.inv.predict_receiver_function`): Thomson-Haskell motion-stress
  propagator, full reverberations, per-layer density, adaptive FFT padding;
  validated against PyGRT; about 0.5 ms for a 67-layer model.
- `seisforge ant aftan-cycle-set` to inspect or override a phase-cycle
  choice.
- `--overwrite` for `seisforge rf hkseq`.
- Documentation site (MkDocs) with getting started, conventions, model guide,
  configuration reference, API reference, and references.
- MIT license; GitHub Actions for lint, tests (Python 3.10 and 3.12), and the
  documentation build.
- **Software provenance in every output** (`seisforge.provenance.software_provenance`):
  SeisForge version, git commit and dirty flag (`"unknown"` outside a source
  checkout), and the Python, numpy, scipy, obspy, xarray, disba and numba
  versions are written to inversion NetCDF attrs, `resolved_*.yaml`, AFTAN
  NetCDF attrs, `.dat` headers and `aftan.log`, Eikonal event-field and stack
  attrs and `eikonal.log`, H-k result attrs, `*_Hk_QC.dat` and the H-k log,
  the RF QC log, and `rotate_ccf.log`. Attributes are only added; no numbers
  change. `seisforge.__version__` now comes from the installed distribution
  metadata (`pyproject.toml` is the single source; `"0+unknown"` when the
  source tree is imported without installing).
- **Forward-failure counts:** the posterior `*_mcmc.nc` records per chain
  `forward_attempts`, `forward_failures` and `forward_failure_rate` (with
  tempering also for the `T = 1` replica), and the log warns when a chain's
  rate exceeds 1 % (assumed threshold), naming the most frequent errors.

### Changed

- `disba` and `numba` are core dependencies.
- `seisforge rf hkseq` no longer deletes its output directory, refuses an
  output directory that overlaps its inputs, and writes only the requested
  stage's files. The high-frequency QC table is now
  `NET.STA_HFreq_Hk_QC.dat`.
- An unknown inversion `kind` is rejected instead of loading no observations.
- Inversion configurations are validated at setup (parameter references,
  model layout, a valid initial model).
- Tests import the installed package (`pip install -e ".[dev]"` first).
- `seisforge.ant.aftan` no longer re-exports private helpers (`_snr`,
  `_aftan_diagram_snr`, `_period_qc`, `_qc_warnings`,
  `_align_to_period_grid`, `_override_energy_map_config`). They remain in
  their modules (`snr`, `qc`, `products`, `config`) and are not public API.
- Inversion forward wrappers (`predict_dispersion`, `predict_rayleigh_hv`)
  and the HVSR mode seeds turn only disba's solver failures
  (`DispersionError`, `ZeroDivisionError`) into a rejected proposal or a
  missing seed. Other exceptions, for example `TypeError`, `AttributeError`,
  unsorted observation periods or a missing disba, now stop the run instead
  of silently rejecting every proposal; the samplers no longer read a
  `ValueError` raised by the target as `-inf`. Runs without such errors draw
  the same samples bit for bit.

### Fixed

- `seisforge.hvsr` raises `ImportError` with an install hint when `disba` is
  missing, instead of returning NaN root seeds.

### Removed

- `examples/configs/joint_inversion.yml` (no longer loadable;
  `examples/STA001` replaces it).

## 0.1.0 (branch `main`)

Receiver-function QC and sequential H-k stacking, and ZNE to ZRT rotation of
ambient-noise cross-correlation functions.
