# Changelog

Changes that matter for users, newest first. **Changed results** lists changes
that alter numbers produced by an existing configuration; each names the
switch that restores the old behaviour where one exists.

## Unreleased (branch `dev`, version 0.2.0)

### Changed results

- **CCF rotation: R and T along the propagation (Lin et al. 2008).**
  `rotate_zne_ccf_to_zrt`, `rotate_zrt_ccf_to_zne` and `seisforge ant
  rotate-ccf` now point R from the virtual source towards the receiver at
  both stations and T 90 degrees clockwise from R, as in Fig. 2 of Lin,
  Moschetti & Ritzwoller (2008, GJI 173, 281-298) (`radial_convention:
  lin2008`, default). Before, R and T were reversed at both stations. **ZR,
  RZ, ZT and TZ change sign** (exact negation); ZZ, RR, RT, TR and TT are bit
  for bit the same (RR/RT/TR/TT agree with Lin et al. eq. 1 in both
  conventions). Old behaviour: `rotation.radial_convention: legacy` (config),
  `--radial-convention legacy` (CLI) or `radial_convention="legacy"`
  (functions), bit for bit. `rotate-ccf` records the convention in
  `rotate_ccf.log` and SAC `kuser0` (`lin2008`/`legacy`).
  `examples/01_ccf_rotate`: data regenerated in the new convention, with the
  RZ truth now built with the correct phase constant (only `NZ`, `EZ` files
  and `true_zrt.npz`, `expected_result.png` changed; residuals still
  0.029-0.040 of max ZZ).
- **AFTAN ZR/RZ phase constants fixed; station mode picks the constant per
  file.** The automatic `pi_over_4` is now that of the stored component on
  both lag branches: ZZ, RR, TT -1; with lin2008 axes ZR -3, RZ +1; with
  legacy axes ZR +1, RZ -3 (derivation and checks:
  `docs/aftan_phase_constants.md`). The negative-branch ZR <-> RZ swap no longer
  chooses the constant (it only labels the physical component). Station
  mode (no `component` in the config) now defaults to `pi_over_4: auto` and
  reads the component of each file from SAC `kcmpnm` or the file name; when
  neither names one, auto mode stops (set `pi_over_4` to a number to measure
  such files). New key `aftan.radial_convention` (`auto` default: SAC
  `kuser0`, else lin2008 with a warning for ZR/RZ; `lin2008`; `legacy`; a
  value contradicting `kuser0` stops). Pair mode stops when the configured
  component contradicts the file's `kcmpnm` or name. On the diffuse-field
  synthetics at 150 km (`tests/aftan_component_cases.py`), largest
  phase-velocity error before -> after: ZR pair positive 0.006 -> 0.006, ZR
  pair negative and station both branches 0.250 -> 0.006; RZ pair positive
  and station both branches 0.280 -> 0.006, RZ pair negative 0.620 -> 0.006.
  Group velocities are unchanged. ZZ, RR and TT results are bit for bit the
  same in both modes (regression baseline, `examples/02_aftan`,
  `examples/full_chain`, `examples/benchmarks/aftan_vs_fortran`); their
  `.dat` headers now also show `pi_over_4_mode: auto` in station mode,
  `stored_component` and `radial_convention`. No option restores the old
  ZR/RZ constants (they were wrong for every rotation convention); to
  measure ZR/RZ files rotated by SeisForge before this change, set
  `aftan.radial_convention: legacy`. `tests/data/aftan_component_baseline.npz`
  regenerated (only ZR/RZ arrays changed).
- **PMF cleaning window.** `PhaseMatchedFilter.clean_undispersed` now sets the
  window as Fortran AFTAN `tgauss.f` does: after the inner limits are made
  symmetric about the compressed pulse, both outer limits lie one outer
  width beyond them, and both half-Gaussian tapers start at weight 1 on the
  inner limit (`aftan.pmf.clean_window: symmetric`, default). The previous
  code followed PyAFTAN, which adds the outer width to the unsymmetrised
  right outer limit (a wider right taper) and starts the left taper one
  sample outside the inner limit. Only PMF outputs change (`*.pmf.dat`,
  `*.pmf_ftan.nc`, `*.pmf.vph.nc`); basic FTAN is unchanged. Noise-free
  synthetic ZZ CCFs (6-40 s, alpha 20): PMF group velocity changes by up to
  0.19 / 0.08 / 0.04 / 0.02 km/s at 100 / 200 / 400 / 1000 km, phase
  velocity by up to 0.06 / 0.02 / 0.01 / 0.004 km/s; the largest group
  velocity error against the truth drops from 0.127 to 0.093 km/s at
  400 km. Benchmark pair M07A-M15A (`examples/benchmarks/aftan_vs_fortran`):
  up to 0.016 km/s in group and 0.008 km/s in phase velocity inside the
  Fortran reference's period range (5-55 s); the stored outputs were
  regenerated. Old behaviour: `aftan.pmf.clean_window: pyaftan`
  (bit-for-bit). The choice is recorded as `pmf_clean_window` in PMF
  `.dat` headers and dataset attrs.
- **STA001 example: data from a finer layering, graded inversion layering,
  parallel chains.** `tools/make_sta001_synthetic.py` now forwards the truth
  in uniform 1/32 km layers (`TRUTH_DZ_KM`) instead of the inversion's own
  layering, so the example no longer hides its layering error; the data files
  were regenerated (same truth, noise and seed). `examples/STA001/inv.yaml`
  uses `discretization: dz: 0.0625, dz_growth_ratio: 1.1, dz_max: 1.0`
  (35 layers; logged layering error 0.03 sigma in phase velocity, 0.08 sigma
  in H/V) instead of uniform `dz: 0.25` (1.2-1.4 sigma in H/V), and
  `sampler.executor: process` with `max_workers: 4` (bit-for-bit the same
  samples as `serial`, checked on a short run). Measured: split R-hat <= 1.007,
  ESS >= 541, about 80 s on 4 processes. The old data and settings are in git
  history; they were example inputs, not a method, so no option restores them.
- **STA001 example replaced by synthetic data.** `examples/STA001/vph_disp.dat`
  and `hv.dat` had no stated origin or truth; they are now computed by
  `tools/make_sta001_synthetic.py` from a stated truth model with seeded
  noise, written with `true_model.dat`. `inv.yaml` now runs 4 x 16000 steps
  (burn-in 5000, thin 10) with adaptive proposals
  (`discard_early_history_fraction: 0.5`) and converges (split R-hat <= 1.006,
  ESS >= 673, about 400 s on one core); the old 4 x 5000 fixed-step settings
  did not converge on the old data. Results of the example therefore differ
  entirely; the old files are in git history (before this change), not
  behind an option, because they were example inputs, not a method.
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

- **Diffuse-field Rayleigh CCFs.** `seisforge.ant.synthetic.synthetic_rayleigh_ccf_zne`
  builds nine ZNE CCFs from plane Rayleigh waves (retrograde, any H/V),
  checked against Bessel closed forms after the package rotation; it does not
  use the AFTAN `pi_over_4` table.
- **AFTAN phase-constant warning.** `seisforge.ant.aftan.phase_constant_warning`;
  AFTAN writes `phase_velocity_warning` to the `.dat` header and `aftan.log`
  when `pi_over_4` does not match the independently checked value for the
  stored component, or the component is ZR/RZ or unknown (station mode reads
  it from SAC `kcmpnm` or the file name only for this check). No number
  changes. Regression baseline `tests/data/aftan_component_baseline.npz`
  (ZZ/ZR/RZ/RR/TT, pair and station mode, both branches;
  `tools/make_aftan_component_baseline.py`). The ZR/RZ issue it flagged is
  fixed (see "Changed results"); the warning now covers manual values that
  do not fit, unchecked components (ZT/TZ/RT/TR), unknown components, and
  ZR/RZ files whose R/T convention had to be assumed.

- **Runnable examples with a known answer.** `examples/01_ccf_rotate`,
  `02_aftan`, `03_eikonal`, `04_rf_qc_hk`, `05_hvsr_forward`,
  `06_inversion` and `full_chain` (one possible route: CCFs -> AFTAN ->
  Eikonal -> station curve -> inversion, linked by the example's own
  `glue.py` script, not by package code), each with small synthetic data,
  a minimal config (lines to edit marked `# change this`), `run.sh`, an
  expected figure and a README with the same five sections. Data scripts
  `tools/make_example_*.py` (fixed seeds, `tools/example_truth.py` shares the
  06 truth model). Overview: `docs/examples.md`. Smoke tests
  `tests/test_examples.py`; the full-length inversion runs and `full_chain` are marked `slow`
  (`pytest -m slow`, a separate CI step).
- **`seisforge init <workflow> [directory]`** copies an example's config
  template (package `seisforge.templates`, kept identical to `examples/` by
  `tools/sync_templates.py` and a test); refuses to overwrite without
  `--overwrite`.
- **`seisforge.ant.synthetic`**: symmetric synthetic CCF of one mode from a
  phase-velocity curve (`synthetic_ccf`, phase convention shared with
  AFTAN's `pi_over_4`), `group_velocity_from_phase`. Checked against an
  analytic non-dispersive pulse, the envelope arrival at r/U, and AFTAN.
- **`seisforge.ant.rotation.rotate_zrt_ccf_to_zne`**: inverse of the CCF
  rotation (round-trip tested), used to build synthetic ZNE input.
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
- `tools/aftan_cycle_set.py`, a stand-alone script (not a package command)
  to preview or apply a manual phase-cycle shift to saved AFTAN outputs;
  preview by default, `--in-place` to write.
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
- **Adaptive proposals during burn-in** (`sampler.adaptation`): the adaptive
  Metropolis algorithm of Haario et al. (2001) fits the proposal covariance
  to each chain's own history during burn-in and freezes it afterwards, so
  stored draws come from a fixed-kernel chain. It speeds up mixing of
  strongly correlated parameters such as B-spline coefficients (ESS about
  10x higher than the best fixed isotropic step on a correlation-0.99
  Gaussian). The frozen covariance is written to `*_mcmc.nc`
  (`proposal_covariance`, with tempering also `proposal_covariance_tempered`)
  and the settings to its attrs and the log. Off by default; without it
  chains are bit-for-bit those of earlier versions.
- **Leaving the early burn-in history out of the adaptive proposal**
  (`sampler.adaptation.discard_early_history_fraction`, default 0): each
  covariance update uses only the most recent `1 - f` of the states so far,
  so the drift from a distant start no longer widens the learned steps. On a
  Gaussian started 30 sd away, `f = 0.5` gave a frozen covariance within
  6-43 % of the true one instead of about 150 times too large and 6-21 times
  the ESS. Recorded in `*_mcmc.nc` attrs and the log. The default keeps
  earlier adaptive runs bit for bit.
- **Burn-in and acceptance checks in the run log:** warnings when `burn_in`
  is short for `sampler.adaptation` (assumed thresholds), and the
  post-burn-in acceptance rate of every chain with a note when it is below
  0.1 or above 0.6 (assumed).

- **Cached B-spline basis** (`seisforge.inv.bspline`): the basis matrix for
  a given set of depth samples and knots is computed once and reused from a
  bounded LRU cache (256 entries, read-only arrays; `bspline_basis` returns
  copies; `clear_bspline_basis_cache` empties it). Results are bit-for-bit
  unchanged; the STA001 prior evaluation drops from about 1.0 to 0.4 ms.

- **Graded layering** (`discretization.dz_growth_ratio`, `dz_max`): layer
  thickness grows with depth, `h(z) = min(dz_max, dz + (r - 1) z)`, keeping
  segment boundaries, transition supports and `force_depths`. The default
  (`dz_growth_ratio: 1`) is the uniform `dz` layering, bit-for-bit. For
  STA001, `dz: 0.0625, dz_growth_ratio: 1.1, dz_max: 1.0` uses 35 layers
  instead of 61 with a smaller layering error (see `docs/inv_usage.md`).
- **Layering error check** (`seisforge.inv.layering.layering_observable_error`):
  difference in predicted phase velocity and ellipticity between the
  configured layering and a fine uniform reference, in data units and in
  sigma. Posterior runs write it for the initial model, with the layering
  choice, to the log and to `discretization_*` output attributes.

### Changed

- **Examples reorganised.** `examples/configs/` moved to
  `examples/reference_configs/` (`aftan_station.yml` -> `aftan.yml`,
  `hkseq.yml` -> `rf_hkseq.yml`, `interface_model.yml` ->
  `inversion_interface_model.yml`); `examples/STA001/` became
  `examples/06_inversion/` (data in `data/`, short `inv.yaml`/`obs.yaml`
  that load to the same setup as the full `reference_configs/inversion_inv.yaml`
  and `inversion_obs.yaml`; same samples); `examples/AFTAN/` moved unchanged
  to `examples/benchmarks/aftan_vs_fortran/` with a README stating the data
  source (USArray TA, DOI 10.7914/SN/TA). `tools/make_sta001_synthetic.py`
  -> `tools/make_example_06_inversion.py`, `tools/plot_sta001_result.py` ->
  `examples/06_inversion/plot_result.py`.
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
- Inversion configurations (`inv.yaml`, `obs.yaml`) reject unknown keys with
  the section and the closest valid key, instead of silently using a default.
  Section names are lowercase (`model`, `inversion`, `discretization`,
  `constraints`, `soft_priors`, `likelihood`, `sampler`, `observations`,
  `forward`, `run`); the capitalized names and the aliases
  `PhysicalConstraints` and `Data` still load with one
  `ConfigDeprecationWarning` per run. `resolved_*.yaml` and
  `parameterized_vs_model_to_config` write the lowercase names. `forward`
  accepts only `algorithm`, `dc` (and `dt` for dispersion); giving both `hv`
  and `rayleigh_hv` is an error. No numbers change.

### Fixed

- `seisforge.hvsr` raises `ImportError` with an install hint when `disba` is
  missing, instead of returning NaN root seeds.

### Removed

- `examples/configs/joint_inversion.yml` (no longer loadable;
  `examples/STA001` replaces it).

## 0.1.0 (branch `main`)

Receiver-function QC and sequential H-k stacking, and ZNE to ZRT rotation of
ambient-noise cross-correlation functions.
