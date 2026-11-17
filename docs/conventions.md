# Conventions

Conventions shared by all SeisForge workflows. Workflow pages give the details
specific to one method.

## Units and names

| quantity | unit | name suffix |
|---|---|---|
| depth, thickness, distance | km | `_km` |
| velocity (Vs, Vp, phase, group) | km/s | `_km_s` |
| density | g/cm^3 | `rho`, `_g_cm3` |
| period, time, delay | s | `_s` |
| frequency | Hz | `_hz` |
| slowness, ray parameter | s/km | `_s_per_km` |

Units are part of function argument and variable names (`period_s`,
`distance_km`, `ray_parameter_s_per_km`). Configuration keys follow the same
rule where a unit is not obvious (`top_km`, `depth_km`, `width_km`).

## Earth models

- Depth is positive downward from the free surface (0 km). There is no water
  layer yet; models describe land stations.
- A layered model lists layers from the top; the last entry is the half-space.
  `seisforge.inv.LayeredModel` stores a thickness for every entry, with the
  half-space thickness ignored (written as 0). `seisforge.hvsr` models take
  one thickness fewer than properties.
- Vs is the inverted property; Vp and density are derived from it by the
  scaling relations of the [inversion model](model.md#3-interfaces).
- An interface's `width_km` is the 10-90 % transition thickness; `0` is a hard
  step.

## Data files

Single-station observations are three-column text, `period_s  value
one_sigma`, with `#` comments allowed (see the
[configuration reference](inv_usage.md#obsyaml)). Gridded and pair-wise
products (AFTAN, Eikonal) are NetCDF with units and schema versions in their
attributes; see the workflow pages.

## Receiver functions

- Radial is positive away from the source, vertical positive up: the direct P
  pulse and Ps from a velocity increase with depth are positive.
- `gauss_a` (or `a`) is the Gaussian parameter of `exp(-w^2 / (4 a^2))`, as in
  iterative time-domain deconvolution.
- Time is relative to direct P, which is at 0 s.
- Synthetic amplitudes follow either the continuous inverse transform
  (`impulse`) or a unit-peak Gaussian (`gaussian_peak`); see
  [synthetic P receiver functions](rf_workflow.md#synthetic-p-receiver-functions).

## Geography

Eikonal/Helmholtz inputs declare their longitude convention
(`longitude_convention: -180_180` or `0_360`) in the dataset attributes, and
input and grid conventions must match
([Eikonal usage](ant/tomography.md#7-usage)).

## Outputs and reproducibility

- Scientific formulas live in small pure functions (`core`, `hk_core`,
  `taper`, `snr`, `transitions`, `layering`, ...); drivers only orchestrate IO,
  logging, and stage policies.
- Outputs are self-describing: NetCDF attributes and `.dat` headers record the
  choices needed to reproduce them (branch, alpha, taper window, SNR
  definition, phase-cycle shift, ...). Inversion runs also write snapshots of
  their input YAML files.
- Existing outputs are never overwritten or deleted unless explicitly
  requested (`io.overwrite`, `--overwrite`).
- Random-number seeds are configuration values (`sampler.seed`) and are
  recorded with the run.
- Every output records the software that made it, as string attributes from
  `seisforge.provenance.software_provenance()`: `seisforge_version`,
  `seisforge_git_commit`, `seisforge_git_dirty` (`true`/`false`; untracked
  files ignored), `python_version`, and `numpy_version`, `scipy_version`,
  `obspy_version`, `xarray_version`, `disba_version`, `numba_version`. The git
  fields are `unknown` when SeisForge is not imported from its own git
  checkout (for example a wheel install) or `git` is unavailable; a missing
  dependency is `not installed`. They appear in NetCDF attrs (inversion
  `*_mcmc.nc`, `*_vs.nc`, `*_predictive.nc`; AFTAN `*.basic_ftan.nc`,
  `*.vph.nc` and their PMF variants; Eikonal event fields and `stack.nc`; the
  per-RF H-k datasets in `*_Hk_results*.joblib`), in the `Run` section of the
  inversion's `resolved_*.yaml`, as `# key: value` lines in AFTAN `.dat` and
  H-k `*_Hk_QC.dat` headers, and in the logs (`aftan.log`, `eikonal.log`,
  `rotate_ccf.log`, `*_hkSeq.log`, `*_RF_QC.log`). Rotated CCF SAC files carry
  no provenance; their log does. `seisforge.__version__` is read from the
  installed package metadata, so `pyproject.toml` holds the only copy of the
  version number.
- Behaviour kept only to reproduce old results is isolated and labelled as
  compatibility, for example `aftan.time_taper.mode: legacy_edge_fraction`.
