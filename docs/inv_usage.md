# Single-station inversion user guide

This guide explains the current single-station inversion workflow in SeisForge.
It is written for users who need to edit configuration files, understand the
model space, and diagnose prior/posterior results.

The current implementation supports:

- Rayleigh phase-velocity inversion.
- Rayleigh H/V inversion.
- Joint Rayleigh phase velocity + Rayleigh H/V inversion.
- Segment-based Vs parameterization with constant, gradient, and B-spline
  profiles.
- Hard geophysical prior constraints.
- A built-in random-walk Metropolis sampler with multiple chains.
- NetCDF/xarray outputs for MCMC samples, Vs ensembles, and posterior
  predictions.

The example station folder is:

```text
examples/STA001/
  inv.yaml
  obs.yaml
  vph_disp.dat
  hv.dat
```

Run the joint inversion from inside the station folder:

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```

If SeisForge is not installed as a command, use:

```bash
python -m seisforge.cli.main inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```

For prior-only sampling:

```bash
seisforge inv disp-hv --inv inv.yaml -o run_prior --prior-only --progress
```

Prior-only sampling does not use observation data. It samples the model space
defined by parameter bounds and hard physical constraints.

## Configuration split

SeisForge uses two YAML files for the recommended workflow:

- `obs.yaml`: observation files, observation metadata, and forward-model
  numerical settings. This file is usually stable for a station.
- `inv.yaml`: model parameterization, inversion parameters, hard priors,
  likelihood weights, and sampler hyperparameters. This file is expected to be
  edited frequently while testing inversion strategies.

The legacy single-file interface is still available:

```bash
seisforge inv disp-hv -c config.yaml --disp vph_disp.dat --hv hv.dat -o run_disp_hv
```

In the new split-config interface, `--disp` and `--hv` are optional overrides
for file paths declared in `obs.yaml`.

## obs.yaml

`obs.yaml` describes what data are being inverted and how synthetic data should
be computed.

Example:

```yaml
Observations:
  dispersion:
    file: vph_disp.dat
    wave: rayleigh
    kind: phase
    mode: 0

  hv:
    file: hv.dat
    wave: rayleigh
    mode: 0

Forward:
  dispersion:
    algorithm: dunkin
    dc: 0.005
    dt: 0.025

  hv:
    algorithm: dunkin
    dc: 0.005
```

Observation files are three-column text files:

```text
period  observed_value  one_sigma_uncertainty
```

For dispersion:

```text
period  phase_or_group_velocity  velocity_uncertainty
```

For Rayleigh H/V:

```text
period  hv  hv_uncertainty
```

Comments starting with `#` are allowed. Periods are sorted internally after
reading.

### Observation metadata

`wave`, `kind`, and `mode` describe the observation itself.

For dispersion:

- `wave`: currently `rayleigh` or `love` in the forward request object. The
  current joint workflow is designed around Rayleigh phase velocity.
- `kind`: `phase` or `group`.
- `mode`: integer mode index. The current examples use fundamental mode
  `mode: 0`.

For H/V:

- `wave`: should be `rayleigh`.
- `mode`: integer mode index, currently normally `0`.

### Forward settings

Forward settings belong in `Forward`, not directly in the observation metadata.
They describe how the synthetic data are computed with `disba`.

For dispersion:

- `algorithm`: disba algorithm, for example `dunkin`.
- `dc`: phase/group velocity root-search increment.
- `dt`: group-velocity finite-difference period step. This matters for group
  velocity and is passed through the disba group-dispersion wrapper.

For Rayleigh H/V:

- `algorithm`: disba ellipticity algorithm.
- `dc`: ellipticity root-search increment.

## inv.yaml overview

`inv.yaml` contains the parts that are commonly adjusted during inversion
experiments:

```yaml
Model:
Inversion:
Discretization:
Constraints:
Likelihood:
Sampler:
```

Each section is described below.

## Model

`Model` defines how a sampler parameter vector is converted into a Vs profile.
Vp and density are derived from Vs through a scaling rule.

```yaml
Model:
  scaling:
    vp:
      method: brocher2005
    rho:
      method: brocher2005

  segments:
    - top_km: 0.0
      bottom_km: 1.0
      profile:
        type: gradient
        top:
          parameter: sediment.vs_top
        bottom:
          parameter: sediment.vs_bottom
```

### Scaling

Available Vp scaling methods:

- `brocher2005`: empirical Brocher-style relation from Vs to Vp.
- `constant_vp_vs`: fixed Vp/Vs ratio.

Example:

```yaml
vp:
  method: constant_vp_vs
  vp_vs: 1.75
```

Available density scaling methods:

- `brocher2005`: empirical relation from Vs to density.
- `constant`: fixed density.

Example:

```yaml
rho:
  method: constant
  rho: 2.7
```

The default examples use Brocher scaling. This is convenient when only Vs is
inverted. If future workflows invert Vp/Vs or Vp directly, the scaling layer
will need to be extended.

### Segments

The model is built from ordered, contiguous depth segments. Each segment has:

- `top_km`: segment top depth in km.
- `bottom_km`: segment bottom depth in km.
- `profile`: the Vs parameterization inside the segment.

Current segment depths are fixed numeric values. Segment boundaries are
preserved automatically during discretization; a layer will not cross from one
segment into another.

Supported profile types are:

- `constant`
- `gradient`
- `bspline`

### Constant profile

```yaml
profile:
  type: constant
  value:
    parameter: upper_crust.vs
```

`value` can be a fixed number or a parameter reference.

### Gradient profile

```yaml
profile:
  type: gradient
  top:
    parameter: sediment.vs_top
  bottom:
    parameter: sediment.vs_bottom
```

The segment velocity changes linearly from `top` to `bottom`. The values can be
fixed numbers or parameter references.

### B-spline profile

```yaml
profile:
  type: bspline
  degree: 3
  knot_spacing: geometric
  knot_alpha: 1.6
  coefficients:
    - parameter: crust.c0
    - parameter: crust.c1
    - parameter: crust.c2
    - parameter: crust.c3
```

The B-spline is evaluated on the segment-local depth coordinate. Internally,
the basis is built on the segment domain; for geometric knot spacing, the
distribution is controlled on a normalized 0-1 coordinate.

Options:

- `degree`: spline degree. Cubic splines use `degree: 3`.
- `knot_spacing`: currently use `uniform` or `geometric`.
- `knot_alpha`: controls geometric knot concentration when
  `knot_spacing: geometric`.
- `coefficients`: B-spline coefficients, either fixed values or parameter
  references.

B-spline coefficients are basis weights. They are not generally equal to Vs at
specific depths. With an open-clamped spline, the first and last coefficients
have a closer relationship to endpoint values, but interior coefficients are
shape controls rather than directly interpretable layer velocities.

## Inversion parameters

`Inversion.parameters` defines the scalar parameter vector sampled by MCMC.

Example:

```yaml
Inversion:
  parameters:
    - name: crust.c0
      initial: 3.05
      prior:
        type: uniform
        bounds: [2.50, 3.55]
      proposal:
        sigma: 0.035
```

Fields:

- `name`: unique parameter name. This is referenced from `Model` with
  `{parameter: name}`.
- `initial`: initial value. It is used by the `theta0` initialization strategy
  and is also useful as a readable reference model.
- `prior.type`: currently only `uniform` is implemented.
- `prior.bounds`: lower and upper hard bounds for the scalar parameter.
- `proposal.sigma`: random-walk proposal standard deviation for this parameter.

Parameter bounds define scalar support. They are only one part of the prior.
Model-level constraints in `Constraints` further restrict the physically
allowed model space.

`proposal.sigma` is not a prior. It only controls MCMC step size and therefore
sampling efficiency.

## Discretization

`Discretization` controls conversion from a continuous or segmented Vs model to
the finite-layer model passed to disba.

```yaml
Discretization:
  dz: 0.25
  zmax: 15.0
  force_depths: []
```

Fields:

- `dz`: nominal layer thickness in km.
- `zmax`: deepest model depth used for discretization.
- `force_depths`: optional fixed depths that should be inserted as additional
  layer boundaries.

Important: segment boundaries are already preserved automatically. You do not
need to put fixed segment boundaries into `force_depths`. Use `force_depths`
only for extra fixed internal depths that are not already segment boundaries.

This matters for future receiver-function joint inversion. If interface depths
become inversion parameters, hard-coded `force_depths` will be inappropriate.
The model should instead preserve the current segment boundaries dynamically
for each candidate parameter vector.

## Constraints

`Constraints` are hard geophysical priors. A candidate model that violates any
constraint receives log prior `-inf` and is rejected before the likelihood is
allowed to matter.

Example:

```yaml
Constraints:
  constraints:
    - type: positive_velocity
    - type: vp_vs_range
      bounds: [1.45, 2.70]
    - type: monotonic_vs
      depth_range: [0.0, 2.0]
      tolerance: 0.03
      dz: 0.1
```

Current constraint types:

### positive_velocity

Requires Vs, Vp, and density to be finite and positive.

```yaml
- type: positive_velocity
```

This is usually harmless and should normally remain enabled.

### vp_gt_vs

Requires:

```text
Vp - Vs > margin
```

Example:

```yaml
- type: vp_gt_vs
  margin: 0.05
```

This is a numerical/physical safety constraint. It is useful when Vp and Vs can
approach each other because of a scaling choice or a future direct Vp/Vs
parameterization. For many crustal inversions, `vp_vs_range` is more
geophysically interpretable.

### vp_vs_range

Requires the Vp/Vs ratio to lie inside a given range for every layer:

```yaml
- type: vp_vs_range
  bounds: [1.45, 2.70]
```

This is often easier to reason about than `vp_gt_vs`. With Brocher scaling it
acts as a check on derived elastic properties. With future Vp/Vs inversion, it
will become a direct geophysical prior.

Brocher scaling can produce relatively high Vp/Vs in very low-Vs shallow
sediments. Start with a broad range for exploratory runs, then narrow it using
local geological knowledge.

### boundary_non_decreasing_vs

Requires Vs not to drop strongly across segment boundaries:

```yaml
- type: boundary_non_decreasing_vs
  tolerance: 0.05
```

The tolerance is in km/s. If the velocity just below a boundary is lower than
the velocity just above it by more than `tolerance`, the model is rejected.

This is useful for simple monotonic crustal toy models, but it should be used
carefully when low-velocity zones or sharp sediment/crust contrasts are
expected.

### vs_range

Samples the model over a depth range and requires Vs to stay inside bounds:

```yaml
- type: vs_range
  depth_range: [0.0, 15.0]
  bounds: [0.8, 4.6]
  dz: 0.25
```

Fields:

- `depth_range`: optional `[top, bottom]` in km. If omitted, the whole model is
  checked.
- `bounds`: allowed Vs range in km/s.
- `dz`: depth sampling interval for checking the constraint.

### monotonic_vs

Samples the model over a depth range and requires Vs to be non-decreasing with
depth, allowing a finite tolerance:

```yaml
- type: monotonic_vs
  depth_range: [0.0, 2.0]
  tolerance: 0.03
  dz: 0.1
```

The current implementation is depth-range based. This is simple, but not
always ideal. In a realistic model, monotonicity may be appropriate for the
sediment segment, optional in the crust, and usually unnecessary in the mantle.

A likely future extension is segment-level constraints:

```yaml
- type: monotonic_vs
  segment: sediment
  tolerance: 0.03
```

This would be especially useful when segment thicknesses become inversion
parameters, because the constraint would follow the segment rather than a fixed
absolute depth interval.

### max_vs_gradient

Limits the absolute Vs gradient:

```yaml
- type: max_vs_gradient
  depth_range: [0.0, 15.0]
  max_abs: 1.2
  dz: 0.25
```

`max_abs` is in km/s per km. This is useful for preventing extreme spline
oscillations or unrealistic shallow jumps.

## Likelihood

The current likelihood is a weighted Gaussian chi-square:

```yaml
Likelihood:
  dispersion_weight: 1.0
  hv_weight: 1.0
```

Internally:

```text
log L = -0.5 * (w_disp * chi2_disp + w_hv * chi2_hv)
```

The weights are inversion-strategy parameters. They do not change the data or
their uncertainties; they change the relative contribution of each observable
to the posterior.

For joint dispersion + H/V inversion, use the output diagnostics to decide
whether one observable dominates the other. Avoid assigning physical meaning to
a single combined root-mean-square misfit unless the two data types have been
carefully normalized for that purpose.

## Sampler

`Sampler` controls the built-in random-walk Metropolis sampler.

```yaml
Sampler:
  n_steps: 5000
  burn_in: 1000
  thin: 5
  n_chains: 4
  executor: serial
  max_workers: null
  seed: 20260702
  initial_strategy: prior_uniform
  max_initial_attempts: 50000
  proposal_sigma: from_parameters
  use_parameter_bounds: true
  global_jump_interval: 1000
```

### Chain length and storage

- `n_steps`: total Metropolis steps per chain.
- `burn_in`: number of early steps discarded from stored samples.
- `thin`: store every Nth step after burn-in.
- `n_chains`: number of independent chains.

The number of stored draws per chain is approximately:

```text
(n_steps - burn_in) / thin
```

### Executor

`executor` controls how multiple chains are run:

- `serial`: run chains one after another. This is simplest and best for
  debugging.
- `thread`: run chains with Python threads. This can help when much of the time
  is spent in compiled code or external libraries.
- `process`: run chains in separate processes. This is often better for
  CPU-bound Python work, but progress callbacks are currently disabled for this
  executor.

`max_workers` limits the number of concurrent workers. `null` lets Python
choose a default.

### Seed

`seed` controls random-number generation.

MCMC uses randomness for:

- drawing prior-uniform initial points,
- proposing candidate steps,
- accepting or rejecting candidates,
- assigning independent random streams to multiple chains.

With a fixed seed, the same configuration and data should produce reproducible
chains. This is useful for debugging and examples. With `seed: null`, each run
uses a fresh random state.

Recommended practice:

- Use a fixed seed while developing configs or debugging priors.
- Use different seeds, or `null`, for repeated production runs.
- Record the seed used for published or archived results.

### Initialization strategy

Current options:

- `theta0`: start each chain from the `initial` values listed in
  `Inversion.parameters`.
- `prior_uniform`: draw initial points uniformly from parameter bounds and keep
  only points that satisfy the hard prior.

`prior_uniform` is useful when checking whether the prior space is broad enough
and physically valid. It also avoids starting every chain from the same
reference model.

### Proposal sigma

`proposal_sigma` can be:

- `from_parameters`: use each parameter's `proposal.sigma`.
- a scalar: use the same proposal standard deviation for every parameter.
- a list: provide one proposal standard deviation per parameter.

Small proposal sigmas increase acceptance but may mix slowly. Large proposal
sigmas explore faster when accepted but can lead to many rejected proposals.

### Global jumps

`global_jump_interval` keeps a pymcinv-style global uniform proposal idea. Every
N steps, the sampler proposes a candidate uniformly from the full parameter
bounds instead of using the local Gaussian random walk.

This can help a chain escape local regions. If it is too frequent, acceptance
can become very low. Values like 500-2000 are usually more conservative than
using global jumps every few steps.

Set it to `null` to disable global jumps.

## Output files

For a joint dispersion + H/V posterior run, the output directory contains files
like:

```text
disp_hv.log
disp_hv_mcmc.nc
disp_hv_vs.nc
disp_hv_predictive.nc
resolved_config.yaml
resolved_inv.yaml
resolved_obs.yaml
```

Important files:

- `*_mcmc.nc`: parameter samples, log probabilities, acceptance flags, initial
  and final theta values.
- `*_vs.nc`: Vs(z) ensemble and depth-wise summaries, including mean, standard
  deviation, percentiles, and best model when available.
- `*_predictive.nc`: posterior predictive dispersion and/or H/V ensembles.
- `*.log`: run settings, progress messages, acceptance rates, and output paths.
- `resolved_inv.yaml` and `resolved_obs.yaml`: snapshots of the input configs.
- `resolved_config.yaml`: merged internal config with observation arrays
  removed for readability.

For prior-only runs, output names include `_prior_`, for example:

```text
disp_hv_prior_mcmc.nc
disp_hv_prior_vs.nc
```

There is no posterior predictive file for prior-only sampling because no data
likelihood is used.

## How to tune a configuration

Start with prior-only sampling:

```bash
seisforge inv disp-hv --inv inv.yaml -o run_prior --prior-only --progress
```

Inspect the prior Vs(z) ensemble. If the target/reference model is outside the
hard-prior ensemble, do not hide the issue by plotting wider artificial
envelopes. Instead check:

- parameter bounds,
- B-spline segment depth and knot spacing,
- monotonicity tolerance,
- maximum gradient,
- Vs range constraints,
- initialization failures from overly restrictive hard priors.

Then run the posterior:

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```

Use the separate dispersion and H/V residual diagnostics to tune likelihood
weights. A single joint misfit number is usually not very informative when the
observables have different units and sensitivities.

## Segment-thickness inversion

Variable segment thickness is not implemented yet, but the current architecture
is close to supporting it.

Today, segment boundaries are fixed numeric values:

```yaml
segments:
  - top_km: 0.0
    bottom_km: 1.0
```

To invert a boundary depth, the model-mapping layer would need to allow segment
tops and bottoms to reference parameters:

```yaml
segments:
  - name: sediment
    top_km: 0.0
    bottom_km:
      parameter: sediment.bottom

  - name: crust
    top_km:
      parameter: sediment.bottom
    bottom_km: 35.0
```

This is not only a YAML change. It requires:

1. Updating `ModelParameterization.vector_to_model()` so segment boundaries can
   be resolved from theta.
2. Adding validation that resolved segments are ordered, contiguous, positive
   thickness, and within model bounds.
3. Updating constraints so they can refer to segment names instead of fixed
   depth ranges when appropriate.
4. Ensuring discretization always preserves the current candidate segment
   boundaries dynamically.
5. Updating output metadata so variable interfaces can be summarized alongside
   Vs(z).

The change is moderate, not fundamental. The sampler already works on a flat
theta vector, so adding boundary-depth parameters does not require a new MCMC
engine. The main work is in parameter-to-model mapping, validation, and
segment-aware constraints.

For receiver-function joint inversion, this extension will be important:

- sediment thickness can be a parameter,
- Moho depth can be a parameter,
- RF-sensitive discontinuities should be preserved exactly during forward
  modeling,
- monotonicity constraints should often apply within selected segments rather
  than over broad fixed depth intervals.

## Relationship to reference implementations

The current design keeps the useful ideas from the reference codes but avoids
copying their full structure.

From MCMC_Compliance:

- Hard physical checks are treated as part of the prior support.
- Prior-only sampling is a meaningful way to inspect the geophysical model
  space.
- Posterior analysis should focus on physical quantities like Vs(z), not only
  scalar parameter histograms.

From pymcinv:

- The sampler keeps a local random-walk mode plus an occasional global uniform
  jump, analogous to the `step4uwalk` idea.
- Candidate models are checked for physical validity before expensive forward
  calculations dominate the run.
- B-spline coefficients parameterize smooth velocity structure, while the final
  scientific interpretation is usually the model ensemble in depth.

The main difference is that SeisForge now separates:

```text
parameter vector -> model mapping -> hard prior -> forward prediction -> likelihood -> sampler
```

This separation makes it easier to add new parameterizations, new observables,
and new samplers without rewriting the whole inversion workflow.
