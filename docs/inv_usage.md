# Inversion configuration reference

This page lists every configuration key of the single-station inversion. For a
first run see [getting started](inv_getting_started.md); for what the model
keys mean (segments, interfaces, transitions, layering, priors) see the
[model guide](model.md).

The inversion currently supports:

- Rayleigh phase- or group-velocity dispersion, Rayleigh H/V, or both jointly.
- Segmented Vs(z) models with constant, gradient, and B-spline profiles.
- Interfaces with sampled absolute depths, optional gradational widths, and
  per-segment Vp/rho scaling.
- Uniform, truncated normal, and half-normal parameter priors; hard physical
  constraints; a soft curvature prior.
- A random-walk Metropolis sampler with multiple chains (serial, thread, or
  process executors).
- NetCDF outputs for MCMC samples, Vs ensembles, and posterior predictions.

A synthetic P receiver function forward model exists
(`seisforge.inv.predict_receiver_function`, see
[Receiver functions](rf_workflow.md#synthetic-p-receiver-functions)), but RF
data cannot yet be configured as an inversion observable.

## Command line

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o OUTPUT [options]
```

| subcommand | data |
|---|---|
| `disp` | dispersion only |
| `hv` | Rayleigh H/V only |
| `disp-hv` | dispersion and H/V jointly |

| option | meaning |
|---|---|
| `--inv FILE` | inversion YAML (required) |
| `--obs FILE` | observation YAML; omit with `--prior-only` |
| `-o, --output DIR` | output directory (required) |
| `--prior-only` | sample the prior only; no data are used |
| `--with-prior` | run prior sampling before the posterior run |
| `--overwrite` | allow writing into an existing output directory |
| `--progress`, `--no-progress`, `--progress-every N` | progress display |
| `--log-level LEVEL` | file log level (default `INFO`) |

Without an installed entry point, use `python -m seisforge.cli.main inv ...`.
From Python, `seisforge.inv.run_inversion(kind="disp_hv", inv_file=...,
obs_file=..., output=...)` does the same (`kind` is `disp`, `hv`, or
`disp_hv`).

## Section names and unknown keys

Section names are lowercase (`model`, `inversion`, `sampler`, ...), like the
other SeisForge configurations. An unknown key anywhere in `inv.yaml` or
`obs.yaml` is an error that names the file, the section, and the closest
valid key, for example
`Unknown key in sampler: 'burnin' (did you mean 'burn_in'?)`, so a
misspelled option never falls back silently to its default.

The old capitalized section names (`Model`, `Inversion`, `Discretization`,
`Constraints`, `SoftPriors`, `Likelihood`, `Sampler`, `Observations`,
`Forward`) and the aliases `PhysicalConstraints` (for `constraints`) and
`Data` (for `observations`) are deprecated: they still load, with one warning
per run listing the new names. Giving the same section under an old and a new
name is an error. The `resolved_*.yaml` files of a run use the new names and
can be loaded again as inputs.

## obs.yaml

`obs.yaml` describes the data and how synthetics are computed. Observation file
paths belong here, not on the command line.

```yaml
observations:
  dispersion:
    file: vph_disp.dat
    wave: rayleigh
    kind: phase
    mode: 0
  hv:
    file: hv.dat
    wave: rayleigh
    mode: 0

forward:
  dispersion:
    algorithm: dunkin
    dc: 0.005
    dt: 0.025
  hv:
    algorithm: dunkin
    dc: 0.005
```

Observation files are three-column text, `period_s  value  one_sigma`, with
optional `#` comments; periods are sorted after reading. Relative paths are
resolved from the YAML file's folder.

| key | values | meaning |
|---|---|---|
| `observations.dispersion.file` | path | dispersion data (velocity in km/s) |
| `observations.dispersion.wave` | `rayleigh`, `love` | the workflow is designed around Rayleigh |
| `observations.dispersion.kind` | `phase`, `group` | observed quantity |
| `observations.dispersion.mode` | integer | 0 = fundamental mode |
| `observations.hv.file` | path | Rayleigh H/V data |
| `observations.hv.wave`, `.mode` | `rayleigh`, integer | fundamental-mode Rayleigh ellipticity |
| `forward.dispersion.algorithm` | disba algorithm, e.g. `dunkin` | |
| `forward.dispersion.dc` | km/s | root-search velocity increment |
| `forward.dispersion.dt` | s | period step for group velocity |
| `forward.hv.algorithm`, `.dc` | as above | ellipticity calculation |
| `observations.*.noise_parameter` | parameter name | optional; samples a noise scale for this data type (see [Likelihood](#likelihood)) |

## inv.yaml

`inv.yaml` has the sections `model`, `inversion`, `discretization`,
`constraints`, `soft_priors`, `likelihood`, and `sampler`.

### model

Numbers anywhere in `model` may be literals or parameter references
`{parameter: name}` defined in `inversion.parameters`.

**Segment layout** (fixed boundaries; `examples/06_inversion`):

```yaml
model:
  scaling: {vp: {method: brocher2005}, rho: {method: brocher2005}}
  segments:
    - name: sediment          # optional
      top_km: 0.0
      bottom_km: 1.0
      profile: {type: gradient, top: {parameter: sediment.vs_top}, bottom: 2.0}
```

| key | meaning |
|---|---|
| `scaling` | Vp and rho relations (below); default Brocher (2005) |
| `segments[].top_km`, `.bottom_km` | segment depths in km; the first starts at 0 and each bottom equals the next top |
| `segments[].profile` | Vs profile (below) |
| `segments[].name` | optional name, used by `layers_per_segment` |

**Interface layout** (sampled interface depths and widths;
`examples/reference_configs/inversion_interface_model.yml`,
[model guide, section 3](model.md#3-interfaces)):

| key | meaning |
|---|---|
| `zmax_km` | bottom of the parameterized model (top of the half-space), km |
| `scaling` | default scaling for segments without their own |
| `interfaces[].depth_km` | absolute depth of the interface, km (literal or parameter) |
| `interfaces[].width_km` | 10-90 % transition thickness, km; `0` (default) = hard step |
| `interfaces[].shape` | `sigmoid` (default) or `smootherstep` |
| `interfaces[].name` | optional name |
| `segments[]` | one more than interfaces, top to bottom; `profile`, optional `name` and `scaling`; no `top_km`/`bottom_km` |

**Profiles:**

| `type` | keys |
|---|---|
| `constant` | `value` |
| `gradient` | `top`, `bottom`: Vs at the segment's top and bottom |
| `bspline` | `coefficients` (list), `degree` (default 3), `knot_spacing` (`geometric` default, or `uniform`), `knot_alpha` (default 2.0) |

See the [model guide](model.md#meaning-of-the-b-spline) for what B-spline
coefficients and knot settings mean.

#### Scaling

Vp and rho are derived from Vs. `scaling` has a `vp` and a `rho` entry, each
with a `method` and its parameters:

| key | method | parameters |
|---|---|---|
| `vp` | `brocher2005` | none; valid for Vs in 0-4.5 km/s |
| `vp` | `constant_vp_vs` | `vp_vs` |
| `rho` | `brocher2005` | none; from Brocher Vp |
| `rho` | `gardner1974` | `rho = a Vp^b`; `a` (default 1.741), `b` (default 0.25); km/s, g/cm^3 |
| `rho` | `constant` | `rho` in g/cm^3 |

```yaml
scaling:
  vp: {method: constant_vp_vs, vp_vs: 1.8}
  rho: {method: constant, rho: 3.35}
```

In the interface layout a segment's own `scaling` overrides the model default.
Segments whose Vs can exceed 4.5 km/s (mantle) need a relation other than
Brocher.

### inversion parameters

`inversion.parameters` defines theta, in this order.

```yaml
inversion:
  parameters:
    - name: crust.c0
      initial: 3.05
      prior: {type: uniform, bounds: [2.50, 3.55]}
      proposal: {sigma: 0.035}
```

| key | meaning |
|---|---|
| `name` | unique; referenced from `model` as `{parameter: name}` |
| `initial` | starting value for `initial_strategy: theta0`; must lie inside the bounds and give a valid model |
| `prior.bounds` | `[lower, upper]`; required for every prior type; the sampler never leaves them |
| `prior.type` | `uniform` (default), `normal`, or `half_normal` |
| `prior.mean`, `prior.sigma` | for `normal` (truncated to the bounds) |
| `prior.scale` | for `half_normal` (peaks at the lower bound; for interface widths) |
| `proposal.sigma` | random-walk step for this parameter; default 5 % of the bound range |

`proposal.sigma` only affects sampling efficiency, not the prior.

### discretization

Converts the continuous model to forward layers
([model guide, section 4](model.md#4-from-continuous-model-to-forward-layers)).

```yaml
discretization:
  dz: 0.25
  zmax: 15.0
  force_depths: []
  layers_per_segment: {sediment: 8, crust: 30}   # optional
  transition_layers: 16
  layer_value: midpoint
```

| key | default | meaning |
|---|---|---|
| `dz` | 0.5 | target layer thickness in km, for segments without a fixed count |
| `zmax` | model bottom | deepest discretized depth, km |
| `force_depths` | `[]` | extra fixed layer boundaries, km. Hard boundaries and interfaces are always preserved; do not list them here |
| `layers_per_segment` | none | layers per segment: one integer, or a mapping from segment name to count. Recommended when interface depths are sampled |
| `transition_layers` | 16 | equal layers across each gradational interface support (minimum 2) |
| `layer_value` | `midpoint` | `midpoint` or `mean` (depth average) |
| `dz_growth_ratio` | 1 | layer thickness growth with depth (see below); 1 is uniform `dz`, bit-for-bit as before |
| `dz_max` | none | largest layer thickness in km for the graded layering; at least `dz` |

#### Graded layering

With `dz_growth_ratio` r > 1 the target thickness at depth z (km, from the
surface) is

    h(z) = min(dz_max, dz + (r - 1) z),

so layers stacked from the surface are `dz`, `dz r`, `dz r^2`, ... until they
reach `dz_max`. Each segment is split into the fewest layers that keep every
layer at most `h` (equal steps of the stretched coordinate
`integral dz / h`, formula in `seisforge.inv.layering`). Segment boundaries,
interfaces, transition supports, `force_depths` and `zmax` are kept exactly as
with uniform `dz`; transition supports still get `transition_layers` and
segments with a `layers_per_segment` count are not graded.

```yaml
discretization:
  dz: 0.0625          # first-layer thickness, km
  dz_growth_ratio: 1.1
  dz_max: 1.0
  zmax: 15.0
```

Forward cost (dominated by disba's ellipticity) grows linearly with the number
of layers, while the error of piecewise-constant layers is set by thickness
where velocity changes fastest, which is usually near the surface. Graded
layering spends layers there.

#### Choosing a layering

**Rule:** the change in predicted observables caused by the layering should be
well below the data noise. As a working level take a maximum of **0.2 sigma**
over the period range for models that the posterior can reach (assumed value,
not from a source: 0.2 sigma on every datum shifts chi^2 by at most
0.04 per datum).

Check it with `layering_observable_error`, which compares the predicted phase
velocity and ellipticity of the configured layering with a fine uniform
reference layering (default `dz / 4`; give a smaller `reference_dz_km` to be
sure the reference itself has converged):

```python
from seisforge.inv.config import load_inversion_input
from seisforge.inv.layering import layering_observable_error
from seisforge.inv.setup import build_inversion_setup

inp = load_inversion_input(kind="disp_hv", inv_file="inv.yaml", obs_file="obs.yaml")
setup = build_inversion_setup(inp.run_config)
model = setup.parameterization.vector_to_model(setup.parameterization.theta0())
error = layering_observable_error(
    model,
    setup.prior.discretization,
    dispersion=setup.target.dispersion,
    hv=setup.target.hv,
    reference_dz_km=0.03125,
)
print(error.summary())   # max differences in data units and in sigma
```

Run it for a few representative models (for example prior draws from
`seisforge.inv.samplers.draw_valid_initial_thetas`), not only the initial
one. Every posterior run writes the same check for the initial model to the
log and to the `discretization_*` attributes of the output files.

STA001 example (truth model plus 15 prior draws, 14 periods 0.8-7 s,
reference uniform 1/32 km; data noise 0.03 km/s and 0.02):

| layering | layers | max abs d c (km/s) | max abs d H/V |
|---|---|---|---|
| uniform `dz: 0.25` (example setting) | 61 | 0.0056 (0.19 sigma) | 0.036 (1.8 sigma) |
| uniform `dz: 0.125` | 121 | 0.0017 (0.06 sigma) | 0.0096 (0.48 sigma) |
| `dz: 0.0625`, r 1.1, `dz_max: 1.0` | 35 | 0.0020 (0.07 sigma) | 0.0023 (0.12 sigma) |
| `dz: 0.0625`, r 1.2, `dz_max: 1.0` | 27 | 0.0050 (0.17 sigma) | 0.0048 (0.24 sigma) |
| `dz: 0.125`, r 1.2, `dz_max: 1.0` | 23 | 0.0078 (0.26 sigma) | 0.0095 (0.48 sigma) |
| `dz: 0.0625`, r 1.5, `dz_max: 1.0` | 21 | 0.024 (0.80 sigma) | 0.019 (0.95 sigma) |

The H/V error comes almost entirely from the 0-1 km sediment gradient and
falls about fourfold per halving of the layer thickness there; a thin first
layer matters more than the deep layers. The uniform 0.25 km of the example
is above the 0.2 sigma level for H/V (the example data were computed with the
same layering, so the example is self-consistent but not converged with
respect to the continuous model).

### constraints

Hard physical constraints. A model that fails one gets log prior -inf and is
rejected before any forward calculation.

```yaml
constraints:
  constraints:
    - type: positive_velocity
    - type: vp_vs_range
      bounds: [1.45, 2.70]
    - type: monotonic_vs
      depth_range: [0.0, 2.0]
      tolerance: 0.03
      dz: 0.1
```

Depth-sampled constraints evaluate Vs every `dz` km over `depth_range`
(default: the whole model), always including segment edges.

| `type` | keys (default) | requires |
|---|---|---|
| `positive_velocity` | none | Vs, Vp, rho finite and positive. Keep it enabled |
| `vp_gt_vs` | `margin` (0) | `Vp - Vs > margin` in every layer |
| `vp_vs_range` | `bounds` | Vp/Vs inside bounds in every layer; usually more informative than `vp_gt_vs` |
| `boundary_non_decreasing_vs` | `tolerance` (0), km/s | Vs does not drop by more than `tolerance` across a segment boundary or interface |
| `vs_range` | `bounds`, `depth_range`, `dz` (0.1) | Vs inside bounds |
| `monotonic_vs` | `tolerance` (0), `depth_range`, `dz` (0.1) | Vs never decreases by more than `tolerance` between samples |
| `max_vs_gradient` | `max_abs` (km/s per km), `depth_range`, `dz` (0.1) | absolute Vs gradient at most `max_abs` |

Notes:

- With Brocher scaling, `vp_vs_range` checks a derived property; very slow
  sediments can have high Vp/Vs, so start broad.
- `boundary_non_decreasing_vs` suits simple increasing models; use it with
  care where low-velocity zones are expected.
- `monotonic_vs` and `max_vs_gradient` act over fixed depth ranges, not named
  segments, so they do not follow a sampled interface. A global
  `max_vs_gradient` also limits gradational interfaces
  ([model guide, section 5](model.md#5-constraints-and-priors-define-the-model-space)).

### soft_priors

Finite log-prior terms: they do not reject models but prefer some valid models
over others. Prior-only runs include them.

```yaml
soft_priors:
  priors:
    - type: vs_curvature
      depth_range: [0.0, 10.0]
      dz: 0.05
      sigma: 0.20
```

`vs_curvature` adds `-0.5 * (RMS[d2Vs/dz2] / sigma)^2`, evaluated within each
segment separately (segment boundaries are not smoothed). `sigma` is in
km/s/km^2; smaller is smoother. `dz` defaults to 0.05 km; the RMS keeps the
strength roughly independent of `dz`. Use it to discourage short-wavelength
B-spline oscillations, together with the hard constraints rather than instead
of them.

### likelihood

```yaml
likelihood:
  dispersion_weight: 1.0
  hv_weight: 1.0
```

Each data type $k$ contributes a Gaussian term with its residuals $r_i$ and the
one-sigma uncertainties $\sigma_i$ of the data file,
$\chi^2_k = \sum_i (r_i/\sigma_i)^2$:

$$
\log L = \sum_k w_k \left[ -\frac{\chi^2_k}{2 h_k^2} - N_k \ln h_k \right]
$$

with weights $w_k$ (`dispersion_weight`, `hv_weight`, default 1) and noise
scales $h_k$ (default 1, giving the plain $-\tfrac12 \sum_k w_k \chi^2_k$).

**Fixed weights** change the relative influence of the two data types by
hand. Judge them from the separate dispersion and H/V residuals; a single
combined misfit of quantities with different units says little.

**Sampled noise scales** let the data set the balance instead (a hierarchical
noise model, Bodin et al. 2012). Name a parameter that holds $\log_{10} h$ in
the observation, and define it like any other parameter:

```yaml
# obs.yaml
observations:
  dispersion:
    file: vph_disp.dat
    noise_parameter: noise.dispersion
# inv.yaml
inversion:
  parameters:
    - name: noise.dispersion
      initial: 0.0
      prior: {type: uniform, bounds: [-0.5, 0.5]}   # h from 0.32 to 3.2
      proposal: {sigma: 0.03}
```

The data uncertainties become $h\,\sigma_i$: $h > 1$ says the file's
uncertainties are too small for the misfit the model space allows, $h < 1$
that they are too large. The $N_k \ln h_k$ term stops $h$ from growing to
absorb any misfit; at fixed residuals the best value is
$h_k^2 = \chi^2_k / N_k$. A uniform prior on $\log_{10} h$ treats factors
above and below 1 alike.

- A noise parameter must not appear in `model`, and the weight of that data
  type must stay 1: the sampled scale replaces the weight.
- The scale corrects the overall error level of a data type, not correlated
  errors or a wrong model space; read a large $h$ as a sign to check both.
- With noise-free synthetic data every $h$ runs to its lower bound, because
  the true model fits perfectly. Synthetic tests need noise added.
- $\log_{10} h$ is stored in `theta` like any other parameter; the run
  attributes record `dispersion_noise_parameter` and `hv_noise_parameter`.

### sampler

```yaml
sampler:
  n_steps: 5000
  burn_in: 1000
  thin: 5
  n_chains: 4
  executor: thread
  seed: 20260702
  initial_strategy: prior_uniform
  proposal_sigma: from_parameters
  global_jump_interval: 1000
```

| key | default | meaning |
|---|---|---|
| `n_steps` | 1000 | Metropolis steps per chain |
| `burn_in` | 0 | early steps not stored |
| `thin` | 1 | store every Nth step after burn-in; stored draws per chain = `(n_steps - burn_in) / thin` |
| `n_chains` | 1 | independent chains |
| `executor` | `serial` | `serial`, `thread`, or `process` |
| `max_workers` | Python default | concurrent workers for `thread`/`process` |
| `seed` | none | random seed; `null` = fresh random state |
| `initial_strategy` | `theta0` | `theta0` (all chains at `initial`) or `prior_uniform` (valid points drawn from the bounds) |
| `initial_thetas` | none | explicit start: one theta vector, or one per chain; overrides `initial_strategy` |
| `max_initial_attempts` | 10000 | draws allowed for `prior_uniform` |
| `proposal_sigma` | `from_parameters` | `from_parameters`, one number for all parameters, or a list |
| `use_parameter_bounds` | true | keep proposals inside the prior bounds |
| `global_jump_interval` | none | every N steps propose uniformly over the bounds |
| `tempering` | none | parallel tempering: `{}` for the default ladder, `{n_temperatures, max_temperature}`, or `{temperatures: [...]}`; plus `swap_interval` (below) |
| `adaptation` | none (off) | adaptive proposal covariance during burn-in: `{initial_diagonal_steps, update_interval_steps}`, optional `regularization_fraction` and `discard_early_history_fraction` (below) |

**Executor.** `thread` is usually best on one machine: disba and the RF kernel
release the GIL, so chains run in parallel (four threads ran disba about 3.6x
faster than serial), and progress display works. `process` avoids the GIL
completely, which helps when Python-side prior code dominates; it has no
progress display, and each worker compiles the numba kernels once. `serial`
is simplest for debugging.

**Seed.** It controls initial draws, proposals, and acceptance, with an
independent stream per chain. Fix it while developing a configuration and
record it for archived results; use different seeds for repeated production
runs.

**Proposals and global jumps.** Small proposal sigmas give high acceptance but
slow mixing; large ones give many rejections. Acceptance of roughly 0.15-0.4
is healthy. A global jump (the pymcinv `step4uwalk` idea) helps a chain leave
a local region; values of 500-2000 are conservative, and frequent jumps lower
acceptance.

**Parallel tempering.** A chain stuck in a lower mode (a *lagging chain*,
or R-hat far above 1 with chains at different log posteriors) cannot leave it
by small steps. With `tempering`, each chain becomes a ladder of replicas at
temperatures $1 = T_1 < T_2 < \dots < T_n$, sampling
$p(\theta)\, L(\theta)^{1/T_i}$. Hot replicas see a flattened likelihood and
cross between modes; adjacent replicas exchange states every `swap_interval`
steps with probability

$$
\alpha = \min\Big(1,\ \exp\big[(1/T_i - 1/T_{i+1})(\ln L_{i+1} - \ln L_i)\big]\Big),
$$

which keeps the $T = 1$ replica an exact sample of the posterior. Only that
replica is stored (Geyer 1991; Earl & Deem 2005).

```yaml
sampler:
  tempering: {n_temperatures: 5, max_temperature: 30}   # geometric ladder
  # tempering: {temperatures: [1, 3, 10, 30]}            # explicit ladder
  # tempering: {}                                        # 5 temperatures up to 30
```

- Cost: a ladder of $n$ costs about $n$ times a plain run.
- `swap_acceptance(chain, temperature_pair)` in `*_mcmc.nc` and the log show
  how often neighbours exchange. Values well above 0.1 on every pair are
  fine; a pair near 0 needs another temperature between them.
- `max_temperature` must flatten the barrier between modes: roughly, it
  should be comparable to the drop of $\ln L$ between them.
- Replica $i$ proposes with `proposal_sigma` $\times \sqrt{T_i}$ (an assumed
  scaling: a likelihood tempered by $1/T$ widens a Gaussian posterior by
  $\sqrt{T}$).
- Prior-only runs ignore `tempering`: the prior has no likelihood to temper.
- Run untempered first; turn tempering on when the diagnostics show separate
  modes.

**Adaptive proposals.** Parameters that trade off against each other, such
as neighbouring B-spline coefficients, form a narrow, tilted ridge in the
posterior. A proposal that moves each parameter independently must then use
steps as small as the ridge is narrow, and the chain crawls along it: split
R-hat stays far above 1 and the ESS in single digits even after thousands of
steps. Tempering does not help here; it helps with separate modes. With
`adaptation`, each chain learns the shape of the ridge during burn-in
(adaptive Metropolis, Haario et al. 2001):

```yaml
sampler:
  n_steps: 20000
  burn_in: 8000
  adaptation:
    initial_diagonal_steps: 2000   # steps with proposal.sigma before the first update
    update_interval_steps: 200     # steps between updates
    # regularization_fraction: 1.0e-4
    # discard_early_history_fraction: 0.5   # default 0: the whole history
```

- For the first `initial_diagonal_steps` steps the chain uses the diagonal
  `proposal.sigma`. From then on, every `update_interval_steps` steps, the
  proposal covariance becomes
  $C = \frac{2.38^2}{d}\,\big(\mathrm{Cov}(\text{chain so far}) + \varepsilon\,\mathrm{diag}(\sigma^2)\big)$,
  with $d$ the number of parameters and $\sigma$ the starting
  `proposal.sigma`. $2.38^2/d$ is the scaling that is optimal for Gaussian
  posteriors (Gelman et al. 1996); $\varepsilon$ =
  `regularization_fraction` (default $10^{-4}$, assumed) keeps every
  parameter a small step of its own unit even if it has not moved yet.
- The last update falls at or before the end of burn-in; afterwards $C$ is
  frozen, so the stored draws come from an ordinary Metropolis chain with a
  fixed proposal and sample the posterior exactly. `burn_in` must be at least
  `initial_diagonal_steps`, which must exceed the number of parameters.
- Choose `initial_diagonal_steps` long enough for the chain to reach the
  high-probability region (watch the log posterior in the progress lines),
  and leave several updates after it: a covariance estimated from the
  transient start is too wide or wrongly tilted. Keep `proposal.sigma`
  sensible, because it still drives the start.
- `discard_early_history_fraction` $f$ (default 0, the whole history) leaves
  the earliest fraction of the states so far out of every update: after $n$
  steps the covariance uses $x_k, \dots, x_n$ with
  $k = \lfloor f\,(n+1) \rfloor$, so the window grows with the chain and
  $f = 0.5$ uses the most recent half. This removes the drift from a distant
  start, which otherwise makes the learned steps too wide and the
  post-burn-in acceptance low. On a 4-parameter Gaussian started 30 standard
  deviations away (`tests/test_inv_adaptive_proposal.py`), $f = 0.5$ brought
  the frozen covariance within 6-43 % of the true one (whole history: off by
  a factor of about 150), post-burn-in acceptance from 0.01-0.05 to
  0.19-0.36, and ESS up 6-21 times. It needs a longer `burn_in`, because each
  update rests on fewer states.
- The log warns when `burn_in` looks short for the adaptation (assumed
  thresholds): fewer adapted steps than `initial_diagonal_steps`, or fewer
  than 100 states per parameter in the window of the last update. It also
  prints the acceptance rate of each chain after burn-in, where the proposal
  is fixed, with a note below 0.1 or above 0.6 (assumed).
- Each chain adapts on its own history only, so chains stay independent and
  the executors give the same result. With tempering each temperature adapts
  its own covariance from the states it held, swaps included. Global jumps
  keep their uniform proposal. The prior-only run (`--prior-only`,
  `--with-prior`) adapts as well, with its own chains.
- How to read the result: the log prints, per chain, the frozen proposal
  step size of every parameter (`frozen proposal sd`), and `*_mcmc.nc` holds
  the frozen covariance `proposal_covariance(chain, parameter,
  parameter_column)` (with tempering, all temperatures in
  `proposal_covariance_tempered`) and the settings in attrs
  `adaptation_*`. Acceptance after burn-in near 0.2-0.35 is what the scaling
  aims for in many dimensions. Chains whose frozen covariances differ
  strongly did not reach the same region during burn-in; lengthen it. Judge
  the improvement by R-hat and ESS, as without adaptation.

## Output files

For `disp-hv` (other kinds use their own prefix):

| file | content |
|---|---|
| `disp_hv_mcmc.nc` | `theta(chain, draw, parameter)`, `log_prob`, `log_prior`, `log_parameter_prior`, `log_physical_hard_prior`, `log_soft_prior`, `log_likelihood`, acceptance, initial and final theta; `vs_curvature_rms` with the curvature prior; convergence diagnostics `split_rhat(parameter)`, `effective_sample_size(parameter)`, and attrs `max_split_rhat`, `min_effective_sample_size`, `lagging_chains`; with tempering, `swap_acceptance(chain, temperature_pair)` and attr `temperatures`; with `adaptation`, `proposal_covariance(chain, parameter, parameter_column)` (and `proposal_covariance_tempered`) and attrs `adaptation_*`; forward-solver failures `forward_attempts(chain)`, `forward_failures(chain)`, `forward_failure_rate(chain)` (with tempering also `*_cold`) and attr `forward_failure_top_errors` |
| `disp_hv_vs.nc` | Vs(z) per sample and depth-wise `mean`, `std`, `minimum`, `maximum`, `p01`...`p99`, `median`, `best_vs` |
| `disp_hv_predictive.nc` | predicted dispersion and H/V per sample, their percentiles, and the observed values and sigmas |
| `disp_hv.log` | settings, acceptance rates, forward failures, best theta, output paths |
| `resolved_inv.yaml`, `resolved_obs.yaml` | input snapshots |
| `resolved_config.yaml` | merged internal configuration without observation arrays |

Prior-only runs write `disp_hv_prior_mcmc.nc` and `disp_hv_prior_vs.nc` and no
predictive file.

### Convergence diagnostics

Every run reports, per parameter, the split R-hat (Gelman et al. 2013) and the
multi-chain effective sample size (Vehtari et al. 2021), and lists *lagging
chains*: chains whose log posterior stays wholly below that of the best chain
(their 95th percentile under its 5th), i.e. chains stuck in a lower mode.
They need at least two chains and four stored draws per chain.

- R-hat near 1 means the chains sample the same distribution. The log warns
  above 1.1; use 1.01 for final results.
- The ESS counts how many independent draws the correlated chains are worth.
  The log warns below 100; percentiles of the Vs ensemble need several
  hundred.
- A lagging chain is reported, never dropped. Rerun with more steps, more
  chains, or parallel tempering before trusting the posterior.

### Forward-solver failures

disba cannot solve every model inside the prior: its fundamental-mode root
search fails mostly when the half-space is slower than a layer above it or a
layer has Vp/Vs close to 1, and higher modes or H/V can stop short of the
longest periods. Such a proposal gets a log likelihood of `-inf` and is
rejected, which removes that part of the prior from the posterior. The
posterior `*_mcmc.nc` therefore records, per chain:

- `forward_attempts`: evaluations inside the hard prior support, i.e. forward
  solver runs (the starting point and every proposal that passed the prior);
- `forward_failures`: attempts whose forward solver failed;
- `forward_failure_rate`: their ratio.

With parallel tempering these count every replica of the chain; the
`*_cold` variables count the `T = 1` replica alone, the one whose samples are
stored. `attrs["forward_failure_top_errors"]` lists the two most frequent
error messages. The log repeats the counts and warns when a chain's rate
exceeds 1 % (an assumed threshold, `FORWARD_FAILURE_WARNING_RATE`), quoting
those messages. A few failures are harmless; a high rate means the stored
posterior is the prior truncated by the solver: tighten the constraints that
allow such models (for example require Vs to increase into the half-space) or
drop the periods or modes that fail. Prior-only runs call no solver and have
no such variables, and posterior predictive files are not counted.

Errors that are not solver failures (wrong input types, unsorted periods,
programming errors) stop the run with the original exception.

## Tuning a configuration

1. Run `--prior-only` and inspect the prior Vs ensemble. If a plausible
   profile is outside it, revise bounds, B-spline depth and knots,
   monotonicity tolerance, maximum gradient, or Vs range; do not widen plot
   envelopes instead.
2. If `prior_uniform` initialization fails, the hard prior leaves too little
   valid volume; see the [model guide](model.md#5-constraints-and-priors-define-the-model-space).
3. Run the posterior and compare the dispersion and H/V residuals separately
   before changing likelihood weights.
4. Adjust `proposal.sigma` for acceptance of roughly 0.15-0.4.
