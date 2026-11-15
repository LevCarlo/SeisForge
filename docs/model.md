# Inversion model guide

This page explains the model space of the single-station inversion: how a
vector of sampled numbers becomes a Vs(z) profile, how interfaces and
gradational transitions work, how the continuous model becomes the layers the
forward solvers see, and how constraints shape the prior. Every YAML key is
listed in the [configuration reference](inv_usage.md); a first run is in
[getting started](inv_getting_started.md).

## 1. The Model Pipeline

~~~text
inv.yaml scalar parameters (theta)
    -> ModelParameterization.vector_to_model(theta)
    -> ParameterizedVsModel: VsSegment objects separated by Interface objects
    -> ParameterizedVsModel.to_layered_model(DiscretizationConfig)   (inv.layering)
    -> LayeredModel passed to the forward solvers (disba, synthetic P RF)
~~~

The sampler proposes theta only. Model mapping, hard geophysical constraints,
and the forward solvers are separate steps, so any proposed Vs(z) can be built
and inspected without running a forward calculation (section 6).

Vs is the inverted property. Vp and density are derived from Vs by a scaling
relation (section 3) and are not sampled.

## 2. Segments and Profiles

The model is an ordered list of depth **segments**. Each segment has a Vs
**profile**:

| type | inputs | typical use |
| --- | --- | --- |
| `constant` | one velocity | known cap, simple layer, or half-space |
| `gradient` | top and bottom velocities | sediment compaction, linear crust |
| `bspline` | coefficients, degree, knot layout | flexible crustal or mantle profile |

Any number may be fixed (`value: 2.50`) or sampled
(`value: {parameter: upper_crust.vs}`). Only parameter references become
elements of theta; literal values stay fixed for the whole run.

Segment boundaries come either from fixed `top_km`/`bottom_km` values (the
segment layout, used by `examples/STA001`) or from interfaces whose depths can
be sampled (the interface layout, section 3). In both cases a hard boundary is
a model feature: no forward layer ever straddles it.

### Meaning of the B-spline

A B-spline profile is evaluated on the segment's normalized coordinate:

~~~text
x = (z - top) / (bottom - top),  0 <= x <= 1
Vs(z) = sum_i B_i(x) c_i
~~~

The basis is open-clamped, so the first coefficient is the velocity at the
segment top and the last is the velocity at its bottom. Interior
coefficients are overlapping basis weights: they control local shape but are
not velocities at fixed depths.

Knots are placed on `x`, not on depth. With `knot_spacing: geometric`,
`knot_alpha > 1` makes intervals smaller near the segment top (more shallow
flexibility), `0 < knot_alpha < 1` concentrates flexibility toward the bottom,
and `knot_alpha = 1` is uniform. Because the layout is relative, moving a
segment from 2-15 km to 10-40 km keeps the same relative knot layout, and a
segment whose bottom is a sampled interface stretches with it.

## 3. Interfaces

The interface layout describes the model as segments separated by interfaces.
Interface depths are **absolute** depths and may be sampled; segment depths are
derived from them.

~~~yaml
model:
  zmax_km: 80.0                 # bottom of the parameterized model (half-space top)
  scaling:                      # default for segments without their own scaling
    vp: {method: brocher2005}
    rho: {method: brocher2005}
  interfaces:                   # ordered top to bottom
    - name: basement
      depth_km: {parameter: basement.depth}
    - name: moho
      depth_km: {parameter: moho.depth}
      width_km: {parameter: moho.width}   # 10-90 % transition thickness; 0 = hard step
      shape: sigmoid                      # or smootherstep
  segments:                     # one more than interfaces; no top_km/bottom_km
    - name: sediment
      profile: {type: gradient, top: {parameter: sed.vs_top}, bottom: {parameter: sed.vs_bot}}
      scaling:
        vp: {method: constant_vp_vs, vp_vs: 2.2}
        rho: {method: gardner1974}
    - name: crust
      profile: {type: bspline, coefficients: [...]}
    - name: mantle
      profile: {type: gradient, top: 4.4, bottom: {parameter: mantle.vs_bot}}
      scaling:
        vp: {method: constant_vp_vs, vp_vs: 1.8}
        rho: {method: constant, rho: 3.35}
~~~

A complete file is `examples/configs/interface_model.yml`.

**Profiles stretch with the interfaces.** Every profile is defined on its
segment's local coordinate, so when the Moho deepens the crustal B-spline
stretches with it: a coefficient keeps meaning "Vs at this relative position in
the crust". A gradient's `top`/`bottom` are always the values at the segment's
current boundaries.

**Gradational interfaces.** With `width_km > 0`, Vs, Vp, and rho blend the two
neighbouring segments:

$$
v(z) = \big(1 - S(z)\big)\, v_\text{upper}(z) + S(z)\, v_\text{lower}(z)
$$

$S$ is 0.5 at `depth_km`, 0.1 and 0.9 at `depth_km` $\mp$ `width_km`/2, and
exactly 0 / 1 outside a finite support. Two shapes are available
(`seisforge.inv.transitions`):

| shape | form | support half-width | max gradient |
|---|---|---|---|
| `sigmoid` | logistic with steepness `k = 2 ln 9 / w`, truncated and rescaled | `2 w` | about `1.1 dv / w` |
| `smootherstep` | `6x^5 - 15x^4 + 10x^3`, C2-continuous | `0.987 w` | about `0.93 dv / w` |

Outside the support the model is exactly the hard-step model, and as the width
goes to zero the model converges continuously to the hard step. Each neighbour
is evaluated with depths clamped to its own segment, so B-splines never
extrapolate. A support may not reach the surface, overlap a neighbouring
interface's support, or extend below `zmax_km`; such samples are rejected by
the prior (log prior = -inf), not raised.

**Per-segment scaling.** Vp and rho are computed from each segment's own Vs
with that segment's `scaling` (or the model default), then blended with the
same weights as Vs. A crust relation is therefore never applied to sediment
velocities inside a transition, and Vp/Vs inside a transition stays between
the values of the two sides. Brocher (2005) is only valid for Vs <= 4.5 km/s,
so mantle segments need their own relation, as above. The available relations
are listed in the
[configuration reference](inv_usage.md#scaling).

**Priors for interface parameters.**

~~~yaml
- name: moho.depth            # independent estimate, e.g. H-k: 35 +/- 2 km
  initial: 35.0
  prior: {type: normal, mean: 35.0, sigma: 2.0, bounds: [28.0, 45.0]}
  proposal: {sigma: 0.8}
- name: moho.width            # hard step unless the data ask for a gradient
  initial: 0.0
  prior: {type: half_normal, scale: 2.0, bounds: [0.0, 8.0]}
  proposal: {sigma: 0.5}
~~~

`half_normal` peaks at the lower bound, so a hard interface is the default and
a gradational one has to improve the fit enough to pay for its prior cost.
Choose `scale` near the thinnest transition the data can resolve. Depth
parameters need their own `proposal.sigma`, for example about 1 km for the
Moho and 0.02-0.05 km for a sediment base.

At setup, every parameter reference must be defined and the initial values must
give a valid model; both are checked before sampling starts.

**Current limits.** Constraints are defined over fixed depth ranges
(section 5), so a range does not follow a moving interface; choose ranges that
are valid for every allowed interface depth. Interface depths and widths are
reported as ordinary parameters in the MCMC samples; the Vs ensemble file
summarizes Vs(z) only.

## 4. From Continuous Model to Forward Layers

Every forward solver sees only the `LayeredModel`, so the conversion
(`seisforge.inv.layering`) follows fixed rules:

1. A hard interface is always a layer boundary, at its exact depth, whatever
   `dz` is. No layer straddles a velocity jump.
2. The part of each segment outside interface supports is split into layers by
   `dz`, or by a fixed count per segment (`layers_per_segment`). `dz` can
   grow with depth (`dz_growth_ratio`, `dz_max`; see
   [graded layering](inv_usage.md#graded-layering)).
3. The support of each gradational interface gets `transition_layers` equal
   layers.
4. Each layer takes Vs, Vp, and rho from the continuous model at its midpoint
   (`layer_value: midpoint`, default) or as the depth average
   (`layer_value: mean`).
5. The half-space takes the properties at the bottom depth (`zmax`).

~~~yaml
discretization:
  zmax: 80.0
  layers_per_segment: {sediment: 8, crust: 40, mantle: 12}   # or one integer, or use dz
  transition_layers: 16
  layer_value: midpoint
~~~

**Use `layers_per_segment` when interface depths are sampled.** With `dz` the
layer count changes in steps as an interface moves, and so does the forward
prediction. On STA001, moving the sediment base in 1 m steps changed the log
posterior by up to 4.2 at a step with `dz: 0.25`, 12 times the typical change;
with fixed counts the largest step equaled the typical one. A fixed count
moves layer boundaries continuously with the interfaces. Segments missing from
a `layers_per_segment` mapping fall back to `dz`. The steps matter most where
layers are thick compared with the shortest wavelength in the data (shallow
structure, high frequencies); in a crustal-scale synthetic test with receiver
functions (a <= 2.5) and `dz: 1.0`, the steps were negligible.

**Check the conversion.** `layering_report` compares the layered model with
the continuous one:

~~~python
from seisforge.inv import build_inversion_setup, layering_report, load_yaml

setup = build_inversion_setup(load_yaml("examples/configs/interface_model.yml"))
theta = setup.parameterization.theta0()
theta[setup.parameterization.names.index("moho.width")] = 3.0
model = setup.parameterization.vector_to_model(theta)
layered = model.to_layered_model(setup.prior.discretization)
print(layering_report(model, layered).summary())
# 67 layers (min 0.1 km); hard interfaces preserved: True (max offset 0 km);
# transition layers: [16]; max |dVs| 0.08109 km/s;
# vertical time error Vs -5.377e-05, Vp -4.508e-05
~~~

It reports whether every hard interface is a layer boundary, how many layers
resolve each transition, the largest Vs difference inside any layer, and the
relative error of vertical S and P travel time (`integral dz / v`). Vertical
travel time controls receiver-function delays such as `t_Ps`, so keep that
error well below the timing uncertainty of the RF data you will invert.

`LayeredModel.segment_index` records which segment each layer came from, and
`LayeredModel.to_hvsr_model()` converts to the `seisforge.hvsr` diffuse-field
H/V model.

## 5. Constraints and Priors Define the Model Space

The prior has three parts:

1. **Parameter priors** (`inversion.parameters[].prior`): a box in theta, with
   optional normal or half-normal shapes.
2. **Hard constraints** (`constraints`): physical checks on the built model,
   for example Vp/Vs range, Vs range, monotonicity, maximum gradient. A model
   that fails one gets log prior = -inf before any forward calculation.
3. **Soft priors** (`soft_priors`): finite penalties, currently Vs curvature,
   that prefer some valid models over others.

Hard constraints turn a box of B-spline coefficients into a physically
plausible model space. Constraint checks sample Vs(z) on a depth grid that
always includes segment edges, so jumps at segment boundaries are seen even
when `dz` does not land on them. Two interactions with interfaces are worth
knowing:

- `boundary_non_decreasing_vs` compares the two segment profiles at the
  interface depth, so it also applies to gradational interfaces.
- `max_vs_gradient` sees the transition gradient, about `1.1 dv / w` for a
  sigmoid, so it also bounds jump/width. Restrict its `depth_range` if it
  should not act on interfaces.

A prior-only run (`--prior-only`) samples exactly this model space. Run it
first: if a plausible profile is outside the prior ensemble, the data cannot
bring it back.

**`initial` is a starting point, not a prior mean.** A practical two-stage
workflow:

1. Use broad but defensible bounds and physical constraints to explore the
   posterior support. `initial_strategy: prior_uniform` gives diverse valid
   chain starts.
2. Inspect the depth-domain Vs ensemble. If the first-stage posterior is away
   from all bounds, a second run may use focused but buffered bounds.

Do not tighten around a single best model. A posterior that repeatedly touches
a bound is evidence that the bound limits the result. `prior_uniform` can fail
when independent coefficient bounds combined with strict monotonic and
gradient constraints leave very little valid volume; then start from a known
valid `theta0`, revise the constraints or bounds, or simplify the
parameterization before raising `max_initial_attempts`.

## 6. Inspect a Model Before Sampling

`parameterization_from_config` is the same mapping the inversion uses:

~~~python
import matplotlib.pyplot as plt
import numpy as np

from seisforge.inv import build_inversion_setup, load_yaml, parameterization_from_config

config = load_yaml("examples/STA001/inv.yaml")
parameterization = parameterization_from_config(config)
theta0 = parameterization.theta0()
model = parameterization.vector_to_model(theta0)

# Continuous profile, and the finite layers the forward solver receives.
z_km = np.linspace(0.0, model.zmax, 601)
setup = build_inversion_setup(config)
layered = model.to_layered_model(setup.prior.discretization)

# Bounds and every hard constraint are checked together.
prior_result = setup.prior.evaluate(theta0)
print("valid:", prior_result.success)

fig, ax = plt.subplots()
ax.plot(model.evaluate(z_km), z_km, label="continuous model")
ax.stairs(layered.vs, np.r_[layered.layer_tops, model.zmax], orientation="horizontal",
          baseline=None, label="forward layers")
for depth_km in model.discontinuities():
    ax.axhline(depth_km, color="0.5", lw=0.8, ls="--")
ax.invert_yaxis()
ax.set(xlabel="Vs (km/s)", ylabel="Depth (km)")
ax.legend()
plt.show()
~~~

To see what an interior B-spline coefficient does, perturb it and plot again:

~~~python
theta = theta0.copy()
theta[parameterization.names.index("crust.c3")] += 0.15
plt.plot(parameterization.vector_to_model(theta).evaluate(z_km), z_km)
~~~

This shows the coefficient's overlapping footprint rather than assigning it a
single depth.

`parameterized_vs_model_from_config` is only for a fixed model with literal
values; an inversion YAML with parameter references needs
`parameterization_from_config`.

### Fit B-spline coefficients to a reference profile

Given a reference profile `z_ref_km`, `vs_ref_km_s`, fit the part covered by a
B-spline segment (here 1-15 km):

~~~python
from seisforge.inv import fit_bspline_coefficients

mask = (z_ref_km >= 1.0) & (z_ref_km <= 15.0)
fit = fit_bspline_coefficients(
    z_ref_km[mask], vs_ref_km_s[mask],
    n_coefficients=8, degree=3, domain=(1.0, 15.0),
    spacing="geometric", alpha=1.6, bounds=(0.8, 4.6), smoothing=0.0,
)
print(fit.coefficients, fit.rms)
~~~

Use `fit.coefficients` as initial values or as a diagnostic, not as a fixed
answer: initial values choose a starting point, while bounds and constraints
state the prior. The fitting `domain` must match the segment's depth interval.
A small positive `smoothing` penalizes the second difference of the
coefficients; it regularizes this fit only and is not an MCMC prior.

## 7. Checklist Before a Long Run

1. Segments start at 0 km, are contiguous, and reach the required model depth.
2. Every parameter reference has exactly one parameter definition.
3. `theta0` passes `setup.prior.evaluate(theta0)`.
4. `layering_report` shows preserved hard interfaces, resolved transitions, and
   a vertical travel-time error well below the data uncertainty.
5. Sampled interface depths use `layers_per_segment`, not `dz`.
6. Segments whose Vs can exceed 4.5 km/s use a scaling other than Brocher.
7. Constraints express geological knowledge, not merely a preferred answer.
8. B-spline knot geometry and `zmax` cover the depth sensitivity of the data.
9. A prior-only run samples the intended model space before its result is
   compared with a posterior ensemble.

## 8. Design Background

The design keeps ideas from two reference codes without copying their
structure:

- From MCMC_Compliance: hard physical checks are part of the prior support;
  prior-only sampling is a meaningful way to inspect the model space; posterior
  analysis focuses on Vs(z), not only on parameter histograms.
- From pymcinv: a local random walk plus an occasional global uniform jump
  (the `step4uwalk` idea); physical validity is checked before expensive
  forward calculations; B-spline coefficients parameterize smooth structure,
  while interpretation uses the depth ensemble.

SeisForge separates the stages

~~~text
parameter vector -> model mapping -> hard prior -> forward prediction -> likelihood -> sampler
~~~

so new parameterizations, observables, and samplers can be added without
rewriting the workflow.
