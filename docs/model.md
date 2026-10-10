# Inversion Model Configuration

This guide is the source of truth for the current SeisForge model interface. It
shows how to define a segmented Vs(z) model in inv.yaml, inspect it in Python,
and run it through the current CLI.

The model path is deliberately explicit:

~~~text
inv.yaml scalar parameters (theta)
    -> ModelParameterization.vector_to_model(theta)
    -> ParameterizedVsModel: VsSegment objects separated by Interface objects
    -> ParameterizedVsModel.to_layered_model(DiscretizationConfig)   (inv.layering)
    -> LayeredModel passed to the forward solvers (disba, synthetic P RF)
~~~

The sampler proposes theta only. Model mapping, hard geophysical constraints,
and the forward solver are separate, so any proposed Vs(z) can be inspected
before calculating dispersion or H/V.

## 1. Files and Entry Points

A station inversion uses two YAML files:

- inv.yaml contains the model parameterization, scalar priors, discretization,
  hard constraints, likelihood weights, and sampler settings.
- obs.yaml contains observation file paths and forward settings such as wave,
  mode, and disba algorithm.

The model always belongs in inv.yaml. A joint inversion is run as:

~~~bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
~~~

A prior-only exploration uses the same model file without observation data:

~~~bash
seisforge inv disp-hv --inv inv.yaml -o run_prior --prior-only --progress
~~~

The output directory contains NetCDF samples, a depth-domain Vs ensemble,
posterior predictive data when observations are used, resolved YAML snapshots,
and a log.

## 2. Segment Model

Model.segments is an ordered list of contiguous depth intervals. Every segment
contains:

- top_km: segment top in km.
- bottom_km: segment bottom in km.
- profile: a Vs(z) profile defined inside that segment.

The first segment must start at 0 km, and every pair must meet exactly:

~~~text
segment[i].bottom_km == segment[i + 1].top_km
~~~

A segment boundary is a model feature. During discretization SeisForge never
creates a layer that crosses it. Thus an interface at 0.30 km or 2.00 km is
preserved even if Discretization.dz does not divide that depth exactly.

This segment layout fixes every boundary depth. To invert interface depths
(sediment thickness, Moho depth), give gradational transitions, or use
different Vp/rho relations per segment, use the interface layout in
[section 3](#3-interfaces-depth-inversion-and-per-segment-scaling).

### Available profile types

| type | Segment inputs | Typical use |
| --- | --- | --- |
| constant | one velocity | known cap, simple layer, or half-space |
| gradient | top and bottom velocities | sediment compaction or smooth transition |
| bspline | coefficients, degree, knot layout | flexible crustal or mantle profile |

A velocity may be fixed:

~~~yaml
value: 2.50
~~~

or sampled:

~~~yaml
value:
  parameter: upper_crust.vs
~~~

Only values written as parameter references become elements of theta. Literal
values stay fixed for the whole run.

## 3. Interfaces, Depth Inversion, and Per-segment Scaling

The interface layout describes the model as segments separated by interfaces.
Interface depths are **absolute** depths and may be sampled; segment depths are
derived from them.

~~~yaml
Model:
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

~~~text
v(z) = (1 - S(z)) * v_upper(z) + S(z) * v_lower(z)
~~~

`S` is 0.5 at `depth_km`, 0.1 and 0.9 at `depth_km -/+ width_km/2`, and
exactly 0 / 1 outside a finite support. Two shapes are available
(`seisforge.inv.transitions`):

| shape | form | support half-width | max gradient |
|---|---|---|---|
| `sigmoid` | logistic with steepness `k = 2 ln 9 / w`, truncated and rescaled | `2 w` | about `1.1 dv / w` |
| `smootherstep` | `6x^5 - 15x^4 + 10x^3`, C2-continuous | `0.987 w` | about `0.93 dv / w` |

Outside the support the model is exactly the hard-step model. Each neighbour is
evaluated with depths clamped to its own segment, so B-splines never
extrapolate. A support may not reach the surface, overlap a neighbouring
interface's support, or extend below `zmax_km`; such samples are rejected by
the prior (log prior = -inf), not raised.

**Per-segment scaling.** Vp and rho are computed from each segment's own Vs
with that segment's `scaling` (or the model default), then blended with the
same weights as Vs. A crust relation is never applied to sediment velocities
inside a transition. Brocher (2005) is only valid for Vs <= 4.5 km/s, so mantle
segments need their own relation, as above. Available relations:

| key | method | parameters |
|---|---|---|
| vp | `brocher2005` | Vs in 0-4.5 km/s |
| vp | `constant_vp_vs` | `vp_vs` |
| rho | `brocher2005` | from Brocher Vp |
| rho | `gardner1974` | `rho = a Vp^b`, defaults `a = 1.741`, `b = 0.25` (km/s, g/cm^3) |
| rho | `constant` | `rho` |

**Priors for interface parameters.** Besides `uniform`, two prior types help:

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

`bounds` are always required: they are the region the sampler explores and
draws starting points from. `half_normal` peaks at the lower bound, so a hard
interface is the default and a gradational one has to improve the fit enough to
pay for its prior cost. Choose `scale` near the thinnest transition the data
can resolve. Depth parameters need their own `proposal.sigma`, e.g. about 1 km
for the Moho and 0.02-0.05 km for a sediment base.

At setup, every parameter reference must be defined and the initial values must
give a valid model; both are checked before sampling starts.

## 4. From Continuous Model to Forward Layers

Every forward solver sees only the `LayeredModel`, so the conversion
(`seisforge.inv.layering`) follows fixed rules:

1. A hard interface is always a layer boundary, at its exact depth, whatever
   `dz` is. No layer straddles a velocity jump.
2. The part of each segment outside interface supports is split into layers by
   `dz`, or by a fixed count per segment (`layers_per_segment`).
3. The support of each gradational interface gets `transition_layers` equal
   layers.
4. Each layer takes Vs, Vp, and rho from the continuous model at its midpoint
   (`layer_value: midpoint`, default) or as the depth average
   (`layer_value: mean`).
5. The half-space takes the properties at the bottom depth (`zmax`).

~~~yaml
Discretization:
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
a `layers_per_segment` mapping fall back to `dz`.

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

## 5. Tutorial Model: Constant + Gradient + B-spline

The following example has:

1. A fixed near-surface cap from 0.00 to 0.30 km.
2. A gradient sediment package from 0.30 to 2.00 km.
3. A cubic B-spline crust from 2.00 to 15.00 km with eight coefficients.

This is a template, not a universal earth model. Adapt depths, bounds, and
constraints to station geology and period sensitivity.

### 5.1 Complete inv.yaml

~~~yaml
Model:
  # Vs is inverted. Vp and density are derived before calling disba.
  scaling:
    vp:
      method: brocher2005
    rho:
      method: brocher2005

  segments:
    # A fixed shallow cap.
    - top_km: 0.0
      bottom_km: 0.30
      profile:
        type: constant
        value: 1.35

    # A simple sedimentary velocity increase.
    - top_km: 0.30
      bottom_km: 2.00
      profile:
        type: gradient
        top:
          parameter: sediment.vs_top
        bottom:
          parameter: sediment.vs_bottom

    # A flexible crustal profile. Its local depth is 0--13 km.
    - top_km: 2.00
      bottom_km: 15.00
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
          - parameter: crust.c4
          - parameter: crust.c5
          - parameter: crust.c6
          - parameter: crust.c7

Inversion:
  # This list defines the order of theta.
  parameters:
    - name: sediment.vs_top
      initial: 1.50
      prior:
        type: uniform
        bounds: [1.20, 2.10]
      proposal:
        sigma: 0.04

    - name: sediment.vs_bottom
      initial: 2.50
      prior:
        type: uniform
        bounds: [2.00, 3.10]
      proposal:
        sigma: 0.04

    - name: crust.c0
      initial: 2.60
      prior:
        type: uniform
        bounds: [2.20, 3.20]
      proposal:
        sigma: 0.04

    - name: crust.c1
      initial: 2.80
      prior:
        type: uniform
        bounds: [2.30, 3.50]
      proposal:
        sigma: 0.04

    - name: crust.c2
      initial: 3.00
      prior:
        type: uniform
        bounds: [2.50, 3.70]
      proposal:
        sigma: 0.04

    - name: crust.c3
      initial: 3.15
      prior:
        type: uniform
        bounds: [2.70, 3.90]
      proposal:
        sigma: 0.04

    - name: crust.c4
      initial: 3.30
      prior:
        type: uniform
        bounds: [2.80, 4.00]
      proposal:
        sigma: 0.04

    - name: crust.c5
      initial: 3.42
      prior:
        type: uniform
        bounds: [2.90, 4.10]
      proposal:
        sigma: 0.04

    - name: crust.c6
      initial: 3.50
      prior:
        type: uniform
        bounds: [3.00, 4.20]
      proposal:
        sigma: 0.04

    - name: crust.c7
      initial: 3.58
      prior:
        type: uniform
        bounds: [3.05, 4.30]
      proposal:
        sigma: 0.04

Discretization:
  dz: 0.25
  zmax: 15.0
  # Segment boundaries are automatically included. Use this only for an
  # additional fixed depth internal to a segment.
  force_depths: []

Constraints:
  constraints:
    - type: positive_velocity

    # Vp is derived using Model.scaling, then checked as a Vp/Vs ratio.
    - type: vp_vs_range
      bounds: [1.45, 2.70]

    # Allows a maximum downward jump of 0.05 km/s at 0.30 and 2.00 km.
    - type: boundary_non_decreasing_vs
      tolerance: 0.05

    - type: vs_range
      depth_range: [0.0, 15.0]
      bounds: [0.8, 4.6]
      dz: 0.25

    # This applies across all three segments.
    - type: monotonic_vs
      depth_range: [0.0, 15.0]
      tolerance: 0.05
      dz: 0.25

    - type: max_vs_gradient
      depth_range: [0.0, 15.0]
      max_abs: 1.5
      dz: 0.25

Likelihood:
  dispersion_weight: 1.0
  hv_weight: 1.0

Sampler:
  n_steps: 5000
  burn_in: 1000
  thin: 5
  n_chains: 4
  executor: serial
  seed: 20260710

  # Start from the initial values above. prior_uniform is useful only after
  # checking that the broad bounds and hard constraints leave enough legal
  # models to initialize all chains.
  initial_strategy: theta0
  max_initial_attempts: 50000

  proposal_sigma: from_parameters
  use_parameter_bounds: true
  global_jump_interval: 1000
~~~

Every parameter reference in Model.segments must have exactly one matching
entry in Inversion.parameters. Parameter names are arbitrary, but names such
as sediment.*, crust.*, and mantle.* make the NetCDF output readable.

To invert the constant cap rather than fix it, use:

~~~yaml
profile:
  type: constant
  value:
    parameter: cover.vs
~~~

and add cover.vs to Inversion.parameters with initial, prior bounds, and a
proposal sigma.

## 6. Meaning of the B-spline

Within a B-spline segment, SeisForge uses the segment-local normalized
coordinate:

~~~text
x = (z - top_km) / (bottom_km - top_km),  0 <= x <= 1
Vs(z) = sum_i B_i(x) c_i
~~~

The basis is open-clamped. Hence in the eight-coefficient example:

- c0 equals the velocity at the 2.00-km top of the B-spline segment.
- c7 equals the velocity at the 15.00-km bottom of the B-spline segment.
- c1 through c6 are overlapping basis weights. They control local shape but
  are not velocities at fixed physical depths.

With eight coefficients and degree 3, there are four interior knots. Knot
spacing is created on x in [0, 1], not directly on depth in kilometres:

- knot_alpha: 1.0 gives uniform knot intervals.
- knot_alpha > 1.0 gives smaller intervals near the segment top and larger
  intervals at depth, making the shallow part of that segment more flexible.
- 0 < knot_alpha < 1.0 concentrates flexibility toward the segment bottom.

Therefore knot_alpha: 1.6 concentrates flexibility near 2 km in the example,
but only because 2 km is the top of that specific segment. Moving the segment
from 2--15 km to 10--40 km preserves exactly the same relative knot layout.

## 7. Inspect the Initial Model Before Sampling

Use parameterization_from_config for an inversion YAML that contains parameter
references. It is the same mapping used by the inversion setup.

~~~python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seisforge.inv import (
    DiscretizationConfig,
    build_inversion_setup,
    load_yaml,
    parameterization_from_config,
)

config = load_yaml(Path("inv.yaml"))

# Resolve the initial scalar vector into a continuous segmented model.
parameterization = parameterization_from_config(config)
theta0 = parameterization.theta0()
model = parameterization.vector_to_model(theta0)

# Continuous profile: inspect parameterization geometry.
z_km = np.linspace(0.0, model.zmax, 601)
vs_km_s = model.evaluate(z_km)

# Finite layers: this is what the forward solver receives.
disc_cfg = DiscretizationConfig(dz=0.25, zmax=15.0)
layered = model.to_layered_model(disc_cfg)

# Bounds and every hard constraint are checked together.
setup = build_inversion_setup(config)
prior_result = setup.prior.evaluate(theta0)
print(parameterization.names)
print("Initial theta:", theta0)
print("Hard-prior valid:", prior_result.success)
print(
    "Failed constraints:",
    prior_result.constraint_evaluation.failed
    if prior_result.constraint_evaluation
    else prior_result.error,
)

fig, ax = plt.subplots()
ax.plot(vs_km_s, z_km, label="continuous parameterized model")
layer_tops_km = np.r_[0.0, np.cumsum(layered.thickness[:-1])]
ax.step(layered.vs, layer_tops_km, where="pre", label="layered forward model")
for boundary_km in model.discontinuities():
    ax.axhline(boundary_km, color="0.5", lw=0.8, ls="--")
ax.invert_yaxis()
ax.set(xlabel="Vs [km/s]", ylabel="Depth [km]")
ax.legend()
plt.show()
~~~

The smooth curve is the scientific parameterization. The step model is the
finite-layer approximation passed to disba. A smaller dz improves that
approximation but increases the number of layers and the cost of every MCMC
proposal.

To see what an interior coefficient actually does:

~~~python
theta = theta0.copy()
index = parameterization.names.index("crust.c3")
theta[index] += 0.15

perturbed_model = parameterization.vector_to_model(theta)
plt.plot(perturbed_model.evaluate(z_km), z_km, label="c3 + 0.15 km/s")
plt.gca().invert_yaxis()
plt.legend()
plt.show()
~~~

This is the right way to develop intuition for interior coefficients. It shows
their overlapping basis footprint directly rather than assigning them an
incorrect single physical depth.

## 8. Fit B-spline Coefficients to a Reference Profile

Given a sampled reference profile z_ref_km and vs_ref_km_s, fit only the part
covered by the B-spline segment. For the 2--15 km segment:

~~~python
from seisforge.inv import fit_bspline_coefficients

mask = (z_ref_km >= 2.0) & (z_ref_km <= 15.0)
fit = fit_bspline_coefficients(
    z_ref_km[mask],
    vs_ref_km_s[mask],
    n_coefficients=8,
    degree=3,
    domain=(2.0, 15.0),
    spacing="geometric",
    alpha=1.6,
    bounds=(0.8, 4.6),
    smoothing=0.0,
)

print("B-spline coefficients:", fit.coefficients)
print("Fit RMS [km/s]:", fit.rms)
~~~

Use fit.coefficients as a diagnostic or as the initial values for crust.c0
through crust.c7. It is not a data-derived answer to be fixed into the final
inversion: initial values choose a starting point, while bounds and hard
constraints state the actual prior support.

The fitting domain must match the global depth interval of the B-spline
segment. The implementation evaluates the segment internally on local depth
(z - top_km) from 0 to its thickness; both forms yield identical coefficients
because their normalized coordinates are identical.

For a noisy reference profile, a small positive smoothing penalizes the second
difference of the coefficient vector. It regularizes the deterministic fit
only; it is not automatically an MCMC prior.

## 9. Constraints Define the Physical Prior

Scalar uniform bounds define a box in coefficient space. Constraints restrict
that box to a physically accepted model space. Invalid models receive log
prior = -inf before any forward calculation contributes to the likelihood.

Useful current constraints include:

- positive_velocity: all Vs, derived Vp, and derived density are finite and
  positive.
- vp_vs_range: derived Vp/Vs lies within stated bounds. It is often more
  informative than only requiring Vp > Vs when empirical scaling is used.
- boundary_non_decreasing_vs: prevents a downward jump across segment
  boundaries larger than tolerance; it does not require exact continuity. The
  jump compares the two segment profiles at the interface depth, so it applies
  to gradational interfaces too.
- vs_range: checks Vs(z) on a depth grid, including between B-spline
  coefficient endpoints.
- monotonic_vs: requires sampled Vs to be non-decreasing, with an allowed
  negative tolerance.
- max_vs_gradient: limits the absolute sampled gradient in km/s per km.
  Inside a gradational interface the gradient peaks at about `1.1 dv / w`
  (sigmoid), so this constraint also bounds jump/width there. Restrict its
  `depth_range` if it should not act on interfaces.

monotonic_vs currently acts over the stated depth_range, not over a named
segment. If a low-velocity zone is allowed in the crust, omit the global
constraint or restrict its depth range to the sedimentary segment.

Constraint sampling always includes segment edges, even when dz does not land
exactly on them. This is important for both the constant-to-gradient and
gradient-to-B-spline transitions.

## 10. Initial Values, Broad Priors, and Two-stage Inversion

initial is a starting point, not a prior mean. Uniform bounds plus hard
constraints define the actual prior support.

A practical two-stage workflow is:

1. Use broad but defensible bounds and physical constraints to explore the
   posterior support. prior_uniform can provide diverse valid chain starts.
2. Inspect both theta samples and, more importantly, the depth-domain Vs
   ensemble. If the first-stage posterior is away from all bounds, construct a
   second inv.yaml with focused but buffered bounds.

Do not tighten around a single best model. A posterior that repeatedly touches
a stage-one bound is evidence that the first support is limiting the result;
broaden or reconsider that prior before claiming a focused second stage.

prior_uniform can fail when independent coefficient bounds combined with strict
monotonic and gradient constraints leave very little legal volume. In that
case, begin with a known valid theta0, revise the scientific constraints or
bounds, or simplify the parameterization before merely raising
max_initial_attempts.

## 11. CLI and Python APIs

Use the CLI for a reproducible station run:

~~~bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_mixed --progress
~~~

Use the Python mapping for inspection, synthetic studies, and numerical tests:

~~~python
from seisforge.inv import load_yaml, parameterization_from_config

config = load_yaml("inv.yaml")
parameterization = parameterization_from_config(config)
model = parameterization.vector_to_model(parameterization.theta0())
~~~

Do not use parameterized_vs_model_from_config for an inversion YAML with
parameter references. That helper is for a fixed literal model.
parameterization_from_config is the interface that resolves theta into a model.

For programmatic execution, use the run-level driver:

~~~python
from seisforge.inv import run_inversion

result = run_inversion(
    kind="disp_hv",
    inv_file="inv.yaml",
    obs_file="obs.yaml",
    output="run_mixed",
    progress=True,
)
print(result.posterior_vs)
~~~

run_inversion owns configuration loading, setup assembly, sampling, NetCDF
output, resolved-input snapshots, and logging. Model objects themselves do not
read files or write results.

## 12. Checklist Before a Long Run

1. Segments are contiguous, begin at 0 km, and reach the required model depth.
2. Every parameter reference has exactly one parameter definition.
3. theta0 passes setup.prior.evaluate(theta0).
4. `layering_report` shows preserved hard interfaces, resolved transitions, and
   a vertical travel-time error well below the data uncertainty.
5. Sampled interface depths use `layers_per_segment`, not `dz`.
6. Segments whose Vs can exceed 4.5 km/s use a scaling other than Brocher.
7. Constraints express geological knowledge, not merely a preferred answer.
8. B-spline knot geometry and zmax cover the depth sensitivity of the data.
9. A prior-only run samples the intended constrained model space before its
   result is compared with a posterior ensemble.

