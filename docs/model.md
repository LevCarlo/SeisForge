# Inversion Model Configuration

This guide is the source of truth for the current SeisForge model interface. It
shows how to define a segmented Vs(z) model in inv.yaml, inspect it in Python,
and run it through the current CLI.

The model path is deliberately explicit:

~~~text
inv.yaml scalar parameters (theta)
    -> ModelParameterization.vector_to_model(theta)
    -> ParameterizedVsModel with ordered VsSegment objects
    -> ParameterizedVsModel.to_layered_model(DiscretizationConfig)
    -> LayeredModel passed to disba
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

At present top_km and bottom_km must be numeric constants. Segment thicknesses
and interface depths are not yet inverted. They require a separate mapping from
theta to ordered interfaces and segment-aware constraints; they should not be
treated as ordinary velocity coefficients.

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

## 3. Tutorial Model: Constant + Gradient + B-spline

The following example has:

1. A fixed near-surface cap from 0.00 to 0.30 km.
2. A gradient sediment package from 0.30 to 2.00 km.
3. A cubic B-spline crust from 2.00 to 15.00 km with eight coefficients.

This is a template, not a universal earth model. Adapt depths, bounds, and
constraints to station geology and period sensitivity.

### 3.1 Complete inv.yaml

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

## 4. Meaning of the B-spline

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

## 5. Inspect the Initial Model Before Sampling

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

## 6. Fit B-spline Coefficients to a Reference Profile

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

## 7. Constraints Define the Physical Prior

Scalar uniform bounds define a box in coefficient space. Constraints restrict
that box to a physically accepted model space. Invalid models receive log
prior = -inf before any forward calculation contributes to the likelihood.

Useful current constraints include:

- positive_velocity: all Vs, derived Vp, and derived density are finite and
  positive.
- vp_vs_range: derived Vp/Vs lies within stated bounds. It is often more
  informative than only requiring Vp > Vs when empirical scaling is used.
- boundary_non_decreasing_vs: prevents a downward jump across segment
  boundaries larger than tolerance; it does not require exact continuity.
- vs_range: checks Vs(z) on a depth grid, including between B-spline
  coefficient endpoints.
- monotonic_vs: requires sampled Vs to be non-decreasing, with an allowed
  negative tolerance.
- max_vs_gradient: limits the absolute sampled gradient in km/s per km.

monotonic_vs currently acts over the stated depth_range, not over a named
segment. If a low-velocity zone is allowed in the crust, omit the global
constraint or restrict its depth range to the sedimentary segment.

Constraint sampling always includes segment edges, even when dz does not land
exactly on them. This is important for both the constant-to-gradient and
gradient-to-B-spline transitions.

## 8. Initial Values, Broad Priors, and Two-stage Inversion

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

## 9. CLI and Python APIs

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

## 10. Checklist Before a Long Run

1. Segments are contiguous, begin at 0 km, and reach the required model depth.
2. Every parameter reference has exactly one parameter definition.
3. theta0 passes setup.prior.evaluate(theta0).
4. The continuous and discretized models agree at the selected dz.
5. Constraints express geological knowledge, not merely a preferred answer.
6. B-spline knot geometry and zmax cover the depth sensitivity of the data.
7. A prior-only run samples the intended constrained model space before its
   result is compared with a posterior ensemble.

