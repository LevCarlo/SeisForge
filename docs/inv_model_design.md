# Inversion Model Design

The inversion module separates three concepts:

```text
sampler parameters -> parameterized Vs model -> LayeredModel for forward solvers
```

This keeps BayesBay, future B-spline parameterizations, and custom MCMC from
being tied to one fixed layer representation.

## Core Objects

- `LayeredModel`: final elastic model passed to forward solvers such as disba.
- `LayeredVsModel`: simple Vs-primary layered model, useful for the first
  BayesBay synthetic tests.
- `ParameterizedVsModel`: segment-based model that preserves explicit
  interfaces during discretization.
- `VsSegment`: a depth interval with its own Vs profile.
- `ConstantVs` and `GradientVs`: first profile types.
- `BSplineVs`: reserved interface for a future B-spline profile.
- `DiscretizationConfig`: controls conversion from continuous or segment-based
  Vs models into finite layers.
- `ScalingConfig`: derives Vp and rho from Vs using `scaling.py`.

## Interface Preservation

Discretization never creates a layer that crosses a `VsSegment` boundary. This
is important for sharp interfaces such as sediment basement or Moho depth. The
layered model may approximate smooth regions with finite layers, but explicit
interfaces are always inserted as layer boundaries.

## Future MCMC Use

Custom MCMC should operate on parameter vectors and create one of the model
objects above. The sampler only needs a function like:

```python
def log_posterior(theta):
    model = vector_to_model(theta)
    layered = model.to_layered_model(discretization)
    prediction = forward(layered)
    return log_prior(theta) + log_likelihood(prediction)
```

That means the sampler does not need to know whether Vs came from fixed layers,
gradients, B-splines, or a hybrid model with explicit interfaces.

## Forward Interface

`seisforge.inv.forward` wraps disba behind small request/prediction objects:

- `DispersionRequest`: periods, mode, wave type, and phase/group choice.
- `RayleighHVRequest`: periods and mode for Rayleigh ellipticity, used as the
  first CCF-HV forward observable.
- `predict_dispersion`: `LayeredModel -> DispersionPrediction`.
- `predict_rayleigh_hv`: `LayeredModel -> RayleighHVPrediction`.
- `predict_joint`: runs any requested combination.

Forward failures raise `ForwardError`. MCMC code should catch this and assign
`-inf` log probability to the candidate model, rather than letting a sampler
crash.
