# Bayesian Vs(z) Inversion: Overview

`seisforge.inv` estimates the shear-wave velocity profile Vs(z) below one
station from surface-wave dispersion, Rayleigh-wave ellipticity, or both.
The answer is not one model but a sample of the posterior distribution

$$
\boxed{\;p(\theta\mid\mathbf d) \propto p(\theta)\,L(\mathbf d\mid\theta)\;}
$$

over a vector $\theta$ of model parameters, drawn by Markov chain
Monte Carlo. Mapping, prior, layering, forward modelling, likelihood and
sampling are separate steps:

$$
\theta \;\to\; V_s(z) \;\to\; \text{layers} \;\to\; \mathbf g(\theta) \;\to\; L(\mathbf d\mid\theta) \;\to\; \{\theta^{(j)}\}
$$

| page | question | content |
|---|---|---|
| [Model space and prior](model.md) | which profiles are possible, and how probable before the data? | segments, B-splines, interfaces and transitions, Vp and density scaling, parameter priors, hard constraints, curvature prior |
| [Forward modelling and likelihood](forward.md) | how well does a profile fit the data? | layering and its error, dispersion and ellipticity, Gaussian likelihood, weights, sampled noise scale |
| [Sampling](sampler.md) | how are posterior draws produced? | Metropolis, global jumps, adaptive proposals, parallel tempering, starting points |
| [Posterior and convergence](posterior.md) | what do the draws say, and can they be trusted? | Vs ensembles, predicted data, R-hat, ESS, lagging chains, forward failures, synthetic recovery |

## Inputs

The data are curves of value against period with one-sigma errors, in
three-column text files (`period_s value one_sigma`, `#` comments). Where a
dispersion curve comes from is open: an AFTAN measurement, a local curve
from Eikonal or Helmholtz phase-velocity maps or from two-step tomography,
or another program. So is the choice of $\sigma$ (a standard error, a
spread, a floor); it sets the weight of every datum and is part of the
result. The ellipticity observable is the H/V of a Rayleigh mode, not the
diffuse-field H/V spectral ratio
([forward page](forward.md#2-forward-observables)).

## What the method cannot do

Surface-wave data constrain smooth averages of Vs over depth ranges that
grow with period; many profiles fit them equally well. The posterior
expresses this non-uniqueness instead of hiding it, but it is conditional
on the model space: the parameterization, bounds and constraints decide
which profiles are candidates, and the result changes with them. A
prior-only run and a comparison of prior and posterior bands
([posterior page](posterior.md#1-what-the-vs-ensemble-shows)) are
therefore part of every inversion.

## Usage

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o output     # also: inv disp, inv hv
seisforge inv disp-hv --inv inv.yaml -o output_prior --prior-only
```

`obs.yaml` names the data files and the solver settings; `inv.yaml` holds
the model, prior, layering, likelihood and sampler. Each method page ends
with the keys it uses. `examples/06_inversion` is a run on synthetic data
with a known truth;
`examples/reference_configs/inversion_inv.yaml`, `inversion_obs.yaml` and
`inversion_interface_model.yml` are commented complete configurations. From
Python, `seisforge.inv.run_inversion(kind="disp_hv", inv_file=...,
obs_file=..., output=...)` does the same as the command.

Every figure on these pages is drawn by the script of the same name in
`docs/inv/scripts/` from package functions and data with a known answer,
and runs in seconds:

```bash
PYTHONPATH=src python docs/inv/scripts/<figure name>.py
```
