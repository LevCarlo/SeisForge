# Getting started: a single-station inversion

This page runs the example in `examples/06_inversion` (station STA001) from start to finish: a
Bayesian (MCMC) inversion of Rayleigh phase velocity and Rayleigh H/V for
Vs(z) in the upper 15 km. The data are synthetic, computed from a known
model, so the result can be compared with the truth. The posterior run takes
about 80 s and the prior run about 6 s, with the four chains in four
processes (measured on a 10-core Apple M4; on one core expect about four
times as long).

The [model guide](model.md) explains the model space, and the
[configuration reference](inv_usage.md) lists every key.

## 1. The station folder

```text
examples/06_inversion/
  obs.yaml              which data files to use
  inv.yaml              model, priors, constraints, sampler
  data/vph_disp.dat     period_s  phase_velocity_km_s  sigma_km_s
  data/hv.dat           period_s  rayleigh_hv  sigma
  data/true_model.dat   the layered truth model the data were computed from
  run.sh                the commands of this page
  plot_result.py        the figure below
```

`inv.yaml` and `obs.yaml` keep only the keys you must set; the full,
commented versions that load to exactly the same setup are
`examples/reference_configs/inversion_inv.yaml` and `inversion_obs.yaml`.

The two YAML files have separate jobs. `obs.yaml` is usually written once per
station; `inv.yaml` is the file you edit while testing inversion strategies.
Data files have three columns (period, value, one-sigma uncertainty) and may
contain `#` comments.

`inv.yaml` describes a gradient sediment segment (0-1 km) over a cubic B-spline
crust (1-15 km) with eight coefficients, so the sampler explores ten scalar
parameters:

```text
sediment.vs_top, sediment.vs_bottom, crust.c0 ... crust.c7
```

**The data are synthetic.** `tools/make_example_06_inversion.py` sets these ten
parameters to a truth model, forwards it with the disba settings of `obs.yaml`
in uniform 1/32 km layers (much thinner than the inversion's layers, so the
example does not hide the inversion's own layering error), and adds Gaussian
noise (seed 20261115):

| | truth (km/s) |
|---|---|
| sediment, 0-1 km | Vs 1.70 at the surface to 2.60 at 1 km, linear |
| crust, 1-15 km (B-spline coefficients `c0`...`c7`) | 2.65, 2.85, 3.05, 3.20, 3.30, 3.45, 3.60, 3.70 |
| Vp, density | from Vs by Brocher (2005) |

Noise: standard deviation 0.03 km/s on the phase velocity (about 1 %) and
0.02 on the H/V ratio, at 14 periods from 0.8 to 7 s; the files list these as
the uncertainties. All values are assumed, not taken from a real station. The
truth lies inside the prior, and it uses the inversion's own parameterization,
so the model space can represent it exactly (an optimistic test: real Earth
structure is never exactly a B-spline). To regenerate the data, from the
repository root:

```bash
PYTHONPATH=src python tools/make_example_06_inversion.py --overwrite
```

## 2. Sample the prior first

Copy the folder somewhere outside the repository and run, from inside it
(`bash run.sh` runs sections 2-4 in one go):

```bash
seisforge inv disp-hv --inv inv.yaml -o output/prior --prior-only --progress
```

`--prior-only` ignores the data. The chains sample the model space allowed by
the parameter bounds and the hard constraints in `inv.yaml`. This answers
"which Vs(z) profiles does my prior allow?" before any data are involved. If a
plausible profile is outside this ensemble, no amount of data will recover it.

## 3. Run the posterior

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o output/posterior --progress
```

`inv disp` and `inv hv` invert one data type; `inv disp-hv` inverts both.
An existing output directory is not overwritten unless `--overwrite` is given.
The example runs its four chains in four processes (`sampler.executor:
process`, `max_workers: 4`). The executor does not change the numbers: every
chain draws from its own seeded generator, and a short run gave identical
samples with `serial` and `process`. Use `serial` on a machine with one core.

**Layering.** The forward solver sees the model as thin uniform layers. The
example uses 62.5 m layers at the surface, each 10 % thicker than the one
above, at most 1 km (`discretization: dz: 0.0625, dz_growth_ratio: 1.1,
dz_max: 1.0`, 35 layers). Every run logs the change of the predicted data
against 4 times thinner layers, in units of the data sigma:

```text
discretization layering_error_at_initial: 35 layers vs reference 961 (uniform dz 0.01562 km);
  max |d phase velocity| 0.000818 km/s (0.0273 sigma); max |d H/V| 0.00153 (0.0767 sigma)
```

Keep both ratios well below 1 (here below 0.1). With your own data, read this
line first; if a ratio is near or above 1, make `dz` smaller. Uniform 0.25 km
layers, used by this example before, give 1.2-1.4 sigma in H/V. See
[choosing a layering](inv_usage.md#discretization).

The output directory contains:

| file | content |
|---|---|
| `disp_hv_mcmc.nc` | parameter samples `theta(chain, draw, parameter)`, log prior and log likelihood terms, acceptance |
| `disp_hv_vs.nc` | Vs(z) for every sample, plus depth-wise `mean`, `median`, `p05`...`p95`, `best_vs` |
| `disp_hv_predictive.nc` | predicted dispersion and H/V for every sample, with the observations |
| `disp_hv.log` | settings, acceptance rates, best theta |
| `resolved_inv.yaml`, `resolved_obs.yaml` | snapshots of the inputs |

## 4. Look at the result

From inside the copied station folder, draw the figure below (prior and posterior Vs
bands with the truth, and the posterior predictive data):

```bash
python plot_result.py -o result.png
```

and read the diagnostics:

```python
import xarray as xr

mcmc = xr.open_dataset("output/posterior/disp_hv_mcmc.nc")
burn_in = mcmc.attrs["sampler_burn_in"]
print("acceptance after burn-in:", mcmc["accepted"].isel(step=slice(burn_in, None)).mean("step").values)
print("split R-hat:", mcmc.split_rhat.values.round(3))
print("ESS:", mcmc.effective_sample_size.values.round())
```

![STA001 prior and posterior Vs ensembles and posterior predictive data](img/sta001_result.png)

What to check:

- **Acceptance** after burn-in (0.18-0.20 per chain here; the log prints it
  as `post-burn-in acceptance rates`). Roughly 0.15-0.4 is healthy for
  random-walk Metropolis. Much lower means the steps are too large; much
  higher means they are too small and the chains move slowly. The log adds a
  note below 0.1 or above 0.6.
- **Truth.** The truth (black) lies inside the posterior 5-95 % band at every
  depth of the output grid. The band is about a tenth of the prior width at
  the surface and a quarter to a third at 2-8 km, and widens below 10 km
  (0.56 of the prior width at 10 km, 0.69 at 14 km), where the 7 s maximum
  period loses sensitivity. The posterior means of the ten parameters lie
  within 0.85 posterior standard deviations of the truth.
- **Posterior vs prior.** Where the posterior band is as wide as the prior
  band, the data do not constrain that depth range. Compare the bottom of the
  model with the period range of the data before interpreting it.
- **Predictive fit.** The observations should fall inside the posterior
  predictive band, given their error bars. A systematic offset points to a
  model space that cannot fit the data (too narrow bounds, too strict
  constraints, too few B-spline coefficients) rather than to the sampler.
- **Convergence.** `mcmc.split_rhat` should be close to 1 (below 1.1, and
  1.01 for final results) and `mcmc.effective_sample_size` at least a few
  hundred; `mcmc.attrs["lagging_chains"]` lists chains stuck in a lower mode.
  The log file repeats these and warns when they fail. In this run split
  R-hat is 1.000-1.007 and the ESS 541-935 for all ten parameters (4400
  stored draws), with no lagging chain. The prior run reaches R-hat <= 1.012
  and ESS >= 397.
- **Forward failures.** `mcmc.forward_failure_rate` is the fraction of
  proposals inside the prior for which disba found no solution (rejected).
  It should be near 0; the log warns above 1 % and names the most frequent
  error. See [forward-solver failures](inv_usage.md#forward-solver-failures).

**How the example converges.** Neighbouring B-spline coefficients trade off
against each other, so `inv.yaml` turns on adaptive proposals
(`sampler.adaptation`, see [the sampler reference](inv_usage.md#sampler)):
4 chains x 16000 steps, the first 5000 burn-in; 1500 steps with the diagonal
`proposal.sigma`, then a covariance update every 250 steps from the most
recent half of the chain so far (`discard_early_history_fraction: 0.5`).
With the whole history (`discard_early_history_fraction: 0`) the steps
learned from the drift away from the random starting points were wider and
the run mixed worse: on 4 x 20000 steps (burn-in 8000, updates every 200
steps after 2000), ESS 447-803 against 800-969 and
post-burn-in acceptance 0.10-0.15 against 0.17-0.20. The earlier fixed-step
settings (4 x 5000 steps, no adaptation) did not converge on the previous
example data (R-hat up to 1.77).

## 5. Next steps

- Change the model: segment depths, profile types, bounds, and constraints.
  The [model guide](model.md) explains how these define the prior.
- Invert interface depths (sediment thickness, Moho depth) or a gradational
  interface: [model guide, section 3](model.md#3-interfaces).
  `examples/reference_configs/inversion_interface_model.yml` is a complete example.
- Inspect a model in Python before a long run:
  [model guide, section 6](model.md#6-inspect-a-model-before-sampling).
- Drive a run from Python instead of the CLI:

```python
from seisforge.inv import run_inversion

result = run_inversion(kind="disp_hv", inv_file="inv.yaml", obs_file="obs.yaml",
                       output="output/posterior", progress=True)
```
