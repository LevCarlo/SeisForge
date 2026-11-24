# Forward Modelling and Likelihood

!!! overview "Method at a glance"

    For a model inside the prior, the likelihood is computed in three steps:

    1. **Layers.** The continuous model becomes a stack of homogeneous
       layers over a half-space, fine enough that the layering changes the
       predictions by much less than the data errors (§1).
    2. **Predictions.** disba computes surface-wave dispersion (phase or
       group velocity) and the Rayleigh-wave ellipticity of the layered
       model at the observed periods (§2).
    3. **Misfit.** Each data type contributes an independent Gaussian term,
       with a fixed weight or a sampled scale on its uncertainties (§3).

    A model for which the solver finds no solution gets
    $\ln L = -\infty$.

## 1. From a continuous model to layers

The solvers take homogeneous layers. The conversion follows fixed rules:

1. A hard interface is always a layer boundary, at its exact depth; no
   layer straddles a step.
2. Each segment, outside transition supports, is cut into layers either by
   a target thickness (`dz`, optionally growing with depth) or into a fixed
   number of equal layers (`layers_per_segment`).
3. The support of each gradational interface gets `transition_layers`
   equal layers (default 16).
4. A layer takes Vs, Vp and $\rho$ of the continuous model at its midpoint
   (`layer_value: midpoint`, default) or their depth average (`mean`,
   5-point Gauss–Legendre).
5. The half-space below `zmax` takes the properties at `zmax`.

**Graded layering.** Errors of piecewise-constant layers are largest where
velocity changes fastest, usually near the surface, while the solver cost
grows with the number of layers. With a growth ratio $r$ =
`dz_growth_ratio` > 1 the target thickness at depth $z$ is

$$
\boxed{\;h(z) = \min\big(d_{\max},\ d_0 + (r-1)\,z\big)\;}
\label{eq:graded}
$$

($d_0$ = `dz`, $d_{\max}$ = `dz_max`), so layers stacked from the surface
have thicknesses $d_0, d_0 r, d_0 r^2, \dots$ until $d_{\max}$. A region
$[z_a, z_b]$ (segment part between features) gets
$n = \lceil s(z_b) - s(z_a)\rceil$ layers whose boundaries are equally
spaced in the stretched coordinate

$$
s(z) = \int_0^z \frac{dz'}{h(z')} =
\begin{cases}
\dfrac{1}{r-1}\ln\!\Big(1 + \dfrac{(r-1)z}{d_0}\Big), & z \le z_c,\\[8pt]
s(z_c) + \dfrac{z - z_c}{d_{\max}}, & z > z_c,
\end{cases}
\qquad z_c = \frac{d_{\max} - d_0}{r - 1}.
$$

Each layer spans at most one unit of $s$, so it is never thicker than $h$
at its bottom (Appendix A). Region ends, transition supports and
`force_depths` stay exact.

**How fine is fine enough.** The layering is a modelling error. Measured
as the largest change of a prediction against a much finer uniform
layering, in units of the data error $\sigma$, it should stay well below
1; as a working level, below 0.2 (an assumed value: a shift of $0.2\sigma$
in every datum adds at most $0.04$ to $\chi^2$ per datum). Midpoint
sampling is second order, so the error falls as the square of the layer
thickness.

![Layering error](img/layering_convergence.svg#only-light)
![Layering error](img/layering_convergence-dark.svg#only-dark)

*Truth model of `examples/06_inversion` at its 14 periods (0.8–7 s; data
errors 0.03 km/s and 0.02). (a) Uniform 0.25 km layers (60) and the
graded layering of the example ($d_0$ = 62.5 m, $r$ = 1.1,
$d_{\max}$ = 1 km; 34 layers). (b) Largest change of phase velocity and
ellipticity against uniform 1/64 km layers, divided by $\sigma$. Uniform
layering converges as $n^{-2}$; 240 uniform layers and 34 graded ones give
an ellipticity error below $0.1\sigma$. The graded phase-velocity error
levels off near $0.03\sigma$, set by the 1 km layers at depth. Script:
`docs/inv/scripts/layering_convergence.py`.*

Every posterior run logs this measure for its starting model, against
layers four times thinner than `dz` (`layering_error_at_initial`); the
function `layering_observable_error` computes it for any model.
`layering_report` gives the geometric view: interface preservation, layers
per transition, the largest Vs error inside a layer, and the relative
error of the vertical travel time $\int dz/v$ that controls
receiver-function delays.

**Sampled interface depths.** With `dz` the number of layers in a segment
changes by one whenever an interface crosses a multiple of the target
thickness, so the predictions, and the likelihood, jump as the depth moves
continuously. A fixed count per segment moves all boundaries continuously
with the interface. Use `layers_per_segment` when interface depths are
parameters.

## 2. Forward observables

The data are values at periods $T_i$ with one-sigma errors $\sigma_i$, in
three-column files (`period_s value one_sigma`). Where they come from
(AFTAN, Eikonal or Helmholtz maps, two-step tomography, other software) is
the user's choice, and so is what is passed as $\sigma$: a standard error,
a spread, or a floor are different statements about the data.

Predictions come from disba (a Python port of the surface-wave codes of
Computer Programs in Seismology, Herrmann) for the layered model:

- **Dispersion:** phase or group velocity, Rayleigh or Love, any mode
  (default: fundamental-mode Rayleigh phase velocity). The root search
  steps in velocity by `dc` (default 0.005 km/s); group velocity is
  differenced with period step `dt` (default 0.025 s).
- **Ellipticity:** the ratio $|u_x/u_z|$ of horizontal to vertical surface
  displacement of a Rayleigh mode (default: fundamental). This is the `hv`
  observable. It is the H/V of a single Rayleigh mode, not the
  diffuse-field H/V spectral ratio of `seisforge.hvsr`, which also contains
  Love waves and body waves; data must be the matching quantity.

For a homogeneous half-space both are period independent. With
$\gamma = c^2/V_s^2$, $q = \sqrt{1 - c^2/V_p^2}$ and $s = \sqrt{1 - \gamma}$,

$$
(2 - \gamma)^2 = 4qs, \qquad
\boxed{\;\frac{H}{V} = \frac{|2 - \gamma - 2qs|}{q\,\gamma}\;}
\label{eq:halfspace-hv}
$$

(Appendix B). For a Poisson solid ($V_p/V_s = \sqrt3$) these give
$c = 0.91940\,V_s$ and $H/V = 0.68125$.

**Failures.** disba does not solve every model inside the prior. Its
root search fails mostly when the half-space is slower than a layer above
it or a layer has $V_p/V_s$ near 1, and the ellipticity or higher modes
can stop short of the longest periods. Such a model gets
$\ln L = -\infty$ and is rejected, which removes it from the posterior as
if it were outside the prior. The posterior page shows how this is
[counted](posterior.md#4-forward-failures).

## 3. The likelihood

Data type $k$ (dispersion, ellipticity) has $N_k$ data, residuals
$r_i = d_i - g_i(\theta)$ and errors $\sigma_i$, assumed
independent and Gaussian: $\chi^2_k = \sum_i (r_i/\sigma_i)^2$. With a weight
$w_k$ and a scale $h_k$ on the errors,

$$
\boxed{\;\ln L(\theta) = \sum_k w_k\left[-\frac{\chi^2_k}{2h_k^2} - N_k\ln h_k\right]\;}
\label{eq:likelihood}
$$

up to a constant. The default $w_k = h_k = 1$ is the plain
$-\tfrac12\sum_k\chi^2_k$.

**Fixed weights** (`dispersion_weight`, `hv_weight`) scale a data type's
influence by hand; $w_k = 0$ drops it. A weight is a statement that the
given errors of that type are wrong by a factor $1/\sqrt{w_k}$.

**Sampled noise scale.** Instead, $\log_{10} h_k$ can be a parameter
(hierarchical noise model, Bodin et al. 2012): the errors become
$h_k\sigma_i$ and the data set their level. The $N_k\ln h_k$ term is the
normalization of the Gaussian; without it $h_k$ would grow to absorb any
misfit. At fixed residuals, $\ln L$ is largest at

$$
h_k^2 = \chi^2_k / N_k
$$

(Appendix C), and the posterior of $h_k$ narrows as $N_k$ grows. A
uniform prior on $\log_{10} h$ treats factors above and below 1 alike. The
scale corrects the overall error level only: correlated errors, a wrong
mode or a model space that cannot fit the data also raise $h$.

![Noise scale](img/noise_scale.svg#only-light)
![Noise scale](img/noise_scale-dark.svg#only-dark)

*Residuals drawn with 1.5 times the stated errors. (a) $\ln L(h)$ of the
package (dots) and of eq. $\eqref{eq:likelihood}$ (lines) agree exactly;
the maximum is at $\sqrt{\chi^2/N}$ (1.35 for 14 data, 1.43 for 140).
(b) Posterior of $\log_{10} h$ under a uniform prior. Script:
`docs/inv/scripts/noise_scale.py`.*

A sampled scale replaces the weight (the weight must stay 1), and the
parameter may not appear in the model. With noise-free synthetic data
$h$ runs to its lower bound, because the true model fits exactly.

<div class="assumptions" markdown>

| assumption | fails when | consequence |
|---|---|---|
| 1-D, isotropic, elastic layered medium below the station | lateral variation within the sensitivity area; anisotropy | the data average structure that the model cannot hold |
| errors independent and Gaussian with the stated $\sigma$ | errors correlated between periods (smoothed curves, shared processing) | the posterior is too narrow |
| the observed mode is the predicted one | an overtone or a mix of modes in the data | biased Vs; a large noise scale |
| layering error $\ll \sigma$ | thin layers needed at short periods | a systematic misfit that the posterior treats as structure |

</div>

!!! verified "Verified"

    disba wrappers and request settings (`tests/test_inv_forward.py`);
    failure policy: solver failures become $-\infty$, other errors stop the
    run (`tests/test_inv_forward_failures.py`); Gaussian likelihood, weights
    and the self-consistent zero (`tests/test_inv_likelihood.py`); noise
    scale, its maximum at $h^2 = \chi^2/N$ and $h = 1$ as the plain
    likelihood bit for bit (`tests/test_inv_noise_scale.py`); layering
    rules (`tests/test_inv_interfaces.py`) and convergence of the graded
    layering (`tests/test_inv_layering_convergence.py`). The phase velocity
    and ellipticity of three half-spaces ($V_p/V_s$ = 1.73, 2, 3) agree with
    eq. $\eqref{eq:halfspace-hv}$ to $2\times10^{-6}$
    (`docs/inv/scripts/halfspace_check.py`).

!!! unverified "Not verified"

    disba against an independent code for layered models (only the
    half-space limit is checked here); Love waves and higher modes in an
    inversion; correlated data errors; real data.

## 4. Usage

`obs.yaml` holds the data and the solver settings; `inv.yaml` the layering
and the likelihood (complete files: `examples/reference_configs/inversion_obs.yaml`,
`inversion_inv.yaml`):

```yaml
# obs.yaml
observations:
  dispersion: {file: data/vph_disp.dat, wave: rayleigh, kind: phase, mode: 0}
  hv: {file: data/hv.dat, noise_parameter: noise.hv}    # optional sampled scale
forward:
  dispersion: {algorithm: dunkin, dc: 0.005, dt: 0.025}
  hv: {algorithm: dunkin, dc: 0.005}
```

```yaml
# inv.yaml
discretization:
  dz: 0.0625
  dz_growth_ratio: 1.1
  dz_max: 1.0
  zmax: 15.0
  # layers_per_segment: {sediment: 8, crust: 30}   # when interface depths are sampled
  transition_layers: 16
  layer_value: midpoint
likelihood:
  dispersion_weight: 1.0
inversion:
  parameters:
    - {name: noise.hv, initial: 0.0, prior: {bounds: [-0.5, 0.5]}, proposal: {sigma: 0.03}}
```

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o output   # or: inv disp, inv hv
```

## Appendix A: graded layer boundaries

With $g = r - 1$, $h(z) = d_0 + gz$ below the cap, so
$s(z) = \int_0^z dz'/(d_0 + gz') = g^{-1}\ln(1 + gz/d_0)$; beyond $z_c$,
$h = d_{\max}$ and $s$ grows linearly. Between two boundaries one unit of
$s$ apart, $\int_{z_k}^{z_{k+1}} dz/h = 1$ with $h$ increasing, so
$z_{k+1} - z_k \le h(z_{k+1})$. From the surface, $s = k$ gives
$z_k = d_0 (r^k - 1)/g$ and layer thicknesses $d_0 r^k$, the geometric
series of eq. $\eqref{eq:graded}$. Using $n = \lceil s(z_b) - s(z_a)\rceil$
equal steps of at most one unit fits a whole number of layers into any
region (derived here; the code subtracts $10^{-9}$ before the ceiling so
that an exact integer does not gain a sliver layer).

## Appendix B: the Rayleigh wave of a half-space

Take $x$ along propagation, $z$ down, and potentials
$\phi = A e^{-kqz}e^{ikx}$, $\psi = B e^{-ksz}e^{ikx}$ with
$u_x = \phi_x - \psi_z$, $u_z = \phi_z + \psi_x$. The free surface has
$\sigma_{xz} = 0$:

$$
-2ik^2 qA - k^2(1 + s^2)B = 0 \;\Rightarrow\; B = -\frac{2iq}{2 - \gamma}A,
$$

using $1 + s^2 = 2 - \gamma$. At $z = 0$,

$$
u_x = ikA\left(1 - \frac{2qs}{2 - \gamma}\right),\qquad
u_z = kqA\,\frac{\gamma}{2 - \gamma},
$$

whose ratio of moduli is eq. $\eqref{eq:halfspace-hv}$. The condition
$\sigma_{zz} = 0$ gives the Rayleigh equation $(2-\gamma)^2 = 4qs$
(Aki & Richards 2002). $u_x$ and $u_z$ are $90°$ out of phase,
so the particle motion is an ellipse (derived here).

## Appendix C: the best noise scale

For one data type, $\partial \ln L/\partial h = w(\chi^2/h^3 - N/h) = 0$
gives $h^2 = \chi^2/N$, and $\partial^2\ln L/\partial h^2 = -2wN/h^2 < 0$
there. In $u = \ln h$, $\ln L = -\tfrac{w}{2}\chi^2 e^{-2u} - wNu$, whose
curvature at the maximum is $-2wN$: the posterior of $\ln h$ under a flat
prior has a width of about $1/\sqrt{2N}$ (derived here).

## References

- Aki, K., & Richards, P. G. (2002). *Quantitative Seismology* (2nd ed.).
  University Science Books.
- Bodin, T., Sambridge, M., Tkalčić, H., Arroucau, P., Gallagher, K., &
  Rawlinson, N. (2012). Transdimensional inversion of receiver functions
  and surface wave dispersion. *Journal of Geophysical Research*, 117,
  B02301. [doi:10.1029/2011JB008560](https://doi.org/10.1029/2011JB008560)
- disba (K. Luu): <https://github.com/keurfonluu/disba>
