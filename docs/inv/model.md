# Model Space and Prior

!!! overview "Method at a glance"

    The sampler moves a vector $\theta$ of scalars; this page is
    the map from $\theta$ to an Earth model and the prior on it.

    1. **Profile.** Depth is cut into segments; each has a Vs profile
       (constant, linear, or B-spline) whose numbers are literals or entries
       of $\theta$ (§1).
    2. **Interfaces.** Segment boundaries are hard steps or smooth
       transitions of width $w$; their depths and widths may be sampled (§2).
    3. **Vp and density** follow from Vs by an empirical relation chosen per
       segment (§3).
    4. **Prior.** Parameter priors define a box, hard constraints cut
       physically implausible models out of it, and an optional soft term
       prefers smooth profiles (§4).

    The prior on Vs(z) that results is not the prior on
    $\theta$: it is what a prior-only run samples.

## 1. Segments and profiles

Depth $z$ (km, positive down) is divided into segments
$[z_t, z_b]$. On each, with the local coordinate
$x = (z - z_t)/(z_b - z_t) \in [0, 1]$, Vs is

$$
V_s(z) = c, \qquad
V_s(z) = c_t + (c_b - c_t)\,x, \qquad
V_s(z) = \sum_{i=0}^{n-1} c_i\,B_{i,p}(x)
$$

for the `constant`, `gradient` and `bspline` profiles. Every number may be a
literal or a sampled parameter; only parameters enter
$\theta$.

**B-spline basis.** $B_{i,p}$ are B-splines of degree $p$ (default 3,
lowered to $n-1$ if there are fewer coefficients) on an open clamped knot
vector: $p+1$ knots at 0 and at 1, and $n - p - 1$ interior knots. With
`knot_spacing: geometric` the $n-p$ knot intervals have lengths
proportional to $\alpha^j$, $j = 0, \dots, n-p-1$ (`knot_alpha`; uniform
for $\alpha = 1$), so $\alpha > 1$ puts short intervals, and more freedom,
at the top of the segment. Three properties fix what a coefficient means:

$$
B_{i,p}(x) \ge 0,\qquad \sum_i B_{i,p}(x) = 1,\qquad
V_s(z_t) = c_0,\quad V_s(z_b) = c_{n-1} .
$$

So the end coefficients are the velocities at the segment ends, an
interior coefficient changes Vs only where its basis function is non-zero
($p + 1$ knot intervals), and Vs at any depth lies between the smallest and
largest coefficient active there. Interior coefficients are weights, not
velocities at given depths.

![B-spline basis](img/bspline_basis.svg#only-light)
![B-spline basis](img/bspline_basis-dark.svg#only-dark)

*(a) The eight cubic basis functions of the crust segment of
`examples/06_inversion` (1–15 km, $\alpha = 1.6$; interior knots at 1.9,
3.3, 5.6 and 9.2 km); their sum is 1 to $10^{-15}$. (b) Raising $c_3$ by
0.15 km/s changes the profile by exactly $0.15\,B_3(z)$. Script:
`docs/inv/scripts/bspline_basis.py`.*

Because the knots sit on $x$, not on depth, a profile keeps its shape
relative to its segment: when a sampled interface moves, the profile
stretches with it.

## 2. Interfaces

Segments are separated by interfaces at absolute depth $z_i$. Either the
depths are fixed (`top_km`/`bottom_km` per segment) or they are listed as
`interfaces` with a model bottom `zmax_km`; in the second form $z_i$ and a
width $w$ may be parameters. A **hard** interface ($w = 0$) is a step. A
**gradational** one blends the two neighbouring segments, each evaluated
with depth clamped to its own interval:

$$
v(z) = \big[1 - S(z)\big]\,v_{\text{upper}}(z) + S(z)\,v_{\text{lower}}(z),
\qquad v \in \{V_s, V_p, \rho\} .
$$

$S$ rises from 0 to 1, with $S(z_i) = 0.5$ and 10 % and 90 % at
$z_i \mp w/2$, and is exactly 0 or 1 outside the support
$[z_i - h, z_i + h]$ (Appendix A):

| `shape` | $S$ inside the support | half-support $h$ |
|---|---|---|
| `sigmoid` (default) | logistic $\ell(u) = 1/(1+e^{-u})$ with $k = 2\ln 9/w$, rescaled: $\dfrac{\ell(k(z - z_i)) - \ell(-kh)}{\ell(kh) - \ell(-kh)}$ | $2w$ |
| `smootherstep` | $6\xi^5 - 15\xi^4 + 10\xi^3$, $\xi = (z - z_i + h)/2h$ | $0.98673\,w$ |

The steepest gradient, for profiles that are flat across the support and a
contrast $\Delta v$, is

$$
\boxed{\;\max\left|\frac{dv}{dz}\right| \approx 1.099\,\frac{\Delta v}{w}\ \ (\texttt{sigmoid}),
\qquad 0.950\,\frac{\Delta v}{w}\ \ (\texttt{smootherstep})\;}
\label{eq:max-gradient}
$$

![Transition weights](img/transition_shapes.svg#only-light)
![Transition weights](img/transition_shapes-dark.svg#only-dark)

*(a) $S(z)$ for $w$ = 2 km; dots mark the 10/50/90 % points, shading the
supports. The truncated logistic gives 0.0999 and 0.9001 at
$z_i \mp w/2$. (b) $w\,dS/dz$ from finite differences against the maxima of
eq. $\eqref{eq:max-gradient}$ (dashed), which agree to $3\times10^{-4}$.
Script: `docs/inv/scripts/transition_shapes.py`.*

A support may not reach the surface, overlap a neighbouring support, or
pass the model bottom. Such a $\theta$ has no model and is
outside the prior (log prior $= -\infty$); it is not an error. As
$w \to 0$ the model tends to the hard step, which is the value at $w = 0$.

## 3. Vp and density

Vs is the only inverted property. Vp and $\rho$ (km/s, g/cm³) come from
each segment's own Vs by its own relation (`scaling`, else the model
default), and are then blended with the same $S$ as Vs:

| relation | formula | range |
|---|---|---|
| `vp: brocher2005` | $V_p = 0.9409 + 2.0947V_s - 0.8206V_s^2 + 0.2683V_s^3 - 0.0251V_s^4$ (Brocher 2005, eq. 9) | $0 \le V_s \le 4.5$ |
| `rho: brocher2005` | $\rho = 1.6612V_p - 0.4721V_p^2 + 0.0671V_p^3 - 0.0043V_p^4 + 0.000106V_p^5$ (Brocher 2005, eq. 1) | from Brocher $V_p$ |
| `vp: constant_vp_vs` | $V_p = \kappa V_s$ | |
| `rho: gardner1974` | $\rho = a V_p^{\,b}$, defaults $a = 1.741$, $b = 0.25$ | |
| `rho: constant` | $\rho$ fixed | |

The Gardner default $a$ is Gardner et al.'s $\rho = 0.23\,V_p^{1/4}$
with $V_p$ in ft/s, converted to km/s: $0.23 \cdot (1000/0.3048)^{1/4} =
1.7407$. The often-quoted m/s form $0.31\,V_p^{1/4}$ has a rounded
coefficient (0.3095), which is why $0.31 \cdot 1000^{1/4}$ gives 1.743;
Brocher (2005) rounds to 1.74. A Brocher segment whose Vs
leaves 0–4.5 km/s has no model, so Brocher scaling also bounds Vs; a mantle
segment needs another relation. Scaling each side before blending keeps a
crustal relation off sediment velocities inside a transition.

## 4. The prior

The log prior is a sum of three parts, up to a constant:

$$
\boxed{\;\ln p(\theta) = \sum_j \ln p_j(\theta_j)
+ \ln \mathbb 1_C\big(m(\theta)\big)
- \tfrac12\left(\frac{\kappa_{\text{rms}}(m(\theta))}{\sigma_\kappa}\right)^2\;}
\label{eq:prior}
$$

**Parameter priors** $p_j$. Every parameter has bounds $[l_j, u_j]$, which
are its support and the box the sampler never leaves. Inside it the
density is `uniform`, `normal` ($-\tfrac12[(\theta - \mu)/\sigma]^2$,
truncated to the bounds; for an independent estimate such as a Moho depth),
or `half_normal` ($-\tfrac12[(\theta - l)/s]^2$, peaking at the lower
bound; for transition widths, so that a hard step is the default and a
gradual one must improve the fit enough to pay for itself). `initial` is
only a starting point.

**Hard constraints** $\mathbb 1_C$ are 1 if the model $m(\theta)$
passes every check and 0 otherwise; failing models are rejected before any
forward calculation. Layer checks use the forward layers
([forward page §1](forward.md#1-from-a-continuous-model-to-layers)). Depth
checks are statements about the continuous profile $V_s(z)$ on the closed
interval `depth_range` $= [z_a, z_b]$ (default: the whole model), so they
do not depend on any depth spacing or on where an interface falls:

| `type` | condition |
|---|---|
| `positive_velocity` | Vs, Vp, $\rho$ finite and positive in every layer |
| `vp_gt_vs` | $V_p - V_s >$ `margin` in every layer |
| `vp_vs_range` | $V_p/V_s$ within `bounds` in every layer |
| `boundary_non_decreasing_vs` | at each segment boundary, upper-side Vs minus lower-side Vs $\le$ `tolerance` |
| `vs_range` | $l \le V_s(z) \le u$ for all $z \in [z_a, z_b]$, both sides of every hard interface included (`bounds` $= [l, u]$) |
| `monotonic_vs` | $dV_s/dz \ge -r$ wherever $V_s$ is smooth (`max_decrease_km_s_per_km` $= r$, default 0) |
| `max_vs_gradient` | $\lvert dV_s/dz\rvert \le g$ wherever $V_s$ is smooth (`max_abs_km_s_per_km` $= g$) |

"Wherever $V_s$ is smooth" means: inside a segment, the analytic derivative
of its profile; inside the support of a gradational interface, the
derivative of the blend,

$$
\frac{dV_s}{dz} = S'(z)\,\big[v_{\text{lower}}(z) - v_{\text{upper}}(z)\big]
+ \big[1 - S(z)\big]\,v_{\text{upper}}'(z) + S(z)\,v_{\text{lower}}'(z),
$$

which for flat neighbours peaks at eq. $\eqref{eq:max-gradient}$, so
`max_vs_gradient` also bounds $\Delta v/w$. A hard interface is a jump,
not a gradient: neither slope check sees it. In particular `monotonic_vs`
cannot stop Vs from dropping all at once at a hard interface; bound such
drops with `boundary_non_decreasing_vs`.

The extremes are found piece by piece: the range is cut at segment
boundaries, transition-support edges and B-spline knots, and inside each
piece the roots of the derivative are solved for
(`seisforge.inv.profile_analysis`). The sample points that bracket them are
fixed to the pieces, so they move with the interfaces and knots. Reported
extremes are values the profile attains; on 300 prior draws of the example
the extremes of Vs agree with sampling every 0.5 m to $10^{-15}$ km/s.

**Compatibility: grid evaluation.** Until continuous evaluation became the
default (see the changelog), the depth checks
sampled Vs every `dz` over `depth_range` plus every segment edge (where the
lower segment's value was taken), allowed `monotonic_vs` a drop of
`tolerance` per sample step, and bounded `max_vs_gradient` by difference
quotients between neighbouring samples (`max_abs`). Decisions then depended
on `dz` and on the grid position of interfaces: a hard jump counted as its
size divided by the distance to the next sample, the allowed total decrease
grew with the number of steps, and the upper segment's value at its bottom
was never checked. That evaluation is kept, with its old keys, to reproduce
old results:

```yaml
constraints:
  evaluation: grid      # default: continuous
  constraints:
    - {type: monotonic_vs, depth_range: [0.0, 15.0], tolerance: 0.06, dz: 0.25}
```

Under the default, the old keys `dz`, `tolerance` (of `monotonic_vs`) and
`max_abs` stop the run with a message naming both ways forward. A grid
`tolerance` per `dz` corresponds roughly to
`max_decrease_km_s_per_km` $=$ `tolerance`/`dz` (0.06/0.25 km = 0.24 km/s
per km); `max_abs` becomes `max_abs_km_s_per_km`, now without the jumps.

**Soft prior.** `vs_curvature` adds the last term of eq. $\eqref{eq:prior}$,
with

$$
\kappa_{\text{rms}}^2 = \frac{1}{L}\sum_{\text{segments}}\int
\left(\frac{\partial^2 V_s}{\partial z^2}\right)^2 dz ,
$$

evaluated inside each segment separately (boundaries and transitions are
not differenced across), over `depth_range` of total length $L$, on a grid
of spacing `dz` (default 0.05 km). $\sigma_\kappa$ is in km/s/km²; a linear
profile costs nothing.

**The prior on Vs(z).** Equation $\eqref{eq:prior}$ is a prior on
$\theta$. The prior on the profile follows from the map
$m(\theta)$ and is not uniform. For a B-spline with independent
uniform coefficients alone, $V_s(z) = \sum_i c_i B_i$ is a sum of uniform
variables: its mean is $\sum_i B_i (l_i + u_i)/2$ and its variance
$\sum_i B_i^2 (u_i - l_i)^2/12$, narrower where several basis functions
overlap. Constraints then remove most of the box.

![Prior ensemble](img/prior_ensemble.svg#only-light)
![Prior ensemble](img/prior_ensemble-dark.svg#only-dark)

*Prior of `examples/06_inversion/inv.yaml`. (a) Uniform draws from the
10-parameter box, and the 10 % of them that pass the six hard constraints
(exact samples of the constrained prior). (b) Vs at 6 km under the box
alone against its exact density, the convolution of the uniform densities
of $B_i c_i$; sample mean and standard deviation agree with the closed form
to 0.001 km/s. Script: `docs/inv/scripts/prior_ensemble.py`.*

A profile outside the prior ensemble cannot appear in the posterior,
whatever the data. Sampling the prior alone (`--prior-only`) shows this
ensemble before any data are used.

<div class="assumptions" markdown>

| assumption | fails when | consequence |
|---|---|---|
| structure is representable by the segments and knots | real gradients are sharper than the knot spacing, or interfaces sit where no segment boundary is | a smoothed profile fits the data; the posterior is narrow but biased |
| empirical Vp and $\rho$ | Vp/Vs or density differ from the relation (fluids, unusual lithology) | Vs absorbs the difference |
| constraints describe geology, not the expected answer | bounds or constraints exclude the true structure | the posterior piles up at a bound |
| constraint depth ranges are fixed | a sampled interface moves through a range | the check acts on different segments for different $\theta$ |
| a drop at a hard interface is allowed by the slope checks | velocity inversions are expected only within segments | `monotonic_vs` passes a model that drops at an interface unless `boundary_non_decreasing_vs` bounds the jump |

</div>

!!! verified "Verified"

    Partition of unity, geometric knots, coefficient fitting and the basis
    cache (`tests/test_inv_bspline.py`); transition weights (compact,
    monotone, hard-step limit), clamped evaluation, support checks,
    per-segment scaling and Gardner density (`tests/test_inv_interfaces.py`);
    parameter priors and the map $\theta \to$ model
    (`tests/test_inv_parameterization.py`); the curvature prior, zero for a
    linear profile and not differenced across segments
    (`tests/test_inv_soft_priors.py`); Brocher density as the polynomial of
    Brocher Vp, and the Gardner default as the ft/s relation in km/s
    (`tests/test_inv_scaling.py`). Depth constraints
    (`tests/test_inv_constraints_continuous.py`): decisions the same for
    grid spacings 0.1, 0.25 and 1 km and for interface shifts below one
    step, on cases where the grid evaluation changes its decision; the
    transition slope equal to eq. $\eqref{eq:max-gradient}$ to $10^{-12}$;
    extremes of Vs and of its slope against dense sampling of 10 example
    prior draws; the analytic slopes of B-splines and transition weights
    against finite differences (`tests/test_inv_bspline.py`,
    `tests/test_inv_interfaces.py`); and `evaluation: grid` reproducing all
    12 231 decisions (453 models × 27 constraints) of the code before the
    change (`tests/data/constraint_grid_baseline.npz`). The figures check
    the basis properties, eq. $\eqref{eq:max-gradient}$ and the box-prior
    density.

!!! unverified "Not verified"

    The polynomial coefficients of §3 were not re-checked against Brocher
    (2005) for this page. The extreme search of the depth constraints can
    miss two roots of the derivative that fall between neighbouring bracket
    samples (an extreme of size about $|V_s''|\,\delta^2/8$ for sample
    spacing $\delta$); no such case was found, but it is not excluded. How much the parameterization biases a real
    profile has only been tested on synthetic models that the
    parameterization represents exactly.

## 5. Usage

The keys of this page (complete files:
`examples/reference_configs/inversion_inv.yaml` for fixed segments,
`inversion_interface_model.yml` for sampled interfaces):

```yaml
model:
  zmax_km: 60.0
  scaling: {vp: {method: brocher2005}, rho: {method: brocher2005}}
  interfaces:
    - {name: moho, depth_km: {parameter: moho.depth}, width_km: {parameter: moho.width}, shape: sigmoid}
  segments:
    - name: crust
      profile: {type: bspline, degree: 3, knot_spacing: geometric, knot_alpha: 1.6,
                coefficients: [{parameter: c0}, {parameter: c1}, {parameter: c2}, {parameter: c3}]}
    - name: mantle
      profile: {type: gradient, top: 4.4, bottom: {parameter: mantle.vs}}
      scaling: {vp: {method: constant_vp_vs, vp_vs: 1.8}, rho: {method: constant, rho: 3.35}}
inversion:
  parameters:
    - {name: moho.depth, initial: 35.0, prior: {type: normal, mean: 35.0, sigma: 2.0, bounds: [28.0, 45.0]}}
    - {name: moho.width, initial: 0.0, prior: {type: half_normal, scale: 2.0, bounds: [0.0, 8.0]}}
    - {name: c0, initial: 3.2, prior: {bounds: [2.6, 3.8]}}
    # ... c1-c3, mantle.vs
constraints:
  constraints:
    - {type: positive_velocity}
    - {type: vp_vs_range, bounds: [1.45, 2.7]}
    - {type: monotonic_vs, depth_range: [0.0, 30.0], max_decrease_km_s_per_km: 0.1}
soft_priors:
  priors:
    - {type: vs_curvature, sigma: 0.2, depth_range: [0.0, 30.0], dz: 0.05}
```

To inspect a model before sampling:

```python
from seisforge.inv import build_inversion_setup, load_yaml

setup = build_inversion_setup(load_yaml("inv.yaml"))
theta = setup.parameterization.theta0()
result = setup.prior.evaluate(theta)          # .success, .model, .layered_model
vs = result.model.evaluate([0.0, 5.0, 10.0])  # continuous Vs(z), km/s
```

## Appendix A: transition weights

**Sigmoid.** The logistic $\ell(ku)$, $u = z - z_i$, reaches 0.1 and 0.9
at $ku = \mp\ln 9$; requiring this at $u = \mp w/2$ gives $k = 2\ln 9/w$.
Truncating at $h = 2w$, where $\ell(-kh) = \ell(-4\ln 9) = 1.5\times10^{-4}$,
and rescaling to $[0, 1]$ makes the support compact at the price of moving
the 10/90 % points by $1.2\times10^{-4}$. The slope of the logistic peaks
at $u = 0$ with $dS/dz = k/4 = \ln 9/(2w) = 1.0986/w$ (times
$1/[\ell(kh) - \ell(-kh)] = 1.0003$).

**Smootherstep.** $S(\xi) = 6\xi^5 - 15\xi^4 + 10\xi^3$ has
$S' = 30\xi^2(1-\xi)^2$ and $S'' = 60\xi(1-\xi)(1-2\xi)$, both zero at
$\xi = 0, 1$, so $v$ is $C^2$ across the support edges. Solving
$S(\xi) = 0.1$ gives $\xi_{10} = 0.24664$, so the 10–90 % width is
$(1 - 2\xi_{10})\cdot 2h = 0.50673 \cdot 2h$, hence $h = 0.98673\,w$. The
slope peaks at $\xi = 1/2$: $S' = 30/16$, so
$dS/dz = 1.875/(2h) = 0.9501/w$.

With flat neighbours $dv/dz = \Delta v\,dS/dz$, which gives
eq. $\eqref{eq:max-gradient}$ (derived here).

## References

- Brocher, T. M. (2005). Empirical relations between elastic wavespeeds and
  density in the Earth's crust. *Bulletin of the Seismological Society of
  America*, 95(6), 2081–2092.
  [doi:10.1785/0120050077](https://doi.org/10.1785/0120050077)
- Gardner, G. H. F., Gardner, L. W., & Gregory, A. R. (1974). Formation
  velocity and density—the diagnostic basics for stratigraphic traps.
  *Geophysics*, 39(6), 770–780.
  [doi:10.1190/1.1440465](https://doi.org/10.1190/1.1440465)
