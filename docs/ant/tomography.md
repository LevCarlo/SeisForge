# Surface-Wave Eikonal and Helmholtz Tomography

!!! overview "Method at a glance"

    Eikonal tomography (Lin, Ritzwoller & Snieder 2009) turns path-average phase
    velocities of many station pairs, at one period, into a map of local phase
    velocity without a tomographic inversion. The whole method is four steps:

    1. **Travel times.** Take one station $s$ as a virtual source. Its phase
       travel time to each receiver $j$ is $\tau_j = d_{sj}/\bar c_{sj}$, with
       $d_{sj}$ the geodesic distance and $\bar c_{sj}$ the measured
       path-average phase velocity.
    2. **Travel-time surface.** Interpolate the $\tau_j$ onto a grid. The
       contours of $\tau(\mathbf x)$ are the wavefronts of this source.
    3. **Local slowness.** The spacing of the wavefronts gives the local speed:
       $1/c = |\nabla\tau|$ (§1). With amplitudes, a Helmholtz correction can be
       added (§2).
    4. **Stack.** Repeat for every station as source and average the slowness in
       each cell. The scatter between sources gives an uncertainty (§5).

    There is no ray tracing and no regularisation. Resolution is set by the
    station spacing and the wavelength.

![Eikonal tomography of a known medium](img/eikonal_synthetic.svg#only-light)
![Eikonal tomography of a known medium](img/eikonal_synthetic-dark.svg#only-dark)

*Synthetic test with an exact answer: phase velocity increasing linearly to
the north, for which the Eikonal equation has a closed-form solution
(circular rays,
$\tau = |G|^{-1}\operatorname{arccosh}[1 + G^2\ell^2/(2v_1v_2)]$, with $\ell$
the straight distance between the two points and $G$ the gradient), 140
randomly placed stations, $T$ = 20 s, package defaults. (a) Travel-time
surface of one source. (b) Its apparent velocity; the blank disc is the
near-source mask. (c) Stack of all sources. (d) Error: 0.005 km/s RMS in the
interior, but large in the outer ring, where the interpolation extrapolates
beyond the stations. These edge cells pass every default test. Script:
`docs/ant/scripts/eikonal_synthetic.py`.*

## 1. Why |∇τ| is the local slowness

At one frequency $\omega$, a fundamental-mode surface wave is treated as a
2-D scalar wave $u = A\,e^{-i\omega\tau}$ on the surface, whose local speed
is the phase velocity $c(\mathbf x)$ of the structure below. The wave
equation splits into two parts
([Appendix](#appendix-from-the-helmholtz-equation-to-the-eikonal-equation)):

$$
|\nabla\tau|^2 = \frac{1}{c^2} + \frac{\nabla^2 A}{\omega^2 A},
\qquad
\nabla\cdot(A^2\,\nabla\tau) = 0 .
$$

The first equation is exact for this 2-D wave. Dropping its amplitude term
gives the **Eikonal equation**

$$
\boxed{\;|\nabla\tau| = \frac{1}{c}\;}
$$

The magnitude of $\nabla\tau$ is the local slowness, and its direction is the
local propagation direction. The Eikonal equation holds when
$|\nabla^2A/A| \ll k^2 = (2\pi/\lambda)^2$, that is, when the amplitude does
not curve on the scale of a wavelength. It fails behind anomalies of about
one wavelength or smaller, where wavefronts focus or defocus, and where waves
from several directions interfere. $1/|\nabla\tau|$ is therefore an
**apparent** phase velocity, which can differ from the structural $c$
(Wielandt 1993).

The 2-D description neglects mode coupling and the depth dependence of the
wavefield. Its error for strong lateral gradients is not quantified here.

## 2. The Helmholtz correction

Keeping the amplitude term gives the structural phase velocity (Lin &
Ritzwoller 2011):

$$
\boxed{\;\frac{1}{c^2} = |\nabla\tau|^2 - \frac{\nabla^2 A}{\omega^2 A}\;}
$$

$A$ must be the amplitude of a wave from one source, interpolated like
$\tau$; its normalisation cancels. The price is the second derivative of a
measured amplitude, which is much noisier than the first derivative of a
travel time. The correction needs reliable amplitudes and dense stations,
and it is off by default. Whether the amplitudes of noise correlations,
after the usual temporal and spectral normalisation, are suitable has not
been examined here.

## 3. Building the travel-time surface

**Cycle skips** (periods $\ge$ 20 s). An AFTAN measurement on the wrong
$2\pi$ branch shifts $\tau_j$ by whole periods. Stations are visited in order
of distance from a seed station near the array centre. Each station $j$ is
predicted from its nearest already accepted station $n$, assuming the same
apparent velocity along both paths, $\tau_j^{\text{pred}} = d_{sj}\,\tau_n/d_{sn}$.
It is then shifted by whole periods to the prediction and rejected if the
residual remains above 10 s:

$$
\tau_j \to \tau_j + \operatorname{round}\!\left(\frac{\tau_j^{\text{pred}} - \tau_j}{T}\right)T .
$$

Several seeds are tried, and the one that accepts the most stations wins
(ties: smallest total residual). The correction assumes the apparent
velocity changes little between neighbouring stations. At short periods,
where one period is a small fraction of $\tau$, it is not applied by default.

**Interpolation.** The $\tau_j$ and the source itself ($\tau = 0$) are
interpolated onto a regular longitude-latitude grid, in local kilometres.
Two interpolants are available. Both pass through the data:

- a thin-plate spline (SciPy `RBFInterpolator`, the default), which
  minimises the bending energy in one global solve;
- GMT `surface` (Smith & Wessel 1990), a minimum-curvature spline on the
  grid, applied after a block median, optionally in tension.

The surface must be smooth enough to differentiate but must not invent
structure. This is the main free choice of the method. As a stability check,
the same data are interpolated a second time with smoothing 0.2 (RBF) or
tension 0.2 (GMT), and cells where the two surfaces differ by more than 2 s
are rejected: where the interpolant matters, the gradient is not
constrained by data. On an analytic field with stations about 40 km apart,
the two backends agree to 0.012 km/s inside the array, and each is within
0.025 km/s of the truth.

**Derivatives.** $\nabla\tau$ is computed by second-order finite differences
in arc length: $x$ along each parallel, $y$ along the meridian. Parallels
converge towards the pole, which adds a term to the Laplacian:

$$
\nabla^2 f = f_{xx} + f_{yy} - \frac{\tan\varphi}{N(\varphi)}\,f_y ,
$$

with $\varphi$ the latitude and $N$ the prime-vertical radius of the WGS84
ellipsoid. Against the analytic Laplacian of epicentral distance, this form
is accurate to 0.12 % at 0–64° latitude.

Each cell then has the slowness $s = |\nabla\tau|$, the apparent velocity
$1/s$, the propagation azimuth $\operatorname{atan2}(\partial_x\tau,
\partial_y\tau)$ (clockwise from north), and its deviation from the
great-circle azimuth, which diagnoses off-path propagation.

## 4. Which cells of one source's field are kept

Each cell gets the first reason that applies, in this order (the code is the
`rejection_reason` written to the event-field files; 8 marks a source with
too few data):

| code | test (default) | rejects | why |
|---|---|---|---|
| 4 | non-finite, or next to a non-finite cell | holes | the difference stencil would use them |
| 1 | two interpolants differ by > 2 s | unstable cells | gradient not constrained by data |
| 2 | no station within 150 km and not all 4 quadrants occupied within 250 km | cells far from stations | extrapolation |
| 3 | apparent velocity outside 2–5 km/s | wild gradients | implausible for crustal periods (assumed range) |
| 5 | distance to source < 3 wavelengths ($\lambda = 3.5\ \text{km/s}\times T$) | near-source cells | wavefront strongly curved; far-field phase of the CCF poor |
| 6 | $\lvert\nabla^2\tau\rvert$ > 0.005 s/km² | strongly curved wavefronts | focusing, interference |
| 7 | outermost grid cell | the boundary | one-sided differences |

A circular wavefront at distance $R$ from its source has
$\nabla^2\tau = 1/(cR)$, so the curvature limit corresponds to
$R \approx 57$ km at 3.5 km/s. Beyond the near-source mask it acts on
focusing and interference. A source enters the stack only if at least 5 % of
the grid passes. The Helmholtz value of a cell is rejected in addition if
the amplitude is not positive, if the two amplitude interpolants differ by
more than 1 %, if $|\nabla^2A/(\omega^2A)| > 1/(4\ \text{km/s})^2$, or if the
corrected $1/c^2$ is not positive.

## 5. Stacking and its uncertainty

In each cell the accepted single-source slownesses $s_e$ are combined:

1. **Directional weights.** $w_e = 1/n_e$, where $n_e$ counts the sources
   (itself included) whose great-circle azimuth at this cell is within ±20°
   of its own. Sources with no such neighbour are dropped. Weights above the
   mean plus 3 standard deviations are capped, and the weights are
   normalised to sum to 1. A cluster of stations in one direction thus counts
   as one direction.
2. **Minimum count.** Cells with fewer than 50 accepted sources are left
   empty.
3. **Outliers.** Sources more than 2 weighted standard deviations from the
   weighted mean are removed once.
4. **Mean.** $\bar s = \sum_e w_e s_e$ and $c = 1/\bar s$. Slowness is
   averaged because it is what the gradient measures. Averaging velocities
   would give a higher value, by about the relative variance (Jensen's
   inequality).

Two spreads are written, with $c_e = 1/s_e$, $\bar c_w$ their weighted mean,
and $n$ the number of contributing sources:

$$
\sigma_c = \Big[\tfrac{n}{n-1}\sum_e w_e (c_e - \bar c_w)^2\Big]^{1/2},
\qquad
\sigma_{\bar c} = \frac{\sigma_c}{\sqrt{n_{\text{eff}}}},\quad n_{\text{eff}} = \frac{1}{\sum_e w_e^2}.
$$

- $\sigma_c$ is the scatter of single-source estimates. It contains the
  source-specific errors and any azimuthal anisotropy, and it does not
  shrink as sources are added.
- $\sigma_{\bar c}$ is the standard error of the mean if the sources were
  independent. They are not: all sources share the same stations, and
  neighbouring sources see almost the same wavefront. $\sigma_{\bar c}$ is
  therefore a lower bound and becomes implausibly small where many sources
  overlap. A floor is advisable when it is passed to a depth inversion.
- Neither contains errors common to all sources: a wrong phase constant, or
  a biased interpolant at the edges (panel d).

Each cell also records its azimuthal coverage, 360° minus the largest gap
between contributing source azimuths. How Lin et al. (2009) scale the spread
for correlated sources, and how that compares with $\sigma_{\bar c}$, has not
been checked against the paper.

## 6. Limits

<div class="assumptions" markdown>

| assumption | fails when | symptom | guard |
|---|---|---|---|
| amplitude term negligible | anomalies $\lesssim\lambda$; focusing; interference | biased apparent velocity, patterns aligned with source direction | Helmholtz correction; azimuthal averaging |
| $\tau$ smooth and single-valued | multipathing, cycle skips, near the source | stripes or rings, large $\nabla^2\tau$ | cycle-skip correction, curvature limit, near-source mask |
| stations sample the wavefront densely | spacing comparable to $\lambda$; gaps; edges | smooth but wrong surface | stability check, coverage, boundary mask |
| path averages correct | wrong $2\pi$ branch or phase constant; short paths | offsets of $\tau$; bias common to all sources | cycle-skip correction (long periods only) |

</div>

- **Short periods** suffer from the station spacing and are not corrected for
  cycle skips by default.
- **Long periods** suffer from the near-source mask: at 40 s it is 420 km,
  so in a small array few cells survive for any source.
- **Edges** are optimistic. In the synthetic test, cells passing every
  default test have errors above 0.08 km/s there.
- **Anisotropy** makes slowness depend on direction. The isotropic stack is
  correct only for even azimuthal coverage, and anisotropy is not estimated.

!!! verified ""

    Verified: the analytic field above, recovery of a constant gradient,
    removal of an injected one-period cycle skip, the Helmholtz formula, the
    directional weights, the ellipsoidal Laplacian, and the agreement of the two
    interpolants (`tests/test_ant_eikonal.py`).

!!! unverified ""

    Not verified: real data,
    Helmholtz on noise-correlation amplitudes, and anisotropic media.


## 7. Usage

```bash
seisforge ant eikonal -c eikonal.yml [--stage all|fields|stack]
```

`--stage fields` builds the single-source fields, and `--stage stack`
re-stacks existing fields after a change of the stacking parameters. The
input is a NetCDF dataset written by the user's own script from the pair
measurements:

| variable | dims | content |
|---|---|---|
| `period_s` | `period` | increasing integer periods |
| `source_station`, `receiver_station` | `path` | station names, each directed pair once |
| `source_longitude_deg`, `source_latitude_deg`, `receiver_longitude_deg`, `receiver_latitude_deg` | `path` | coordinates |
| `phase_velocity_km_s` | `path, period` | path-average phase velocity; NaN if missing |
| optional `amplitude`, `snr`, `measurement_valid`, `distance_km` | | `amplitude` is required for Helmholtz |

The global attributes are `schema_version: 1`, `component`, and
`longitude_convention` (`-180_180` or `0_360`, matching the grid). With
`input.min_snr` they also include `snr_scale` (`linear` or `db`).

The config sets the periods (`input.periods_s`), the grid, and the parameters
quoted above under `cycle_skip`, `interpolation` (`method: scipy_rbf` or
`gmt_surface`, which needs `pip install -e ".[gmt]"`), `eikonal`,
`helmholtz.enabled`, `qc` and `stacking`. Outputs are
`event_fields/<period>s.nc`, with per-cell values and rejection reasons for
every source, `eikonal_stack.nc`, and `eikonal.log`. Existing outputs are kept
unless `output.overwrite: true`. A runnable example with a known answer is
`examples/03_eikonal`; every key is in
`examples/reference_configs/eikonal.yml`.

## Appendix: from the Helmholtz equation to the Eikonal equation

The 2-D wave of §1 satisfies $\nabla^2 u + (\omega^2/c^2)\,u = 0$. With
$u = A\,e^{-i\omega\tau}$ (time dependence $e^{+i\omega t}$, so crests at
$\omega t - \omega\tau$ = const move outwards),

$$
\nabla u = (\nabla A - i\omega A\nabla\tau)\,e^{-i\omega\tau},
$$

$$
\nabla^2 u = \left[\nabla^2 A - \omega^2 A\,|\nabla\tau|^2 - i\omega\,(2\nabla A\cdot\nabla\tau + A\,\nabla^2\tau)\right]e^{-i\omega\tau}.
$$

Inserting this into the wave equation and dividing by $A\,e^{-i\omega\tau}$,
the real part gives
$|\nabla\tau|^2 = 1/c^2 + \nabla^2A/(\omega^2A)$. The imaginary part gives
$2\nabla A\cdot\nabla\tau + A\nabla^2\tau = 0$, i.e.
$\nabla\cdot(A^2\nabla\tau) = 0$: energy is conserved along rays, which ties
the amplitude to the focusing of the wavefronts.

## References

- Lin, F.-C., Ritzwoller, M. H., & Snieder, R. (2009). Eikonal tomography:
  surface wave tomography by phase front tracking across a regional
  broad-band seismic array. *Geophysical Journal International*, 177(3),
  1091–1110.
  [doi:10.1111/j.1365-246X.2009.04105.x](https://doi.org/10.1111/j.1365-246X.2009.04105.x)
- Lin, F.-C., & Ritzwoller, M. H. (2011). Helmholtz surface wave tomography
  for isotropic and azimuthally anisotropic structure. *Geophysical Journal
  International*, 186(3), 1104–1120.
  [doi:10.1111/j.1365-246X.2011.05070.x](https://doi.org/10.1111/j.1365-246X.2011.05070.x)
- Smith, W. H. F., & Wessel, P. (1990). Gridding with continuous curvature
  splines in tension. *Geophysics*, 55(3), 293–305.
  [doi:10.1190/1.1442837](https://doi.org/10.1190/1.1442837)
- Wielandt, E. (1993). Propagation and structural interpretation of
  non-plane waves. *Geophysical Journal International*, 113(1), 45–53.
  [doi:10.1111/j.1365-246X.1993.tb02527.x](https://doi.org/10.1111/j.1365-246X.1993.tb02527.x)
