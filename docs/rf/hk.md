# H-k Stacking

!!! overview "Method at a glance"

    A P wave crossing the base of a layer (the Moho, or the base of a
    sediment) converts to S (Ps) and reverberates between the interface and
    the free surface (PpPs, and PpSs + PsPs). Their delays after direct P
    depend on the layer thickness $H$ and $\kappa = V_P/V_S$. H-k stacking
    (Zhu & Kanamori 2000) searches a grid of $(H, \kappa)$ for the pair
    whose predicted delays line up with positive Ps and PpPs and negative
    PpSs + PsPs. In SeisForge:

    1. **One stack per RF.** For every $(H, \kappa)$, sum the RF amplitudes
       at the three predicted delays with weights (§1–§2). The grid maximum
       is that RF's pick.
    2. **Station estimate.** Drop RFs whose stack is dominated by direct P,
       screen the other picks, and take their median and scaled MAD (§4).
    3. **Two layers (optional).** Pick a sediment layer from high-frequency
       RFs first, then stack the crust below it (§5).

    $V_P$ (or $V_S$) is assumed, not resolved (§3).

![H-k stack of one synthetic RF](img/hk_synthetic.svg#only-light)
![H-k stack of one synthetic RF](img/hk_synthetic-dark.svg#only-dark)

*One noise-free synthetic RF (35 km crust, $\kappa$ = 1.75, $p$ = 0.06 s/km,
$a$ = 2.5) stacked with the true $V_P$ = 6.3 km/s. (a)–(c) The amplitude at
each phase delay; each phase alone is a band along a curve of constant
delay. The dashed lines are the tangents of eq. $\eqref{eq:tradeoff}$ through
the truth (circle). (d) The weighted stack peaks at the truth (35.0 km,
1.750). Script: `docs/rf/scripts/hk_synthetic.py`.*

## 1. Phase delays

With the vertical slownesses
$\eta_s = \sqrt{V_S^{-2} - p^2}$ and $\eta_p = \sqrt{V_P^{-2} - p^2}$ of a
layer of thickness $H$, the delays after direct P are

--8<-- "docs/rf/generated/hk_delays.tex"

([Appendix A](#appendix-a-delays-from-ray-geometry)). PpSs and PsPs have
the same legs in a different order, so they arrive together. Either $V_P$
is fixed and $V_S = V_P/\kappa$, or $V_S$ is fixed and $V_P = \kappa V_S$.
A post-critical ray ($p \ge 1/V$) has no real delay.

## 2. The stack

$$
\boxed{\;S(H,\kappa) = w_1\,A(t_{\mathrm{Ps}}) + w_2\,A(t_{\mathrm{PpPs}}) - w_3\,A(t_{\mathrm{PsPs}})\;}
\label{eq:stack}
$$

The minus sign makes the negative PpSs + PsPs pulse add: the third term is
$-w_3$ times the amplitude at $t_{\mathrm{PsPs}}$, not a sum of two phases.
$A(t)$ is the RF linearly interpolated between the two samples around
$t$. A delay outside the trace, or a post-critical one, contributes zero.
The number of such grid points is recorded per phase. A nonzero count
means the grid reaches beyond the trace or beyond critical incidence.
The default weights $(0.5, 0.2, 0.3)$ are assumed values.

Before SeisForge 0.2.0, $A(t)$ was the nearest sample. The stack was then
constant over runs of grid points that map to the same samples, and the
maximum went to the first point of such a run, the lowest $\kappa$: for
the weak-contrast sediment of §5, $S$ at $H$ = 2 km was the same for
$\kappa$ = 1.76–2.1 and the pick was 1.76 (truth 2.0).

**Direct-P exclusion.** Where $t_{\mathrm{Ps}}$ is shorter than
`min_ps_delay_s`, all three delays fall on the direct-P pulse, and as
$H \to 0$ the stack tends to $(w_1 + w_2 - w_3)\,A(0) = 0.4\,A_P$ with the
default weights. Those grid points are not picked (hatched in the
[two-layer figure](#5-sediment-then-crust)); their number is recorded. If
the stack there is higher than the pick, no conversion in the grid stacks
above direct P, and the RF is flagged `direct_p_dominates` (§4). For a
Gaussian $a$, the direct pulse $e^{-a^2t^2}$ falls to 10 % at
$\sqrt{\ln 10}/a$, 0.30 s for $a$ = 5 and 0.61 s for $a$ = 2.5, the values
of the reference config (assumed). A sediment whose Ps arrives earlier is
not resolved at that $a$.

Before stacking, each RF is multiplied by
$c(p) = 151.5478\,p^2 + 3.2896\,p + 0.2618$ ($p$ in s/km), the
"amplitude correction for Ps" of Seispy's `hk.py` (Xu & He 2023). Its
derivation is not given there and has not been traced. A constant factor
scales the whole stack of one RF, so it does not move that RF's pick. It
changes the peak value $S_{\max}$, which enters the Mahalanobis weights of
§4.

## 3. What one stack resolves

**Trade-off.** Along a curve of constant delay, $t = H\,\tau(\kappa)$, so
$dH/d\kappa = -H\,\tau'/\tau$. With $V_P$ fixed, at vertical incidence:

--8<-- "docs/rf/generated/hk_tradeoff.tex"

At $H$ = 35 km and $\kappa$ = 1.75 the slopes are $-47$, $-13$ and $-20$ km
per unit $\kappa$ ($-46$, $-14$, $-21$ at $p$ = 0.06 s/km). Ps alone fixes
only a line. The multiples cross it at a different angle and fix the point,
so the result relies on multiples, which are the weakest and most easily
disturbed phases. At $p = 0$, Ps and PpPs together give
$\kappa = (t_{\mathrm{PpPs}} + t_{\mathrm{Ps}})/(t_{\mathrm{PpPs}} - t_{\mathrm{Ps}})$,
independent of $V_P$ (derived here).

**Assumed $V_P$.** At fixed $\kappa$ and delay,
$d\ln H/d\ln V_P = 1$ at vertical incidence for all three phases, and
0.91–1.09 at $p$ = 0.06 s/km (derived here). A 5 % error in $V_P$ gives a
5 % error in $H$, about 1.7 km at 35 km. $\kappa$ is nearly independent of
$V_P$.

The model is a flat layer with constant velocities:

<div class="assumptions" markdown>

| assumption | fails when |
|---|---|
| flat, isotropic layers | dipping interfaces or anisotropy: delays and amplitudes vary with back azimuth |
| sharp interface | a gradational Moho weakens the multiples, especially at high frequency |
| constant $V_P$ in the layer | $H$ scales with the assumed average $V_P$ |
| no other arrivals at the predicted delays | sediment reverberations or intracrustal interfaces overlap the Moho phases |

</div>

## 4. From single-RF picks to a station estimate

Each RF $j$ gives a pick $(H_j, \kappa_j)$ with peak value $S_j$.

1. RFs flagged `direct_p_dominates` (§2) are dropped. If they are more than
   half of the RFs, the stage has no estimate (the half is an assumed
   value; it reproduces the median rule of earlier versions, where such RFs
   picked $H$ near 0).
2. The **Mahalanobis** screen keeps the other picks with
   $d_j^2 = \Delta\mathbf x_j^{\mathsf T}\,\Sigma^{-1}\Delta\mathbf x_j < r^2$,
   with $\mathbf x_j = (H_j, \kappa_j)$, and the mean and covariance
   $\Sigma$ weighted by $S_j$.
3. From the $n$ kept picks,

$$
\hat H = \operatorname{median}_j H_j \pm 1.4826\,\operatorname{median}_j |H_j - \hat H|,
$$

and likewise for $\kappa$, for every $n$. $1.4826\,\mathrm{MAD}$ estimates
the standard deviation of Gaussian scatter (Rousseeuw & Croux 1993) and is
insensitive to a minority of outliers. The spread is the scatter of
single-RF picks, not the uncertainty of the estimate; for Gaussian picks
the standard error of the median is about $1.25/\sqrt n$ times the
spread.

Before SeisForge 0.2.0 the estimate was the mean $\pm$ standard deviation
for $n < 15$ and the median $\pm$ unscaled MAD for $n \ge 15$, a factor of
about 1.5 apart in the spread. An energy screen and an elliptical MAD screen
were computed, logged and plotted, but did not change the estimate; they
were removed, and their config keys raise an error.

For a bivariate Gaussian cloud, $d^2$ is $\chi^2$ with two degrees of
freedom, so the screen keeps

$$
P(d < r) = 1 - e^{-r^2/2}
\label{eq:fraction}
$$

([Appendix B](#appendix-b-fraction-kept-by-a-mahalanobis-screen)), which is
39 % at the default $r = 1$. The Mahalanobis screen follows the cloud's
correlation, which an H-k trade-off makes strong.

![From picks to a station estimate](img/hk_cluster.svg#only-light)
![From picks to a station estimate](img/hk_cluster-dark.svg#only-dark)

*(a) Picks of 60 noisy synthetic RFs (15 % white noise, $p$ = 0.045–0.078
s/km; truth 35 km, 1.75), the Mahalanobis screen at the reference-config
value $r = 1$, and the estimate $34.75 \pm 0.44$ km, $1.760 \pm 0.015$. The
screen keeps 30 of 60 (50 %, not 39 %): the outliers inflate the
covariance. (b) Monte Carlo through the package function against eq.
$\eqref{eq:fraction}$. Script: `docs/rf/scripts/hk_cluster.py`.*

## 5. Sediment, then crust

A sediment layer adds its own Ps and multiples within a few seconds of direct
P. These are resolved only by high-frequency RFs. With `--mode both`, the
sediment is stacked first as one layer on the high-frequency RFs, giving
$(h_0, \kappa_0)$. If the sediment stage has an estimate (§4) and
$h_0$ exceeds `two_layer_min_thickness_km` (0.4 km in the reference
config, an assumed value), the crust is then stacked on the low-frequency
RFs with $H$ the depth of the Moho. Each delay
of eq. $\eqref{eq:delays}$ becomes the sum over the two layers:

$$
t = h_0\,\tau(\eta_{s,0}, \eta_{p,0}) + (H - h_0)\,\tau(\eta_s, \eta_p),
$$

with the sediment velocities from $\kappa_0$ and the fixed sediment $V_S$
(or $V_P$). Otherwise the crust is stacked as one layer.

![Sediment and crust](img/hk_two_layer.svg#only-light)
![Sediment and crust](img/hk_two_layer-dark.svg#only-dark)

*Noise-free synthetic RFs, $p$ = 0.06 s/km, $dt$ = 0.1 s, Moho at 35 km
under a 2 km top layer. (a) $a$ = 5 RFs: slow sediment ($V_S$ 1.8 km/s,
$\kappa$ 2.2) and weak contrast ($V_S$ 2.5 km/s, $\kappa$ 2.0). (b), (c)
Their sediment stacks with the true $V_S$ fixed, over $H$ from 0 with
`min_ps_delay_s` = 0.3 s; hatched points are not picked. Picks (2.00 km,
2.15) and (2.00 km, 1.95). (d)–(f) Slow sediment: the $a$ = 2.5 RF, the
crust stacked as one layer (36.4 km, 1.78), and stacked under the pick of
(b) (35.0 km, 1.75). Script: `docs/rf/scripts/hk_two_layer.py`.*

**Thin-layer limit.** As $H \to 0$ all three delays go to zero, and
$S \to (w_1 + w_2 - w_3)\,A(0) = 0.4\,A_P$ with the default weights: a
ridge at the small-$H$ edge of the grid, carried by direct P. Before
SeisForge 0.2.0 nothing excluded it. For the weak contrast of panel (c) the
grid from 0 picked (0.2 km, 1.5); the outcome also depended on the
sampling interval and the top-layer density. That ridge was also what made
a station without sediment fall back to one crustal layer (picks near 0,
median below 0.4 km). Now the ridge region is excluded from the pick, and
a station without sediment is recognised because direct P dominates its
sediment stacks (§2, §4); on synthetic RFs of the crust alone the ridge is
7–9 times higher than any conversion in the grid, against 0.5–1.0 for the
sediments of this figure.

$\kappa$ of a thin layer is resolved less well than $H$: in (b) and (c) the
picks are 0.05 below the truth (0.01 at $dt$ = 0.01 s), from interference
of the sediment phases with each other and with the tail of direct P. For
a very weak contrast ($V_S$ 2.8 km/s over 3.6 km/s, 3 km) $H$ is still
found, but the direct-P ridge is higher, so such a layer counts as not
resolved.

Ignoring the sediment biases the crust: in (e) the sediment delays are
absorbed into $H$ and $\kappa$.

!!! verified "Verified"

    - The delays, trade-off slopes and $V_P$ sensitivity are derived with
      SymPy (`hk_derivation.py`) and match `hk_core.hk_phase_delays_s` to
      $10^{-12}$ s and finite differences to $10^{-6}$.
    - `tests/test_rf_hk_core.py`: delays at vertical incidence against the
      closed form; two layers with identical properties reduce to one;
      interpolation is exact for a linear trace and matches `np.interp`;
      delays outside the trace contribute zero and are counted; grid steps
      equal `delta` exactly; the scaled MAD estimates a Gaussian $\sigma$.
    - `tests/test_rf_hk_synthetic.py`, on RFs from the independent
      propagator of [synthetic RFs](synthetic.md): one crustal layer at
      $p$ = 0.04, 0.06, 0.08 recovered exactly on the grid ($dt$ = 0.05 s;
      within two grid steps at $dt$ = 0.1 s); the weak-contrast sediment
      of §5 from $H$ = 0 and from 0.5 km within 0.05 km and 0.06 in
      $\kappa$; the direct-P ridge excluded and flagged; `rf hkseq --mode
      both` on a sediment-over-crust station recovers the sediment (2 km,
      2.2) within 0.05 and the Moho within 0.2 km and 0.01, and on a
      crust-only station falls back to one layer and recovers the Moho
      within one grid step.
    - `tests/test_hk_qc.py`: the estimate is median $\pm$ 1.4826 MAD for
      $n$ = 5, 14, 15, 40.
    - `tests/test_rf_synthetic.py`: Ps, PpPs and PsPs pulses of the
      synthetic RF are within one sample of these delays.
    - On synthetic data with a known answer: the figures above, and
      `examples/04_rf_qc_hk` (34.80 ± 0.30 km, 1.760 ± 0.015; truth 35 km,
      1.75).

!!! unverified "Not verified"

    - The two-layer stage is checked only on noise-free synthetics, and
      with the true sediment $V_S$.
    - The direct-P dominance rule and the half-of-RFs rule are checked on
      noise-free synthetics only; with noise, weak sediments near the
      threshold may flip between one and two layers.
    - The origin and the physical meaning of the Seispy amplitude
      correction.
    - The spread is the scatter of single-RF picks, not a calibrated
      uncertainty of the estimate.
    - No test on real data.

## 6. Usage

```bash
seisforge rf hkseq -c hkseq.yml --mode both|high|low [--plot] [--overwrite]
```

`high` stacks the sediment only, `low` the crust as one layer, `both` the
sediment and then the crust (§5). The keys this page refers to:

```yaml
IO:
  ROOT: .
  HighFreq_RF: output/Gauss_5.0_CLEAN   # SAC files *.sac, user0 = p (s/km)
  LowFreq_RF: output/Gauss_2.5_CLEAN
  SAVE: output/hkSeq                    # inside ROOT, not overlapping the inputs
  FIGURE: output/figures
  LOG: output/logs
Meta: {station: RF01, network: XX}
Model:
  Sediment:
    H: {min: 0.0, max: 5.0, delta: 0.05}   # km
    k: {min: 1.5, max: 3.0, delta: 0.01}
    Vp: null
    Vs: 2.5                                # fixed: Vp = k Vs (assumed value)
    weight: [0.5, 0.2, 0.3]                # w1, w2, w3 of eq. (stack)
    min_ps_delay_s: 0.3                    # §2, assumed: a = 5 pulse at 10 %
    two_layer_min_thickness_km: 0.4        # §5, assumed
  Crust:
    H: {min: 25.0, max: 50.0, delta: 0.2}
    k: {min: 1.6, max: 2.0, delta: 0.01}
    Vp: 6.3                                # fixed: Vs = Vp / k (assumed value)
    Vs: null
    weight: [0.5, 0.2, 0.3]
    min_ps_delay_s: 0.6                    # §2, assumed: a = 2.5 pulse at 10 %
QC:
  LowFreq: {mahalanobis_dist_threshold: 1.0}
  HighFreq: {mahalanobis_dist_threshold: 1.0}
```

Every key is required except `IO.OVERWRITE`; exactly one of `Vp` and `Vs`
is set. Unknown keys are refused, and `energy_threshold` and
`ell_mad_sigma` stop the run with a message that they were removed. A grid
runs from `min` in steps of exactly `delta` and ends at `max` if `max` is
on the grid, otherwise at the last point below it (0 to 1 in steps of 0.3
gives 0, 0.3, 0.6, 0.9).

Outputs in `SAVE`, per stage (`HFreq` sediment, `LFreq` crust):
`NET.STA_<stage>_Hk_results.joblib` (one xarray Dataset per RF: the three
amplitude grids, delays and stack, with the pick, the settings and the
software versions as attributes), `NET.STA_<stage>_Hk_results_QC.joblib`
(the kept ones), and `NET.STA_<stage>_Hk_QC.dat` (file, $H$, $\kappa$,
$S_{\max}$ of the kept RFs, under a header with the software versions, the
grid and settings, the number of RFs dominated by direct P, and the
estimate `H_km` and `k`). Each per-RF dataset records `min_ps_delay_s`,
`n_excluded_direct_p` and `direct_p_dominates`. The estimate is also in
`LOG/NET.STA_hkSeq.log`.
Existing outputs of the requested stage stop the run unless `--overwrite`
is given.

Example with a known answer: `examples/04_rf_qc_hk`; reference config:
`examples/reference_configs/rf_hkseq.yml`.

## Appendix A: Delays from ray geometry

A plane P wave with horizontal slowness $p$ reaches the point $x$ of the
interface at $t_0 + p\,x$. A leg through the layer with velocity $v$ and
angle $\theta$ from the vertical ($\sin\theta = pv$) takes
$H/(v\cos\theta)$ and moves $H\tan\theta$ sideways. Every phase ends at the
station. Its time is therefore the arrival time at its starting point on the
interface plus the leg times, and that starting point lies
$H\sum\tan\theta$ before the station. Per leg,

$$
\frac{H}{v\cos\theta} - pH\tan\theta = \frac{H\,(1 - \sin^2\theta)}{v\cos\theta} = H\,\frac{\cos\theta}{v} = H\,\eta_v .
$$

So a phase with legs $\{v_l\}$ arrives $H(\sum_l \eta_{v_l} - \eta_p)$ after
direct P, which has one P leg. The legs are S for Ps; P, P, S for PpPs; and
P, S, S for PpSs and PsPs. This gives eq. $\eqref{eq:delays}$.
`hk_derivation.py` repeats this sum symbolically from $H/(v\cos\theta)$ and
$H\tan\theta$.

## Appendix B: Fraction kept by a Mahalanobis screen

For $\mathbf x \sim \mathcal N(\boldsymbol\mu, \Sigma)$, the whitened
vector $\Sigma^{-1/2}(\mathbf x - \boldsymbol\mu)$ is a standard bivariate
normal, and $d$ is its length. In polar coordinates its density is
$\rho\,e^{-\rho^2/2}$, so

$$
P(d < r) = \int_0^r \rho\,e^{-\rho^2/2}\,d\rho = 1 - e^{-r^2/2} .
$$

The sample covariance follows the cloud's correlation, so the fraction is
the same for any correlation (panel b of the cluster figure).

## References

- Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute
  deviation. *Journal of the American Statistical Association*, 88(424),
  1273–1283.
  [doi:10.1080/01621459.1993.10476408](https://doi.org/10.1080/01621459.1993.10476408)
- Xu, M., & He, J. (2023). Seispy: Python module for batch calculation and
  postprocessing of receiver functions. *Seismological Research Letters*,
  94(2A), 935–943.
  [doi:10.1785/0220220288](https://doi.org/10.1785/0220220288)
- Zhu, L., & Kanamori, H. (2000). Moho depth variation in southern
  California from teleseismic receiver functions. *Journal of Geophysical
  Research*, 105(B2), 2969–2980.
  [doi:10.1029/1999JB900322](https://doi.org/10.1029/1999JB900322)
