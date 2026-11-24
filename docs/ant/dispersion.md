# AFTAN: Frequency-Time Analysis and Phase-Matched Filtering

!!! note "Notes for reference"
    Notes written while implementing `seisforge.ant.aftan`, following the
    Fortran AFTAN of Levshin, Ritzwoller and co-workers (Bensen et al. 2007)
    and PyAFTAN; for reference only.

!!! overview "Method at a glance"

    FTAN (Dziewonski et al. 1969) and PMF (Herrin & Goforth 1977) were first
    applied to earthquake surface waves. Here we only focus on ambient-noise
    cross-correlations (CCFs), and §1–§5 use the ZZ component.

## 1. How group and phase velocity are measured

![FTAN of a synthetic CCF](img/ftan_synthetic.svg#only-light)
![FTAN of a synthetic CCF](img/ftan_synthetic-dark.svg#only-dark)

*(a) The positive lag of a synthetic ZZ correlation between two stations
400 km apart. (b) Its FTAN diagram: the envelope after narrow-band
filtering, one column per period, plotted against group velocity $r/t$. The
picks follow the true group velocity, and the phase velocities fall on the
true curve. (c) Errors stay below 0.03 km/s. Script:
`docs/ant/scripts/ftan_synthetic.py`.*

The correlation in (a) is a dispersed wave train: each frequency travels
with its own speed (in this example long periods arrive first).
Two speeds describe it:

- the **group velocity** $U(\omega)$, the speed of the energy of a narrow
  frequency band;
- the **phase velocity** $c(\omega) = \omega/k$, the speed of the crests.

Frequency-time analysis (FTAN) separates the frequencies and measures both.

**Step 1: filter.** The trace $x(t)$ is made analytic: in the spectrum
$\hat x(\omega) = \int x(t)\,e^{-i\omega t}\,dt$ the negative frequencies
are removed and the positive ones doubled. It is then filtered with a
Gaussian centred at $\omega_0$ (Dziewonski et al. 1969):

$$
G(\omega;\omega_0) = \exp\!\left[-\alpha\left(\frac{\omega - \omega_0}{\omega_0}\right)^2\right],
$$

$$
S(t;\omega_0) = \frac{1}{2\pi}\int_0^\infty 2\hat x(\omega)\,G(\omega;\omega_0)\,e^{i\omega t}\,d\omega
= A\,e^{i\varphi} .
$$

The filtered signal $S$ is complex. Its modulus $A$ is the envelope of the
band and its argument $\varphi$ the phase. The parameter $\alpha$ sets the
width of the band (§2). Repeating this for a set of $\omega_0$ gives the FTAN
diagram, panel (b).

**Step 2: group velocity from the envelope peak.** The energy of the band
arrives when its envelope peaks, at $t_g$. With $r$ the interstation
distance,

$$
U(\omega_0) = \frac{r}{t_g} .
$$

That the envelope peak travels at $d\omega/dk$ is shown in
[Appendix B](#appendix-b-the-envelope-peak-travels-at-the-group-velocity).

**Step 3: phase velocity from the phase at the peak.** A narrow-band wave
that has travelled the distance $r$ has the phase
$\varphi = \omega t - kr + \phi_0$. Read at $t_g$ and solved for
$k = \omega/c$,

$$
\boxed{\;c = \frac{\omega\, r}{\omega\,t_g - \varphi(t_g) + \phi_0 + 2\pi N}\;},
\qquad N\in\mathbb Z .
$$

- $\omega t_g$ is the phase accumulated over the travel time. It would equal
  $kr$ if crests travelled with the energy ($c = U$).
- $\omega t_g - \varphi$ measures how far the crests have moved relative to
  the energy.
- $\phi_0$ is the initial phase of the signal. For a ZZ noise correlation
  $\phi_0 = \pi/4$
  ([Appendix A](#appendix-a-the-phase-of-a-zz-noise-correlation)). If the
  input is the empirical Green's function $-dC/dt$ instead of the correlation
  $C$, then $\phi_0 = -\pi/4$. SeisForge measures $C$ itself.
- $2\pi N$ is the unknown number of whole cycles, because a phase is known
  only modulo $2\pi$ (§4).

An error $\delta t$ in $t_g$ cancels to first order, because $\varphi$
changes by $\omega\,\delta t$ as well.

These three steps are the whole method. The rest of the page deals with
choosing $\alpha$ (§2), turning the picks into a continuous curve (§3),
finding $N$ (§4), and cleaning the trace before measuring (§5).

## 2. The filter width α

### What α means

In the form $\exp[-(\omega-\omega_0)^2/2\sigma_\omega^2]$ the filter has

$$
\sigma_\omega = \frac{\omega_0}{\sqrt{2\alpha}},\qquad
\sigma_t = \frac{\sqrt{2\alpha}}{\omega_0},\qquad
\sigma_\omega\,\sigma_t = 1 ,
$$

where $\sigma_t$ is the width of the filter's impulse response. So $\alpha$
fixes the **relative** bandwidth, $\sigma_\omega/\omega_0 = 1/\sqrt{2\alpha}$,
which is the same for every filter (a constant-Q bank). It also fixes the
length of the impulse response in periods, $\sigma_t/T_0 = \sqrt{2\alpha}/2\pi$.
At $\alpha = 20$ the relative bandwidth is 16 % and the response lasts about
one period. For these amplitude widths $\sigma_\omega\sigma_t = 1$, which is
the uncertainty-principle minimum (1/2 for widths of $|G|^2$); only a
Gaussian reaches it, which is why a Gaussian is used.

![Gaussian filters](img/gaussian_filters.svg#only-light)
![Gaussian filters](img/gaussian_filters-dark.svg#only-dark)

*(a) Frequency response at $T_0$ = 10 s for three values of $\alpha$. (b)
Impulse response; the envelope agrees with the closed form
$e^{-\omega_0^2 t^2/4\alpha}$ (dashed) to 0.1 %. (c) The trade-off between
time and frequency width. Script: `docs/ant/scripts/gaussian_filters.py`.*

### Too wide a band: the frequency is biased

If the amplitude spectrum $a(\omega)$ of the signal is not flat across the
filter, the filtered energy is centred away from $\omega_0$. For
$a \propto e^{\beta(\omega - \omega_0)}$, with $\beta = d\ln a/d\omega$, the
product of the exponential and the Gaussian is a Gaussian centred at

$$
\omega_c = \omega_0 + \beta\,\sigma_\omega^2,\qquad
\frac{\omega_c - \omega_0}{\omega_0} = \frac{1}{2\alpha}\,\frac{d\ln a}{d\ln\omega}.
$$

A spectrum falling as $\omega^{-2}$, for example, lowers the effective
frequency by $1/\alpha$, which is 5 % at $\alpha = 20$. The measured $t_g$
belongs to $\omega_c$, not to $\omega_0$. The measurement is therefore
assigned to the **instantaneous frequency** at the peak,

$$
\omega_{\text{inst}} = \left.\frac{\partial\varphi}{\partial t}\right|_{t_g},
\qquad T_{\text{inst}} = \frac{2\pi}{\omega_{\text{inst}}} ,
$$

and $\omega_{\text{inst}}$ is the $\omega$ used in the phase-velocity formula.
Outputs keep both $T_{\text{inst}}$ and the filter period $T_0$.

### Too narrow a band: the packet is smeared

A narrow band means a long impulse response, $\sigma_t \propto \sqrt\alpha$.
At the same time the dispersion inside the band spreads the packet: the
group delay changes across the band by about $r\,|D|\,\sigma_\omega$, with
$D = d(1/U)/d\omega$. The filtered packet therefore has a width of roughly

$$
w^2 \approx \sigma_t^2 + (r D \sigma_\omega)^2 = \frac{2\alpha}{\omega_0^2} + \frac{r^2 D^2\omega_0^2}{2\alpha},
$$

which is smallest at

$$
\alpha_{\text{opt}} = \frac{r\,|D|\,\omega_0^2}{2}
= \pi\,\frac{r/U}{T_0}\,\left|\frac{d\ln U}{d\ln\omega}\right| .
$$

The best $\alpha$ is proportional to the number of periods in the travel
time multiplied by the logarithmic dispersion. It grows with distance, and
it is larger where $U$ changes quickly.

### Choosing α in practice

$\alpha$ is one number per trace: it can depend on distance but not on
period. This is an implementation choice ($\alpha_{\text{opt}}$ itself varies
with period). The choices offered (`alpha.mode`):

| mode | $\alpha$ | origin |
|---|---|---|
| `constant` (default) | 20 | PyAFTAN |
| `aftan_distance_scaling` | $20\,f\sqrt{r/1000\ \text{km}}$ | Fortran AFTAN, factor $f$ |
| `pyftan_constant` | $20 f$ | PyAFTAN, factor $f$ |
| `empirical_distance_table` | linear in $r$ through (0, 5), (100, 8), (250, 12), (500, 20), (1000, 25), (2000, 35), (4000, 50), (20000, 75) km | empirical; nodes configurable |

All of these grow more slowly with distance than $\alpha_{\text{opt}}$,
which is proportional to $r$. Packet width is not the only concern: a wide
band also biases the frequency, which argues for a larger $\alpha$ at short
distance. The values in the table are empirical, not derived.

## 3. Picking a continuous dispersion curve

**Refining each peak.** The envelope is sampled at $\Delta t$. The peak
sample $n$ and its two neighbours define a parabola, whose vertex lies at

$$
\delta = \frac{A_{n-1} - A_{n+1}}{2\,(A_{n-1} - 2A_n + A_{n+1})}
$$

samples from $n$, so $t_g = (n + \delta)\,\Delta t$. The phase and
$\omega_{\text{inst}}$ are read at the same point, from a parabola through
the unwrapped phases of the three samples.

**Several maxima.** A column of the FTAN diagram can have several local
maxima: the fundamental mode, overtones, other arrivals and noise. All of
them are kept, and the largest is the first-pass pick.

**Jumps.** When two arrivals overlap, the largest peak can switch from one
to the other between neighbouring periods. As in AFTAN, a trigger measures
the curvature of the curve; this implementation uses $U$ against $x = \ln T$
(Fortran `trigger.f` differentiates in $\omega$ with another normalisation,
so thresholds do not carry over):

$$
\kappa_i = \frac{1}{x_{i+1}-x_{i-1}}\left[\frac{U_{i+1}-U_i}{x_{i+1}-x_i} - \frac{U_i - U_{i-1}}{x_i - x_{i-1}}\right].
$$

A sample is flagged with the sign of $\kappa_i$ when $|\kappa_i|$ exceeds a
threshold (50 for basic FTAN, 20 for PMF). A jump is a place where the flag
changes sign. Between two jumps a few periods apart, each pick is replaced
by the local maximum closest in $U$ to the previous period's pick. The
replacement is kept only if no flags remain. Finally, only the longest run
of unflagged periods is kept. This removes isolated spikes, but it cannot
tell which of two continuous branches is the fundamental mode.

**Input taper.** Before filtering, each lag branch is multiplied by a window
that equals 1 for group velocities between $v_{\min}$ and $v_{\max}$, that
is for $t \in [r/v_{\max}, r/v_{\min}]$. Outside that interval it has
half-cosine ramps of length $T_{\max}$, as in Fortran AFTAN. This removes the
energy near zero lag and the late noise without suppressing early arrivals
at short distance.

**Signal-to-noise ratio.** For each pick, the branch is filtered again at
$T_{\text{inst}}$, without the input taper. The signal is the envelope
maximum in $[t_g - T, t_g + T]$. The noise is the RMS envelope in a window
that starts 500 s after the signal window and is 500 s long. The windows of
Bensen et al. (2007) and the AFTAN definition are available as alternatives.

**Enough wavelengths.** The initial phase $\phi_0$ holds in the far field,
and the two lag branches must not overlap. Both need enough wavelengths
along the path. SeisForge (not AFTAN) caps the longest period at
$T_{\max}^{\text{eff}} = \min(T_{\max},\ r/(n_\lambda v_{\text{ref}}))$, with
defaults $n_\lambda = 0.5$ and $v_{\text{ref}} = 4$ km/s. This is a
permissive limit. A stricter one, typically three wavelengths, is usually
applied when measurements are selected for tomography. The far-field error
of $\phi_0$ is only 0.3 % in $c$ at one wavelength (Appendix A), so the
stricter cut guards mainly against overlapping branches and the finite
window.

## 4. The 2π ambiguity

Every $N$ in the phase-velocity formula reproduces the measured phase
exactly. In slowness the branches are

$$
\frac{1}{c_N} = \frac{1}{c_0} + N\,\frac{T}{r},
\qquad
\frac{\Delta c}{c} \approx \frac{\lambda}{r}\quad\text{between neighbours},
$$

with $\lambda = cT$ the wavelength. A phase measured at a single distance has
no memory of how many crests passed, so the ambiguity is inherent in the
method. One trace does contain one more relation, $dk/d\omega = 1/U$, which
links the phase at neighbouring frequencies. It fixes $N$ from period to
period but not overall. The overall $N$ needs outside information, a
reference curve.

![2 pi branches](img/phase_branches.svg#only-light)
![2 pi branches](img/phase_branches-dark.svg#only-dark)

*(a) Branches $N = -2\ldots2$ of the measured curve (r = 400 km). (b) Branch
spacing $\lambda/r$: at short periods and long paths, neighbouring branches
are only a few percent apart. (c) Branch chosen for a reference that is
uniformly too fast or too slow by $\varepsilon$. Script:
`docs/ant/scripts/phase_branches.py`.*

**Choosing N.**

1. **Anchor** at the longest period, where branches are farthest apart.
   Choose the $N$ that is closest to the reference $c_{\text{ref}}$ (a
   reference curve, else a constant 4 km/s).
2. **Walk to shorter periods** using the group velocity. Predict the
   wavenumber by integrating $dk/d\omega = 1/U$ with the trapezoidal rule,
   $k_i = k_{i+1} + (\omega_i - \omega_{i+1})\,\tfrac12(1/U_i + 1/U_{i+1})$
   with $k_{i+1} = \omega_{i+1}/c_{i+1}$, and choose the $N_i$ closest to it.
3. **Shift the whole curve** (an addition in SeisForge, used only when a
   reference curve is given, not with a constant reference velocity). Among
   shifts of $-2$ to $+2$ cycles, choose the one that minimises the mean
   relative misfit to $c_{\text{ref}}$ over all periods. Step 1 uses one
   period, so a reference that is poor there can be outvoted by the rest of
   the band.

**How good must the reference be?** Per period, branch $N = 0$ is the
closest if the relative reference error satisfies
$|\varepsilon| < \lambda/2r$. For the whole-curve shift with a uniform error
$\varepsilon$, the correct branch beats its neighbours when
$|\varepsilon| \lesssim \tfrac12\langle\lambda/r\rangle$. In panel (c) this
estimate is 10.4 %, and the choice actually switches at $-10.5$ % and about
$+12$ %. The tolerance is set by the long periods, where $\lambda/r$ is
large, and it shrinks as $1/r$. At 1000 km and $T \le 20$ s,
$\lambda/r \lesssim 7$ %, so a 4 % error in the reference can already select
the wrong branch. Step 2 has its own condition: the integrated slowness error
over one period step must stay below half a cycle,
$r\,|\Delta\omega|\,|\delta(1/U)| < \pi$. A jump in $U$ that survives §3
propagates a cycle error to all shorter periods.

## 5. Phase-matched filtering

A basic FTAN pick is only as clean as the trace. PMF (Herrin & Goforth 1977;
Levshin & Ritzwoller 2001) uses the first-pass group velocity to remove the
dispersion. The surface wave then becomes a short pulse that can be
separated from noise and other arrivals, which stay spread out. The trace is
cleaned, the dispersion restored, and §1–§4 run again. Let the first pass
give $t_g(\omega) = r/U(\omega)$ on $[\omega_a, \omega_b]$, with $\omega_a$
the lowest frequency.

**(1) Phase matching.** Integrate the group delay:

$$
\Phi(\omega) = \int_{\omega_a}^{\omega} \big[t_g(\omega') - t_g(\omega_a)\big]\,d\omega' ,
$$

using a cubic spline of $t_g$ in $\omega$, with $\Phi = 0$ outside the band.
Multiply the analytic spectrum by $e^{i\Phi}$. Write the signal spectrum as
$a(\omega)\,e^{-i\psi(\omega)}$ with $\psi = kr - \phi_0$, so that
$\psi' = r\,dk/d\omega = t_g$. Then

$$
-\psi(\omega) + \Phi(\omega) = -\psi(\omega_a) - t_g(\omega_a)\,(\omega - \omega_a),
$$

which is a **linear** phase. Every frequency now arrives at the same time,
$t_g(\omega_a)$, so the surface wave is a pulse there. The lower limit and
the subtracted $t_g(\omega_a)$ add only a constant and a term linear in
$\omega$ to $\Phi$. They shift the pulse in time, and step (3) removes them
again, so their choice does not affect the result.

A cosine band taper is applied at the same time. Its corners are
$f_{1,2} = (1\mp\beta)/T_{\max}$, $f_3 = 1/T_{\min}$ and
$f_4 = (1+\beta)/T_{\min}$, with $\beta = 1/\sqrt{2\alpha}$. The taper is
therefore 0.5 at $1/T_{\max}$.

**(2) Window.** Let $n_p$ be the sample of the envelope maximum of the
compressed signal. On each side, take the envelope minima that are more than
5 s from $n_p$ and below 0.2 of the peak amplitude: $n_{l1}, n_{r1}$ are the
nearest such minima and $n_{l2}, n_{r2}$ the next ones. Define

$$
h = \max(n_p - n_{l1},\ n_{r1} - n_p),\qquad
w = \max(n_{l1} - n_{l2},\ n_{r2} - n_{r1}) + 1,
$$

and multiply by

$$
W(n) =
\begin{cases}
1, & |n - n_p| \le h,\\[2pt]
\exp\!\left[-\dfrac{(|n - n_p| - h)^2}{2\,w^2}\right], & |n - n_p| > h .
\end{cases}
$$

The window is symmetric about the peak, as in the Fortran routine
`tgauss.f`. Its flat part reaches the farther of the two inner minima, and
its taper width is the larger of the two gaps between inner and outer
minimum. A missing outer minimum is replaced by the trace end, and a side
with no qualifying minimum is not tapered. PyAFTAN's variant has a wider
right taper.

![PMF cleaning window](img/pmf_window.svg#only-light)
![PMF cleaning window](img/pmf_window-dark.svg#only-dark)

*The window on a compressed noisy CCF. It is built here from the rule above
and agrees with the package's window sample by sample. Script:
`docs/ant/scripts/pmf_window.py`.*

**(3) Redispersion.** Multiply the spectrum of the windowed signal by
$e^{-i\Phi}$ and take the real part. The result is the original surface
wave with the energy away from the pulse removed.

**(4) Second pass.** Run §1–§4 on the cleaned trace, over the period range
that the first pass covered, with a lower jump threshold.

PMF can only sharpen the first pass: if the first-pass $U$ follows a wrong
branch (an overtone or another arrival), PMF isolates that branch.

![Phase-matched filtering](img/pmf_comparison.svg#only-light)
![Phase-matched filtering](img/pmf_comparison-dark.svg#only-dark)

*(a) Synthetic CCF plus a faster interfering arrival and noise. (b) After
phase matching, the surface wave is a pulse at $t_g(\omega_a)$ (dotted). (c)
The redispersed output has lost the interfering arrival and most of the
noise. (d) Group velocities. Script: `docs/ant/scripts/pmf_comparison.py`.*

**Long-period edge.** At the long-period end of its band, the PMF group
velocity is biased low. On noise-free synthetics with $\alpha = 20$ the error
exceeds 0.03 km/s above about $0.75\,T_{\max}$ and reaches about 0.09 km/s at
$T_{\max}$, nearly independent of distance. Basic FTAN stays within
0.03 km/s, and the phase velocity is not affected. PyAFTAN shows the same
bias on the same trace. The cause is the window: its width is set by the
envelope minima of the compressed pulse, which follow the short periods of
the band, so it removes part of the long-period energy. The band taper,
which is 0.5 at $1/T_{\max}$, adds to the bias. A wider window reduces the
bias but keeps more noise. **PMF group velocities are therefore best used up
to about $0.7\,T_{\max}$** of the PMF band. These numbers come from
noise-free synthetic data only.

## 6. Other components (open question)

The diffuse-field average of
[Appendix A](#appendix-a-the-phase-of-a-zz-noise-correlation) applies to
the horizontal components too. A retrograde Rayleigh wave has horizontal
motion $H = i\varepsilon\,\hat d\,Z$, where $\varepsilon > 0$ is the H/V
ratio. Use the radial axes of [the rotation page](rotation.md#1-conventions),
which point along the propagation at both stations, and
$\langle\cos a\,e^{-ikr\cos a}\rangle = -iJ_1(kr)$. Then

$$
\hat C_{ZR} = \big\langle\overline{Z_A}\,(i\varepsilon\cos a)\,Z_B\big\rangle = +\varepsilon J_1,\qquad
\hat C_{RZ} = \big\langle\overline{(i\varepsilon\cos a)\,Z_A}\,Z_B\big\rangle = -\varepsilon J_1,
$$

$\hat C_{RR} = \varepsilon^2(J_0 - J_2)/2$, and for Love waves on TT,
$(J_0 - J_2)/2$ (Lin et al. 2008; Lin et al. 2014 for ZR and RZ). With
$J_n(x) \simeq \sqrt{2/\pi x}\cos(x - n\pi/2 - \pi/4)$ the far-field constants
are as follows. $p$ is the value the code adds as $p\,\pi/4$ to the measured
phase, so $p = -4\phi_0/\pi$.

| component | far-field $\hat C$ | $\phi_0$ | $p$ |
|---|---|---|---|
| ZZ, RR, TT | $+\cos(kr - \pi/4)$ | $+\pi/4$ | $-1$ |
| ZR | $+\cos(kr - 3\pi/4)$ | $+3\pi/4$ | $-3$ |
| RZ | $-\cos(kr - 3\pi/4)$ | $-\pi/4$ | $+1$ |

$\hat C_{RZ} = -\hat C_{ZR}$ because the factor $i$ of the horizontal motion
is conjugated when it sits at the source. The constants belong to the stored component and are the same on
both lag branches. They are not taken from the reciprocal component, because
the time-reversed negative lag of ZR for pair AB is $-$RZ for pair BA: the
radial axis at B changes direction when B becomes the source.

What is established and what is not:

- The table is checked on synthetic diffuse fields
  (`tests/test_ant_aftan_components.py`), not on real data.
- The PMF stage has not been checked on ZR and RZ.
- ZT, TZ, RT and TR have no automatic constant.
- The derivation assumes a laterally homogeneous medium and an isotropic
  noise field. How well the ZR/RZ and RR constants hold on real
  correlations is open.

## 7. Usage

```bash
seisforge ant aftan -c aftan.yml [--write-energy-map] [--plot-energy-map]
```

One run measures the CCFs of one virtual source. The keys this page refers
to, with their defaults where they have one:

```yaml
station: XX.S00                 # virtual source
io:
  input_dir: data/ccf
  sac_pattern: "*_ZZ.SAC"       # files to measure
  output_dir: output
  overwrite: false
aftan:
  branch: both                  # positive | negative | both | stack (ZZ, RR, TT only)
  min_period: 2.0               # s
  max_period: 8.0
  period_sampling: {mode: uniform, uniform: {step: 0.5}}   # or list, geomspace
  velocity_window: {min: 1.5, max: 4.0}                    # km/s; input taper (§3)
  pi_over_4: {mode: auto}       # p of §6 from the component; or a number
  prediction_file: ref.dat      # period_s velocity_km_s; reference for §4
  basic:
    alpha: {mode: constant, constant: {value: 20.0}}       # §2 table
    trig_threshold: 50.0
  pmf:
    enabled: false
    alpha: {mode: constant, constant: {value: 20.0}}
    trig_threshold: 20.0
    window_factor: 1.0          # scales h and w of §5
    clean_window: symmetric     # or pyaftan
  snr: {definition: local}      # or pyftan, aftan
phase_cycle_selection: {enabled: true, max_shift: 2}       # §4 step 3
short_distance_period_guard: {enabled: true, reference_velocity: 4.0, max_period_nwl: 0.5}
```

- **Input.** SAC files holding a two-sided CCF, with `b`, `delta` and either
  `dist` or `evla`/`evlo` (virtual source) and `stla`/`stlo`. The component
  is read from `kcmpnm`, else from the file name. Instead of `station` and
  `sac_pattern`, a pair can be given as `source_station`, `receiver_station`
  and `components`, with `io.input_root_datadir`, `io.output_root_datadir`
  and `io.input_template`; files are then read from
  `<input_root_datadir>/<source>/<source>_<receiver>/`.
- **Output** per file and branch: `<name>.<branch>.dat` with columns
  `target_period_s instant_period_s group_velocity_km_s phase_velocity_km_s
  amplitude snr` and a header recording the settings; `<name>.<branch>.pmf.dat`
  when PMF is on; `aftan.log`. The energy-map options write the FTAN diagram
  (`.basic_ftan.nc`, `.png`) and, with `energy_map.phase_velocity: true`, the
  $2\pi$ candidates (`.vph.nc`).
- **Manual cycle choice.** `python tools/aftan_cycle_set.py <file>.vph.nc
  --shift k` previews a shift of $k$ cycles; `--in-place` applies it to the
  `.vph.nc` and `.dat` files and records it.

A runnable example with a known answer is `examples/02_aftan`; every key is
listed in `examples/reference_configs/aftan.yml`.

## Appendix A: The phase of a ZZ noise correlation

This appendix derives $\phi_0 = \pi/4$. It assumes a two-dimensional,
isotropic noise field: plane Rayleigh waves of equal strength from every
horizontal direction, in a laterally homogeneous medium. Real noise fields
deviate from this, and the deviation biases the phase.

**Convention.** With $\hat x(\omega) = \int x(t)\,e^{-i\omega t}\,dt$ and
inverse kernel $e^{+i\omega t}$ (the `numpy.fft` convention), a wave
$\cos(\omega t - kr + \phi_0)$ whose crests reach distance $r$ at $t = r/c$
has the spectrum $e^{-i(kr - \phi_0)}$ at $\omega > 0$.

**Diffuse field.** With station A at the origin and B at $r\hat x$, a plane
wave travelling in direction $\hat d$ has the vertical motion
$W e^{-ik\,\hat d\cdot\mathbf x}$. Averaging the correlation spectrum
$\overline{Z_A}Z_B$ over the angle $a$ between $\hat d$ and $\hat x$ gives

$$
\hat C_{ZZ}(\omega) = W\,\frac{1}{2\pi}\int_0^{2\pi} e^{-ikr\cos a}\,da = W\,J_0(kr),
$$

which is the integral representation of the Bessel function $J_0$. This is
the spatial-autocorrelation result of Aki (1957).

**Far field.** For $kr \gg 1$ (DLMF 10.17),

$$
J_0(x) \simeq \sqrt{\frac{2}{\pi x}}\cos\!\left(x - \frac{\pi}{4}\right)
= \sqrt{\frac{1}{2\pi x}}\left[e^{\,i(x - \pi/4)} + e^{-i(x - \pi/4)}\right].
$$

Under the inverse transform, the second term becomes
$e^{i(\omega t - kr + \pi/4)}$. This is a wave arriving at positive lag
with $\phi_0 = +\pi/4$. The first term is its mirror image at negative lag,
with the same constant after time reversal. Physically, only waves
travelling close to the line through A and B add coherently. Their path
difference grows quadratically with the angle away from that line, so the
sum is a Fresnel integral, which contributes $e^{\pm i\pi/4}$ (the
stationary-phase argument of Snieder 2004).

**Correlation or Green's function.** The empirical Green's function is the
negative time derivative of the correlation, $G = -dC/dt$ (Lin et al. 2008,
sec. 2). Differentiation multiplies the spectrum by $i\omega$ (a phase of
$+\pi/2$), and the minus sign adds $\pi$. So $G$ has
$\phi_0 = \pi/4 + \pi/2 - \pi = -\pi/4$.

**How far is "far field".** The next term of the expansion (DLMF 10.17),
$J_0(x) \simeq \sqrt{2/\pi x}\,[\cos(x-\pi/4) + \tfrac{1}{8x}\sin(x-\pi/4)]$,
is a phase error of about $1/(8kr)$. Since $\delta\varphi = \delta(kr)$, the
relative phase-velocity error is

$$
\left|\frac{\delta c}{c}\right| \approx \frac{1}{8\,(kr)^2} = \frac{1}{32\pi^2}\left(\frac{\lambda}{r}\right)^2 ,
$$

which is 0.3 % at one wavelength and 0.04 % at three. This estimate is
derived here, not taken from a source.

## Appendix B: The envelope peak travels at the group velocity

Write the spectrum of one lag branch as $a(\omega)\,e^{-i\psi(\omega)}$ with
$\psi = kr - \phi_0$. Near $\omega_0$, expand
$\psi \simeq \psi_0 + \psi'(\omega-\omega_0) + \tfrac12\psi''(\omega-\omega_0)^2$
and take $a$ as constant across the band. The integral for $S$ in §1 is then
the integral of a Gaussian, and

$$
A(t;\omega_0) \propto \exp\!\left[-\frac{(t - \psi')^2}{2\,w^2}\right],\qquad
\psi' = r\,\frac{dk}{d\omega} = \frac{r}{U(\omega_0)} ,
$$

where $w$ is the packet width of §2. The envelope peaks at $t_g = r/U$. The
term $\psi''$, the change of group delay across the band, only widens the
packet. The approximation fails where $U$ bends sharply within one band,
which is where the group-velocity error in the opening figure is largest
(15–25 s).

## References

- Aki, K. (1957). Space and time spectra of stationary stochastic waves,
  with special reference to microtremors. *Bulletin of the Earthquake
  Research Institute*, 35, 415–456.
- Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
  Moschetti, M. P., Shapiro, N. M., & Yang, Y. (2007). Processing seismic
  ambient noise data to obtain reliable broad-band surface wave dispersion
  measurements. *Geophysical Journal International*, 169(3), 1239–1260.
  [doi:10.1111/j.1365-246X.2007.03374.x](https://doi.org/10.1111/j.1365-246X.2007.03374.x)
- DLMF: NIST Digital Library of Mathematical Functions, §10.17, Asymptotic
  expansions for large argument. <https://dlmf.nist.gov/10.17>
- Dziewonski, A., Bloch, S., & Landisman, M. (1969). A technique for the
  analysis of transient seismic signals. *Bulletin of the Seismological
  Society of America*, 59(1), 427–444.
  [doi:10.1785/BSSA0590010427](https://doi.org/10.1785/BSSA0590010427)
- Herrin, E., & Goforth, T. (1977). Phase-matched filters: application to
  the study of Rayleigh waves. *Bulletin of the Seismological Society of
  America*, 67(5), 1259–1275.
  [doi:10.1785/BSSA0670051259](https://doi.org/10.1785/BSSA0670051259)
- Levshin, A. L., & Ritzwoller, M. H. (2001). Automated detection,
  extraction, and measurement of regional surface waves. *Pure and Applied
  Geophysics*, 158(8), 1531–1545.
  [doi:10.1007/PL00001233](https://doi.org/10.1007/PL00001233)
- Lin, F.-C., Moschetti, M. P., & Ritzwoller, M. H. (2008). Surface wave
  tomography of the western United States from ambient seismic noise:
  Rayleigh and Love wave phase velocity maps. *Geophysical Journal
  International*, 173(1), 281–298.
  [doi:10.1111/j.1365-246X.2008.03720.x](https://doi.org/10.1111/j.1365-246X.2008.03720.x)
- Lin, F.-C., Tsai, V. C., & Schmandt, B. (2014). 3-D crustal structure of
  the western United States: application of Rayleigh-wave ellipticity
  extracted from noise cross-correlations. *Geophysical Journal
  International*, 198(2), 656–670.
  [doi:10.1093/gji/ggu160](https://doi.org/10.1093/gji/ggu160)
- Snieder, R. (2004). Extracting the Green's function from the correlation
  of coda waves: a derivation based on stationary phase. *Physical Review
  E*, 69, 046610.
  [doi:10.1103/PhysRevE.69.046610](https://doi.org/10.1103/PhysRevE.69.046610)
