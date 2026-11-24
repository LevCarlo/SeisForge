# Surface Green Function of a Layered Half-Space

!!! overview "Method at a glance"

    The [diffuse-field H/V](index.md) needs the surface compliance $C(k)$,
    the surface displacement per unit surface stress, for horizontal
    wavenumbers $k$ at one frequency. `seisforge.hvsr` evaluates it in four
    steps:

    1. **Plane waves.** In each layer, P, SV and SH waves going up and down,
       with vertical wavenumbers on the radiation branch (§1).
    2. **Boundary system.** Free surface and welded interfaces give one
       linear system; each wave's amplitude is referred to the interface from
       which it decays, so no exponential grows (§2).
    3. **Poles.** Surface-wave wavenumbers $k_n$ from `disba` seeds, refined
       on $C$ itself; residues by a central difference (§3).
    4. **Body-wave integral.** Adaptive Gauss–Kronrod on segments between
       branch points, on a path just below the real axis (§4).

## 1. Conventions

Time dependence $e^{-i\omega t}$, waves $e^{ikx}$, $z$ positive down from the
free surface. In a layer with velocity $v$ the vertical wavenumber is

$$
\gamma_v = \sqrt{k^2-\omega^2/v^2},\qquad
\operatorname{Re}\gamma_v \ge 0,\quad \operatorname{Im}\gamma_v<0 \text{ if } \operatorname{Re}\gamma_v=0,
$$

so that $e^{-\gamma_v z}$ either decays with depth or carries energy
downwards. Numerically, the principal square root (which has
$\operatorname{Re}\ge0$) is negated only when $\operatorname{Im}\gamma_v>0$
and $\operatorname{Re}\gamma_v$ is within the round-off of the argument,
$|\operatorname{Re}\gamma_v|\,|\gamma_v| \le 16\,\epsilon\,(|k|^2+\omega^2/v^2)$
($d\gamma = da/2\gamma$ for $a=k^2-\omega^2/v^2$). The test is relative,
so it does not depend on the units of $k$; Python and Fortran use the same
one. (Before 0.2.0 Python used an absolute $10^{-8}$ and Fortran
$32\epsilon$; on the paths used, real $k$ and $k(1-i\varepsilon)$, neither
ever chose differently, so no result changed.) The P-SV state vector is
$(u_x,u_z,\sigma_{xz},\sigma_{zz})$; with $e^{sz}$, $s=\mp\gamma$, a P wave and
an SV wave contribute the columns

$$
\mathbf p = \begin{pmatrix} ik\\ s\\ 2i\mu ks\\ (\lambda+2\mu)s^2-\lambda k^2\end{pmatrix},
\qquad
\mathbf s = \begin{pmatrix} -s\\ ik\\ -\mu(s^2+k^2)\\ 2i\mu ks\end{pmatrix},
$$

($\mu=\rho\beta^2$, $\lambda=\rho\alpha^2-2\mu$), and SH contributes
$(u_y,\sigma_{yz})=(1,\ \mu s)$. $C(k)$ is the $2\times2$ matrix of surface
$(u_x,u_z)$ for unit surface $(\sigma_{xz},\sigma_{zz})$, and $C_{yy}$ the SH
scalar. Units: km, km/s, g/cm³, Hz throughout; $C$ is used only in ratios.

## 2. Boundary system

Layer $j$ (thickness $h_j$) has four P-SV amplitudes: down-going waves
$\propto e^{-\gamma(z-z_j^{\text{top}})}$ and up-going waves
$\propto e^{+\gamma(z-z_j^{\text{bot}})}$. Every exponential then has
modulus at most 1 inside its layer. The half-space has two down-going
waves. The unknowns, $4(N-1)+2$ for $N-1$ finite layers, satisfy

- two rows at $z=0$: $(\sigma_{xz},\sigma_{zz})$ equal to the unit load, and
- four rows per interface: continuity of the whole state vector,

solved once for both loads (SH: $2(N-1)+1$ unknowns, one surface row, two
per interface). A Thomson–Haskell product would carry
$e^{\gamma h}$ factors that overflow for thick layers or high frequency;
this system has none, and stays finite for, e.g., a 100 km layer at 50 Hz
and $k$ = 400 km⁻¹.

At $k=0$ the problem is one-dimensional and has a closed form for a layer
over a half-space ([Appendix](#appendix-normal-incidence-closed-form)): with
$Z=\rho v$ and $\theta=\omega H/v_1$,

$$
\boxed{\;-\omega Z_N \operatorname{Im}C(0) = \frac{Z_N^2}{Z_N^2\cos^2\theta + Z_1^2\sin^2\theta}\;}
\label{eq:normal}
$$

with $v=\beta$ for $C_{xx}$ and $C_{yy}$ and $v=\alpha$ for $C_{zz}$. It
peaks at $f_n=(2n+1)v_1/4H$ with value $(Z_N/Z_1)^2$.

![Normal-incidence compliance against the closed form](img/normal_incidence.svg#only-light)
![Normal-incidence compliance against the closed form](img/normal_incidence-dark.svg#only-dark)

*Model of the [H/V page](index.md) (50 m, $\beta_1$ = 0.2, $\alpha_1$ =
0.6 km/s over $\beta_N$ = 1.0, $\alpha_N$ = 2.0 km/s). Package
`psv_surface_compliance` and `sh_surface_compliance` at $k=0$ against
eq. $\eqref{eq:normal}$: maximum relative difference $1.1\times10^{-14}$.
Dotted: $(2n+1)\beta_1/4H$ and $\alpha_1/4H$. Script:
`docs/hvsr/scripts/normal_incidence.py`.*

## 3. Surface-wave poles

For $k>\omega/\beta_N$ the Rayleigh poles are the zeros of
$\operatorname{Re}[1/\det C(k)]$ and the Love poles those of
$\operatorname{Re}[1/C_{yy}(k)]$. Finding them:

1. **Seeds.** `disba` (Dunkin's method, phase-velocity step
   `dispersion_step_km_s` = 0.001 km/s) traces mode $n$ over all periods in
   one call, so each period starts from the previous root. Where a higher
   mode does not exist the seed is missing and the mode is skipped there.
2. **Refinement.** Within $\pm3\times$ the step around the seed in phase
   velocity, the function is sampled (33 points plus the seed, then geometric
   steps around the seed) and every sign change is refined with Brent's method; the root nearest the seed is kept.
   Without a sign change, the smallest $|\cdot|$ is accepted if it is below
   1 % of its value at the seed.
3. **Residue.** With $h=10^{-5}k_n$ (`residue_step_fraction`),

    $$
    \operatorname{Res}_{k_n} C \approx \tfrac{h}{2}\,[C(k_n+h)-C(k_n-h)],
    $$

    exact for $R_n/(k-k_n)$ plus a constant, with error $O(h^2 C''_{\text{reg}})$.

A mode whose root cannot be refined, or whose diagonal residue term is
complex or negative beyond $10^{-8}$, is left out at that frequency and
recorded: the frequency's diagnostic becomes `incomplete` and lists the wave,
the mode index and the reason. A mode with no seed at a frequency (beyond its
cut-off) is not a failure and is not recorded.

![The compliance along k at 1 Hz](img/k_plane.svg#only-light)
![The compliance along k at 1 Hz](img/k_plane-dark.svg#only-dark)

*$|C|$ at 1 Hz against horizontal slowness $k/\omega$. Shaded: the
body-wave interval, split at $1/\alpha_N$. The Rayleigh and Love poles (red)
lie beyond $1/\beta_N$, where $C_{xx}$, $C_{zz}$ and $C_{yy}$ are real to
$1.5\times10^{-11}$ (checked in the script). The peak inside the interval is
the near-real pole behind the [former H/V peak spikes](index.md#4-peak-height-near-f0-t21-resolved).
Script: `docs/hvsr/scripts/k_plane.py`.*

## 4. Body-wave integral

With $u=k\beta_N/\omega\in(0,1)$, the interval is first split at
$u=\beta_N/v_j$ for every P and S velocity $v_j$ of the model that falls
inside it: these are the branch points, where some $\gamma_{v_j}$ changes from
imaginary to real and the integrand has a square-root kink. On each piece a
15-point Gauss–Kronrod rule (QUADPACK QK15) gives the integral and, from the
embedded 7-point Gauss rule, an error estimate $|K_{15}-G_7|$. The piece with
the largest estimate relative to the target is bisected, the left half
taking its place in the list and the right half appended, until for each of
the three terms ($C_{xx}$, $C_{zz}$, $C_{yy}$) the summed estimate is below
`body_quadrature_rtol` ($10^{-6}$) times $|\operatorname{Im}\int|$ of that
term. The estimate is pessimistic: on the documented models the actual error
against scipy's QAGS is about $10^{-8}$.

The error is measured on the complex integrand. A pole $k_p$ just off the
real axis makes $\operatorname{Im}[kC]$ a peak that can be far narrower than
the nodes (see [T21](index.md#4-peak-height-near-f0-t21-resolved)), but
$\operatorname{Re}[kC]$ has a $1/(k-k_p)$ tail that the rule sees at any
distance, at least as large as the peak's contribution; the bisection then
closes in on the pole. If `body_quadrature_max_intervals` (4000) is reached
first, the integral is kept and the frequency is `incomplete`.

The path is $k(1-i\varepsilon)$ with $\varepsilon$ =
`contour_shift_fraction` = $10^{-8}$, the outgoing side of the cuts. The
result approaches the real-axis limit linearly in $\varepsilon$, with
$|\delta I/I|$ between 10 and 80 $\varepsilon$ on the documented models; the
default keeps that bias below the tolerance (it was $10^{-5}$ before 0.2.0,
a bias up to $8\times10^{-4}$). Negative energy beyond round-off is an error.

Cost: 600–1300 integrand evaluations per frequency for the one-layer model,
up to 4500 at 10 Hz for the four-layer one (the fixed rule used 100–150).

## 5. The Fortran kernel

`src/seisforge/hvsr/_fortran_src/hvsr_core.f90` implements §1–4 with the
same columns, branch rule, boundary system and adaptive quadrature (same
rule, same choice of interval, same list order; Python passes it the
segments and the `disba` seeds). It returns a skip code for every frequency
and mode and a convergence flag for every frequency, which Python turns into
the same diagnostics as the native solver. Its root refinement differs: 65
samples and bisection, without the seed-neighbour search and the
minimisation fallback, so it finds fewer roots of closely spaced high modes
(10 of 20 × 20 Rayleigh mode-frequency pairs on the four-layer test model at
3.6–10 Hz). Before 0.2.0 these were dropped silently, which is the
$2\times10^{-4}$ H/V difference to the native solver once quoted here; they
are now `incomplete`. Where both include the same modes, body-wave terms
agree to $10^{-10}$ and surface-wave terms and H/V to $10^{-9}$ (measured
$2\times10^{-11}$ and $7\times10^{-11}$).

!!! verified ""

    Verified: the radiation branch, including a decaying wave just off the
    cut in any units (`tests/test_hvsr_diagnostics.py`); the body-wave
    integral against scipy's QAGS near the leaky pole and the linear
    $\varepsilon$ bias; the half-space compliance at $k=0$
    ($1/\rho\beta\omega$, $1/\rho\alpha\omega$), finiteness of the system for a
    thick layer at high frequency, frequency independence of half-space
    body-wave H/V, and body-wave H/V against HV-DFA 1.0 within 0.03 % at 0.2
    and 1 Hz (stored values; `tests/test_hvsr_body_waves.py`); the Fortran
    agreement above (`tests/test_hvsr_fortran.py`, only with the extension
    built); eq. $\eqref{eq:normal}$ and the reality of $C$ beyond
    $\omega/\beta_N$ (scripts above). All on synthetic models.

!!! unverified ""

    Not verified: the residue step against an analytic residue of a layered
    model; that the error estimate always sees a near-real pole (argued in
    §4 from the $1/(k-k_p)$ tail, tested on one model); the Fortran kernel on this
    documentation's figures (they use the native solver); the missing
    Fortran fallbacks for high-mode roots (recorded, not fixed).

## Usage

The functions above are in `seisforge.hvsr.propagator`
(`psv_surface_compliance(model, frequency_hz, k)`,
`sh_surface_compliance`), `seisforge.hvsr.modes`
(`surface_wave_green_contributions`) and `seisforge.hvsr.body_waves`
(`body_wave_green_contributions`, `predict_body_wave_hvsr`). The numerical
controls of §3–4 are fields of `NativeDiffuseFieldSolver` and
`FortranDiffuseFieldSolver`: `body_quadrature_rtol` ($10^{-6}$),
`body_quadrature_max_intervals` (4000), `contour_shift_fraction`
($10^{-8}$), `dispersion_step_km_s` (0.001), `residue_step_fraction`
($10^{-5}$). `BodyWaveSettings` (`relative_tolerance`, `max_intervals`,
`contour_shift_fraction`) holds the same body-wave controls for
`body_wave_green_contributions`; `predict_body_wave_hvsr` takes
`horizontal_definition` as a keyword.

## Appendix: normal-incidence closed form

At $k=0$ the SH problem in the layer is $u=Ae^{i\kappa z}+Be^{-i\kappa z}$,
$\kappa=\omega/\beta_1$, and in the half-space a down-going wave with
$\sigma_{yz}=i\omega Z_N u$ (from $s=-\gamma=i\omega/\beta_N$). Carrying this
impedance to the surface through the layer,

$$
\frac{\sigma_{yz}(0)}{u(0)} = i\omega Z_s,\qquad
Z_s = Z_1\,\frac{Z_N\cos\theta - iZ_1\sin\theta}{Z_1\cos\theta - iZ_N\sin\theta},
$$

so $C(0)=1/(i\omega Z_s)$ and
$-\omega\operatorname{Im}C(0) = \operatorname{Re}(1/Z_s) =
Z_N/(Z_N^2\cos^2\theta+Z_1^2\sin^2\theta)$, which is eq. $\eqref{eq:normal}$.
SV ($C_{xx}$) is the same problem; P ($C_{zz}$) replaces $\beta$ by
$\alpha$. Derived here; a half-space ($Z_1=Z_N$) gives
$C(0)=-i/(\omega\rho\beta)$.

## References

- Dunkin, J. W. (1965). Computation of modal solutions in layered, elastic
  media at high frequencies. *Bulletin of the Seismological Society of
  America*, 55(2), 335–358.
  [doi:10.1785/BSSA0550020335](https://doi.org/10.1785/BSSA0550020335)
- [disba](https://github.com/keurfonluu/disba): the dispersion code used for
  the mode seeds.
- References for the diffuse-field relations: [H/V page](index.md#references).
