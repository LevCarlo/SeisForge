# Diffuse-Field H/V

!!! overview "Method at a glance"

    In a diffuse wavefield the mean power of motion at a point is proportional
    to the imaginary part of the Green function with source and receiver at
    that point (Sánchez-Sesma et al. 2011). For a station on a layered
    half-space, `seisforge.hvsr` predicts H/V in four steps:

    1. **Surface compliance.** For each horizontal wavenumber $k$, solve the
       layered model for the surface displacement under a unit surface stress:
       the $2\times2$ P-SV matrix $C(k)$ and the SH scalar $C_{yy}(k)$
       ([Green-function page](green_function.md)).
    2. **Green function at the source.** Its imaginary part is a sum of
       surface-wave pole residues (Rayleigh, Love) and a body-wave integral
       over $0<k<\omega/\beta_N$, where $\beta_N$ is the half-space S velocity (§2).
    3. **Azimuthal average.** Each horizontal direction receives half of the
       radial (P-SV) and half of the transverse (SH) term.
    4. **Ratio.** H/V is the square root of horizontal over vertical power (§1).

    The input is the layered model alone; no source distribution enters.

![H/V of a layer over a half-space and its six terms](img/hvsr_contributions.svg#only-light)
![H/V of a layer over a half-space and its six terms](img/hvsr_contributions-dark.svg#only-dark)

*One layer (50 m, $\beta_1$ = 0.2 km/s, $\alpha_1$ = 0.6 km/s,
$\rho_1$ = 1.8 g/cm³) over a half-space ($\beta_N$ = 1.0, $\alpha_N$ = 2.0,
$\rho_N$ = 2.2), native solver with default settings. (a) H/V; the peak,
12.0 at 0.96 Hz on this grid, lies within 5 % of $f_0=\beta_1/4H$ = 1 Hz
(its height converged: §4).
(b) The six terms of eq. $\eqref{eq:energies}$. The script also checks the
scaling $H\to2H$, $f\to f/2$ ($4\times10^{-13}$) and the √2 between the two
horizontal definitions. Script: `docs/hvsr/scripts/hvsr_contributions.py`.*

## 1. Why H/V is a ratio of Green functions

In a diffuse (equipartitioned) field, the power spectrum of displacement
component $i$ at $\mathbf x$ is

$$
E_i(\mathbf x,\omega) \propto \operatorname{Im} G_{ii}(\mathbf x,\mathbf x;\omega),
$$

with a factor common to all components. H/V is therefore

$$
\boxed{\;\frac{H}{V} = \sqrt{\frac{\operatorname{Im}G_{11}+\operatorname{Im}G_{22}}{\operatorname{Im}G_{33}}}
= \sqrt{\frac{2\operatorname{Im}G_{11}}{\operatorname{Im}G_{33}}}\;}
\label{eq:hv}
$$

since a 1-D medium makes the two horizontal components equal. This
`horizontal_definition="sum"` (default) is the definition of Sánchez-Sesma
et al. (2011) and HV-DFA. Observations defined with the mean of the two
horizontal powers correspond to `"mean"`, which divides eq. $\eqref{eq:hv}$ by
$\sqrt2$.

## 2. Surface waves and body waves

Let $C_{ij}(k)$ be the surface displacement per unit surface stress
$\sigma_{iz}$; the applied load is $-\sigma_{iz}$, hence the minus signs
below. A Hankel transform at zero offset, the azimuthal average, and the
Sokhotski–Plemelj formula at each pole
([Appendix A](#appendix-a-poles-and-the-body-wave-interval)) give
$\operatorname{Im}G_{ii} = E_{ii}/2\pi$ with

$$
\begin{aligned}
E_{33} &= \underbrace{-\operatorname{Im}\!\int_0^{\omega/\beta_N}\! k\,C_{zz}\,dk}_{\text{P-SV body}}
\;\underbrace{-\;\pi\sum_{n\in R} k_n \operatorname{Res}_{k_n} C_{zz}}_{\text{Rayleigh}},\\
2E_{11} &= \underbrace{-\operatorname{Im}\!\int_0^{\omega/\beta_N}\! k\,C_{xx}\,dk}_{\text{P-SV body}}
\;\underbrace{-\;\operatorname{Im}\!\int_0^{\omega/\beta_N}\! k\,C_{yy}\,dk}_{\text{SH body}}\\
&\quad\;\underbrace{-\;\pi\sum_{n\in R} k_n \operatorname{Res}_{k_n} C_{xx}}_{\text{Rayleigh}}
\;\underbrace{-\;\pi\sum_{n\in L} k_n \operatorname{Res}_{k_n} C_{yy}}_{\text{Love}},
\end{aligned}
\label{eq:energies}
$$

and $H/V=\sqrt{2E_{11}/E_{33}}$. `GreenFunctionContributions` stores the six
terms, each horizontal one already halved (`rayleigh_horizontal`,
`love_horizontal`, `psv_horizontal`, `sh_horizontal`; `rayleigh_vertical`,
`psv_vertical`).

The split is physical. For $k<\omega/\beta_N$ at least one wave radiates
into the half-space: this interval holds all body waves, including leaky
modes. For $k>\omega/\beta_N$ nothing radiates, $C$ is real, and power
enters $\operatorname{Im}G$ only through the surface-wave poles $k_n$.

## 3. Three limits with known answers

**One Rayleigh mode gives its ellipticity.** The residue of $C$ at a mode has rank one, and its diagonal entries are in
the ratio $u_x^2:u_z^2$ of the mode's surface displacement
([Appendix B](#appendix-b-one-mode-gives-its-ellipticity)). With
$\chi=|u_x/u_z|$, eq. $\eqref{eq:energies}$ reduces to

$$
\boxed{\;\frac{H}{V}\Big|_{\text{one Rayleigh mode}} = \chi\;}
$$

The H/V peak of the fundamental mode is the frequency at which $u_z$ passes
through zero, and the trough the one at which $u_x$ does.

![Rayleigh-only H/V equals the ellipticity](img/rayleigh_ellipticity.svg#only-light)
![Rayleigh-only H/V equals the ellipticity](img/rayleigh_ellipticity-dark.svg#only-dark)

*Native solver with only Rayleigh mode 0, against `disba.Ellipticity` (a
separate code): relative difference below $6\times10^{-5}$ except at the
$u_z=0$ singularity next to $f_0$ (362 vs 353). Poisson half-space
($\alpha=\sqrt3\,\beta$): 0.681250 against the ellipticity from the Rayleigh
equation solved in the script with potentials. Script:
`docs/hvsr/scripts/rayleigh_ellipticity.py`.*

**A half-space has no length scale.** $C(k)$ depends on $k$ and $\omega$
only through $k/\omega$ and $\omega H/v$, so every term of
eq. $\eqref{eq:energies}$ scales with the same power of $\omega$ when there are
no layers: H/V of a half-space is independent of frequency (Poisson solid:
1.3256 with surface waves, 2.097 from body waves alone; package values, not
compared with a published one). In a layered model, scaling all thicknesses
by $\lambda$ moves every feature of the curve from $f$ to $f/\lambda$.

**A soft layer resonates at $f_0=\beta_1/4H$.** At vertical incidence the
SH and SV surface compliances peak at $f_n=(2n+1)\beta_1/4H$, with power
amplification $(\rho_N\beta_N/\rho_1\beta_1)^2$
([Green-function page](green_function.md#2-boundary-system)). The H/V peak lies
near $f_0$ but is not that resonance: H/V also contains oblique body waves
and surface waves. In the first figure the Love term dominates the horizontal power
around $f_0$, and the vertical Rayleigh term passes through zero at
0.99 Hz, so there the vertical power, the denominator, is the P-SV
body-wave term alone.

## 4. Peak height near f0 (T21, resolved)

![H/V near f0: fixed and adaptive quadrature](img/peak_convergence.svg#only-light)
![H/V near f0: fixed and adaptive quadrature](img/peak_convergence-dark.svg#only-dark)

*(a) H/V between 0.9 and 1.1 Hz from the fixed Gauss–Legendre rule used
before 0.2.0 (50, 100, 200 nodes per segment, rebuilt in the script) and
from the adaptive quadrature now used. (b) The P-SV horizontal integrand at
0.955 Hz and the old 50 nodes. Script:
`docs/hvsr/scripts/peak_convergence.py`.*

Before 0.2.0 the H/V of this model changed by up to a factor of 3 between
0.90 and 1.00 Hz when the quadrature order changed (spikes of 9–29). The
cause is a pole of $C(k)$ just off the real axis inside the body-wave
interval: Rayleigh mode 1, whose cut-off ($\omega/k=\beta_N$) is near
1.08 Hz, continues below it as a leaky mode with apparent velocity
1.2–1.5 km/s, between $\beta_N$ and $\alpha_N$. It radiates S waves into the
half-space, but its leakage vanishes near $f^*$ = 0.9524 Hz: the half width of
the integrand's peak goes to zero like $(f-f^*)^2$ (the script fits both
sides), down to $5\times10^{-5}$ of the interval at 0.955 Hz, against a node
spacing of $1.6\times10^{-2}$. A fixed rule samples only its flanks, at
positions that move with the order. The integral itself is finite and
smooth in $f$: as the peak narrows it grows, and its area tends to
$\pi$ times the residue, as for a surface-wave pole.

The integral is now adaptive ([Green-function page, §4](green_function.md#4-body-wave-integral)):
Gauss–Kronrod on each segment, bisected until the error estimate of each term is
below $10^{-6}$ of the term. The error is measured on the complex integrand,
whose real part has a $1/(k-k_p)$ tail at any distance from a near-real pole,
so the pole cannot hide between nodes. With it, H/V near $f_0$ changes by
less than $10^{-8}$ when the tolerance goes from $10^{-6}$ to $10^{-9}$; the
peak is 12.1 at 0.97 Hz. The fixed rule was also off elsewhere: by 4 % in
the body-wave terms at 0.9 and 1.0 Hz here, and by 16 % and 7 % at 3 and
10 Hz on the four-layer test model, where surface waves dominate and H/V
changed by at most 0.14 %.

!!! verified ""

    Verified: body-wave terms at 0.955 and 0.97 Hz against scipy's QAGS
    (a different adaptive rule with extrapolation) to $10^{-6}$; convergence
    of H/V over 0.94–0.97 Hz with the tolerance, and no spikes
    (`tests/test_hvsr_body_waves.py`; the convergence test is marked slow).

!!! unverified ""

    Not verified: how often such near-real poles occur in other models
    (one model examined); whether HV-DFA, which samples the real axis
    uniformly (`-nks`), resolves them.

## 5. Solvers

All three take an `ElasticLayerModel` (km, km/s, g/cm³; thicknesses of the
finite layers only, the last property entry is the half-space), frequencies
in Hz, and `DiffuseFieldSettings`, and are called through
`predict_diffuse_field_hvsr(model, frequencies_hz, solver=..., settings=...)`.

| solver | what computes eq. $\eqref{eq:energies}$ | needs |
|---|---|---|
| `NativeDiffuseFieldSolver` | Python ([Green-function page](green_function.md)) | `disba` (mode seeds) |
| `FortranDiffuseFieldSolver` | the same equations in an f2py kernel; Python passes the `disba` seeds | optional extension: `pip install -e '.[hvsr-fortran]'`, `python tools/build_hvsr_fortran.py` |
| `HVDFAReferenceSolver` | the external program HV-DFA (García-Jerez et al. 2016) | a local HV-DFA executable |

The native and Fortran solvers return the six terms; HV-DFA returns H/V
only. The Fortran kernel was written from the equations on these pages.
HV-DFA is distributed under CC BY-NC; SeisForge does not include or build
it, and the adapter only runs a user-supplied binary (SI units, a private
temporary directory per call) for regression tests.

The settings mean:

| key | default | native and Fortran | HV-DFA |
|---|---|---|---|
| `max_rayleigh_modes`, `max_love_modes` | 20, 20 | modes $0,1,\dots$; stop at the first mode with no root at any frequency | `-nmr`, `-nml` |
| `include_body_waves` | true | the body-wave integral of eq. $\eqref{eq:energies}$ | `-nks 0` when false |
| `horizontal_definition` | `sum` | eq. $\eqref{eq:hv}$ | `mean`: output divided by √2 |

These choose *what* is computed and are shared by all solvers. How
accurately is a field of the solver, so a control a solver would ignore
cannot be set: `NativeDiffuseFieldSolver` and `FortranDiffuseFieldSolver`
take `body_quadrature_rtol` and the other controls of the
[Green-function page](green_function.md#usage); `HVDFAReferenceSolver` takes
`body_wave_wavenumbers` (100, `-nks`), `precision_percent` (0.1, `-prec`),
`sh_damping` and `psv_damping` (0, `-ash`, `-apsv`).

0 for all modes gives body waves only (`approximation="body_waves_only"`);
`include_body_waves: false` gives surface waves only.

Every frequency carries a diagnostic: `ok`; `incomplete` when H/V is given
but a term is missing or not converged, with the term and reason listed in
`diagnostics[i].issues` (a surface-wave mode whose root or residue failed,
with its index; a body-wave integral that reached
`body_quadrature_max_intervals`); or `invalid` (NaN). The same records are in
`contributions.issues`. HV-DFA returns no diagnostics, so its frequencies
are `ok` when the program succeeds.

<div class="assumptions" markdown>

| assumption | fails when | symptom |
|---|---|---|
| diffuse field: sources everywhere, power equipartitioned | dominant nearby or directional sources; short records | measured H/V differs in shape, not only in level |
| 1-D layering | lateral changes within about a wavelength | H/V differs between azimuths or nearby stations |
| elastic medium | frequencies where attenuation matters | predicted peaks too high and narrow |

</div>

!!! verified ""

    Verified: eq. $\eqref{eq:hv}$ and the √2 of `"mean"`
    (`tests/test_hvsr_forward.py`); full native H/V within 1 % of HV-DFA 1.0
    at 0.2 and 1 Hz on a four-layer model, and body-wave-only H/V within
    0.03 %, against values from one HV-DFA run stored in the tests
    (`tests/test_hvsr_native.py`, `tests/test_hvsr_body_waves.py`); Fortran
    against native, body-wave terms to $10^{-10}$, surface-wave terms and H/V
    to $10^{-9}$ where both include the same modes, and the modes each leaves
    out recorded (`tests/test_hvsr_fortran.py`, runs only when the extension
    is built); `incomplete` status for a left-out mode or an unconverged
    integral (`tests/test_hvsr_diagnostics.py`);
    the three limits of §3 (scripts above). All checks use synthetic models.

!!! unverified ""

    Not verified: real data; models with low-velocity zones; the HV-DFA
    regression `test_upstream_hvdfa_regression` runs only with
    `SEISFORGE_HVDFA_EXECUTABLE` set and checks only that H/V is finite and
    positive.

## 6. Usage

```bash
seisforge init hvsr-forward my_site && cd my_site && python hvsr_forward.py config.yml
```

`seisforge init` copies `config.yml` and the script `hvsr_forward.py`, which
calls the native solver with its default accuracy and writes the solver
controls and any `incomplete`/`invalid` frequencies into the output header. The model file has one row per layer,
`thickness_km vp_km_s vs_km_s density_g_cm3`, the last row the half-space
(thickness ignored). The config:

```yaml
model_file: data/model.txt
frequency_hz: {min: 0.2, max: 10.0, count: 100}   # log-spaced
output_file: output/hvsr.dat                      # not overwritten if present
settings: {include_body_waves: true}              # optional, DiffuseFieldSettings
```

Example with a known answer: `examples/05_hvsr_forward` (the model of the
figures).

## Appendix A: poles and the body-wave interval

A point stress at the origin of the surface gives, at zero offset,
$u_z = \frac{1}{2\pi}\int_0^\infty k\,C_{zz}(k)\,dk$ for a vertical stress and
$u_x = \frac{1}{2\pi}\int_0^\infty k\,\tfrac12[C_{xx}+C_{yy}](k)\,dk$ for a
horizontal one: the radial and transverse responses each enter with
$\langle\cos^2\rangle=\langle\sin^2\rangle=\tfrac12$. For the time
dependence $e^{-i\omega t}$ outgoing waves put the poles $k_n$ just above the
real axis, so the path passes below them. Near a pole
$C\approx R_n/(k-k_n)$, and

$$
\frac{1}{x-i0-k_n} = \mathrm{P}\frac{1}{x-k_n} + i\pi\,\delta(x-k_n).
$$

Above $\omega/\beta_N$, $C$ is real between the poles, so the principal
value adds nothing to the imaginary part, and each pole contributes
$\pi k_n R_n$, with real $R_n$. Below $\omega/\beta_N$ the integrand is
complex and is integrated directly. With $G=-C$ for the applied load, this
gives eq. $\eqref{eq:energies}$ (derived here; the $1/2\pi$ cancels in H/V).

## Appendix B: one mode gives its ellipticity

At a pole the homogeneous boundary problem has one solution, the mode, with
surface displacement $\mathbf a=(u_x,u_z)$. Near the pole every surface
stress excites only that mode, so $C\approx\mathbf a\,\mathbf b^{\mathsf T}
/(k-k_n)$: the residue has rank one, with $\mathbf b$ the mode's excitation
by $(\sigma_{xz},\sigma_{zz})$. By reciprocity $|b_x/b_z|=|a_x/a_z|=\chi$,
so $\operatorname{Res}C_{xx}/\operatorname{Res}C_{zz}=\chi^2$, and
eq. $\eqref{eq:energies}$ with one Rayleigh mode gives
$H/V=\sqrt{2\cdot\tfrac12\chi^2}=\chi$ (derived here; the ellipticity figure
checks it numerically).

## References

- Sánchez-Sesma, F. J., et al. (2011). A theory for microtremor H/V spectral
  ratio: application for a layered medium. *Geophysical Journal
  International*, 186(1), 221–225.
  [doi:10.1111/j.1365-246X.2011.05064.x](https://doi.org/10.1111/j.1365-246X.2011.05064.x)
- García-Jerez, A., Piña-Flores, J., Sánchez-Sesma, F. J., Luzón, F., &
  Perton, M. (2016). A computer code for forward calculation and inversion of
  the H/V spectral ratio under the diffuse field assumption. *Computers &
  Geosciences*, 97, 67–78.
  [doi:10.1016/j.cageo.2016.06.016](https://doi.org/10.1016/j.cageo.2016.06.016)
  Program: [HV-DFA](https://github.com/agarcia-jerez/HV-DFA).
