# Synthetic P Receiver Functions

!!! overview "Method at a glance"

    A plane P wave with horizontal slowness $p$ arrives from the half-space
    below a stack of flat, isotropic layers. A P receiver function is the
    radial ground motion deconvolved by the vertical. For one plane wave it
    is the spectral ratio of the two:

    1. **Layers.** In each layer, write the motion and traction of the four
       plane waves (down and up P and SV) as the columns of a matrix (§1).
    2. **Propagate.** Start from the two traction-free surface states and
       carry them down to the half-space, one layer at a time (§2).
    3. **Solve.** Combine the two states so that the half-space holds a unit
       up-going P and no up-going S. This gives the surface radial $R(\omega)$
       and vertical $Z(\omega)$ (§2).
    4. **Filter.** $\mathrm{RF}(\omega) = \dfrac{R}{Z}\,e^{-\omega^2/4a^2}$,
       transformed to time on an FFT grid long enough that late
       reverberations do not wrap around (§3, §4).

    The result contains every conversion and reverberation in the stack.
    Each layer uses its own $V_P$, $V_S$ and density $\rho$.

![A synthetic receiver function](img/synthetic_rf.svg#only-light)
![A synthetic receiver function](img/synthetic_rf-dark.svg#only-dark)

*(a) RF of a 35 km crust ($V_P$ 6.3, $V_S$ 3.6 km/s, $\rho$ 2.8 g/cm³)
over a mantle half-space ($V_P$ 8.0, $V_S$ 4.5, $\rho$ 3.3), $p$ = 0.06 s/km,
$a$ = 2.5. The ticks are the ray-theory delays of [H-k stacking](hk.md#1-phase-delays).
Each pulse peaks within one sample of its tick, with the expected sign. (b)
Direct-P amplitude of a half-space against eq. $\eqref{eq:halfspace}$.
Script: `docs/rf/scripts/synthetic_rf.py`.*

## 1. Plane waves in a layer

Depth $z$ is positive downward and the wave travels towards $+x$. Plane
waves are $e^{i\omega(px + qz - t)}$, with $q = +\eta$ for down-going and
$q = -\eta$ for up-going waves, where $\eta_v = \sqrt{v^{-2} - p^2}$. P
motion is along the slowness vector, $(u_x, u_z) = V_P\,(p, q)$, and SV
motion is perpendicular to it, $V_S\,(q, -p)$. Hooke's law with
$\partial_x \to i\omega p$, $\partial_z \to i\omega q$ gives the tractions
divided by $i\omega$:

$$
\frac{\sigma_{xz}}{i\omega} = \mu\,(q\,u_x + p\,u_z),\qquad
\frac{\sigma_{zz}}{i\omega} = \lambda\,(p\,u_x + q\,u_z) + 2\mu\,q\,u_z ,
$$

with $\mu = \rho V_S^2$ and $\lambda = \rho V_P^2 - 2\mu$. The four motion-stress
vectors $(u_x, u_z, \sigma_{xz}/i\omega, \sigma_{zz}/i\omega)$ of down P, down
SV, up P and up SV are the columns of $E_k$ for layer $k$. With the traction
divided by $i\omega$, $E_k$ does not depend on frequency.

## 2. From the surface to the half-space

Let $\mathbf c_k$ be the four wave amplitudes at the top of layer $k$.
Crossing the layer multiplies them by
$\Lambda_k = \operatorname{diag}(e^{i\omega\eta_p h_k}, e^{i\omega\eta_s h_k}, e^{-i\omega\eta_p h_k}, e^{-i\omega\eta_s h_k})$.
The motion-stress vector is continuous at the interface below, so

$$
\mathbf c_{k+1} = E_{k+1}^{-1}E_k\,\Lambda_k\,\mathbf c_k .
$$

For $p < 1/V_P$ in every layer all $\eta$ are real. The transfer matrices
$E_{k+1}^{-1}E_k$ are then real, are computed once, and each layer costs one
phase multiplication per frequency. The phases are not growing exponentials,
so the product stays stable.

At the free surface the traction vanishes. Two independent surface states
remain: unit $u_x$ and unit $u_z$, with amplitudes $\mathbf s_x$ and
$\mathbf s_z$ (the first two columns of $E_0^{-1}$). Propagating both to the
half-space gives $\mathbf X$ and $\mathbf Z$. The surface motion
$u_x\mathbf s_x + u_z\mathbf s_z$ must produce a unit up-going P (index 3)
and no up-going S (index 4) in the half-space:

$$
u_x X_3 + u_z Z_3 = 1,\qquad u_x X_4 + u_z Z_4 = 0
\quad\Rightarrow\quad
R = u_x = \frac{Z_4}{D},\quad Z = -u_z = \frac{X_4}{D},\quad D = X_3 Z_4 - Z_3 X_4 ,
$$

with $Z$ positive up. Waves going down in the half-space are left free,
since they leave the stack.

For a half-space alone this is the free-surface reflection of a P wave
([Appendix](#appendix-the-half-space-ratio)):

--8<-- "docs/rf/generated/halfspace_ratio.tex"

This is the amplitude of the direct-P pulse. It is zero at vertical
incidence and grows with $p$, which is why RF amplitudes depend on the ray
parameter.

## 3. Amplitude and time conventions

$\mathrm{RF}(\omega) = (R/Z)\,e^{-\omega^2/4a^2}$. A spike of amplitude $A$
in $R/Z$ becomes a Gaussian in time:

--8<-- "docs/rf/generated/gaussian_pulse.tex"

Its half-width at half maximum is $\sqrt{\ln 2}/a \approx 0.83/a$ (0.33 s
at $a$ = 2.5). Two amplitude scalings are offered:

| `amplitude` | spike $A$ peaks at | matches |
|---|---|---|
| `impulse` (default) | $A\,a/\sqrt\pi$ | continuous inverse transform; units 1/s |
| `gaussian_peak` | $A$ | iterative deconvolution with a unit-peak Gaussian |

Observed RFs must be compared in the scaling in which they were made. The
physics uses $e^{-i\omega t}$, and the spectrum is conjugated before
NumPy's inverse FFT ($e^{+i\omega t}$). The trace is shifted by
`time_shift_s`, so that `time_s = n dt - time_shift_s` with direct P at 0 s.

## 4. FFT length

The discrete inverse transform is periodic with the FFT length
$T = n_{\mathrm{fft}}\,\Delta t$. Energy arriving after $T$ wraps into the
window. The default length covers $\max(2\,T_w,\ T_w + 60\ \mathrm{s})$
for a window $T_w$, rounded up to a fast FFT size. The peak of the last
quarter of the padding is then compared with the window's peak, skipping
the last $4/a$, which holds the wrapped early flank of direct P. If the
ratio exceeds `tail_tolerance` ($10^{-3}$, an assumed value), the length is
doubled. Doubling halves the frequency step, so the existing frequencies
are reused and only the new ones are computed. After 16 times the original
length the function raises an error instead of returning a wrapped trace:
then $Z(\omega)$ nearly vanishes and $R/Z$ rings. An explicit `n_fft` is
used as given. Only frequencies where the Gaussian exceeds $10^{-12}$ are
computed.

![Wrap-around](img/synthetic_wraparound.svg#only-light)
![Wrap-around](img/synthetic_wraparound-dark.svg#only-dark)

*A 20 s window of the crust of the first figure. With the FFT as long as
the window, the PpSs + PsPs pulse at 19 s and later reverberations wrap into
the start of the window. The error is 26 % of the peak, against $8\times10^{-8}$
with the default. Reference: a 131072-point FFT. Script:
`docs/rf/scripts/synthetic_wraparound.py`.*

<div class="assumptions" markdown>

| assumption | fails when |
|---|---|
| flat, isotropic, elastic layers | dipping layers, anisotropy, attenuation (none modelled) |
| solid layers, $0 < V_S < V_P$ | water or other fluid layers (rejected) |
| $p < 1/V_P$ in every layer and the half-space | evanescent P, e.g. a fast half-space at large $p$ (rejected) |
| one plane P wave, radial over vertical | S receiver functions; the source-side and noise parts of real deconvolutions |

</div>

!!! verified "Verified"

    - `synthetic_derivation.py` derives the half-space ratio and the Gaussian
      pulse with SymPy and matches `p_wave_surface_response` to
      $10^{-14}$ and the `impulse` peak to $10^{-13}$.
    - `tests/test_rf_synthetic.py`: the half-space ratio; Ps, PpPs and PsPs
      times and signs against ray theory; identical layers equal a
      half-space; a vanishing layer converges to the model without it;
      density changes the conversions but not direct P; `gaussian_peak`
      recovers a spike's amplitude; the default FFT matches a 65536-point FFT
      to $10^{-4}$ of the peak, also for a 15 s window; a ringing model and
      evanescent P raise errors.
    - Against PyGRT 0.17.2 (`grt rcvfn`, an independent generalized
      reflection-transmission code): the direct-P amplitude and the Ps/P,
      PpPs/P and PsPs/P ratios of the crust of the first figure are pinned
      in `test_pygrt_reference_amplitudes` to $2\times10^{-4}$ and
      $10^{-3}$.

!!! unverified "Not verified"

    - A wider PyGRT comparison (six models including arbitrary density, 12
      layers and a 0.9 km/s sediment, $p$ = 0.04 and 0.075) found waveform
      agreement to 3–8 × 10⁻⁴ of the peak after a constant 2.8 ms delay in
      PyGRT. It was run once offline and cannot be repeated from the
      repository.
    - No comparison with observed RFs.

## 5. Usage

```python
from seisforge.rf.synthetic import receiver_function

time_s, rf = receiver_function(
    thickness_km, vp_km_s, vs_km_s, rho_g_cm3,   # layers from the top; last = half-space
    0.06,                                        # p (s/km)
    dt_s=0.05, n_samples=1200, gauss_a=2.5,
    time_shift_s=5.0, amplitude="gaussian_peak",
)
```

`seisforge.inv.predict_receiver_function(model, ReceiverFunctionRequest(...))`
applies it to an inversion `LayeredModel` and raises `ForwardError` on
invalid input ([API](../api/rf.md)).

## Appendix: The half-space ratio

With no layers, the surface holds the incident up-going P (amplitude 1) and
the reflected down-going P and SV (amplitudes $A$, $B$). The two traction
conditions $\sigma_{xz} = \sigma_{zz} = 0$ at $z = 0$ are two linear
equations for $A$ and $B$. The surface motion is the sum of the three
waves, and $R/Z = u_x/(-u_z)$. `synthetic_derivation.py` solves this with
SymPy from the eigenvectors of §1. The result reduces to eq.
$\eqref{eq:halfspace}$, so $A$ and $B$ need not be written out.

## References

- Aki, K., & Richards, P. G. (2002). *Quantitative Seismology* (2nd ed.),
  ch. 7. University Science Books.
- Haskell, N. A. (1962). Crustal reflection of plane P and SV waves.
  *Journal of Geophysical Research*, 67(12), 4751–4768.
  [doi:10.1029/JZ067i012p04751](https://doi.org/10.1029/JZ067i012p04751)
- Kennett, B. L. N. (1983). *Seismic Wave Propagation in Stratified Media*.
  Cambridge University Press.
- Ligorría, J. P., & Ammon, C. J. (1999). Iterative deconvolution and
  receiver-function estimation. *Bulletin of the Seismological Society of
  America*, 89(5), 1395–1400.
  [doi:10.1785/BSSA0890051395](https://doi.org/10.1785/BSSA0890051395)
- Thomson, W. T. (1950). Transmission of elastic waves through a stratified
  solid medium. *Journal of Applied Physics*, 21(2), 89–93.
  [doi:10.1063/1.1699629](https://doi.org/10.1063/1.1699629)
