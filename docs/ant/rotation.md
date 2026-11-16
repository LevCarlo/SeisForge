# Rotation of Multi-Component Cross-Correlations (ZNE → ZRT)

!!! note "Draft"
    Method page under review. Statements that could not be checked against
    the code or a source are marked **Unverified**.

Ambient-noise cross-correlations are computed between the recorded channels
of two stations: vertical (Z), north (N) and east (E). Surface waves,
however, are polarised relative to the *path*: Rayleigh waves move in the
vertical plane that contains the path (Z and radial, R), Love waves
perpendicular to it (transverse, T). Only after the horizontal channels are
turned into path coordinates do the two wave types separate. This page
derives that rotation for the full $3\times3$ correlation tensor, states
exactly which axes SeisForge uses, and lists the identities that check it.

The executed SymPy version of every step is on the page
[Rotation: executable derivation](scripts/rotation_derivation.py).

## 1. Conventions

### Cross-correlation and component order

For a virtual source A and a receiver B, the correlation of component $a$ at
A with component $b$ at B is

$$
C_{ab}(t) = \int u_a^{A}(\tau)\, u_b^{B}(\tau + t)\, d\tau ,
\qquad
\hat C_{ab}(\omega) = \overline{\hat u_a^{A}(\omega)}\, \hat u_b^{B}(\omega),
$$

with the Fourier transform $\hat f(\omega) = \int f(t)\,e^{-i\omega t}\,dt$
(the convention of `numpy.fft`/`scipy.fft`, used throughout the AFTAN code).
The **first letter is always the virtual source A, the second the receiver
B**: `NZ` is N at A correlated with Z at B. A wave that leaves A and reaches
B after a time $r/U$ produces a peak at **positive** lag $t = r/U$.

!!! warning "Unverified: input CCFs"
    SeisForge does not compute the correlations; it reads them from SAC files.
    The definition above is what the rotation and the AFTAN positive branch
    assume (it is the definition used by the synthetic CCFs in
    `seisforge.ant.synthetic` and in the [phase-constant
    analysis](../aftan_phase_constants.md)). A correlation code that uses the
    opposite lag convention swaps the two branches; a code that stores the
    receiver channel first swaps the meaning of the letters. Neither is
    detected by the package. Check your correlation code against this
    definition.

### Angles

Both angles are measured clockwise from north, as in seismology:

- $\theta$, the **azimuth**: direction at A towards B;
- $\psi$, the **back azimuth**: direction at B towards A.

On a sphere the great circle through A and B meets the two local meridians at
different angles, so $\psi \neq \theta + 180°$ in general. The difference
grows with distance and latitude; in `examples/01_ccf_rotate` (a short
path near 30° N) the log reports $\theta = 46.02°$ and $\psi = 226.32°$,
0.3° away from $\theta + 180°$. The package therefore takes both angles as
inputs, from the SAC headers `az`/`baz`, else from the coordinates
(`evla/evlo` = A, `stla/stlo` = B, via ObsPy's `gps2dist_azimuth`), else from
the configuration.

### The rotated axes used by SeisForge

Writing horizontal unit vectors as (north, east) components, the package uses
by default (`radial_convention: lin2008`)

| | radial $\hat R$ | transverse $\hat T$ |
|---|---|---|
| virtual source A | $(\cos\theta,\, \sin\theta)$ | $(-\sin\theta,\, \cos\theta)$ |
| receiver B | $(-\cos\psi,\, -\sin\psi)$ | $(\sin\psi,\, -\cos\psi)$ |

These are `seisforge.ant.rotation.horizontal_axes`, and
`rotate_zne_ccf_to_zrt` projects on them. Two consequences decide the signs
of the mixed components:

1. **At A, $\hat R_A$ points towards B; at B, $\hat R_B$ points away from A.**
   On a flat Earth ($\psi = \theta + \pi$) both equal $+\hat x$, where
   $\hat x$ is the unit vector from A to B. So the radial axis at *both*
   stations points **along the propagation of the positive-lag wave**, from
   the virtual source towards the receiver. This is the geometry of Fig. 2 of
   Lin, Moschetti & Ritzwoller (2008).
2. **$\hat T$ is $\hat R$ turned by +90°** (clockwise seen from above) at
   both stations, as in that figure; on a flat Earth $\hat T_A = \hat T_B$.
   Both 2×2 blocks are proper rotations (determinant +1), so (R, T) has the
   same handedness as (N, E).

![Station geometry and the R/T axes](img/rotation_geometry.png)

*The axes in this figure are computed by `rotate_zne_ccf_to_zrt` itself
(script `docs/ant/scripts/rotation_geometry.py`), not drawn by hand.*

### Relation to Lin et al. (2008) and the legacy convention

The coefficients of TT, RR, TR and RT in eq. 1 of Lin et al. (2008) are
exactly those of §2 below (checked on the [executable
page](scripts/rotation_derivation.py)). Those four products contain two
horizontal axes each, so they would be the same if R and T were reversed at
both stations; the direction of the axes is fixed by their Fig. 2, and it
matters only for ZR, RZ, ZT and TZ, which contain one horizontal axis.

SeisForge before this change used the opposite axes: R and T reversed at both
stations (R pointing from the receiver back towards the virtual source). The
option `radial_convention: legacy` (`rotate-ccf` config key under `rotation:`,
CLI `--radial-convention legacy`, argument of `rotate_zne_ccf_to_zrt` and
`rotate_zrt_ccf_to_zne`) restores it:

| component | lin2008 vs legacy |
|---|---|
| ZZ, RR, RT, TR, TT | identical, bit for bit (same floating-point operations) |
| ZR, RZ, ZT, TZ | opposite sign (negation is exact in floating point) |

`rotate-ccf` writes the convention used into `rotate_ccf.log` and into SAC
`kuser0` (`lin2008` or `legacy`) of every output file. AFTAN reads `kuser0` to
choose the ZR/RZ phase constants ([Dispersion](dispersion.md)). Files rotated
by SeisForge before the change carry no such mark and use the legacy axes; for
ZR/RZ from such files set `aftan.radial_convention: legacy` (AFTAN warns when
it has to assume the default for ZR/RZ).

Other codes may use yet other conventions (flipping only $\hat R$ changes the
sign of ZR, RZ, RT and TR). Results from different codes can be compared only
after this sign bookkeeping.

## 2. The forward rotation

A correlation is bilinear in the two recordings. If the motion at A is
projected on a unit vector $\hat u$ and the motion at B on $\hat v$, then

$$
C_{uv}(t) = \int (\hat u\cdot \mathbf{u}^A)(\tau)\,(\hat v\cdot\mathbf{u}^B)(\tau+t)\,d\tau
= \sum_{i,j\in\{Z,N,E\}} u_i\, C_{ij}(t)\, v_j .
$$

Put the new unit vectors of A as rows of a matrix $S$ and those of B as rows
of a matrix $\mathsf R$ (both in Z, N, E order):

$$
S = \begin{pmatrix} 1 & 0 & 0\\ 0 & \cos\theta & \sin\theta\\ 0 & -\sin\theta & \cos\theta \end{pmatrix},
\qquad
\mathsf R = \begin{pmatrix} 1 & 0 & 0\\ 0 & -\cos\psi & -\sin\psi\\ 0 & \sin\psi & -\cos\psi \end{pmatrix}.
$$

(legacy: both multiplied from the left by $D = \mathrm{diag}(1,-1,-1)$, so
$C'_\text{legacy} = D\,C'\,D$.)

Then, with $C$ the tensor whose row index is the source component and column
index the receiver component,

$$
\boxed{\;C' = S\,C\,\mathsf R^{\mathsf T}\;}
\qquad\text{i.e.}\qquad
C'_{ab} = \sum_{ij} S_{ai}\, C_{ij}\, \mathsf R_{bj},
\quad a,b\in\{Z,R,T\}.
$$

The order matters: $S$ (angle $\theta$) acts on the first letter, $\mathsf R$
(angle $\psi$) on the second. Swapping them is the most common way to get a
rotation that looks right for ZZ, RR, TT and wrong for everything else.

Writing out the nine products (the vertical row and column pass through
unchanged):

$$
\begin{aligned}
C_{ZZ}' &= C_{ZZ}\\
C_{ZR}' &= -\cos\psi\, C_{ZN} - \sin\psi\, C_{ZE}\\
C_{ZT}' &= \sin\psi\, C_{ZN} - \cos\psi\, C_{ZE}\\
C_{RZ}' &= \cos\theta\, C_{NZ} + \sin\theta\, C_{EZ}\\
C_{TZ}' &= -\sin\theta\, C_{NZ} + \cos\theta\, C_{EZ}\\
C_{RR}' &= -\cos\theta\cos\psi\, C_{NN} - \cos\theta\sin\psi\, C_{NE} - \sin\theta\cos\psi\, C_{EN} - \sin\theta\sin\psi\, C_{EE}\\
C_{RT}' &= \phantom{-}\cos\theta\sin\psi\, C_{NN} - \cos\theta\cos\psi\, C_{NE} + \sin\theta\sin\psi\, C_{EN} - \sin\theta\cos\psi\, C_{EE}\\
C_{TR}' &= \phantom{-}\sin\theta\cos\psi\, C_{NN} + \sin\theta\sin\psi\, C_{NE} - \cos\theta\cos\psi\, C_{EN} - \cos\theta\sin\psi\, C_{EE}\\
C_{TT}' &= -\sin\theta\sin\psi\, C_{NN} + \sin\theta\cos\psi\, C_{NE} + \cos\theta\sin\psi\, C_{EN} - \cos\theta\cos\psi\, C_{EE}
\end{aligned}
$$

These are, term by term, the expressions in `rotate_zne_ccf_to_zrt`; the
executable derivation evaluates both for 200 random angle pairs and tensors,
in both conventions, and finds them equal to $10^{-15}$. Each output needs only some inputs (ZR
needs ZN and ZE, RR needs the four horizontal pairs, ...); the package uses
this (`required_zne_components`) to read only the files a request needs.

On a flat Earth, $\cos\psi = -\cos\theta$ and $\sin\psi = -\sin\theta$, and
the horizontal block becomes a similarity transform with one angle:

$$
\begin{aligned}
C_{ZR}' &= \cos\theta\, C_{ZN} + \sin\theta\, C_{ZE},\\
C_{RR}' &= \cos^2\theta\, C_{NN} + \sin\theta\cos\theta\,(C_{NE}+C_{EN}) + \sin^2\theta\, C_{EE},\\
C_{TT}' &= \sin^2\theta\, C_{NN} - \sin\theta\cos\theta\,(C_{NE}+C_{EN}) + \cos^2\theta\, C_{EE}.
\end{aligned}
$$

The familiar "RR is the projection on the path" picture holds only in this
limit; on a sphere the two stations see the path under different azimuths.

## 3. The inverse rotation

$S$ and $\mathsf R$ are orthogonal ($S S^{\mathsf T} = \mathsf R \mathsf
R^{\mathsf T} = I$), so

$$
C = S^{\mathsf T}\, C'\, \mathsf R ,
\qquad
C_{ij} = \sum_{a,b} S_{ai}\, C'_{ab}\, \mathsf R_{bj}.
$$

This is `rotate_zrt_ccf_to_zne`. It needs all nine ZRT components and returns
all nine ZNE components; it is used to build synthetic ZNE input with a
known ZRT answer (`examples/01_ccf_rotate`, `tests/test_rotation.py`).

## 4. What the rotation must preserve (checks)

An exact rotation cannot create or destroy correlation; three identities hold
at every lag, for any $\theta$ and $\psi$:

$$
\begin{aligned}
C_{ZR}'^2 + C_{ZT}'^2 &= C_{ZN}^2 + C_{ZE}^2, \qquad
C_{RZ}'^2 + C_{TZ}'^2 = C_{NZ}^2 + C_{EZ}^2,\\
\textstyle\sum_{a,b\in\{R,T\}} C_{ab}'^2 &= \textstyle\sum_{i,j\in\{N,E\}} C_{ij}^2,\\
C_{RR}'C_{TT}' - C_{RT}'C_{TR}' &= C_{NN}C_{EE} - C_{NE}C_{EN}.
\end{aligned}
$$

The last one holds because both rotations have determinant +1. On a flat
Earth two more hold, the trace $C_{RR}'+C_{TT}' = C_{NN}+C_{EE}$ and the
antisymmetric part $C_{RT}'-C_{TR}' = C_{NE}-C_{EN}$; on a sphere they do
not, and using them as a check there would flag correct data.

Checks in the repository:

| check | what it shows |
|---|---|
| [executable derivation](scripts/rotation_derivation.py) | symbolic $C'=SC\mathsf R^{\mathsf T}$ equals the code; inverse; the invariants above |
| `tests/test_rotation.py` | identity angles, required inputs, round trip ZNE → ZRT → ZNE in both conventions, a north-pointing path; R along the propagation at both stations; Lin et al. (2008) eq. 1; legacy equals the old formulas bit for bit and lin2008 differs from them only in the sign of ZR, RZ, ZT, TZ |
| `tests/test_ant_synthetic.py::test_plane_wave_rayleigh_ccfs_match_the_bessel_closed_forms_after_rotation` | a diffuse Rayleigh field, rotated, gives the Bessel-function forms of [Dispersion §1](dispersion.md#1-the-phase-of-a-noise-correlation) (ZR $= +\varepsilon J_1$, RZ $= -\varepsilon J_1$ in lin2008; opposite in legacy) |
| `examples/01_ccf_rotate` | a known ZRT field turned into noisy ZNE and back; residual 3-4 noise standard deviations |

## 5. What to expect after rotation, and what an angle error does

In a laterally homogeneous, isotropic medium a diffuse field gives Rayleigh
energy on ZZ, ZR, RZ, RR, Love energy on TT, and nothing on the four mixed
components ZT, TZ, RT, TR (apart from a near-field Rayleigh term on TT).
Energy on the mixed components therefore measures what is *not* in that
picture: anisotropy, lateral heterogeneity, an uneven noise field, or a wrong
angle.

The last of these can be quantified. Differentiating the rows of $S$ and
$\mathsf R$,

$$
\frac{\partial \hat R_B}{\partial\psi} = \hat T_B,\quad
\frac{\partial \hat T_B}{\partial\psi} = -\hat R_B,\qquad
\frac{\partial \hat R_A}{\partial\theta} = \hat T_A,\quad
\frac{\partial \hat T_A}{\partial\theta} = -\hat R_A .
$$

An error $\delta\psi$ in the receiver angle (a wrong back azimuth, or a
horizontal sensor misoriented by $\delta\psi$) therefore gives, to first
order and with zero true RT,

$$
C_{RT}^{\text{meas}} \approx -\,\delta\psi\; C_{RR},\qquad
C_{ZT}^{\text{meas}} \approx -\,\delta\psi\; C_{ZR},
$$

and an error $\delta\theta$ at the source leaks $-\delta\theta\,C_{RR}$ into
TR and $-\delta\theta\,C_{RZ}$ into TZ. A 5° misorientation leaks about 9 %
of the Rayleigh amplitude into the transverse components; the RR, TT
amplitudes change only at second order ($\delta^2/2 \approx 0.4\,\%$). This
is also why using $\theta + 180°$ instead of the true back azimuth is harmless
on short paths and not on long ones. (The four derivative identities hold in
both conventions and are checked symbolically on the executable page.)

### Reversing source and receiver

The lag definition gives $C_{ab}^{BA}(t) = C_{ba}^{AB}(-t)$ for the
*recorded* components. After rotation an extra sign appears: as the receiver
of pair AB, B's radial axis points away from A (along A → B); as the virtual
source of pair BA it points towards A (along B → A). Hence (in either
convention)

$$
C_{ZR}^{AB}(-t) = -\,C_{RZ}^{BA}(t),\qquad
C_{RR}^{AB}(-t) = C_{RR}^{BA}(t),
$$

and in general each R or T letter that changes role contributes a factor
$-1$. AFTAN therefore takes the phase constant of the *stored* component on the
negative branch too, not that of the reciprocal one; see the
[phase-constant derivation](../aftan_phase_constants.md).

For the same reason the rotation never reads a missing component from the
reciprocal pair BA: even for $ZZ$, $NN$ or $EE$ that file is the
time-reversed correlation, $C^{BA}_{ii}(t) = C^{AB}_{ii}(-t)$, so it would
swap the causal and acausal branches. A missing input file is an error.

## 6. Running it

`seisforge ant rotate-ccf -c config.yml` reads the nine (or fewer) ZNE SAC
files of each pair, resolves the angles as described in §1, and writes ZRT
SAC files plus `rotate_ccf.log`. `rotation.radial_convention` (default
`lin2008`, or `legacy`) selects the axes of §1; the value is in the log and
in SAC `kuser0` of each output. The configuration keys and file layout are
on the [previous page](../ant_workflow.md) until they are moved here.

## References

- Aki, K., & Richards, P. G. (2002). *Quantitative Seismology* (2nd ed.).
  University Science Books. (Surface-wave polarisation, ch. 7.)
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
