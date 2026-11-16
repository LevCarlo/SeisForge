# AFTAN Phase Constants: Derivation and History

!!! success "Status: fixed"
    Until the lin2008 change (CHANGELOG, "Changed results") the automatic
    `pi_over_4` values for **ZR and RZ**, the negative-branch ZR <-> RZ swap,
    and station mode's fixed `-1` did not match the package rotation; phase
    velocities of the affected cases were wrong by 0.25-0.6 km/s at 150 km.
    This page derives the constants now in `seisforge.ant.aftan.phase`, shows
    how they were checked, and records what was wrong before.

## What `pi_over_4` is

`AFTANMeasurement` adds `p * pi / 4` (`p = pi_over_4`) to the FTAN phase picked
at the group arrival `t = r / U` and computes the phase velocity from
`c = omega r / (omega t - phase)` (plus a 2 pi cycle). With the Fourier kernel
`exp(+i omega t)` used throughout, a causal branch

    g(t) = cos(omega (t - r / c) + phi0)

has analytic-signal phase `omega t - omega r / c + phi0` at `t`, so it is
measured at `c` exactly when

    p = -4 phi0 / pi        (mod 8).

## Correlation, not Green's function

Lin, Moschetti & Ritzwoller (2008, sec. 2) note that the empirical Green's
function is the negative time derivative of the correlation,
`G_AB(t) = -dC_AB(t)/dt`, a phase difference of pi/2, and that Bensen et al.
(2007) treated the two as the same. SeisForge's AFTAN runs the Gaussian
filter bank directly on the input trace (`AFTANMeasurement._ftan_complex`)
and adds `p * pi / 4` to the phase of its analytic signal (`_interp`); it
never differentiates. So the constants are for the **correlation**, and the
ZZ value `p = -1` (`phi0 = +pi/4`, from `J0`) already contains the pi/2: for
`-dC/dt` the ZZ constant would be `+1` (`phi0 = pi/4 + pi/2 - pi = -pi/4`).

Checked numerically (ZZ diffuse-field CCF, 150 km, positive lag, largest
phase-velocity error in km/s):

| input | p = -3 | p = -1 | p = +1 | p = +3 |
|---|---|---|---|---|
| `C(t)` | 0.246 | **0.001** | 0.286 | 0.626 |
| `-dC/dt` | 0.605 | 0.240 | **0.001** | 0.277 |

(`tests/test_ant_aftan_components.py::test_empirical_greens_function_needs_the_pi_over_2_shifted_constant`.
Only the positive lag: `-dC/dt` of a symmetric `C` is antisymmetric, so its
time-reversed negative lag has the opposite shift.)

## Conventions the constants depend on

- **CCF**: `C_ab(t) = sum_tau u_a^A(tau) u_b^B(tau + t)` for virtual source A
  (first letter) and receiver B (second letter); positive lag is propagation
  A -> B.
- **Rotation** (`seisforge.ant.rotation`): with azimuth `theta` (A to B) and
  back azimuth `psi = theta + 180 deg` (flat Earth), the default `lin2008`
  axes are `R_A = (cos theta, sin theta)` and `R_B = (-cos psi, -sin psi)` in
  (N, E): both equal `+x`, the unit vector from A to B, **along the
  propagation** (Lin et al. 2008, Fig. 2). The `legacy` axes (SeisForge
  before the change) are both `-x`. The transverse axes are equal at the two
  stations in either convention.
- **Polarisation**: fundamental-mode Rayleigh waves are retrograde at the
  free surface (Aki & Richards 2002, sec. 7.2.2). For a plane wave travelling
  along `d`, `z = cos(omega t - k d.x)` and the horizontal motion along `d` is
  `cos(omega t - k d.x + pi/2)`. In complex form `Z = exp(-i k d.x)`,
  `H = i epsilon d Z`, `epsilon > 0`.

## Analytic derivation

Average the plane-wave products over directions `d` at angle `a` from `x`
(diffuse field; Snieder 2004; Lin et al. 2008; Lin, Tsai & Schmandt 2014),
with `X = k r`:

    <exp(-i X cos a)>          = J0(X)
    <cos a exp(-i X cos a)>    = -i J1(X)
    <cos^2 a exp(-i X cos a)>  = (J0(X) - J2(X)) / 2
    <sin^2 a exp(-i X cos a)>  = (J0(X) + J2(X)) / 2 = J1(X) / X

The spectrum of `C_ab` is `conj(U_a^A) U_b^B`. With `s = +1` for lin2008 and
`s = -1` for legacy, `d.R_A = d.R_B = s cos a`:

| Component | Product averaged | Result |
|---|---|---|
| ZZ | `exp(-i X cos a)` | `J0` |
| ZR | `i epsilon s cos a exp(...)` | `+s epsilon J1` |
| RZ | `conj(i epsilon s cos a) exp(...)` = `-i epsilon s cos a exp(...)` | `-s epsilon J1` |
| RR | `epsilon^2 cos^2 a exp(...)` | `epsilon^2 (J0 - J2) / 2` |
| TT, Rayleigh | `epsilon^2 sin^2 a exp(...)` | `epsilon^2 J1 / X` (near field only) |
| TT, Love (motion along `T`) | `cos^2 a exp(...)` | `(J0 - J2) / 2` |

Far from the source `J_n(X) ~ sqrt(2 / (pi X)) cos(X - n pi / 2 - pi / 4)`.
Writing `cos(X - theta) = [exp(i(X - theta)) + exp(-i(X - theta))] / 2`, the
term `exp(-i(k r - theta))` times `exp(i omega t)` is the causal arrival
`cos(omega (t - r / c) + theta)`; the other term is the acausal arrival, which
after time reversal has the same `phi0 = theta`. So **both lag branches carry
the same constant**, that of the stored component:

| Component | Far-field form (lin2008) | `phi0` | `p`, lin2008 | `p`, legacy |
|---|---|---|---|---|
| ZZ | `+cos(X - pi/4)` | `+pi/4` | **-1** | **-1** |
| RR | `+cos(X - pi/4)` (since `J2 ~ -J0`) | `+pi/4` | **-1** | **-1** |
| TT (Love) | `+cos(X - pi/4)` | `+pi/4` | **-1** | **-1** |
| ZR | `+cos(X - 3 pi/4)` | `+3 pi/4` | **-3** | **+1** |
| RZ | `-cos(X - 3 pi/4)` = `cos(X + pi/4)` | `-pi/4` | **+1** | **-3** |

These are `AUTOMATIC_PI_OVER_4` in `seisforge/ant/aftan/phase.py`.

## Numerical check, independent of the table

The synthetic CCFs (`seisforge.ant.synthetic.synthetic_rayleigh_ccf_zne`) use
only the three conventions above: the retrograde polarisation (textbook
physics), the CCF lag definition, and the package rotation (the thing the
table must match). They never read `AUTOMATIC_PI_OVER_4` or `pi_over_4`. The
direction average is checked against the Bessel closed forms with
`scipy.special.jv` in both rotation conventions
(`tests/test_ant_synthetic.py::test_plane_wave_rayleigh_ccfs_match_the_bessel_closed_forms_after_rotation`),
so the analytic table and the numerical one are separate routes to the same
constants; the ZZ value `p = -1` is the established AFTAN value and agrees
with the Fortran AFTAN comparison in `examples/benchmarks/aftan_vs_fortran`,
which anchors the overall sign.

Running AFTAN (station mode, manual `pi_over_4`) on these CCFs with every `p`
in {-3, -1, +1, +3, +5} picks, at 80, 150 and 300 km and on both branches,
exactly the `p` of the last table, in both conventions. At 150 km the error is
0.001-0.010 km/s at the right `p` and 0.25-0.63 km/s at every other value.

With the automatic constants (`tests/test_ant_aftan_components.py`), largest
`|c_measured - c_true|` in km/s at 150 km, 5-14 s, true curve given as
`prediction_file` so that the 2 pi cycle choice does not enter; "before" is
the code before the fix on CCFs rotated with the then-current (legacy)
axes, "after" the fixed code with the default lin2008 rotation:

| Stored component | Pair, + | Pair, - | Station, + | Station, - |
|---|---|---|---|---|
| ZZ | 0.001 -> 0.001 | 0.001 -> 0.001 | 0.001 -> 0.001 | 0.001 -> 0.001 |
| RR | 0.010 -> 0.010 | 0.010 -> 0.010 | 0.010 -> 0.010 | 0.010 -> 0.010 |
| TT | 0.002 -> 0.002 | 0.002 -> 0.002 | 0.002 -> 0.002 | 0.002 -> 0.002 |
| ZR | 0.006 -> 0.006 | **0.250** -> 0.006 | **0.250** -> 0.006 | **0.250** -> 0.006 |
| RZ | **0.280** -> 0.006 | **0.620** -> 0.006 | **0.280** -> 0.006 | **0.280** -> 0.006 |

TT here is a Love-like waveform built with the ZZ phase constant
(`synthetic_ccf`), so its row only pins the code path; the Love-wave check is
in [RR and TT](#rr-and-tt). The tests also run every component, both modes
and both branches on legacy-rotated CCFs with `kuser0: legacy` (all within
0.03 km/s), check that pair and station mode give bit-identical results on
the same file, and that ZZ, RR and TT are bit for bit as before
(`tests/data/aftan_component_baseline.npz`). The test tolerance, 0.03 km/s,
separates the right constant (<= 0.010 km/s) from a pi/2 error (about
0.25 km/s) with margin on both sides.

## Negative branch and reciprocity

The time-reversed negative lag of `ZR` for pair A_B is the physical `RZ` of
pair B_A only up to a sign: as receiver of A_B, `R_B` points away from A; as
source of B_A, it points towards A (in either convention). Hence
`ZR_AB(-t) = -RZ_BA(t)`, a pi shift, which turns RZ's constant into ZR's
(`+1 + 4 = 5 = -3 mod 8` in lin2008). AFTAN therefore takes the constant of
the **stored** component on the negative branch; `physical_component_for_branch`
still labels the branch as the reciprocal component of the reversed pair
(in `aftan.log`), but that label never chooses `pi_over_4`.

## Which convention a file has

ZZ, RR and TT do not depend on the R/T convention; ZR and RZ do (their
constants differ by pi between the two). `seisforge ant rotate-ccf` writes
the convention into SAC `kuser0`. AFTAN (`aftan.radial_convention`):

- `auto` (default): `kuser0` if it holds `lin2008` or `legacy`, else
  `lin2008` with a `phase_velocity_warning` on ZR/RZ results;
- `lin2008` or `legacy`: used when the file has no `kuser0`; a file whose
  `kuser0` says otherwise stops the run.

**Files rotated by SeisForge before the change** have no `kuser0` and legacy
axes: measure their ZR/RZ with `aftan.radial_convention: legacy` (then ZR +1,
RZ -3, both checked above). Without it, auto mode assumes lin2008, warns, and
the phase velocity is pi off (about 0.6 km/s at 150 km;
`tests/test_ant_aftan_components.py::test_unrecorded_legacy_files_need_the_config_option`).
Files from other codes need the same care; see
[Rotation](ant/rotation.md#relation-to-lin-et-al-2008-and-the-legacy-convention).

## RR and TT

- **RR**: `-1` is right on both branches: analytic `(J0 - J2) / 2`, and the
  diffuse-field synthetic gives 0.006-0.010 km/s error (table above).
- **TT**: `-1` is right for Love waves: analytic `(J0 - J2) / 2` with
  transverse polarisation; a scratch run of the same plane-wave average with
  Love polarisation gave 0.010 km/s at `p = -1` and >= 0.25 km/s otherwise
  (not in the test suite). A Rayleigh field leaks only the near-field
  `J1 / (k r)` term into TT.
- **ZT, TZ, RT, TR**: no automatic value exists and none was checked; a
  manual value gets a "has not been checked" warning.

## History: what was wrong

- `_RAYLEIGH_PI_OVER_4` (`ZZ -1, ZR +1, RZ -1, RR -1, TT -1`) and the
  negative-branch swap (`physical_component_for_branch` used to choose the
  constant) were added together in commit `7d5c2de` (2026-07-10, message
  "optimize seisforge inv"), with no derivation, reference or test of the
  ZR/RZ values. Before that, AFTAN used a single `pi_over_4: -1.0`.
- With the rotation of that time (legacy axes) the right values were ZR +1,
  RZ -3. The table had RZ -1, a pi/2 error. **No R/T convention made it
  right**: ZR and RZ are `J1`-type and differ by 0 or pi under any choice of
  radial directions, never by pi/2 as in the table (`+1` vs `-1`); reversing
  the lag definition only swaps the branches, which carry the same constant.
- The negative-branch swap used RZ's constant for the negative lag of ZR (and
  ZR's for RZ) without the sign of `ZR_AB(-t) = -RZ_BA(t)`; with the table's
  RZ value this gave ZR negative -1 and RZ negative +1, both wrong.
- Station mode used `-1` for every file, right for ZZ/RR/TT only.
- Task T19 added the warning, the baseline and `xfail` tests without changing
  numbers; the lin2008 change (CHANGELOG, "Changed results") switched the
  rotation to Lin et al. (2008) Fig. 2, replaced the table by
  `AUTOMATIC_PI_OVER_4`, dropped the swap from the constant lookup and made
  station mode look the component up per file. The old ZR/RZ constants are not
  available behind an option: they were wrong for every rotation convention.

## Not verified

Real data; spherical geometry (flat-earth plane waves only); short distances
where `k r` is small and the far-field constants do not hold (about
`1 / (8 k r)` rad); higher modes or prograde motion (which flip ZR and RZ
together); ZT/TZ/RT/TR; the PMF stage on ZR/RZ (only the basic FTAN stage was
checked); Lin et al. (2008) sec. 2 and Fig. 2 were not re-read for this page
(the convention and the pi/2 statement follow the check made when the change
was planned; the DOI was checked against Crossref).

## References

- Aki, K., Richards, P. G. (2002). *Quantitative Seismology*, 2nd ed., sec. 7.2.
- Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
  Moschetti, M. P., Shapiro, N. M., Yang, Y. (2007). Processing seismic
  ambient noise data to obtain reliable broad-band surface wave dispersion
  measurements. *Geophys. J. Int.* 169, 1239-1260,
  doi:10.1111/j.1365-246X.2007.03374.x.
- Snieder, R. (2004). Extracting the Green's function from the correlation of
  coda waves: a derivation based on stationary phase. *Phys. Rev. E* 69,
  046610, doi:10.1103/PhysRevE.69.046610.
- Lin, F.-C., Moschetti, M. P., Ritzwoller, M. H. (2008). Surface wave
  tomography of the western United States from ambient seismic noise: Rayleigh
  and Love wave phase velocity maps. *Geophys. J. Int.* 173, 281-298,
  doi:10.1111/j.1365-246X.2008.03720.x.
- Lin, F.-C., Tsai, V. C., Schmandt, B. (2014). 3-D crustal structure of the
  western United States: application of Rayleigh-wave ellipticity extracted
  from noise cross-correlations. *Geophys. J. Int.* 198, 656-670,
  doi:10.1093/gji/ggu160.
