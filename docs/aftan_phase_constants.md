# AFTAN Phase Constants per Component (Known Issue)

!!! warning "Status: known issue, not fixed"
    The automatic `pi_over_4` values for **ZR and RZ**, and the negative-branch
    ZR <-> RZ swap in pair mode, do not match this package's CCF rotation.
    Phase velocities of the affected cases below are wrong by 0.25-0.6 km/s at
    150 km. **Group velocities are not affected**, and ZZ, RR and TT are
    correct. No number has been changed yet; affected measurements carry a
    `phase_velocity_warning` in the `.dat` header and a warning in
    `aftan.log`. A follow-up task applies the [recommended fix](#recommended-fix).

## Affected cases

Synthetic CCFs with a known phase velocity (diffuse Rayleigh field, 150 km,
5-14 s, true curve given as `prediction_file` so that the 2 pi cycle choice does
not enter); largest `|c_measured - c_true|` in km/s, current code:

| Stored component | Pair, positive | Pair, negative | Station, positive | Station, negative |
|---|---|---|---|---|
| ZZ | 0.001 | 0.001 | 0.001 | 0.001 |
| RR | 0.010 | 0.010 | 0.010 | 0.010 |
| TT | 0.002 | 0.002 | 0.002 | 0.002 |
| ZR | 0.006 | **0.25** | **0.25** | **0.25** |
| RZ | **0.28** | **0.62** | **0.28** | **0.28** |

"Pair" is a config with `component(s)` (automatic `pi_over_4`); "station" is a
config without them, which uses `pi_over_4 = -1` for every file. TT here is a
Love-like waveform built with the ZZ phase constant (`synthetic_ccf`), so its
row only pins the code path; the Love-wave check is in [RR and TT](#rr-and-tt).
The cases are `tests/aftan_component_cases.py`; the wrong ones are
`xfail(strict=True)` in `tests/test_ant_aftan_components.py`.

## What `pi_over_4` is

`AFTANMeasurement` adds `p * pi / 4` (`p = pi_over_4`) to the FTAN phase picked
at the group arrival `t = r / U` and computes the phase velocity from
`c = omega r / (omega t - phase)` (plus a 2 pi cycle). With the Fourier kernel
`exp(+i omega t)` used throughout, a causal branch

    g(t) = cos(omega (t - r / c) + phi0)

has analytic-signal phase `omega t - omega r / c + phi0` at `t`, so it is
measured at `c` exactly when

    p = -4 phi0 / pi        (mod 8).

The ZZ value `p = -1` (`phi0 = +pi/4`) is the long-standing AFTAN choice and
agrees with the Fortran AFTAN comparison in `examples/benchmarks/aftan_vs_fortran`.

## Conventions the constants depend on

- **CCF**: `C_ab(t) = sum_tau u_a^A(tau) u_b^B(tau + t)` for virtual source A
  (first letter) and receiver B (second letter); positive lag is propagation
  A -> B. This is what the AFTAN positive branch assumes.
- **Rotation** (`seisforge.ant.rotation.rotate_zne_ccf_to_zrt`): with
  azimuth `theta` (A to B) and back azimuth `psi = theta + 180 deg`, the radial
  unit vectors are `R_A = (-cos theta, -sin theta)` and
  `R_B = (cos psi, sin psi)` in (N, E). Both equal `-x`, where `x` is the unit
  vector from A to B: **both radial axes point from the receiver towards the
  source.** The transverse axes are equal too: `T_A = T_B`.
- **Polarisation**: fundamental-mode Rayleigh waves are retrograde at the
  free surface (Aki & Richards 2002, sec. 7.2.2). For a plane wave travelling
  along `d`, `z = cos(omega t - k d.x)` and the horizontal motion along `d` is
  `cos(omega t - k d.x + pi/2)`: radial (along propagation) leads vertical by
  pi/2. In complex form `Z = exp(-i k d.x)`, `H = i epsilon d Z`, `epsilon > 0`.

## Analytic derivation

Average the plane-wave products over directions `d` at angle `alpha` from `x`
(diffuse field; Snieder 2004; Lin et al. 2008; Lin, Tsai & Schmandt 2014),
with `X = k r`:

    <exp(-i X cos a)>          = J0(X)
    <cos a exp(-i X cos a)>    = -i J1(X)
    <cos^2 a exp(-i X cos a)>  = (J0(X) - J2(X)) / 2
    <sin^2 a exp(-i X cos a)>  = (J0(X) + J2(X)) / 2 = J1(X) / X

The spectrum of `C_ab` is `conj(U_a^A) U_b^B`. With `d.R_A = d.R_B = -cos a`:

| Component | Product averaged | Result |
|---|---|---|
| ZZ | `exp(-i X cos a)` | `J0` |
| ZR | `i epsilon (-cos a) exp(...)` | `-epsilon J1` |
| RZ | `conj(i epsilon (-cos a)) exp(...)` = `i epsilon cos a exp(...)` | `+epsilon J1` |
| RR | `epsilon^2 cos^2 a exp(...)` | `epsilon^2 (J0 - J2) / 2` |
| TT, Rayleigh | `epsilon^2 sin^2 a exp(...)` | `epsilon^2 J1 / X` (near field only) |
| TT, Love (motion along `T`) | `cos^2 a exp(...)` | `(J0 - J2) / 2` |

Far from the source `J_n(X) ~ sqrt(2 / (pi X)) cos(X - n pi / 2 - pi / 4)`.
Writing `cos(X - theta) = [exp(i(X - theta)) + exp(-i(X - theta))] / 2`, the
term `exp(-i(k r - theta))` times `exp(i omega t)` is the causal arrival
`cos(omega (t - r / c) + theta)`; the other term is the acausal arrival, which
after time reversal has the same `phi0 = theta`. So **both lag branches carry
the same constant**:

| Component | Far-field form | `phi0` | `p` |
|---|---|---|---|
| ZZ | `+cos(X - pi/4)` | `+pi/4` | **-1** |
| RR | `+cos(X - pi/4)` (since `J2 ~ -J0`) | `+pi/4` | **-1** |
| TT (Love) | `+cos(X - pi/4)` | `+pi/4` | **-1** |
| ZR | `-cos(X - 3 pi/4)` = `cos(X + pi/4)` | `-pi/4` | **+1** |
| RZ | `+cos(X - 3 pi/4)` | `+3 pi/4` | **-3** (= +5) |

The current table has ZR `+1` (right) and RZ `-1` (wrong), and pair mode on the
negative branch uses the other component's value (ZR negative -1, RZ negative
+1: both wrong).

## Is the check independent of the table?

The synthetic CCFs (`seisforge.ant.synthetic.synthetic_rayleigh_ccf_zne`) use
only the three conventions above: the retrograde polarisation (textbook
physics), the CCF lag definition (shared with the AFTAN positive branch, which
is not under test), and the package rotation (the thing the table must match).
They do not read `_RAYLEIGH_PI_OVER_4`, `pi_over_4` or the negative-branch
swap. The direction average is checked against the Bessel closed forms above,
computed with `scipy.special.jv`
(`tests/test_ant_synthetic.py::test_plane_wave_rayleigh_ccfs_match_the_bessel_closed_forms_after_rotation`),
so the numerical table and the analytic derivation are two separate routes to
the same constants. The ZZ result `p = -1` reproduces the established AFTAN
value, which anchors the overall sign.

Running AFTAN on these CCFs with every `p` in {-3, -1, +1, +3, +5} at 200 km
gives the smallest error (0.002-0.006 km/s) at exactly the `p` of the last table
on both branches, and 0.19-0.45 km/s elsewhere. The older `synthetic_ccf` builds
its phase from a given `pi_over_4` and is therefore *not* an independent check
of the table (its docstring says so).

## Where the current values come from

- `_RAYLEIGH_PI_OVER_4` (`ZZ -1, ZR +1, RZ -1, RR -1, TT -1`) and the
  negative-branch swap (`physical_component_for_branch`) were added together in
  commit `7d5c2de` (2026-07-10, message "optimize seisforge inv"), with no
  derivation, reference or test of the ZR/RZ values. Before that, AFTAN used a
  single `pi_over_4: -1.0` for everything.
- The rotation formulas date from commit `8865b26` and have not changed since.

**No sign convention makes the table right.** Each of ZR and RZ is ZZ shifted
by +-pi/2 (they are `J1`-type); reversing one radial axis adds pi. So under any
choice of radial directions, ZR and RZ differ by 0 or pi, never by pi/2 as in
the table (`+1` vs `-1`). Reversing the CCF lag definition only swaps the two
branches, which carry the same constant, so it does not help either. A
prograde polarisation (some higher modes or strong sediments) flips ZR and RZ
together (ZR -3, RZ +1); their difference stays pi.

**The swap needs a sign.** The negative lag of `ZR` for pair A_B is the
physical `RZ` of pair B_A only if station B's radial axis is the same whether
B acts as receiver or as source. In this rotation it is not: as receiver of
A_B, `R_B` points towards A; as source of B_A, it points away from A. Hence
`ZR_AB(-t) = -RZ_BA(t)` (a pi shift), and the swap without that sign gives the
wrong constant. With radial axes pointing *away from the other station* at both
ends, the plain swap would be right, but then ZR = RZ = -3, not the table.

## RR and TT

- **RR**: `-1` is right on both branches: analytic `(J0 - J2) / 2`, and the
  diffuse-field synthetic gives 0.006-0.010 km/s error (table above).
- **TT**: `-1` is right for Love waves: analytic `(J0 - J2) / 2` with
  transverse polarisation; a scratch run of the same plane-wave average with
  Love polarisation gave 0.010 km/s at `p = -1` and >= 0.25 km/s otherwise
  (not in the test suite). A Rayleigh field leaks only the near-field
  `J1 / (k r)` term into TT.
- **ZT, TZ, RT, TR**: no automatic value exists and none was checked.

## Recommended fix

A follow-up task (it changes ZR and RZ results; "Changed results" in
`CHANGELOG.md`):

1. `ant/aftan/phase.py`: look the constant up by the **stored** component on
   both branches, with `ZZ -1, RR -1, TT -1, ZR +1, RZ -3` (these are
   `_CHECKED_PI_OVER_4`; merge the two tables). Keep
   `physical_component_for_branch` for labels and logs, documenting that the
   physical reciprocal component carries a sign: `ZR_AB(-t) = -RZ_BA(t)`.
2. Station mode (`ant/aftan/driver.py`, `config.py`): read the component per
   file with `io._component_from_sac` (SAC `kcmpnm`, else the one ZRT token in
   the file name), default `pi_over_4` to `auto` there too, stop with an
   explanation when auto mode finds no component, and pass the component to the
   existing `_aftan_config_for_branch` so both modes share one path.
3. Tests: drop the `xfail` marks in `tests/test_ant_aftan_components.py`, take
   the ZR/RZ cases out of `UNCHANGED`, and keep ZZ/RR/TT bit for bit against
   `tests/data/aftan_component_baseline.npz`.
4. `phase_constant_warning`: drop the "under revision" note; keep the mismatch
   warning (it also catches manual values).
5. Old ZR/RZ phase velocities were wrong; no option to restore them is
   recommended.

## Not verified

Real data; spherical geometry (flat-earth plane waves only); short distances
where `k r` is small and the far-field constants do not hold (about
`1 / (8 k r)` rad); higher modes or prograde motion; ZT/TZ/RT/TR; the PMF
stage on ZR/RZ (only the basic FTAN stage was checked).

## References

- Aki, K., Richards, P. G. (2002). *Quantitative Seismology*, 2nd ed., sec. 7.2.
- Snieder, R. (2004). Extracting the Green's function from the correlation of
  coda waves: a derivation based on stationary phase. *Phys. Rev. E* 69, 046610.
- Lin, F.-C., Moschetti, M. P., Ritzwoller, M. H. (2008). Surface wave
  tomography of the western United States from ambient seismic noise: Rayleigh
  and Love wave phase velocity maps. *Geophys. J. Int.* 173, 281-298.
- Lin, F.-C., Tsai, V. C., Schmandt, B. (2014). 3-D crustal structure of the
  western United States: application of Rayleigh-wave ellipticity extracted
  from noise cross-correlations. *Geophys. J. Int.* 198, 656-670.
