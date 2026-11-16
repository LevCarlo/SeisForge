# Ambient-Noise Tools: Overview

!!! note "Draft"
    These method pages replace the older workflow pages step by step. The
    older pages (listed under *Previous pages* in the navigation) still hold
    the configuration keys and command details.

`seisforge.ant` is a small set of tools for ambient-noise tomography (ANT),
not a complete pipeline. It has three independent tools. Each can be used on
its own: its input may come from other software, and its output can go to
other software. SeisForge does not compute cross-correlations and does not do
path (ray) tomography.

| tool | page | input | output |
|---|---|---|---|
| CCF rotation (`rotate-ccf`) | [Rotation of multi-component CCFs](rotation.md) | nine ZNE CCFs of a pair, azimuth $\theta$, back azimuth $\psi$ | nine ZRT CCFs |
| FTAN/PMF dispersion (`aftan`) | [Dispersion measurement](dispersion.md) | one CCF (any component), distance $r$ | group and phase velocity vs period, per lag branch |
| Eikonal/Helmholtz imaging (`eikonal`) | [Eikonal and Helmholtz tomography](tomography.md) | path-average phase velocities of many station pairs | local phase-velocity maps $c(x, T)$ with a spread |

## Where they fit in common ANT workflows

Two common routes from pair dispersion to a depth model; neither is the only
or the recommended one:

- **(a)** pair dispersion → path tomography (outside SeisForge) → phase- or
  group-velocity maps → depth inversion at each grid point.
- **(b)** pair dispersion → Eikonal/Helmholtz phase-velocity maps → a
  dispersion curve at a station or grid point → depth inversion
  (`seisforge inv` is one choice).

The FTAN/PMF tool serves both routes; the rotation is needed only for
multi-component work (R and T components). Two optional helper commands link
the tools for route (b): `seisforge ant collect-dispersion` (AFTAN `.dat`
files → Eikonal input) and `seisforge ant extract-station-curve` (Eikonal stack
→ `period velocity sigma` at one station). `examples/full_chain` demonstrates
route (b).

## Conventions to watch when using the tools together

- **Components and branches.** The rotation fixes which way R and T point (by
  default both radial axes point from the virtual source towards the
  receiver, along the propagation, as in Lin et al. 2008, Fig. 2; the
  `legacy` option reverses them). The phase constant AFTAN removes depends on
  the component and, for ZR and RZ, on this convention; `rotate-ccf` records
  it in SAC `kuser0` and AFTAN reads it
  ([derivation](../aftan_phase_constants.md)). CCFs rotated elsewhere need the
  convention checked.
- **Negative lags.** A CCF has a causal branch (A → B) and an acausal one
  (B → A). A negative-branch measurement belongs to the reversed station pair
  when it is passed on to tomography.
- **Periods.** AFTAN reports measurements at instantaneous periods; a
  tomography code usually wants fixed periods, so the curves are resampled
  (the helper above interpolates linearly, without extrapolation).
- **Cycle choices.** AFTAN picks the $2\pi$ branch of each pair against a
  reference curve; the Eikonal tool can additionally correct whole-period
  travel-time errors between neighbouring stations (long periods only).
- **Uncertainty.** AFTAN gives none for phase velocity; the Eikonal stack
  gives a spread of single-source estimates. Whatever is passed to a depth
  inversion as one sigma (standard error, spread, a floor) is a choice to
  state.

## Checks shared by all pages

Every figure on these pages is made by a script in `docs/ant/scripts/` with
the same name as the image, from package functions and synthetic data with a
known answer, and runs in seconds:

```bash
PYTHONPATH=src python docs/ant/scripts/<figure name>.py
```

The SymPy derivation of the rotation
([executable page](scripts/rotation_derivation.py)) is executed on every
documentation build.
