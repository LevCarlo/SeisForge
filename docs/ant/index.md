# Ambient-Noise Tools: Overview

`seisforge.ant` is a small set of independent tools for ambient-noise
tomography (ANT), not a single rigid pipeline. Each tool can be used on its
own: its input may come from other software, and its output can go to other
software. SeisForge does not handle raw waveform preparation (resampling or
instrument-response removal) and does not do path (ray) tomography.

| tool | page | input | output |
|---|---|---|---|
| ZZ CCF & stacking (`ccf`, `stack-ccf`, `export-sac`) | [ZZ cross-correlation and stacking](ccf.md) | pre-prepared daily SDS MiniSEED + station table | daily and stacked ZZ NetCDF4/HDF5 (optional SAC export) |
| CCF rotation (`rotate-ccf`) | [Multi-component CCFs: ZNE → ZRT](rotation.md) | nine ZNE CCFs of a pair, azimuth $\theta$, back azimuth $\psi$ | nine ZRT CCFs |
| FTAN/PMF dispersion (`aftan`) | [AFTAN](dispersion.md) | one CCF (any component), distance $r$ | group and phase velocity vs period, per lag branch |
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
multi-component work (R and T components). Linking the tools (file formats,
resampling, choosing an uncertainty) is done by the user's own scripts;
`examples/full_chain` has one example of route (b).

## Conventions to watch when using the tools together

- **Components and branches.** The rotation fixes which way R and T point
  (both radial axes from the virtual source towards the receiver, along the
  propagation, as in Lin et al. 2008, Fig. 2). The phase constant AFTAN
  removes depends on the component and, for ZR and RZ, on this convention
  ([derivation](dispersion.md#6-other-components-open-question)). ZR/RZ rotated by other
  codes need the convention checked (negate if R is reversed).
- **Negative lags.** A CCF has a causal branch (A → B) and an acausal one
  (B → A). A negative-branch measurement belongs to the reversed station pair
  when it is passed on to tomography.
- **Periods.** AFTAN reports measurements at instantaneous periods; a
  tomography code usually wants fixed periods, so the curves are resampled
  (`examples/full_chain/glue.py` interpolates linearly, without
  extrapolation).
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

`docs/ant/scripts/rotation_derivation.py` derives the rotation formulas with
SymPy and writes them to `docs/ant/generated/`, from where the
[rotation page](rotation.md) includes them.
