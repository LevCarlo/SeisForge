# Diffuse-field H/V forward modelling

`seisforge.hvsr` provides an array-based forward interface without a CLI or a
configuration-file layer. Models use SeisForge units: thickness in km,
velocities in km/s, and density in g/cm³. The final property entry is the
half-space, so `thickness_km` is one element shorter than the other arrays.

Three backends are available. `NativeDiffuseFieldSolver` is the normal SeisForge
backend. It finds surface-wave roots with the optional `disba` dependency,
evaluates their Green-function pole residues, and integrates P-SV/SH body waves
on the finite complex-wavenumber contour. `FortranDiffuseFieldSolver` implements
the same SeisForge equations in an optional f2py extension. It uses `disba` only
to track phase-velocity seeds; root refinement, pole residues, stable boundary
solves, and body-wave quadrature are evaluated by the Fortran kernel.
`HVDFAReferenceSolver` is a safe
adapter for the independently distributed
[HV-DFA](https://github.com/agarcia-jerez/HV-DFA) executable and is intended for
regression tests. SeisForge does not vendor or compile HV-DFA because its CC
BY-NC licence differs from SeisForge's distribution requirements.

```python
import numpy as np

from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    NativeDiffuseFieldSolver,
    predict_diffuse_field_hvsr,
)

model = ElasticLayerModel(
    thickness_km=[0.5, 1.0, 3.0],
    vp_km_s=[1.8, 2.2, 3.2, 5.5],
    vs_km_s=[0.5, 1.0, 1.8, 3.2],
    density_g_cm3=[1.9, 2.1, 2.4, 2.7],
)
frequencies = np.geomspace(0.1, 10.0, 100)
solver = NativeDiffuseFieldSolver()

prediction = predict_diffuse_field_hvsr(
    model,
    frequencies,
    solver=solver,
    settings=DiffuseFieldSettings(
        max_rayleigh_modes=20,
        max_love_modes=20,
        body_wave_wavenumbers=100,
    ),
)
```

## Optional Fortran backend

The Fortran source is an independent implementation of the published DFA
relations and the documented SeisForge numerical contract. It does not copy or
vendor the differently licensed HV-DFA source code.

Build the extension in a development checkout with a Fortran compiler plus the
Meson f2py build dependencies:

```bash
pip install -e '.[hvsr-fortran]'
python tools/build_hvsr_fortran.py
```

Then select it explicitly; the API and settings are shared with the native
solver:

```python
from seisforge.hvsr import FortranDiffuseFieldSolver

prediction = predict_diffuse_field_hvsr(
    model,
    frequencies,
    solver=FortranDiffuseFieldSolver(),
    settings=DiffuseFieldSettings(
        max_rayleigh_modes=5,
        max_love_modes=5,
        body_wave_wavenumbers=100,
    ),
)
```

The extension is optional: importing `seisforge.hvsr` still works when it has
not been built, while calling `FortranDiffuseFieldSolver` raises a focused
installation error. For MCMC, parallelize independent models/walkers and keep
each forward call single-threaded to avoid nested parallelism.

The reference adapter converts units internally, sorts frequencies for HV-DFA and then
restores the caller's order. Every call uses a private temporary directory, so
parallel evaluations cannot collide through the original program's fixed
output filenames. Output is validated even when HV-DFA reports an error with a
zero process exit status.

`horizontal_definition="sum"` follows Sánchez-Sesma et al. (2011) and HV-DFA:

\[
H/V = \sqrt{\frac{2(G^R_{11}+G^{PSV}_{11}+G^L_{11}+G^{SH}_{11})}
                         {G^R_{33}+G^{PSV}_{33}}}.
\]

Use `horizontal_definition="mean"` only when observations are defined using
the mean of the two horizontal powers; its amplitude is smaller by √2.

For reproducible inference, record the executable version or source commit,
compiler, `DiffuseFieldSettings`, and the horizontal convention beside each
forward dataset. The solver intentionally defaults to one OpenMP thread:
parallelism should normally be applied over models by the inversion driver,
not nested inside each small forward call.

The native solver returns all six component arrays in
`prediction.contributions`, making Rayleigh, Love, P-SV, and SH terms directly
inspectable. Install the optional dependency with `pip install -e '.[hvsr]'`.
Its body-wave quadrature is split at every layer's P/S branch point; the
`body_wave_wavenumbers` setting is retained as an effort hint rather than a
literal number of equally spaced samples. `NativeDiffuseFieldSolver` exposes
advanced quadrature, contour, dispersion, and residue controls for convergence
studies, but their defaults should normally be retained.

The implementation has been regression-tested against HV-DFA 1.0 component
outputs. For the documented four-layer example, complete H/V differs by less
than 1% at 0.2 and 1 Hz. A 40-frequency 0.2--5 Hz calculation takes about four
seconds on the development machine after dependency import/JIT warm-up. This is
already safe for forward studies, while large MCMC runs should add model-level
parallelism and caching only after profiling their actual parameterization.

On the development Apple Silicon machine, the six-layer paper example with 40
frequencies, five Rayleigh modes, five Love modes, and the default 48-point
per-segment quadrature takes about 0.12 s with `FortranDiffuseFieldSolver`,
compared with about 3.25 s for `NativeDiffuseFieldSolver`. The two curves and
their peak values agree to numerical precision for that benchmark.
