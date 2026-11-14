---
name: research-code-architecture-trial
description: Trial revision of research-code-architecture, used in place of it in this project. Use when designing, reviewing, or refactoring scientific Python code (data-processing pipelines, forward models, inversions) where numerical methods, conventions, verification, provenance, configuration, IO, drivers, and performance must stay clearly separated and reproducible.
---

# Research Code Architecture

Use this skill for scientific/research codebases where correctness, reproducibility, and inspectable intermediate products matter as much as software polish.

## Core Principle

Keep scientific assumptions visible and checked. Do not hide numerical choices behind clever abstractions, broad config objects, or IO-heavy pipelines, and do not trust a numerical method that nothing independent has checked.

## Apply, Do Not Copy

This skill is a set of principles, not a template.

- Before changing an existing project, read its layout, conventions, docs, and tests. Keep what already works; change only what violates a principle here.
- Decide which sections apply (see the project-type checklists below) and record the decisions in the project's own `CLAUDE.md` or development docs. Later work follows the project's record, not this file.
- Grow structure with the code. Do not create empty layers, packages, or config sections for work that does not exist yet.
- Examples here illustrate an idea; they are not names or stages to reproduce.

## Layers

The usual separation, in whatever form fits the project:

```text
models/config  -> validated parameters and immutable domain objects
core           -> pure scientific/numerical functions
io             -> file formats, path resolution, metadata, provenance
driver         -> workflow orchestration and stage-level error handling
cli            -> argument parsing and command dispatch
tests          -> independent scientific checks plus workflow smoke tests
```

A formula found in a driver, IO module, CLI, or sampler glue belongs in a core module.

## Scientific Functions

Scientific functions should be small, direct, and reviewable.

- Accept arrays, scalars, or typed domain objects; avoid paths, argparse objects, and global state.
- Return arrays, scalars, or simple result objects; avoid writing files.
- Pass scientific parameters explicitly (`decay_cutoff`, `rel_step`) instead of a whole application config.
- Keep the formula in the function body, close to its name. A reader should see the method without following layers of indirection.
- Validate inputs at the top, separately from the computation; do not bury the formula in protection code.
- Do not mutate inputs. Prefer frozen domain objects (read-only arrays) and functions that return modified copies.
- State the range of validity and the reference (paper, equation) in the docstring.
- Make the failure policy explicit: raise, return NaN, or return a sentinel (e.g. `-inf` log-likelihood). Choose one per interface and document it; never let overflow silently become NaN downstream.

## Units and Conventions

Units and conventions cause more silent errors than layering does.

Units: pick one of these per project and record it.

- Suffix style: every quantity carries its unit in its name (`period_s`, `distance_km`, `slowness_s_per_km`). Suits code that passes many loose scalars and arrays.
- Interface-unit style: a fixed unit system at every public interface and inside domain objects (e.g. km, km/s, g/cm^3, s, Hz), stated in docstrings; suffixes only where a value departs from it or is ambiguous (`depth_m`). Suits code built around model objects, where suffixes on every variable make formulas harder to read.

Either way, a function that needs other units converts at its own boundary and says so.

Conventions: write them in the module docstring wherever they apply.

- Coordinate directions (z up or down), sign of each output component.
- Fourier/time convention (`exp(-i omega t)` vs `exp(+i omega t)`) and how it maps to the FFT library used.
- Normalization (by maximum, by zero lag, by incident amplitude) and what amplitudes therefore mean.
- Branch or mode definitions (e.g. which root a "fundamental mode" is).

## Verification

Every numerical method needs at least one check that does not reuse its own derivation:

1. Analytic solution or limit (static half-space, deep/shallow-water limit, known wave speed).
2. Conservation law or identity (energy flux balance, reciprocity, a residual of the defining equation).
3. Independent implementation (simpler formulation, reference code run locally, published values).
4. Regression baseline from a tagged earlier version, for behavior that must not drift.

Test practice:

- Each tolerance has a reason, written next to it when not obvious (compiler flags, solver tolerance, discretization).
- Bug fixes come with a test that fails without the fix.
- Test edge regimes, not only typical inputs: deep models, high frequencies, zero values, empty layers.
- Keep the unit-test suite fast; put long benchmarks in scripts or notebooks.
- Numbers quoted in docs or discussion (error sizes, speeds) come from a computation over the relevant range, not from a few samples.

## Reproducibility and Changed Results

- Keep a regression baseline generated by a script from a tagged version; regenerate it only with that script, never edit it to make a test pass.
- A change that alters numbers existing code or configs produce gets a "Changed results" entry in the changelog, naming the option that restores the old behavior if one exists.
- When old results must stay reproducible, keep the old behavior behind an explicitly named option; never change a default silently.
- Mark parameter values by provenance where they are defined: taken from a source (cite it) or assumed (say so).

## Observables Shaped by Processing

When the processing changes the waveform or value being fitted (filters, whitening, deconvolution, normalization, stacking), the same processing object must be applied to observed data and to predictions.

- Put the shaping steps in one shared, validated object; keep data-only steps (event selection, QC, stacking) outside it.
- Presets for published flows are named after their source and documented as such.
- Test that the data path and the synthetic path agree on the same input.

## Config

Configuration names should expose the scientific meaning.

- Prefer `far_field_buffer_wavelengths` over vague distance multipliers, `bandpass_width_fraction` plus a documented formula over hidden filter constants.
- Validate configs at setup (references resolve, initial state valid) before any expensive work.
- Keep compatibility switches explicit and quarantined under a compatibility section.

## IO and Provenance

Treat IO as infrastructure, not science.

- Put path resolution, file indexing, and format handling (NetCDF/HDF5/Zarr/Parquet/SAC) in IO modules, with an explicit existing-output policy.
- Every generated dataset includes enough attrs/columns to reconstruct key choices (method, parameters, code version or stage).
- Prefer structured outputs over ad hoc text when later stages consume the data; do not make later stages parse logs.

## Project-Type Checklists

Read the ones that apply.

### Data-processing pipelines

- Split work into stages that each produce an inspectable artifact; one stage can rerun without destroying unrelated outputs.
- Add diagnostics for expensive stages: counts, elapsed time, status, failure reason.
- For HPC, prefer scheduler array jobs over in-process multiprocessing unless shared-memory assumptions are explicit.

### Forward models

- One domain model shared by all solvers, validated on construction.
- Each solver documents its conventions, validity range, and how it was verified (keep that list in the docs).
- Guard against numerical breakdown in extreme regimes (overflow, evanescence, wrap-around) with a documented, tested cut-off rather than a silent NaN.

### Inversion / inference

- Keep parameterization (theta -> model), prior, forward, likelihood, and sampler separate; the theta -> model mapping does not depend on the sampler library.
- Physical validity checks belong to the prior support and run before expensive forward calls.
- Support a prior-only run and inspect it before trusting a posterior.
- Make noise models explicit (fixed, hierarchical, correlated) and justify them per data type.

## Performance

Optimize after the scientific contract is pinned by tests.

- First add timing and task-size diagnostics; improve chunking/load balancing before changing algorithms.
- Cache or preload only when metadata records the parameters that make the cache valid.
- Do not trade scientific clarity for speed unless benchmarks justify it, and keep the verification tests passing.

## Packaging and Licensing

- Prefer a `src/` layout and run tests against the installed package, so missing modules or data files fail a test.
- Register compiled extensions and their sources in the build configuration; keep build environments outside the source tree when the build system requires it.
- Check the license of reference code. Code under an incompatible license (e.g. GPL into an MIT package) may be run locally for comparison but not copied or ported.

## Review Checklist

When reviewing or refactoring, ask:

- Can I identify the scientific equations without reading IO or orchestration code?
- Are units and conventions (signs, time convention, normalization) stated where they matter?
- Does each numerical method have an independent check, and do tolerances have reasons?
- Would a change in results show up in a test and in the changelog?
- Are assumed parameter values marked as assumptions?
- Are observed and predicted data processed by the same shaping steps?
- Are intermediate files self-describing enough to debug weeks later?
- Are compatibility choices isolated, and is structure no larger than the code needs?
