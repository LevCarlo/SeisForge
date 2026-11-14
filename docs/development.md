# Development guide

How SeisForge code is organized and what every change has to satisfy. The
short version for coding agents is `CLAUDE.md` in the repository root; this
page is the reference.

## Environment, tests and lint

```bash
conda activate seis
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m pytest -q
```

SeisForge uses the `src/` layout, and pytest imports the **installed** package
(there is no `pythonpath` override). Install once, editable for development,
before running the tests. Tests then exercise what users get from
`pip install`, so a module or data file missing from the package metadata fails
a test instead of being hidden by the source tree. To check the packaging
itself, install non-editable (`pip install ".[dev]"`) into a fresh environment
and run the suite.

GitHub Actions runs both on Python 3.10 and 3.12 for pushes to `dev` and
`main` and for pull requests (`.github/workflows/ci.yml`). Ruff rules are
pinned in `pyproject.toml` so new ruff releases do not change what CI enforces.

Two test groups are skipped unless their dependencies are available:

- Fortran HVSR regressions need the f2py extension:
  `python -m pip install -e ".[hvsr-fortran]"` then
  `python tools/build_hvsr_fortran.py`.
- The HVDFA upstream regression needs `SEISFORGE_HVDFA_EXECUTABLE` pointing to
  the reference executable.

## Code layout

Scientific choices must stay visible: equations live in small functions, not
inside IO or orchestration code. Each workflow package follows the same split:

| Layer | Examples | Contains | Must not |
|---|---|---|---|
| config/models | `ant/aftan/config.py`, `ant/aftan/models.py`, `inv/model.py` | YAML parsing, validation, frozen dataclasses | know about file layouts of other stages |
| core | `ant/aftan/core.py`, `ant/aftan/taper.py`, `rf/hk_core.py`, `rf/synthetic.py`, `inv/layering.py` | pure scientific functions on arrays | read or write files, use global state |
| products/io | `ant/aftan/products.py`, `ant/aftan/io.py`, `inv/io.py` | xarray datasets, file formats, provenance attrs | contain science |
| driver | `ant/aftan/driver.py`, `inv/driver.py`, `rf/hk_workflow.py` | orchestration, overwrite policy, logs | contain any formula |
| cli | `cli/ant.py`, `cli/rf.py`, `cli/inv.py` | argument parsing and dispatch | contain any formula |

For the inversion, the parameter-to-model mapping, prior, forward wrappers,
likelihood, and sampler stay separate; the mapping does not depend on the
sampler.

The older receiver-function modules (`rf/hk.py`, `rf/hkSeq.py`, `rf/qc.py`,
`rf/rf.py`, `rf/traces.py`) predate this layout. Refactor the part you touch
before building new code on it.

## Scientific functions

- Keep the formula in the function body, close to its name; a reader should
  see the method without following several layers of indirection.
- Validate inputs at the top of the function, separately from the computation.
- Pass the scientific parameters explicitly (`gauss_a`, `tail_tolerance`,
  `ramp_s`), not a whole application config.
- State physical conventions in the module docstring: coordinate direction,
  time convention, sign of each output component, normalization. See
  `rf/synthetic.py`.
- Cite the method's source and its range of validity in the docstring.
- Make the failure policy explicit per interface: raise (for example
  `ForwardError`), return NaN, or return a sentinel such as a log likelihood
  of `-inf`. Never let overflow become a silent NaN downstream.
- Do not mutate inputs; return modified copies.
- Optimize only after the scientific result is pinned by a test, and keep the
  test passing.

## Units and naming

- Units are part of names: `period_s`, `distance_km`, `velocity_km_s`,
  `slowness_s_per_km`, `frequency_hz`. Configuration keys follow the same rule
  where the unit is not obvious (`top_km`, `width_km`). The full table is in
  [Conventions](conventions.md).
- Internal units: km, km/s, g/cm^3, s, Hz, s/km. A solver that needs other
  units converts at its own boundary and says so.
- Names carry scientific meaning: `tail_tolerance`, `transition_layers`,
  `legacy_edge_fraction`, not `factor`, `eps`, `n`.

## Numerical changes and reproducibility

- A change that alters numbers produced by existing code or configurations
  gets an entry under **Changed results** in `CHANGELOG.md`, naming the option
  that restores the old behavior if there is one.
- When old results must stay reproducible, keep the old behavior behind an
  explicitly named compatibility option (for example
  `aftan.time_taper.mode: legacy_edge_fraction`); never change a default
  silently.
- Record the new choice in output attrs or headers, so a file shows how it was
  made.
- Mark parameter values by provenance where they are defined: taken from a
  source (cite it) or assumed (say so).

## Tests

- Every numerical method has at least one independent check, not just "it
  runs": an analytic limit, an identity or conservation law, an independent
  implementation, or a regression baseline. The workflow pages list the
  checks in place (for example the validation table of the synthetic
  receiver functions).
- Every tolerance has a reason, written next to it when it is not obvious.
- New behavior comes with tests in the same change. Bug fixes come with a
  test that fails without the fix.
- Test edge regimes, not only typical inputs: thin and thick layers, high
  frequencies, zero widths, empty selections.
- Numbers quoted in docs or commit messages (error sizes, speeds) come from a
  computation over the relevant range, not from a few samples.
- Tests stay fast (the suite runs in seconds); long checks go into `tools/`
  scripts or notebooks. Notebooks under `notebook/` are not tracked; anything
  a test or script depends on belongs in `src/`, `tests/` or `tools/`.

## Style

- Ruff enforces lint rules (`pyproject.toml`). Match the surrounding code's
  idiom and comment density; comments explain why, not what.
- Public functions and classes have type hints and a docstring (numpy section
  style where sections are needed); private helpers start with `_` and are not
  exported from a package `__init__.py`.

## Dependencies and licensing

- SeisForge is MIT licensed. GPL code may be run locally as a reference but
  never copied or ported into the repository.
- Prefer numpy/scipy over a new dependency; a new runtime dependency needs a
  reason in the commit message and an entry with a lower bound in
  `pyproject.toml`.

## Documentation

- Workflow pages describe methods, conventions, options, and how each method
  was validated. Update them with any change to the method.
- `CHANGELOG.md` records user-visible changes, newest first, and is included
  as the Changelog page.
- The pages under `docs/` are built with MkDocs:

  ```bash
  python -m pip install -e ".[docs]"
  mkdocs serve          # http://127.0.0.1:8000, live reload
  mkdocs build --strict
  ```

  `mkdocs.yml` sets `strict: true`: a new page must be added to `nav`, and
  broken internal links fail the build. The theme is MkDocs Material.
- Formulas are LaTeX between `$...$` (inline) or `$$...$$` (display),
  rendered by MathJax; a literal dollar sign in prose needs escaping (`\$`).
- CI builds the site on every push and pull request
  (`.github/workflows/docs.yml`) and publishes pushes to `dev` to
  <https://levcarlo.github.io/SeisForge/>. Publishing needs GitHub Pages
  enabled once: repository Settings -> Pages -> Build and deployment ->
  Source: GitHub Actions.
- The API reference pages (`docs/api/`) are generated from docstrings by
  mkdocstrings, reading `src/` statically (the package does not need to be
  importable). To document a new public function, add it to the `members`
  list of the matching `docs/api/*.md` page.

## Git workflow

- Develop on `dev`. `modern` is archived at tag `archive/modern-2026-11`;
  `main` holds the pre-refactor code.
- Commit messages: a short summary line, then what changed and why.
