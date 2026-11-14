# SeisForge

Seismic processing and single-station inversion: ambient noise (CCF rotation,
AFTAN, Eikonal/Helmholtz), receiver functions (QC, H-k, synthetic P RF), HVSR
forward modelling, and Bayesian Vs(z) inversion. Full rules:
`docs/development.md`; shared conventions: `docs/conventions.md`.

## Commands

```bash
conda activate seis
python -m pip install -e ".[dev]"   # tests run against the installed package
python -m pytest -q
python -m ruff check src tests
mkdocs build --strict               # needs ".[docs]"
```

## Rules

- Keep layers apart: config/models (validated frozen dataclasses), core (pure
  functions on arrays, no file IO), io/products (formats, provenance attrs),
  driver (orchestration), cli. Formulas never go in drivers, IO, CLI or
  sampler glue.
- Units are part of names (`period_s`, `distance_km`, `velocity_km_s`,
  `slowness_s_per_km`); internal units are km, km/s, g/cm^3, s, Hz, s/km.
  Convert at a solver's own boundary and say so.
- Put physical conventions (signs, time convention, normalization) and
  references in module docstrings. Do not mutate inputs.
- A change that alters existing numbers needs a `CHANGELOG.md` entry under
  "Changed results" and, when old results must stay reproducible, an explicit
  option that restores them. Never change a default silently.
- Every new or changed numerical method ships with an independent check
  (analytic limit, identity, independent implementation, or a regression
  baseline); tolerances carry a reason. Bug fixes come with a test that fails
  without the fix.
- Outputs are self-describing (attrs/headers record the choices) and existing
  outputs are never overwritten unless asked (`--overwrite`, `io.overwrite`).
- Mark assumed values (not taken from a source) where they are defined.
- Never copy GPL code into the repository (SeisForge is MIT); run it locally
  as a reference only.
- The older RF modules (`rf/hk.py`, `rf/hkSeq.py`, `rf/qc.py`, `rf/rf.py`,
  `rf/traces.py`) predate these rules: refactor the part you touch before
  building on it.
- A new public function goes into its package `__all__` and the matching
  `docs/api/*.md` page; a new docs page goes into `nav` in `mkdocs.yml`.
- When designing or refactoring, load the project skill
  `research-code-architecture-trial` (`.claude/skills/`), shared with
  ComplyBay; use it instead of the personal `research-code-architecture`.
- Work on `dev`; `modern` is archived (tag `archive/modern-2026-11`) and
  `main` is the pre-refactor code. Commit or push only when the user asks.
