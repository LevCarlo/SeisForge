# Development

## Tests and lint

```bash
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

GitHub Actions runs both on Python 3.10 and 3.12 for pushes to `main` and
`modern` and for pull requests (`.github/workflows/ci.yml`). Ruff rules are
pinned in `pyproject.toml` so new ruff releases do not change what CI enforces.

Two test groups are skipped unless their dependencies are available:

- Fortran HVSR regressions need the f2py extension:
  `python -m pip install -e ".[hvsr-fortran]"` then
  `python tools/build_hvsr_fortran.py`.
- The HVDFA upstream regression needs `SEISFORGE_HVDFA_EXECUTABLE` pointing to
  the reference executable.

## Documentation site

The pages under `docs/` are built with MkDocs:

```bash
python -m pip install -e ".[docs]"
mkdocs serve          # http://127.0.0.1:8000, live reload
mkdocs build --strict
```

`mkdocs.yml` sets `strict: true`: a new page must be added to `nav`, and broken
internal links fail the build. CI builds the site on every push and pull
request (`.github/workflows/docs.yml`). Pushes to `main` also publish it to
GitHub Pages once Pages is enabled for the repository with **Source: GitHub
Actions**.

The API reference pages (`docs/api/`) are generated from docstrings by
mkdocstrings, reading `src/` statically (the package does not need to be
importable). Docstrings use the numpy section style where they have sections.
To document a new public function, add it to the `members` list of the
matching `docs/api/*.md` page. `CHANGELOG.md` is included as the Changelog
page.

## Code layout

Each workflow package follows the same split:

| Layer | Examples | Contains |
|---|---|---|
| config/models | `ant/aftan/config.py`, `ant/aftan/models.py` | YAML parsing, validation, frozen dataclasses |
| core | `ant/aftan/core.py`, `ant/aftan/taper.py`, `rf/hk_core.py`, `ant/eikonal/core.py` | pure scientific functions on arrays |
| products/io | `ant/aftan/products.py`, `ant/aftan/io.py`, `inv/io.py` | xarray datasets, file formats, provenance attrs |
| driver | `ant/aftan/driver.py`, `inv/driver.py`, `rf/hk_workflow.hkSeq` | orchestration, overwrite policy, logs |
| cli | `cli/ant.py`, `cli/rf.py`, `cli/inv.py` | argument parsing and dispatch |

When a change alters numerical results, document it in the workflow page,
record the new choice in output attrs, and keep the old behavior reachable
only through an explicitly labeled compatibility option if old results must
stay reproducible. Add an entry under **Changed results** in `CHANGELOG.md`,
naming the option that restores the old behavior.
