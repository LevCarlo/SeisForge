# SeisForge

A Python toolkit for seismic processing and analysis, including:

- Ambient noise: ZNE→ZRT CCF rotation, AFTAN group/phase dispersion
  measurement, and Eikonal/Helmholtz surface-wave tomography.
- Receiver functions: QC, sequential H-k stacking, and synthetic P receiver
  functions for layered models.
- HVSR: diffuse-field H/V forward modelling (native Python, optional f2py
  Fortran extension).
- Inversion: single-station Bayesian (MCMC) inversion of dispersion and H/V
  for Vs(z).

Released under the MIT License (see `LICENSE`).

## Installation

For local development:

```bash
conda activate seis
python -m pip install -e ".[dev]"
```

Core dependencies include `disba`, which the inversion (`seisforge.inv`) and
native HVSR forward modelling need. Optional extras:

- `hvsr-fortran`: build tools for the f2py HVSR extension
  (`python tools/build_hvsr_fortran.py`).
- `gmt`: `pygmt`, for the GMT `surface` interpolator in Eikonal tomography.
- `docs`: Zensical (and MkDocs as a fallback) with plugins, to build the documentation site.

## Development

```bash
python -m ruff check src tests
python -m pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same two commands on
Python 3.10 and 3.12 for pushes to `dev` and `main` and for pull requests. The
Fortran-extension and HVDFA regression tests are skipped unless the extension
is built or `SEISFORGE_HVDFA_EXECUTABLE` is set.

## Command Line Interface

SeisForge groups commands by scientific workflow:

```bash
seisforge --help
seisforge rf --help
seisforge rf qc -c examples/reference_configs/rf_qc.yml --plot
seisforge rf hkseq -c examples/reference_configs/rf_hkseq.yml --mode both --plot
seisforge ant rotate-ccf -c examples/reference_configs/ccf_rotate.yml
seisforge ant aftan -c examples/reference_configs/aftan.yml
seisforge ant eikonal -c examples/reference_configs/eikonal.yml
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```

Legacy entry points are still available:

```bash
rfqc -c examples/reference_configs/rf_qc.yml --plot
hkseq -c examples/reference_configs/rf_hkseq.yml --mode both --plot
```

## Examples

Each folder under `examples/` is a self-contained example on synthetic data
with a known answer: data, config, `run.sh`, README and the expected figure.
Copy a folder anywhere and run `bash run.sh`; its README says what the
result should be and which config lines to change for your own data.

| folder | workflow |
|---|---|
| `01_ccf_rotate` | rotate ZNE cross-correlations to ZRT |
| `02_aftan` | group and phase velocity with AFTAN |
| `03_eikonal` | Eikonal phase-velocity maps |
| `04_rf_qc_hk` | receiver-function QC and H-k stacking |
| `05_hvsr_forward` | diffuse-field H/V of a layered model |
| `06_inversion` | Bayesian Vs(z) from phase velocity and H/V |
| `full_chain` | CCFs to Vs(z) along one possible route (Eikonal maps, station inversion) |

`seisforge init <workflow> [directory]` copies an example's config to start
your own project. Full commented configs are in `examples/reference_configs/`;
`examples/benchmarks/` holds validation material. `scripts/personal/` holds the
author's personal processing scripts kept as a backup; they are not examples
and are not tested (see `scripts/personal/README.md`).

## Documentation

Workflow guides live in `docs/` and build into one searchable site with
Zensical (from `mkdocs.yml`):

```bash
python -m pip install -e ".[docs]"
zensical serve   # http://localhost:8000
```

Pushes to `dev` publish the site to GitHub Pages
(`.github/workflows/docs.yml`; enable Pages with **Source: GitHub Actions**).

Single-station inversion:

```bash
cd examples/06_inversion
bash run.sh
```

See `docs/inv_getting_started.md` for a walkthrough of this example,
`docs/model.md` for the model space, and `docs/inv_usage.md` for every
configuration key.

See `docs/ant/tomography.md` for the Xarray input contract and surface-wave
Eikonal/Helmholtz tomography workflow.
