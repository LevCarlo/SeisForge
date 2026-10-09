# SeisForge

A Python toolkit for seismic processing and analysis, including:
- Ambient Noise
- Receiver Functions
- Seafloor Compliance

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

## Development

```bash
python -m ruff check src tests
python -m pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same two commands on
Python 3.10 and 3.12 for pushes to `main`/`modern` and for pull requests. The
Fortran-extension and HVDFA regression tests are skipped unless the extension
is built or `SEISFORGE_HVDFA_EXECUTABLE` is set.

## Command Line Interface

SeisForge groups commands by scientific workflow:

```bash
seisforge --help
seisforge rf --help
seisforge rf qc -c examples/configs/rf_qc.yml --plot
seisforge rf hkseq -c examples/configs/hkseq.yml --mode both --plot
seisforge ant rotate-ccf -c examples/configs/ccf_rotate.yml
seisforge ant aftan -c examples/configs/aftan_station.yml
seisforge ant eikonal -c examples/configs/eikonal.yml
```

Legacy entry points are still available:

```bash
rfqc -c examples/configs/rf_qc.yml --plot
hkseq -c examples/configs/hkseq.yml --mode both --plot
```

## Examples And Docs

Example YAML configs live in `examples/configs/`.

Workflow notes live in `docs/`.

Single-station inversion:

```bash
cd examples/STA001
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```

See `docs/inv_usage.md` for the inversion configuration guide.

See `docs/eikonal_workflow.md` for the Xarray input contract and surface-wave
Eikonal/Helmholtz tomography workflow.
