# SeisForge

SeisForge is a Python toolkit for seismic processing and analysis:

- **Ambient noise**: ZNE→ZRT CCF rotation, AFTAN group/phase dispersion
  measurement with PMF, and Eikonal/Helmholtz surface-wave tomography.
- **Receiver functions**: RF quality control, sequential (sediment, then
  crust) H-k stacking, and synthetic P receiver functions for layered models.
- **HVSR**: diffuse-field H/V forward modelling (native Python, optional f2py
  Fortran extension).
- **Inversion**: single-station Bayesian (MCMC) inversion of dispersion and
  H/V for Vs(z), with segmented models whose interface depths and transition
  widths can be sampled.

## Installation

```bash
conda activate seis
python -m pip install -e ".[dev]"
```

`disba` is a core dependency. Optional extras: `hvsr-fortran` (build tools for
the f2py HVSR extension), `gmt` (`pygmt` for the GMT surface interpolator), and
`docs` (this site).

## Command line

Commands are grouped by workflow; most take a YAML config, the two glue
commands take their few choices as options:

| Command | Purpose | Guide |
|---|---|---|
| `seisforge ant rotate-ccf -c ccf_rotate.yml` | rotate ZNE CCFs to ZRT | [CCF rotation](ant_workflow.md) |
| `seisforge ant aftan -c aftan_station.yml` | AFTAN dispersion picks | [AFTAN](aftan_workflow.md) |
| `seisforge ant aftan-cycle-set PAIR.vph.nc --shift N` | inspect/override a phase-cycle choice | [AFTAN](aftan_workflow.md#phase-cycle-selection) |
| `seisforge ant collect-dispersion 'FTAN/**/*.dat' --periods 8:40:2 --stations stations.txt -o measurements.nc` | gather AFTAN picks into the Eikonal input | [Eikonal](eikonal_workflow.md#collect-aftan-measurements) |
| `seisforge ant eikonal -c eikonal.yml` | Eikonal/Helmholtz phase-velocity maps | [Eikonal](eikonal_workflow.md) |
| `seisforge ant extract-station-curve stack.nc --station STA --stations stations.txt -o vph_disp.dat` | one station's curve for the inversion | [Eikonal](eikonal_workflow.md#extract-a-station-curve-for-the-inversion) |
| `seisforge rf qc -c rf_qc.yml --plot` | RF quality control | [Receiver functions](rf_workflow.md) |
| `seisforge rf hkseq -c hkseq.yml --mode both --plot` | sequential H-k stacking | [Receiver functions](rf_workflow.md#sequential-h-k-config) |
| `seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run` | single-station MCMC inversion | [Inversion](inv_getting_started.md) |

Each folder under `examples/` is a self-contained example with data,
config, `run.sh` and README; full commented configs are in
`examples/reference_configs/`.

## Where to start

- New to the inversion: [getting started](inv_getting_started.md) runs
  `examples/06_inversion` end to end.
- Units, sign conventions, and output policies shared by all workflows:
  [conventions](conventions.md).
- Python functions and classes: [API reference](api/index.md).
- Changes that alter results of existing configurations:
  [changelog](changelog.md).
