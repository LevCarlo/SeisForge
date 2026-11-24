# SeisForge

SeisForge is a Python toolkit for seismic processing and analysis:

<div class="grid cards" markdown>

- **Ambient noise**: ZNE→ZRT CCF rotation, AFTAN group/phase dispersion
  measurement with PMF, and Eikonal/Helmholtz surface-wave tomography.
- **Receiver functions**: RF quality control, sequential (sediment, then
  crust) H-k stacking, and synthetic P receiver functions for layered models.
- **HVSR**: diffuse-field H/V forward modelling (native Python, optional f2py
  Fortran extension).
- **Inversion**: single-station Bayesian (MCMC) inversion of dispersion and
  H/V for Vs(z), with segmented models whose interface depths and transition
  widths can be sampled.

</div>

## Installation

```bash
conda activate seis
python -m pip install -e ".[dev]"
```

`disba` is a core dependency. Optional extras: `hvsr-fortran` (build tools for
the f2py HVSR extension), `gmt` (`pygmt` for the GMT surface interpolator), and
`docs` (this site).

## Command line

Commands are grouped by workflow; most take a YAML config:

| Command | Purpose | Guide |
|---|---|---|
| `seisforge ant rotate-ccf -c ccf_rotate.yml` | rotate ZNE CCFs to ZRT | [CCF rotation](ant/rotation.md#3-usage) |
| `seisforge ant aftan -c aftan_station.yml` | AFTAN dispersion picks | [AFTAN](ant/dispersion.md#7-usage) |
| `seisforge ant eikonal -c eikonal.yml` | Eikonal/Helmholtz phase-velocity maps | [Eikonal](ant/tomography.md#7-usage) |
| `seisforge rf qc -c rf_qc.yml --plot` | RF quality control | [RF quality control](rf/qc.md) |
| `seisforge rf hkseq -c hkseq.yml --mode both --plot` | sequential H-k stacking | [H-k stacking](rf/hk.md#6-usage) |
| `seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run` | single-station MCMC inversion | [Inversion](inv/index.md) |

Each folder under `examples/` is a self-contained example with data,
config, `run.sh` and README: see the [examples overview](examples.md).
`seisforge init <workflow>` copies an example's config.

## Where to start

- New to the inversion: [the inversion overview](inv/index.md) runs
  `examples/06_inversion` end to end.
- Units, sign conventions, and output policies shared by all workflows:
  [conventions](conventions.md).
- Python functions and classes: [API reference](api/index.md).
- Changes that alter results of existing configurations:
  [changelog](changelog.md).


