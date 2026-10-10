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

Commands are grouped by workflow; every command takes a YAML config:

| Command | Purpose | Guide |
|---|---|---|
| `seisforge ant rotate-ccf -c ccf_rotate.yml` | rotate ZNE CCFs to ZRT | [CCF rotation](ant_workflow.md) |
| `seisforge ant aftan -c aftan_station.yml` | AFTAN dispersion picks | [AFTAN](aftan_workflow.md) |
| `seisforge ant aftan-cycle-set PAIR.vph.nc --shift N` | inspect/override a phase-cycle choice | [AFTAN](aftan_workflow.md#phase-cycle-selection) |
| `seisforge ant eikonal -c eikonal.yml` | Eikonal/Helmholtz phase-velocity maps | [Eikonal](eikonal_workflow.md) |
| `seisforge rf qc -c rf_qc.yml --plot` | RF quality control | [Receiver functions](rf_workflow.md) |
| `seisforge rf hkseq -c hkseq.yml --mode both --plot` | sequential H-k stacking | [Receiver functions](rf_workflow.md#sequential-h-k-config) |
| `seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run` | single-station MCMC inversion | [Inversion](inv_usage.md) |

Example configs live in `examples/configs/`; `examples/AFTAN/` and
`examples/STA001/` are small end-to-end examples.

## Design conventions

- Scientific formulas live in small pure functions (`core`, `hk_core`,
  `taper`, `snr`, ...); drivers only orchestrate IO, logging, and stage
  policies.
- Units are part of names: `period_s`, `distance_km`, `velocity_km_s`,
  `slowness_s_per_km`.
- Outputs are self-describing: NetCDF attrs and `.dat` headers record the
  choices needed to reproduce them (branch, alpha, taper window, SNR
  definition, ...).
- Existing outputs are never overwritten or deleted unless explicitly
  requested (`io.overwrite`, `--overwrite`).
- Behavior kept only to reproduce old results is isolated and labeled as
  compatibility (for example `aftan.time_taper.mode: legacy_edge_fraction`).
