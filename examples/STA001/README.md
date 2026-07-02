# Dispersion + Rayleigh H/V inversion example

This folder is laid out like a single station working directory:

- `obs.yaml`: observation files, observation metadata, and forward settings.
- `inv.yaml`: model parameterization, hard prior constraints, likelihood weights, and sampler settings.
- `vph_disp.dat`: three-column Rayleigh phase-velocity observations, `period  velocity  sigma`.
- `hv.dat`: three-column Rayleigh H/V observations, `period  hv  sigma`.

Run from this directory:

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```

Prior-only sampling, useful for checking the hard-prior model space before using data:

```bash
seisforge inv disp-hv --inv inv.yaml -o run_prior --prior-only --progress
```

If the package is not installed as a command yet, use:

```bash
python -m seisforge.cli.main inv disp-hv --inv inv.yaml --obs obs.yaml -o run_disp_hv --progress
```
