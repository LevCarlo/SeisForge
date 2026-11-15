# Dispersion + Rayleigh H/V inversion example

This folder is laid out like a single station working directory:

- `obs.yaml`: observation files, observation metadata, and forward settings.
- `inv.yaml`: model parameterization, hard prior constraints, likelihood weights, and sampler settings.
- `vph_disp.dat`: three-column Rayleigh phase-velocity observations, `period  velocity  sigma`.
- `hv.dat`: three-column Rayleigh H/V observations, `period  hv  sigma`.
- `true_model.dat`: the layered model the data were computed from.

The data are synthetic: `tools/make_sta001_synthetic.py` (repository root)
forwards a stated truth model with the package's own disba wrappers and adds
seeded Gaussian noise; its docstring gives the model, the noise levels and the
seed. Regenerate with
`PYTHONPATH=src python tools/make_sta001_synthetic.py --overwrite`.

The truth is forwarded in uniform 1/32 km layers, much thinner than the
inversion's own layers (`discretization` in `inv.yaml`: 62.5 m at the surface,
growing 10 % per layer to at most 1 km). The log of every run has a line
`discretization layering_error_at_initial` giving the change of the predicted
data against 4 times thinner layers in units of the data sigma; with your own
data keep it well below 1, and make `dz` smaller if it is not.

Run from this directory (about 80 s with four processes; 4 chains in
`sampler.executor: process`):

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
