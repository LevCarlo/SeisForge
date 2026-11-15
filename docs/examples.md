# Examples overview

Every folder under `examples/` is a small, self-contained example: data,
config, `run.sh`, `README.md` and the figure you should get. The data are
synthetic with a known answer, so each README can say exactly what the
result should be. To run one, copy the folder anywhere and run

```bash
bash run.sh
```

| folder | what it does | run time | answer it should find |
|---|---|---|---|
| `01_ccf_rotate` | rotates nine ZNE cross-correlations of one station pair to ZRT | < 1 s | known ZRT wavefield, cross terms at the noise level |
| `02_aftan` | group and phase velocity of three CCFs with AFTAN | 1-2 s | known dispersion curve within 0.014 km/s |
| `03_eikonal` | phase-velocity maps from 49 stations | 4 s | the same velocity everywhere, within 0.019 km/s inside the array |
| `04_rf_qc_hk` | receiver-function QC and H-k stacking | 8 s | H = 35 km, Vp/Vs = 1.75; all bad RFs removed |
| `05_hvsr_forward` | diffuse-field H/V of a layered model | 3 s | peak at f0 = Vs / 4H = 1 Hz |
| `06_inversion` | Bayesian Vs(z) from phase velocity and H/V (walk-through: [Getting started](inv_getting_started.md)) | 90 s | the truth model, inside the posterior band |
| `full_chain` | CCFs -> AFTAN -> collect-dispersion -> Eikonal -> station curve -> inversion | 90 s | the truth model under station XX.A22 |

Run times measured on a 10-core Apple M4. `examples/benchmarks/` holds
validation material (SeisForge AFTAN against Fortran AFTAN and PyAFTAN on a
real USArray cross-correlation), not tutorials.

## Every README has the same five sections

1. **Input**: the files needed, their format and required header fields,
   with a real snippet of each.
2. **Run**: one command.
3. **Output**: the files produced and what each is.
4. **How to tell the result is right**: the expected figure, the expected
   numbers with tolerances, and common wrong-looking results with their
   causes.
5. **Using your own data**: the config lines to change, one by one.

## Two levels of config

- The example's own config keeps only the keys that must be set; lines to
  change for your data are marked `# change this`. Everything else takes
  the package default.
- `examples/reference_configs/` has the full, commented version of every
  config with all keys.

Some keys have no default because they are a scientific choice; they stay
in the example configs with a comment: the inversion model and priors, the
AFTAN `branch`, `period_sampling.mode` and `basic.alpha.mode`, and every key
of the receiver-function QC and H-k configs (those older modules have no
defaults).

## Start your own project from an example

```bash
seisforge init aftan my_project/
```

copies the example's config file(s) into `my_project/` (refusing to
overwrite existing files unless `--overwrite`). Workflows: `ccf-rotate`,
`aftan`, `eikonal`, `rf`, `hvsr-forward`, `inversion`, `full-chain`. Then
follow "Using your own data" in the example's README.

## Regenerating the example data

The scripts in `tools/` (`make_example_*.py`, with `tools/example_truth.py`
for the truth model shared by 02, 03, `full_chain` and 06) write the data
with fixed seeds; each docstring states the truth, the noise and every
assumed value. From the repository root:

```bash
PYTHONPATH=src python tools/make_example_02_aftan.py --overwrite
```

The smoke tests `tests/test_examples.py` run every example's `run.sh` in a
temporary folder and check the numbers its README states.
