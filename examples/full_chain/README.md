# Full chain: from cross-correlations to Vs(z)

One small synthetic data set taken along one possible ambient-noise route
(Eikonal maps, then a depth inversion under one station; path tomography
followed by depth inversion at grid points is another common route, and
SeisForge's tools can serve either):

```text
CCFs -> AFTAN -> [glue.py] -> Eikonal maps -> [glue.py] -> station curve -> inversion
```

`glue.py` is this example's own short script, not a SeisForge command: it
turns the AFTAN `.dat` files into the Eikonal input (one directed path per
lag branch, phase velocity linearly interpolated onto integer periods) and
samples the Eikonal stack under one station into the `period velocity sigma`
file of the inversion. It holds only what this example needs; rewrite it for
your own file names, periods and uncertainty choices.

25 stations on a 5 x 5 grid about 40 km apart. The Earth under them is the
same everywhere: the model of `examples/06_inversion` (a 1 km sediment
gradient from 1.7 to 2.6 km/s over a crust rising from 2.65 to 3.7 km/s at
15 km). So the Vs(z) that should come out under station XX.A22 is that
model, `data/true_model.dat`. Data: `tools/make_example_full_chain.py`.

**H/V is synthetic.** SeisForge does not yet measure H/V (ellipticity) from
data; `data/hv.dat` is computed from the true model with noise. Use H/V from
another program for real data.

Each step has its own example with more explanation: `02_aftan`,
`03_eikonal`, `06_inversion`.

## Input

| file | what it is |
|---|---|
| `data/ccf/<source>_<receiver>_ZZ.SAC` | 300 two-sided ZZ CCFs, one per station pair |
| `data/stations.txt` | `station longitude_deg latitude_deg` |
| `data/reference_phase_velocity.dat` | rough constant phase velocity (2.9 km/s) for AFTAN's 2 pi cycle choice |
| `data/hv.dat` | Rayleigh H/V `period_s hv sigma` (synthetic) |
| `aftan.yml`, `eikonal.yml`, `inv.yaml`, `obs.yaml` | settings of each step |
| `data/true_phase_velocity.dat`, `data/true_model.dat` | the truth (only for checking) |

SAC file names must be `<source>_<receiver>_<component>.SAC` and station
names must not contain `_`. Headers: `b`, `delta`, `dist` (or the
coordinates `evlo/evla` = source, `stlo/stla` = receiver). From
`XX.A00_XX.A10_ZZ.SAC`:

```text
b: -150.0  delta: 0.4  npts: 751  dist: 38.59  evlo: 110.0  evla: 30.0  stlo: 110.4  stla: 30.0
```

`data/stations.txt`:

```text
XX.A00 110.000 30.000
XX.A10 110.400 30.000
```

`data/hv.dat`:

```text
# period_s  rayleigh_hv  sigma
0.80 0.7354 0.020
1.20 0.8245 0.020
```

## Run

```bash
bash run.sh
```

About 90 s (the inversion is most of it; four processes on a 10-core Apple
M4). `run.sh` stops if `output/` exists; delete it to run again.

## Output

| file | what it is |
|---|---|
| `output/aftan/*.dat` | AFTAN group and phase velocity of every CCF and lag branch |
| `output/measurements.nc` | phase velocity of every directed path at 4-8 s |
| `output/eikonal/eikonal_stack.nc` | phase-velocity maps |
| `output/vph_disp.dat` | the phase velocity curve under XX.A22 (`period velocity sigma`) |
| `output/inversion/` | the inversion (see `06_inversion`) |
| `result.png` | station curve and Vs(z), both against the truth |

## How to tell the result is right

`result.png` should look like `expected_result.png`. `run.sh` prints:

| | this example | acceptable |
|---|---|---|
| station curve - truth at 4, 5, 6, 7 s | -0.044, -0.045, -0.033, -0.032 km/s | within the plotted sigma (0.05 km/s) |
| max split R-hat | 1.010 | below 1.05 |
| min effective sample size | 518 | above 400 |
| truth inside the posterior 5-95 % band | 234 of 241 depths | nearly all |
| max \|posterior median - truth\| | 0.137 km/s (at 2.25 km) | about 0.15 km/s |

The truth falls just outside the band in the top 0.25 km and at 2.2-2.5 km,
because the station curve is 1-1.6 % too low (next paragraph). The 8 s map
is empty under XX.A22 (too few distant sources), so the curve has 4-7 s.

Why the curve is low, and why its sigma is set to 0.05 km/s: Eikonal
tomography interpolates each source's travel times between stations. With
stations 40 km apart, the map is 1-2 % too slow near the station rows
(checked against the truth; with 20 km spacing in `03_eikonal` the error is
below 0.7 %), and it is worse at 2-3 s, so those periods are not used. The
stack's own standard error (0.006-0.01 km/s) does not include this error,
so `glue.py` sets a sigma floor of 0.05 km/s (`SIGMA_FLOOR_KM_S`, an
assumed value). The 40 km spacing itself comes from
Eikonal's quality control: cells within about 70 km of a virtual source are
rejected, so with 20 km spacing on a 5 x 5 grid the centre station had no
valid period at all.

Common wrong-looking results:

- **Phase velocity jumps by 0.5-1 km/s on some paths** (AFTAN step):
  wrong 2 pi cycle; give a better `prediction_file` (see `02_aftan`).
- **No period left in `output/vph_disp.dat`** (`glue.py station-curve`): the station has
  too few distant virtual sources; a larger array or lower
  `stacking.min_raw_measurements_per_cell`.
- **Posterior band misses the truth by a lot**: the dispersion sigma is too
  small for the real error of the curve; compare the curve with neighbouring
  stations.

## Using your own data

1. `data/ccf/`: your CCFs, named `<source>_<receiver>_<component>.SAC`.
2. `data/stations.txt`: your stations.
3. `aftan.yml`: `io.input_dir`, `io.sac_pattern`, period band,
   `velocity_window`, `prediction_file` (see `02_aftan`).
4. `glue.py`: `PERIODS_S` (your integer periods; the same list in
   `eikonal.yml` `input.periods_s`), and the file-name parsing if your names
   differ.
5. `eikonal.yml`: `grid` box and `stacking.min_raw_measurements_per_cell`
   (see `03_eikonal`).
6. `run.sh`, `STATION=`: the station to invert; `glue.py`
   `SIGMA_FLOOR_KM_S`: your estimate of the map error.
7. `obs.yaml`: your H/V file; `inv.yaml`: model and priors for your area
   (see `06_inversion`).
