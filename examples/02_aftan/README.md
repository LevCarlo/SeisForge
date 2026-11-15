# 02 AFTAN: group and phase velocity from cross-correlations

Measures Rayleigh-wave group and phase velocity versus period from
cross-correlation functions (CCFs) with AFTAN. The CCFs are synthetic, built
from a known dispersion curve (`tools/make_example_02_aftan.py`), so the
answer is known. Every config key: `examples/reference_configs/aftan.yml`.

## Input

| file | what it is |
|---|---|
| `data/ccf/XX.S00_XX.R0?_ZZ.SAC` | ZZ CCFs, virtual source XX.S00, receivers at 40, 80, 120 km |
| `data/reference_phase_velocity.dat` | rough phase velocity guess (`period_s velocity_km_s`), picks the 2 pi cycle |
| `data/true_dispersion.dat` | the truth (only for checking) |
| `config.yml` | settings |

SAC files: two-sided CCF (negative and positive lags), header `b` (start
time, negative), `delta`, and either `dist` (km) or the coordinates
`evlo/evla` (virtual source) and `stlo/stla` (receiver). From `obspy`:

```text
b: -200.0  delta: 0.2  npts: 2001  dist: 80.07  evlo: 110.0  evla: 30.0  stlo: 110.83  stla: 30.0
```

## Run

```bash
bash run.sh
```

About 1-2 s. Delete `output/` before running again.

## Output

| file | what it is |
|---|---|
| `output/<ccf name>.positive.dat`, `.negative.dat` | one row per period: `target_period_s instant_period_s group_velocity_km_s phase_velocity_km_s amplitude snr`; header lines record the settings |
| `output/aftan.log` | what was measured |
| `result.png` | CCFs, and measured velocities against the truth |

## How to tell the result is right

`result.png` should look like `expected_result.png`. `run.sh` prints, per
file, the largest difference from the truth; on this data:

| | this example | acceptable |
|---|---|---|
| phase velocity | 0.007-0.014 km/s | below 0.03 km/s |
| group velocity, 80 and 120 km | 0.005-0.012 km/s | below 0.03 km/s |
| group velocity, 40 km | 0.03-0.08 km/s at the longest periods | short distance: fewer than 2 wavelengths at 8 s |

Common wrong-looking results:

- **Phase velocity off by a constant-looking step (0.5-1.2 km/s here)** while
  group velocity is right: AFTAN picked the wrong 2 pi cycle. Give a
  `prediction_file` closer to your area (within about 5 % on average).
  Without it, this example is wrong at 80 and 120 km.
- **Group velocity scattered at long periods on short paths**: the path is
  shorter than 2-3 wavelengths; drop those periods.
- **Nothing measured**: `velocity_window` does not contain the arrivals, or
  `sac_pattern` matches no file.

## Using your own data

1. `station`: your virtual-source station name.
2. `io.input_dir`: folder with your CCF SAC files; `io.sac_pattern`: which files.
3. `aftan.branch`: `both` for two-sided CCFs, `stack` to average the two lags.
4. `aftan.min_period`, `aftan.max_period`: your period band.
5. `aftan.period_sampling`: the period step.
6. `aftan.velocity_window`: slowest and fastest expected velocity (km/s).
7. `aftan.basic.alpha`: Gaussian filter width (no default; 20 is common for
   regional distances).
8. `aftan.prediction_file`: a rough phase velocity curve for your area.

The default phase correction is for ZZ CCFs; for other components see
`pi_over_4` in `examples/reference_configs/aftan.yml`.
