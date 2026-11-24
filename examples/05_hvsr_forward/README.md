# 05 H/V forward: the H/V curve of a layered model

Computes the horizontal-to-vertical spectral ratio (H/V) that a diffuse
wavefield would produce on a given layered model (diffuse-field theory,
`seisforge.hvsr`, native solver). Use it to see which H/V peak a model
predicts, for example before an inversion. The example model is one soft
layer (50 m, Vs 0.2 km/s) over rock (Vs 1.0 km/s), whose resonance frequency
is known: f0 = Vs / (4 H) = 1 Hz. Method and solvers: `docs/hvsr/index.md`.

SeisForge has no command for this yet, so `run.sh` calls the short script
`hvsr_forward.py`, which reads `config.yml` and calls the Python API.

## Input

| file | what it is |
|---|---|
| `data/model.txt` | the layered model |
| `config.yml` | which model, which frequencies, where to write |

`model.txt`: one row per layer from the surface down, columns
`thickness_km vp_km_s vs_km_s density_g_cm3`; the last row is the half-space
(thickness 0, ignored). `#` lines are comments:

```text
# thickness_km  vp_km_s  vs_km_s  density_g_cm3
0.05  0.6  0.2  1.8
0.00  2.0  1.0  2.2
```

## Run

```bash
bash run.sh
```

About 3 s.

## Output

| file | what it is |
|---|---|
| `output/hvsr.dat` | `frequency_hz hvsr`, header with model file, settings and software versions |
| `result.png` | the model and the H/V curve |

## How to tell the result is right

`result.png` should look like `expected_result.png`. `run.sh` prints the
peak; on this model it is at 1.011 Hz (acceptable 0.95-1.05 Hz, the
analytic f0 = 1 Hz), with H/V about 1.5 at 0.2 Hz and below 2 above 3 Hz.

The **height** of the peak is not reliable: with these settings the values
within about 5 % of f0 change with `settings.body_wave_wavenumbers`
(spikes of 9-28 at 0.9-1.0 Hz), so read the peak frequency, not its
amplitude.

Common wrong-looking results:

- **Peak at the wrong frequency**: check the units (km, km/s); a thickness
  in metres gives a peak 1000 times too low.
- **"warning: ... Hz"** lines: the solver failed at those frequencies; the
  values there are NaN.
- **Very slow**: many layers or high frequencies; lower `count`, or
  `max_rayleigh_modes`/`max_love_modes` in `settings`.

## Using your own data

1. `data/model.txt`: your layers (`model_file` in `config.yml` if you rename it).
2. `frequency_hz.min`, `max`, `count`: your frequency band.
3. `settings` (optional): numerical controls, defaults listed in `config.yml`.
