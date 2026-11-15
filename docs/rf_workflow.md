# Receiver Function Workflows

SeisForge exposes two receiver-function workflows through the unified CLI
(QC and sequential H-k stacking) and a synthetic P receiver function forward
model in Python ([below](#synthetic-p-receiver-functions)):

```bash
seisforge rf qc -c examples/reference_configs/rf_qc.yml --plot
seisforge rf hkseq -c examples/reference_configs/rf_hkseq.yml --mode both --plot
seisforge rf hkseq -c examples/reference_configs/rf_hkseq.yml --mode low --overwrite
```

The older commands remain available for existing scripts:

```bash
rfqc -c examples/reference_configs/rf_qc.yml --plot
hkseq -c examples/reference_configs/rf_hkseq.yml --mode both --plot
```

## RF QC Config

Use `examples/reference_configs/rf_qc.yml` as the starting template.

Important sections:

- `IO.ROOT`: station-level working directory.
- `RAW_HighFreq_RF` / `RAW_LowFreq_RF`: input RF directories under `ROOT`.
- `CLEAN_HighFreq_RF` / `CLEAN_LowFreq_RF`: output directories under `ROOT`.
- `FIGURE` / `LOG`: output figure and log directories under `ROOT`.
- `META.component`: component token used to find RF files, commonly `R`.
- `QC.HighFreq` and `QC.LowFreq`: selection parameters for each RF band.

The QC sequence is:

1. SNR selection from a SAC header, usually `user2`.
2. Slowness selection from a SAC header, usually `user0`.
3. P-arrival polarity and timing selection.
4. MAD outlier rejection on waveform variance.
5. F-test selection against the mean RF.
6. Cross-correlation selection against the average waveform.

Each step has an `ENABLE` flag. Disabled steps copy the stream through
unchanged, which makes it easy to test one criterion at a time.

## Sequential H-k Config

Use `examples/reference_configs/rf_hkseq.yml` as the starting template.

Important sections:

- `IO.HighFreq_RF`: cleaned high-frequency RF directory, usually from RF QC.
- `IO.LowFreq_RF`: cleaned low-frequency RF directory.
- `Model.Sediment`: H-k search range and velocities for the shallow layer.
- `Model.Crust`: H-k search range and velocities for the crust.
- `QC`: energy, elliptical MAD, and Mahalanobis thresholds for H-k estimates.

The H-k workflow keeps the existing parallel design through `joblib.Parallel`.
High-frequency RFs estimate the sediment layer first when `--mode both` is used;
low-frequency RFs then estimate the crust with either one-layer or two-layer
stacking depending on whether the sediment estimate is valid.

### Outputs And Re-runs

`IO.SAVE` must be a subdirectory of `IO.ROOT` that does not overlap either RF
input directory; otherwise `hkseq` stops before writing anything. Each stage
writes only its own files:

| Stage | Files in `IO.SAVE` | Figure in `IO.FIGURE` (`--plot`) |
|---|---|---|
| high | `NET.STA_HFreq_Hk_results.joblib`, `NET.STA_HFreq_Hk_results_QC.joblib`, `NET.STA_HFreq_Hk_QC.dat` | `NET.STA_HFreq_QC.png` |
| low | `NET.STA_LFreq_Hk_results.joblib`, `NET.STA_LFreq_Hk_results_QC.joblib`, `NET.STA_LFreq_Hk_QC.dat` | `NET.STA_LFreq_QC.png` (`NET.STA_HFreq_LFreq_QC.png` with `--mode both`) |

If any of these files already exist, the run stops with `FileExistsError`.
Pass `--overwrite` (or set `IO.OVERWRITE: true`) to replace them. Re-running
`--mode low` no longer deletes high-frequency results; older versions removed
the whole `IO.SAVE` directory. The high-frequency QC table used to be named
`STA.NET_HFreq_Hk_QC.dat`; it now follows the `NET.STA_` prefix of every other
output.

The run log `IO.LOG/NET.STA_hkSeq.log` is rewritten on every run. Logging goes
through the `seisforge.rf.hk_workflow` logger and no longer reconfigures the
Python root logger.

## H-k Method

The stacking formulas live in `seisforge.rf.hk_core` as pure functions. The
grid has `k = Vp/Vs` on axis 0 and thickness `H` (km) on axis 1. With ray
parameter `p` (s/km, SAC `user0`), the delays relative to direct P are
(Zhu & Kanamori, 2000)

```text
eta_s = sqrt(1/Vs^2 - p^2),  eta_p = sqrt(1/Vp^2 - p^2)
t_Ps   = H (eta_s - eta_p)
t_PpPs = H (eta_s + eta_p)
t_PsPs = 2 H eta_s
```

`Vp` is fixed and `Vs = Vp/k`, unless the config gives `Vs`; then `Vs` is
fixed and `Vp = Vs k`. In two-layer mode (crust beneath a resolved sediment
layer of thickness `h0`), the target layer spans `H - h0` and the sediment
delays for `h0` are added, using the sediment `Vp`/`Vs` and the sediment
`k0` estimate.

Before stacking, RF amplitudes are scaled by Seispy's empirical correction
`151.5478 p^2 + 3.2896 p + 0.2618`. The stack is

```text
S(H, k) = w1 A(t_Ps) + w2 A(t_PpPs) - w3 A(t_PsPs)
```

with `Model.*.weight = [w1, w2, w3]`. `A(t)` is the RF amplitude at the nearest
sample (ties go to the earlier sample). A delay outside the RF trace, or a
post-critical ray (`p > 1/V`), contributes **zero**. Older versions clamped
such delays to the first or last RF sample. Each result dataset records
`n_invalid_Ps`, `n_invalid_PpPs`, and `n_invalid_PsPs`. A nonzero count means
the search grid reaches beyond the RF time window, and the RF should be cut
longer or the `H` range narrowed. Grid points inside the trace give exactly
the same values as before.

The sample lookup now uses O(`n_H * n_k`) memory. The previous
implementation built an `npts x n_H x n_k` distance matrix for each phase.

## Synthetic P Receiver Functions

`seisforge.rf.synthetic` computes P receiver functions for isotropic flat-layered
models, and `seisforge.inv.predict_receiver_function` applies it to a
`LayeredModel` for inversion. The code is a SeisForge implementation (MIT)
written from the published formulas; no external RF package is needed.

**Method.** A plane P wave with horizontal slowness `p` arrives from the
half-space. The Thomson-Haskell propagator for motion-stress vectors
(Haskell 1962; Aki & Richards 2002, ch. 7; Kennett 1983) gives the free-surface
radial and vertical displacements; the receiver function is their spectral
ratio after a Gaussian low-pass filter:

```text
RF(w) = R(w) / Z(w) * exp(-w^2 / (4 a^2))
```

The result includes every reverberation in the stack. Each layer uses its own
Vp, Vs, **and density**, so densities from per-segment scaling (Gardner,
constant mantle density, ...) affect the conversion amplitudes. Packages that
take only Vp/Vs fix density internally and cannot do this.

```python
from seisforge.inv import ReceiverFunctionRequest, predict_receiver_function

request = ReceiverFunctionRequest(
    ray_parameter_s_per_km=0.06,
    dt_s=0.05,
    n_samples=1200,        # 60 s
    gauss_a=2.5,           # exp(-w^2 / (4 a^2))
    time_shift_s=5.0,      # time axis starts at -5 s, direct P at 0 s
    amplitude="impulse",   # or "gaussian_peak"
)
prediction = predict_receiver_function(layered_model, request)
prediction.time_s, prediction.rf
```

**Conventions.**

- Radial is positive away from the source and vertical positive up, so the
  direct P pulse is positive; Ps and PpPs are positive and PsPs negative for a
  velocity increase with depth.
- `gauss_a` is the `a` of `exp(-w^2 / (4 a^2))`, as in iterative time-domain
  deconvolution (Ligorria & Ammon 1999) and seispy.
- `amplitude="impulse"` is the continuous inverse transform (the Gaussian has
  unit area; a spike `A` peaks at `A a / sqrt(pi)`), the convention of Zhu's
  `hk` synthetics. `amplitude="gaussian_peak"` divides by `a / sqrt(pi)` so a
  spike `A` peaks at `A`. Use the convention that matches how the observed RFs
  were normalized.
- The result is periodic in the FFT length, so reverberations arriving after
  it wrap into the window. The FFT covers `max(2 x window, window + 60 s)`
  (rounded to an FFT-friendly length). If the trace has not decayed to
  `tail_tolerance` (default 1e-3) of its peak by the end of the padding, the
  length is doubled, up to 16 times; the doubled grid reuses every existing
  frequency, so only the new half is computed. A trace that still rings, because the
  vertical spectrum is nearly zero and `R/Z` has a near-pole, raises an error
  instead of returning a wrapped result. Against a 131072-point FFT the
  default is accurate to about 1e-4 of the peak even for a 15 s window. An
  explicit `n_fft` is used as given.
- Only frequencies where the Gaussian exceeds 1e-12 are computed.
- Every layer and the half-space must have `p < 1/Vp` (no evanescent P), which
  holds for teleseismic P; otherwise a `ForwardError` is raised.
- Not yet supported: S receiver functions, fluid (water) layers, attenuation.

**Validation** (`tests/test_rf_synthetic.py` and an offline comparison):

| check | result |
|---|---|
| half-space `R/Z` vs the analytic `2 p q_s / (1/Vs^2 - 2 p^2)` | equal to 1e-12 |
| Ps, PpPs, PsPs times vs `hk_core` delays, polarities | within one sample, correct signs |
| identical layers vs a half-space; vanishing layer | equal; converges |
| PyGRT 0.17.2 (`grt rcvfn`, independent generalized R/T code), 6 models incl. arbitrary density, 12 layers, a 0.9 km/s sediment, p = 0.04 and 0.075 | waveforms agree to 3-8e-4 of the peak after a constant 2.8 ms delay inside PyGRT; Ps/P, PpPs/P, PsPs/P agree to 1e-3 (regression test) |
| pyhk 0.1.0 with its own density law `0.77 + 0.32 Vp` | direct P equal, converted phases 5-13 % smaller than both SeisForge and PyGRT |

The pyhk difference is not a density effect (no density law reproduces it),
so pyhk is not used as a reference.

**Cost.** The frequency loop is a numba kernel in the up/down wave basis:
each layer costs a phase multiplication and one real 4x4 transfer matrix, and
the layer phases follow a frequency recurrence (refreshed every 64
frequencies). The kernel releases the GIL, so the `thread` MCMC executor
runs chains in parallel (4 threads: about 3x). Typical times:

| model | 60 s, dt 0.05, a 2.5 | 30 s, dt 0.1, a 1.0 | 40 s, dt 0.02, a 5.0 |
|---|---|---|---|
| ~20 layers | 0.22 ms | 0.11 ms | 0.34 ms |
| ~67 layers | 0.48 ms | 0.19 ms | 0.79 ms |
| ~200 layers | 1.33 ms | 0.48 ms | 2.16 ms |

That is below one 40-period disba dispersion curve on the same model. A
degenerate model with zero vertical surface motion raises `ValueError`
(`ForwardError` through `predict_receiver_function`).

## Code Layout

- `seisforge.rf.traces`: `RFtrace`, `RFstream`, and RF selection methods.
- `seisforge.rf.plotting`: RF plotting helpers.
- `seisforge.rf.qc`: YAML-driven RF QC workflow and `rfqc` entry point.
- `seisforge.rf.hk`: public H-k stacking API.
- `seisforge.rf.synthetic`: synthetic P receiver functions (Thomson-Haskell).
- `seisforge.rf.hk_core`: pure H-k formulas (phase delays, sampling,
  amplitude correction, weighted stack).
- `seisforge.rf.hk_workflow`: `HkStack` classes, H-k QC, and the sequential
  H-k driver.
- `seisforge.rf.rf` and `seisforge.rf.hkSeq`: compatibility wrappers.
