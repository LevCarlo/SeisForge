# Receiver Function Workflows

SeisForge currently exposes two receiver-function workflows through the unified
CLI:

```bash
seisforge rf qc -c examples/configs/rf_qc.yml --plot
seisforge rf hkseq -c examples/configs/hkseq.yml --mode both --plot
seisforge rf hkseq -c examples/configs/hkseq.yml --mode low --overwrite
```

The older commands remain available for existing scripts:

```bash
rfqc -c examples/configs/rf_qc.yml --plot
hkseq -c examples/configs/hkseq.yml --mode both --plot
```

## RF QC Config

Use `examples/configs/rf_qc.yml` as the starting template.

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

Use `examples/configs/hkseq.yml` as the starting template.

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

## Code Layout

- `seisforge.rf.traces`: `RFtrace`, `RFstream`, and RF selection methods.
- `seisforge.rf.plotting`: RF plotting helpers.
- `seisforge.rf.qc`: YAML-driven RF QC workflow and `rfqc` entry point.
- `seisforge.rf.hk`: public H-k stacking API.
- `seisforge.rf.hk_core`: pure H-k formulas (phase delays, sampling,
  amplitude correction, weighted stack).
- `seisforge.rf.hk_workflow`: `HkStack` classes, H-k QC, and the sequential
  H-k driver.
- `seisforge.rf.rf` and `seisforge.rf.hkSeq`: compatibility wrappers.
