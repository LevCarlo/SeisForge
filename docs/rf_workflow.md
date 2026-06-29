# Receiver Function Workflows

SeisForge currently exposes two receiver-function workflows through the unified
CLI:

```bash
seisforge rf qc -c examples/configs/rf_qc.yml --plot
seisforge rf hkseq -c examples/configs/hkseq.yml --mode both --plot
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

## Code Layout

- `seisforge.rf.traces`: `RFtrace`, `RFstream`, and RF selection methods.
- `seisforge.rf.plotting`: RF plotting helpers.
- `seisforge.rf.qc`: YAML-driven RF QC workflow and `rfqc` entry point.
- `seisforge.rf.hk`: public H-k stacking API.
- `seisforge.rf.hk_workflow`: H-k stacking and sequential H-k implementation.
- `seisforge.rf.rf` and `seisforge.rf.hkSeq`: compatibility wrappers.
