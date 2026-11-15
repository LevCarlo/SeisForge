# 04 Receiver functions: quality control and H-k stacking

Receiver functions (RFs) show converted waves from interfaces below a
station. This example first throws out bad RFs (quality control), then finds
crustal thickness H and Vp/Vs ratio k by H-k stacking. The RFs are synthetic
(`tools/make_example_04_rf.py`): 30 good RFs from a 35 km thick crust with
Vp/Vs 1.75, plus 12 deliberately bad ones, each in two frequency bands.
Reference configs: `examples/reference_configs/rf_qc.yml`, `rf_hkseq.yml`;
method: `docs/rf_workflow.md`.

## Input

| file | what it is |
|---|---|
| `data/Gauss_2.5/*.sac`, `data/Gauss_5.0/*.sac` | radial RFs, Gaussian width 2.5 (low frequency) and 5.0 (high frequency) |
| `rf_qc.yml` | quality-control settings |
| `hkseq.yml` | H-k settings |

One SAC file per earthquake and band. Required headers: `b` (time of the
first sample relative to direct P, negative), `delta`, `user0` = ray
parameter in s/km, `user2` = signal-to-noise ratio (any number your RF
program writes; used by the SNR test). File names must contain the
component letter in `META.component` (here `R`). From
`data/Gauss_2.5/EV001.XX.RF01.R.sac`:

```text
b: -10.0  delta: 0.1  npts: 701  user0: 0.0616 (s/km)  user2: 20.9  baz: 115.7  kcmpnm: R
```

## Run

```bash
bash run.sh
```

About 8 s. `run.sh` stops if `output/` exists; delete it to run again (the
QC step itself would silently replace files there).

## Output

| file | what it is |
|---|---|
| `output/Gauss_2.5_CLEAN/`, `output/Gauss_5.0_CLEAN/` | copies of the RFs that passed QC |
| `output/logs/XX.RF01_RF_QC.log` | how many RFs each QC test kept |
| `output/hkSeq/XX.RF01_LFreq_Hk_QC.dat` | H and k of every RF kept by the H-k QC (`filename H_best k_best Hk_energy`) |
| `output/logs/XX.RF01_hkSeq.log` | the final `H = ... ± ...` and `k = ... ± ...` |
| `output/figures/` | QC and H-k figures |
| `result.png` | kept RFs with the predicted Ps, PpPs, PsPs times, and the H-k estimates |

## How to tell the result is right

`result.png` should look like `expected_result.png`: the Ps (about 4.4 s),
PpPs (about 14.7 s) and negative PsPs (about 19 s) arrivals line up with the
ticks. `run.sh` prints:

| | this example | acceptable |
|---|---|---|
| RFs kept by QC (low frequency) | 30 of 42 (every bad one removed) | all 12 `BAD_*` removed |
| H | 35.20 ± 0.20 km | 34.5-35.5 km (truth 35) |
| k | 1.744 ± 0.010 | 1.73-1.77 (truth 1.75) |

The H-k grid steps are 0.2 km and 0.01, so the answer cannot be closer than
that.

Common wrong-looking results:

- **No RF kept**: the SNR or slowness header is missing or in other units
  (`user0` must be s/km, about 0.04-0.08 for teleseismic P), or file names
  lack the component letter.
- **H at the edge of the search range**: widen `Model.Crust.H`.
- **H and k trade off along a line (large spread)**: normal for H-k; the
  PpPs and PsPs multiples are weak or noisy. Check `weight`, or fix k.
- **H off by several km with good-looking RFs**: `Model.Crust.Vp` is wrong
  for your area (H scales with Vp).

## Using your own data

`rf_qc.yml`:

1. `IO.RAW_HighFreq_RF`, `IO.RAW_LowFreq_RF`: your RF folders.
2. `META`: station, network, component letter in the file names.
3. `QC.*.SNR.SAC_HEADER`, `QC.*.SLOWNESS.SAC_HEADER`: where your files keep
   SNR and ray parameter.
4. `QC.*.*.THRESHOLD`: as needed (these are explicit choices; there are no
   defaults).

`hkseq.yml`:

1. `Meta`: station and network.
2. `Model.Crust.H`, `Model.Crust.k`: search ranges.
3. `Model.Crust.Vp`: average crustal Vp for your area (assumed value).
4. With a sedimentary layer, run `--mode both` and set `Model.Sediment`.
