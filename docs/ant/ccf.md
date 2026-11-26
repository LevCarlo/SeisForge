# ZZ Cross-Correlation and Stacking

!!! overview "Scope, Non-Goals, and Architecture at a Glance"

    **Scope.** `seisforge ant ccf`, `seisforge ant stack-ccf`, and
    `seisforge ant export-sac` form a high-throughput vertical-component (**ZZ**)
    ambient-noise cross-correlation and multi-day stacking pipeline designed for
    dense 1-D linear and 2-D nodal arrays (tens to 500+ stations).

    **Non-Goals (current phase).** Raw instrument-specific data preparation
    (format decoding, response deconvolution, sub-sample clock alignment) stays
    outside the `seisforge` package so the core stays instrument-agnostic;
    9-component (ZNE) cross-correlation is intentionally deferred while current
    nodal deployments focus on ZZ Rayleigh-wave imaging.

```
  External Stage A (e.g. scripts/prep_seis_s20_sds.py)
      │  Standard SDS MiniSEED (FLOAT32, exact k/fs grid, uniform response_state)
      ▼
  seisforge ant ccf  ──►  daily_ccf/ccf_ZZ_YYYY-MM-DD.nc   (unnormalized ccf_sum + n_windows)
      │  Day-outer parallel (ProcessPoolExecutor or Slurm --date array)
      │  Optional target_sampling_rate_hz + max_distance_km + BLAS GEMM / CuPy GPU
      ▼
  seisforge ant stack-ccf  ──►  stacked_ZZ.nc              (linear | 1-D pws | 2-D tf_pws)
      │  Streaming float64 linear/PWS, or out-of-core pair-chunked tf_pws
      │  Lazy HDF5 pair-slice reads + CPU multiprocessing or GPU-resident CuPy
      ▼
  seisforge ant export-sac  ──►  sac_ZZ/<source>/<source>_<receiver>_ZZ.sac
         Selective post-processing converter for AFTAN or external tools
```

## 1. Quick Start and Usage

```bash
# Daily ZZ cross-correlation (workstation multi-day pool or HPC Slurm job array)
seisforge ant ccf -c ccf.yml --max-workers 8
seisforge ant ccf -c ccf.yml --date 2026-03-10

# Multi-day stacking (CPU parallel or NVIDIA GPU for 2-D tf_pws)
seisforge ant stack-ccf -c ccf_stack.yml --max-workers 32
seisforge ant stack-ccf -c ccf_stack.yml --backend cupy

# Selective SAC export grouped by virtual source station
seisforge ant export-sac -i stacked_ZZ.nc -o sac_ZZ --min-distance-km 5 --layout by-source
```

Minimal production `ccf.yml` (50 Hz shared SDS → 10 Hz ANT, ≤ 150 km):

```yaml
input:
  sds_root: /path/to/SDS_50Hz
  station_table: ./stations.csv
  sampling_rate_hz: 50.0
  response_state: velocity
  valid_passband_hz: [0.02, 20.0]
  gap_guard_s: 2.0
processing:
  cc_type: cc
  target_sampling_rate_hz: 10.0
  window_length_s: 1800.0
  max_lag_s: 100.0
  lag_guard_s: 20.0
  max_distance_km: 150.0
  ram_half_width_s: 15.0
  ram_bandpass: {freq_min_hz: 0.05, freq_max_hz: 2.0}
  whiten: {taper_min_hz: 0.04, freq_min_hz: 0.05, freq_max_hz: 2.0, taper_max_hz: 2.5, smooth_half_width_hz: 0.01}
output:
  dir: ./output/daily_ccf
run:
  start_date: "2026-03-08"
  end_date: "2026-03-22"
  max_workers: 4
```

Runnable example and full annotated configs: `examples/07_ccf/` (`ccf.yml`,
`ccf_stack.yml`, `stations.csv`, `run.sh`).

## 2. Annotated Configuration Reference

Configurations use progressive disclosure: set **Layer 1** to run, choose your
scientific combination in **Layer 2**, and tune **Layer 3** only for large-N
HPC or GPU runs. Unknown YAML keys are rejected with typo suggestions.

### 2.1 Daily CCF (`ccf.yml`)

#### Layer 1 — Core Inputs and Run Window

| Key | Default | Role & Contract |
|---|---|---|
| `input.sds_root` | *(required)* | Root of standard SDS tree (`YEAR/NET/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY`). |
| `input.station_table` | *(required)* | CSV with `network,station,location,channel,latitude_deg,longitude_deg` (+ optional `elevation_m,response_state,instrument_model`). For ZZ runs, filter to vertical channel only (e.g. `BHZ`). |
| `input.sampling_rate_hz` | *(required)* | Exact sampling rate (Hz) of the input SDS traces. Every trace must match. |
| `input.response_state` | *(required)* | One uniform state: `gain_only`, `velocity`, `displacement`, or `acceleration`. Mixed states in `station_table` are refused. |
| `input.valid_passband_hz` | `null` | Optional `[f_lo, f_hi]` Hz usable passband of Stage-A SDS. When set, `whiten` corners must stay inside it. |
| `input.gap_guard_s` | `0.0` | Extra margin (s) dilated around every SDS gap in `valid_mask` to absorb Stage-A filter/taper edge transients (recommended: `2.0`). |
| `input.allow_missing_next_day` | `true` | If `true`, skip cross-midnight windows cleanly when day $D+1$ is absent; if `false`, raise `FileNotFoundError`. |
| `output.dir` | *(required)* | Output directory for `ccf_ZZ_YYYY-MM-DD.nc` and `ccf.log`. |
| `run.start_date`<br>`run.end_date`<br>`run.dates` | *(required)* | Inclusive UTC date range (`YYYY-MM-DD`) or explicit list. Overridable via `--date` for Slurm job arrays. |

#### Layer 2 — Scientific Methods and Parameter Combinations

| Key | Default | Method Trade-offs & Guidance |
|---|---|---|
| `processing.cc_type` | `cc` | **`cc`**: classical normalized cross-correlation (Bensen et al. 2007), paired with continuous `ram_half_width_s` + `whiten`. **`pcc2`**: phase cross-correlation with $\nu = 2$ (Schimmel 1999; Ventosa et al. 2019), amplitude-invariant by construction (RAM is skipped automatically). Uses the exact two-real-channel `rfft` identity $\text{PCC2}_{AB}(t) = \frac{1}{L}[\text{CC}(\cos\phi_A,\cos\phi_B) + \text{CC}(\sin\phi_A,\sin\phi_B)]$ to reuse the same blocked GEMM path at nearly the same speed as `cc`. |
| `processing.target_sampling_rate_hz` | `null` | Optional anti-alias polyphase downsampling rate (Hz) applied per contiguous valid segment on the exact $k / f_{\text{target}}$ grid after reading SDS. Must be $\le$ `input.sampling_rate_hz` with a clean rational ratio. Use `10.0` when feeding a shared 50 Hz SDS archive into a `0.05–2.0 Hz` ANT run. |
| `processing.window_length_s`<br>`processing.step_s` | `500.0`<br>`null` | Segment length $L$ (s) and stride (`null` = non-overlapping). Typical nodal settings: `1800.0` (48 windows/day; loses only 1/48 of a day on a short gap) or `3600.0` (24 windows/day). Any window touching an invalid or non-finite sample is zeroed via `np.where` and excluded from `n_windows`. |
| `processing.max_lag_s`<br>`processing.lag_guard_s` | `60.0`<br>`20.0` | Published lag half-width $M$ and extra guard band $G$ (s) retained in daily files so Hilbert / Stockwell transforms in `stack-ccf` absorb edge truncation before trimming to $\pm M$. Rule of thumb: $M \ge d_{\max} / v_{\min} + 30\text{ s}$ (e.g. `100.0` s for $d_{\max} \le 150\text{ km}$). |
| `processing.ram_half_width_s`<br>`processing.ram_bandpass` | `null`<br>`null` | Continuous-waveform running-absolute-mean (RAM) normalization (`float64` `uniform_filter1d`, mask dilated by half-width). Set half-width to $\approx 0.5\,T_{\max}$ (e.g. `15.0` s for 20 s max period). `ram_bandpass` filters *only* the weight-estimation copy: for short-period 5 s nodal geophones, align it to the target working band (e.g. `0.05–2.0 Hz`) rather than teleseismic 15–50 s. Ignored for `pcc2`. |
| `processing.pcc2_bandpass` | `null` | Optional bandpass applied before computing instantaneous phase in `pcc2`. Usually omitted when `whiten` is already active. |
| `processing.whiten`<br>`processing.whiten_mode` | `null`<br>`vector_norm` | Four-corner cosine-tapered spectral whitening (`taper_min_hz < freq_min_hz < freq_max_hz < taper_max_hz`, `smooth_half_width_hz`, `water_level`) applied on unpadded $L$-sample windows *before* zero-padding to avoid padding leakage. `vector_norm` ($\sqrt{\text{smooth}(\|S\|^2)}$) is slightly smoother than `classic` ($\text{smooth}(\|S\|)$). Keep `taper_max_hz <= 0.5 * target_sampling_rate_hz`. |
| `processing.max_distance_km`<br>`processing.include_autocorr` | `null`<br>`true` | Upper WGS84 interstation distance cutoff (km) applied before IFFT and storage (e.g. `150.0` on dense arrays to avoid over-averaged long paths and cut pair count 2–3×). `include_autocorr` toggles $(i, i)$ diagonal pairs. |

#### Layer 3 — Advanced and HPC Tuning

| Key | Default | Notes |
|---|---|---|
| `processing.taper_fraction` | `0.05` | Tukey cosine taper fraction at each window end. |
| `processing.circular`<br>`processing.nfft` | `false`<br>`null` | Linear correlation (`circular: false`) requires $N_{\text{fft}} \ge L + M + G$ to prevent wrap-around in the guard band (`nfft: null` selects `next_fast_len`). |
| `processing.block_size` | `64` | Source stations per GEMM tile. Lower to `32` on 400+ station CPU nodes to halve temporary GEMM buffer RAM. |
| `processing.spectrum_dtype` | `complex64` | `complex64` (production; halves spectrum RAM/bandwidth, accuracy ~`1e-5`) or `complex128` (reference). |
| `output.overwrite`<br>`output.compression_level` | `false`<br>`4` | `overwrite: false` enables triple-hash resume; gzip level `0..9`. |
| `run.max_workers`<br>`run.backend` | `1`<br>`numpy` | Multi-day `ProcessPoolExecutor` (BLAS locked to 1 thread/worker) and `numpy` (CPU) or `cupy` (NVIDIA GPU). |

### 2.2 Multi-Day Stacking (`ccf_stack.yml`)

| `stack.method` | Formula / Mechanism | Best For | Speed (40k pairs × 60 d, 10 Hz) |
|---|---|---|---|
| **`linear`** | $\sum_d w_d\,\text{ccf\_sum}_d / \sum_d w_d\,n_{w,d}$ (`float64`) | Production baseline; preserves true relative amplitudes and dispersion waveform fidelity. | ~1–2 min (CPU) |
| **`pws`** (1-D) | $S_{\text{pws}}(t) = \|c(t)\|^\nu S_{\text{lin}}(t)$, $c(t) = \frac{1}{N_d}\sum_{d:\,n_{w,d}>0} a_d/\|a_d\|$ (Schimmel & Paulssen 1997) | Fast noise-floor suppression (~4–5× SNR boost), though broadband phase weighting can slightly narrow dispersive packets. | ~2–3 min (CPU) |
| **`tf_pws`** (2-D) | $\mathcal{S}[S_{\text{tf-pws}}](\tau, f) = \|c(\tau, f)\|^\nu \mathcal{S}[S_{\text{lin}}](\tau, f)$ via Stockwell (1996) transform & exact inverse (Schimmel & Gallart 2007) | Strong SNR boost (~3–4×) while preserving dispersive wave-packet shape; matches ANCC `TF_PWS` (`-I 1`). | ~15–30 min (64–96 CPU cores, `complex64`) or ~3–8 min (1 GPU) |

| Key | Default | Notes |
|---|---|---|
| `input.daily_dir` | *(required)* | Directory containing `ccf_ZZ_YYYY-MM-DD.nc` files. Optional `start_date`/`end_date`/`dates` subsets days. |
| `stack.method` | `linear` | `linear`, `pws`, or `tf_pws`. Mixing `cc` and `pcc2` daily files raises `ValueError`. |
| `stack.day_weighting` | `windows` | `windows` (window-weighted mean, default) or `equal` (equal weight per valid day; matches ANCC `TF_PWS`). |
| `stack.pws_power` | `2.0` | Coherence exponent $\nu$ for `pws` and `tf_pws` (set `1.0` to match ANCC `TF_PWS` default). |
| `stack.trim_guard` | `false` | Trim `lag_guard_s` to $\pm\text{max\_lag\_s}$ for `linear` (`pws` and `tf_pws` always trim). |
| `stack.tf_pws_freq_min_hz`<br>`stack.tf_pws_freq_max_hz` | `null`<br>`null` | Restricts the Stockwell frequency axis to the band of interest (strongly recommended; e.g. `0.05`–`2.0` Hz). |
| `stack.tf_pws_sigma`<br>`stack.tf_pws_smooth_samples`<br>`stack.tf_pws_normalize_weights` | `1.0`<br>`0`<br>`false` | Generalized Stockwell Gaussian width, optional time-axis weight smoothing half-width (samples), and per-frequency max normalization. |
| `stack.tf_pws_pair_chunk_size`<br>`stack.tf_pws_dtype` | `64`<br>`complex128` | Pair chunk upper bound (auto-capped to 1 GiB/worker) and Stockwell precision (`complex64` halves RAM/VRAM and unlocks peak GPU speed). |
| `stack.max_workers`<br>`stack.backend` | `1`<br>`numpy` | CPU multi-process workers (overridable via `--max-workers`) or NVIDIA GPU `cupy` (`--backend cupy`; runs sequentially on the GPU to avoid multi-process VRAM contention). |
| `output.file`<br>`output.overwrite`<br>`output.compression_level` | *(required)*<br>`false`<br>`4` | Output stacked NetCDF4/HDF5 path and `stack_ccf.log`. |

### 2.3 Assumed Defaults

<div class="assumptions" markdown>

| parameter | default | note |
|---|---|---|
| `grid_tolerance_samples` | `1e-3` | assumed sample-alignment tolerance |
| `taper_fraction` | `0.05` | assumed Tukey taper fraction per window end |
| `whiten.water_level` | `1e-6` | assumed relative denominator floor |
| `ram` / `pcc2` / `pws` `water_level` | `1e-12` | assumed zero-amplitude guard |
| `block_size` | `64` | assumed source-station GEMM block size |
| `tf_pws_pair_chunk_size` | `64` | assumed pair chunk cap (auto-capped to 1 GiB/worker) |
| `compression_level` | `4` | assumed NetCDF4 gzip level |

</div>

## 3. Output Data Contract

- **Daily (`ccf_ZZ_YYYY-MM-DD.nc`, `stage="ccf_daily"`)**: dimensions
  `(pair, lag_s)` including `lag_guard_s`; stores `ccf_sum(pair, lag_s)`
  (`0.0` when `n_windows == 0` so daily sums add directly), `n_windows(pair)`,
  WGS84 geometry (`distance_km`, `azimuth_deg`, `back_azimuth_deg`, station
  coordinates), software provenance, and resume hashes.
- **Stacked (`stacked_ZZ.nc`, `stage="ccf_stacked"`)**: dimensions
  `(pair, lag_s)`; stores normalized `ccf(pair, lag_s)` (`NaN` for zero-count
  pairs), total `n_windows(pair)`, `n_days(pair)`, geometry, stacking
  parameters, and `dates`.
- **Exported SAC (`seisforge ant export-sac`)**: filters by station list,
  `--min-distance-km`, `--max-distance-km`, `--min-n-windows`, and
  `--include-autocorr`. Default layout `--layout by-source` writes
  `{source}/{source}_{receiver}_{component}.sac` with headers `b`, `e`, `delta`,
  `dist`, `az`, `baz`, `evla`, `evlo`, `stla`, `stlo`, `knetwk`, `kstnm`,
  `kevnm` (`NET.STA`), and `kcmpnm`.

## 4. System Architecture and Design Decisions (ADRs)

**ADR-1: Decoupled Stage-A prep with an explicit input contract.**
*Context:* Nodal formats, clock-drift quirks, and pole-zero conventions vary
wildly by manufacturer, and MiniSEED headers do not record response state or
effective passband.
*Decision:* Preparation runs externally (with cross-midnight padding to absorb
filter/taper edge transients and multi-trace daily MiniSEEDs that never
zero-fill gaps). `seisforge ant ccf` enforces a strict entry contract: identical
`sampling_rate_hz`, trace start times aligned to $k / f_s$ within
`grid_tolerance_samples`, and one uniform `response_state` across all stations
(refusing mixed states that introduce $90^\circ$–$180^\circ$ phase bias).

**ADR-2: Two-level stacking in NetCDF4/HDF5 instead of hourly temporary SAC files.**
*Context:* Writing per-hour, per-pair temporary SAC files on a 400–500 station
array generates tens of millions of tiny files that saturate shared HPC
filesystems (Lustre/GPFS) and forces 2-D `tf_pws` to transform ~1,000+ hourly
traces per pair.
*Decision:* `ant ccf` writes one compressed daily NetCDF4/HDF5 file containing
unnormalized `ccf_sum(pair, lag_s)` and integer `n_windows(pair)`.
`ant stack-ccf` streams those daily files into `linear`, 1-D `pws`, or 2-D
`tf_pws`—cutting `tf_pws` Stockwell transforms by ~20× while eliminating
small-file I/O storms.

**ADR-3: Atomic writes and triple-hash idempotent resume.**
*Context:* HPC jobs are routinely preempted or time-limited; naive "skip if
file exists" logic silently mixes stale parameters or partial files.
*Decision:* Every NetCDF file is written to a same-directory temporary file and
committed via atomic `os.replace`. Daily files embed three 16-char SHA-256
hashes—`param_hash` (scientific parameters only; runtime knobs `max_workers`,
`block_size`, `backend` are excluded), `station_list_hash`, and
`input_fingerprint` (file paths, sizes, and `mtime_ns`). A day is resumed only
when all three match.

**ADR-4: Canonical upper-triangle storage and explicit lag sign.**
For virtual source $A$ and receiver $B$ ($i \le j$),

$$
\boxed{\;C_{AB}(t) = \sum_{\tau} u_A(\tau)\,u_B(\tau + t),
\qquad
\widehat{C}_{AB}(f) = \overline{\widehat{u}_A(f)}\,\widehat{u}_B(f)\;}
\label{eq:ccf-def}
$$

Positive lag $t > 0$ represents propagation from $A$ to $B$, matching
`np.correlate(u_B, u_A, "full")`, ANCC v6.8, and [ZNE → ZRT rotation](rotation.md).
Reversing a ZZ pair is pure time reversal: $C_{ZZ}^{BA}(t) = C_{ZZ}^{AB}(-t)$
(`reverse_ccf_zz`). Pairs beyond `max_distance_km` are pruned *before*
correlation and storage.

**ADR-5: Selective `by-source` SAC export with standard receiver/source headers.**
Flat directories with 40,000+ SAC files stall directory scans. `export-sac`
defaults to `--layout by-source` (`{source}/{source}_{receiver}_{component}.sac`)
with `--layout flat` and `--filename-template` overrides, writing pure receiver
station name to `KSTNM`, receiver network to `KNETWK`, and full `NET.STA`
virtual source to `KEVNM` (preserving source network across multi-network
arrays).

## 5. Performance Engineering and Scaling

Following a roofline view (Williams, Waterman & Patterson 2009), each stage of
the pipeline targets its dominant bottleneck:

| Stage | Primary Bottleneck | Engineering Optimization in SeisForge |
|---|---|---|
| **SDS ingest & prep** | Disk I/O + redundant archives | Single shared high-rate SDS (e.g. 50 Hz for HVSR) + on-the-fly `target_sampling_rate_hz` (Kaiser $\beta = 8$ polyphase decimation on exact $k / f_{\text{target}}$ grid). `50 Hz → 10 Hz` shrinks FFTs and daily NetCDF size **5×** and speeds up `tf_pws` **4–16×** with correlation `r = 1.0000` in the target band. |
| **Daily correlation (`ccf`)** | Compute-bound $O(N^2 N_w N_f)$ | Source-blocked upper-triangle BLAS Level-3 matrix multiply (`cgemm`/`zgemm`, `block_size`), `max_distance_km` pair pruning before IFFT/storage, two-real-channel `rfft` decomposition for `pcc2`, day-level parallelism (`--max-workers` with BLAS locked to 1 thread, or Slurm `--date` job arrays), and optional NVIDIA GPU `--backend cupy`. |
| **Linear & 1-D PWS stacking** | I/O-bound sequential scan | Single-pass `float64` streaming accumulators holding one day in memory at a time; takes ~1–2 min for 40,000 pairs × 60 days. |
| **2-D `tf_pws` stacking** | Memory-bandwidth & FFT-bound | Contiguous doubled-spectrum slicing + frequency-batched IFFTs (`freq_block=32`) + cached Gaussian window tables + **1 GiB per-worker out-of-core pair chunking** + lazy HDF5 pair-slice reads (`read_netcdf_pair_slice`). Scales across CPU cores (`--max-workers`) or runs GPU-resident (`--backend cupy`) with optional `tf_pws_dtype: complex64` (2× less RAM/VRAM, peak FP32 cuFFT throughput). |

### Resource Planning Reference (410 Stations, 60 Days, ZZ, ≤ 150 km)

Assumptions: 50 Hz shared SDS downsampled via `target_sampling_rate_hz: 10.0`,
`window_length_s: 1800.0` (48 windows/day), `max_lag_s: 100.0`,
`lag_guard_s: 20.0`, `max_distance_km: 150.0` (~40,000 surviving pairs):

- **Peak RAM per `ccf` worker**: ~6–8 GB (`cc`, `block_size: 32`, `complex64`);
  ~9–11 GB (`pcc2`). Size `--max-workers` as `floor(node_RAM_GB / 10)`.
- **Peak RAM per `tf_pws` worker**: automatically capped at ~1 GiB working set
  regardless of `tf_pws_pair_chunk_size`.
- **Disk footprint**: ~300–450 MB/day compressed daily NetCDF (~20–27 GB for 60
  days); ~300–400 MB for final `stacked_ZZ.nc`.
- **Throughput order of magnitude** (same hardware class, two-level daily
  stacking vs. hourly single-level stacking):
  - 64–96 core CPU node: `ccf` ~8–15 min (3 nodes via Slurm `--date` array:
    ~5 min), `linear`/`pws` ~1 min, `2-D tf_pws` (`complex64`) ~15–30 min.
  - Single RTX 4090 / A100 GPU (`--backend cupy`, `tf_pws_dtype: complex64`):
    `2-D tf_pws` ~3–8 min.

## 6. Verification

!!! verified "Verified"

    - `tests/test_ant_ccf_core.py`: checks lag sign against
      `np.correlate(u_B, u_A, "full")`, known-delay impulse recovery,
      wrap-around-free guard bands, whitening passband flatness and zero
      padding leakage, continuous RAM mask dilation, blocked upper-triangle
      GEMM across block sizes against an independent `float64` brute-force
      reference (`1e-11` for `complex128`, `1e-5` for `complex64`), two-real-channel
      PCC2 against direct complex Schimmel (1999), reverse-pair symmetry,
      unequal-window and zero-count linear stacking, Stockwell exact
      invertibility, and batch/streaming PWS and `tf_pws` equivalence.
    - `tests/test_ant_ccf_io.py` and `tests/test_ant_ccf_driver.py`: check
      YAML unknown-key and passband validation, mixed `response_state`
      rejection, synthetic cross-midnight SDS reading and grid checks,
      anti-alias `target_sampling_rate_hz` downsampling, atomic NetCDF4
      round-trips, triple-hash resume, parallel `tf_pws` (`max_workers > 1`),
      `complex64` Stockwell accuracy, mocked CuPy CLI dispatch, and SAC export
      headers (`knetwk`, `kstnm`, `kevnm`) and layouts (`by-source`, `flat`).
    - Real-data end-to-end validation on a 6-station Guangxi SEIS-S20 linear
      nodal subset (15 pairs, 9.8–130.3 km, 15 UTC days): `50 Hz -> 10 Hz`
      downsampling matched 50 Hz CCFs to `r = 1.0000` (0.1–0.5 Hz); `cc` vs
      `pcc2` matched to `r = 0.998–0.999`; linear and 2-D `tf_pws` stacks
      matched full-deployment ANCC v6.8 references with zero lag shift
      (`median |dt| = 0.00 s`, `median r = 0.92–0.93`, up to `r = 0.98` at
      130 km).

!!! unverified "Not yet verified"

    - Multi-process `max_workers > 1` and real NVIDIA GPU `backend: cupy`
      execution have not yet been benchmarked on 400–500 station production
      deployments.
    - Formal amplitude / relative-L2 benchmark against ANCC and NoisePy on
      identical day windows remains pending.

## References

- Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
  Moschetti, M. P., Shapiro, N. M., & Yang, Y. (2007). Processing seismic
  ambient noise data to obtain reliable broad-band surface wave dispersion
  measurements. *Geophysical Journal International*, 169(3), 1239–1260.
  [doi:10.1111/j.1365-246X.2007.03374.x](https://doi.org/10.1111/j.1365-246X.2007.03374.x)
- Jiang, C., & Denolle, M. A. (2020). NoisePy: A new high-performance Python
  tool for ambient-noise seismology. *Seismological Research Letters*, 91(3),
  1853–1866.
  [doi:10.1785/0220190364](https://doi.org/10.1785/0220190364)
- Schimmel, M. (1999). Phase cross-correlations: Design, comparisons, and
  applications. *Bulletin of the Seismological Society of America*, 89(5),
  1366–1378.
  [doi:10.1785/BSSA0890051366](https://doi.org/10.1785/BSSA0890051366)
- Schimmel, M., & Paulssen, H. (1997). Noise reduction and detection of weak,
  coherent signals through phase-weighted stacks. *Geophysical Journal
  International*, 130(2), 497–505.
  [doi:10.1111/j.1365-246X.1997.tb05664.x](https://doi.org/10.1111/j.1365-246X.1997.tb05664.x)
- Schimmel, M., & Gallart, J. (2007). Frequency-dependent phase coherence for
  noise suppression in seismic array data. *Journal of Geophysical Research*,
  112, B04303.
  [doi:10.1029/2006JB004680](https://doi.org/10.1029/2006JB004680)
- Stockwell, R. G., Mansinha, L., & Lowe, R. P. (1996). Localization of the
  complex spectrum: the S transform. *IEEE Transactions on Signal Processing*,
  44(4), 998–1001.
  [doi:10.1109/78.492555](https://doi.org/10.1109/78.492555)
- Ventosa, S., Schimmel, M., & Stutzmann, E. (2019). Towards the processing of
  large data volumes with phase cross-correlation. *Seismological Research
  Letters*, 90(4), 1663–1669.
  [doi:10.1785/0220190022](https://doi.org/10.1785/0220190022)
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An insightful
  visual performance model for multicore architectures. *Communications of the
  ACM*, 52(4), 65–76.
  [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
