# ZZ Cross-Correlation and Stacking

!!! overview "Method at a glance"

    `seisforge ant ccf` computes daily vertical-component (ZZ) ambient-noise
    cross-correlation sums from pre-prepared standard SDS MiniSEED archives,
    using a day-outer, source-blocked upper-triangle GEMM loop.
    `seisforge ant stack-ccf` streams those daily NetCDF4/HDF5 files into a
    multi-day linear or phase-weighted stack, and `seisforge ant export-sac`
    selectively converts NetCDF4/HDF5 products to SAC files for AFTAN or
    external tools. Data preparation (resampling, absolute time-grid alignment,
    instrument-response correction) stays outside SeisForge; the CCF entry
    point enforces an explicit input contract instead.

## 1. Conventions and input contract

**Correlation and lag sign.** For virtual source $A$ and receiver $B$,

$$
\boxed{\;C_{AB}(t) = \sum_{\tau} u_A(\tau)\,u_B(\tau + t),
\qquad
\widehat{C}_{AB}(f) = \overline{\widehat{u}_A(f)}\,\widehat{u}_B(f)\;}
\label{eq:ccf-def}
$$

Positive lag $t > 0$ corresponds to wave propagation from $A$ to $B$, matching
`np.correlate(u_B, u_A, "full")` and [ZNE → ZRT rotation](rotation.md).
Only canonical upper-triangle pairs $i \le j$ are stored; reversing a ZZ pair
is pure time reversal: $C_{ZZ}^{BA}(t) = C_{ZZ}^{AB}(-t)$ (`reverse_ccf_zz`).

**Input contract (checked at entry).** Because MiniSEED headers do not carry
response state, effective passband, or preparation filter transients, the
config and station table declare them explicitly:

- Standard SDS layout (`YEAR/NET/STA/CHA.D/NET.STA.LOC.CHA.D.YEAR.DOY`) with
  identical `sampling_rate_hz` across every trace.
- Every trace start time aligned to the UTC epoch grid $k / f_s$ within
  `grid_tolerance_samples` (assumed default: $10^{-3}$ samples).
- One uniform `response_state` across all stations (`gain_only`, `velocity`,
  `displacement`, or `acceleration`); mixed response states are refused because
  mixing uncorrected geophones with deconvolved broadband stations introduces
  $90^\circ$–$180^\circ$ phase bias.
- Optional `valid_passband_hz`: when set, the whitening taper corners
  `[taper_min_hz, taper_max_hz]` must stay inside it.
- Optional `gap_guard_s`: dilates the invalid mask around every SDS trace gap
  to absorb resampling/deconvolution edge transients.
- Optional `processing.target_sampling_rate_hz`: anti-alias polyphase
  downsampling (`scipy.signal.resample_poly`, Kaiser $\beta = 8$) applied per
  contiguous valid segment on the exact $k / f_{\text{target}}$ grid after
  reading SDS. Useful when one higher-rate SDS archive (e.g., 50 Hz for HVSR)
  feeds lower-band ANT correlation (e.g., 10 Hz for 0.05–4 Hz).

## 2. Preprocessing and correlation

**Continuous-waveform RAM (`cc_type: cc`).** Running-absolute-mean temporal
normalization (Bensen et al. 2007) runs on the continuous day waveform before
windowing to avoid per-window edge artefacts, using `float64`
`uniform_filter1d` (avoiding `float32` cumulative-sum drift). Invalid samples
are dilated outward by the RAM half-width (`expand_mask_for_ram`).

**Windowing and spectral whitening.** Continuous traces are sliced on the
window grid (`window_length_s`, `step_s`), demeaned, and tapered with a Tukey
window (`taper_fraction`). Any window containing an invalid or non-finite
sample is zeroed via `np.where` (preventing `NaN * 0`). Spectral whitening runs
on the unpadded $L$-sample window with a four-corner cosine taper before
zero-padding to `nfft`, preventing zero-padding spectral leakage. All
`nfft // 2 + 1` one-sided `rfft` bins are kept (no frequency cropping).

**Wrap-around-free padding and guard band.** Linear correlation requires

$$
\boxed{\;N_{\text{fft}} \ge L + M + G\;}
\label{eq:ccf-nfft}
$$

where $L$, $M$, and $G$ are the window length, `max_lag_s`, and `lag_guard_s`
in samples. Including $G$ prevents circular wrap-around from contaminating the
guard band used later by the Hilbert transform in PWS.

**PCC2 (`cc_type: pcc2`).** For phase cross-correlation with exponent $\nu = 2$
(Schimmel 1999; Ventosa, Schimmel & Stutzmann 2019), writing the unit-modulus
analytic signal as $u = a / |a| = \cos\phi + i\sin\phi$,

$$
\boxed{\;\text{PCC2}_{AB}(t)
= \frac{1}{L}\sum_{\tau}\operatorname{Re}\!\left[\overline{u_A(\tau)}\,u_B(\tau + t)\right]
= \frac{1}{L}\Bigl[\text{CC}(\cos\phi_A, \cos\phi_B)(t)
  + \text{CC}(\sin\phi_A, \sin\phi_B)(t)\Bigr]\;}
\label{eq:ccf-pcc2}
$$

Stacking $\cos\phi/\sqrt{L}$ and $\sin\phi/\sqrt{L}$ along the window axis
reuses the same real `rfft` + blocked upper-triangle GEMM path without a
separate complex FFT pipeline; each valid time window still counts as $1$ in
`n_windows`. Samples with $|a| \le \text{water\_level}\cdot\max|a|$ are
zeroed.

**Zero-count semantics and resume.** Daily NetCDF4/HDF5 files store the
unnormalized window sum `ccf_sum(pair, lag_s)` and integer `n_windows(pair)`.
Pairs with `n_windows == 0` store `0.0` so daily sums add directly; readers
and stackers convert zero-count pairs to `NaN`. A day is skipped on resume only
when the output exists and `param_hash`, `station_list_hash`, and
`input_fingerprint` all match.

## 3. Multi-day stacking and SAC export

**Linear stack (`method: linear`).** Accumulated in `float64` across days via
`LinearStackAccumulator`: either window-weighted
($\sum_d \text{ccf\_sum}_d / \sum_d n_{w,d}$, default) or equal-day-weighted.
Mixing `cc` and `pcc2` daily files raises `ValueError`.

**Phase-Weighted Stack (`method: pws`, experimental).** Follows Schimmel &
Paulssen (1997) with daily mean CCFs $s_d(t)$ as stacking units:

$$
c(t) = \frac{1}{N_{\text{days}}}\sum_{d:\,n_{w,d}>0}
\frac{s_d(t) + i\,\mathcal{H}[s_d](t)}{\bigl|s_d(t) + i\,\mathcal{H}[s_d](t)\bigr|},
\qquad
S_{\text{pws}}(t) = |c(t)|^{\nu}\,S_{\text{lin}}(t).
$$

Missing days ($n_{w,d} = 0$) are excluded from $c(t)$ rather than treated as
zero phase. The Hilbert transform runs over $\pm(\text{max\_lag\_s} +
\text{lag\_guard\_s})$, and the final PWS output is trimmed to
$\pm\text{max\_lag\_s}$.

**Time-Frequency Phase-Weighted Stack (`method: tf_pws`, experimental).**
Follows Stockwell, Mansinha & Lowe (1996) and Schimmel & Gallart (2007) by
evaluating phase coherence in the 2-D Stockwell time-frequency plane
$S_d(\tau, f)$ instead of the 1-D broadband analytic signal:

$$
c(\tau, f) = \frac{1}{N_{\text{days}}}\sum_{d:\,n_{w,d}>0}
\frac{S_d(\tau, f)}{\bigl|S_d(\tau, f)\bigr|},
\qquad
\mathcal{S}[S_{\text{tf-pws}}](\tau, f) = |c(\tau, f)|^{\nu}\,\mathcal{S}[S_{\text{lin}}](\tau, f).
$$

The weighted time-frequency spectrum is inverted back to the time domain using
the exact Stockwell (1996) inverse transform (time summation per frequency bin
followed by $\operatorname{Re}[\text{IFFT}]$) and trimmed to $\pm\text{max\_lag\_s}$.
Optional frequency bounds (`tf_pws_freq_min_hz`, `tf_pws_freq_max_hz`) restrict
the S-transform to the band of interest; in `run_stack_config`, pairs are
processed in memory-bounded chunks (`tf_pws_pair_chunk_size`, assumed default:
`64`, automatically capped to a 1 GiB working-set budget per worker) and can
run across multiple CPU processes (`stack.max_workers` / `--max-workers`) using
lazy HDF5 pair-slice reads, or on an NVIDIA GPU (`stack.backend: cupy` /
`--backend cupy`, optionally with `tf_pws_dtype: complex64` for 2× smaller
memory footprint and peak FP32 GPU throughput). To match ANCC's `TF_PWS`
defaults, set `pws_power: 1.0` and `day_weighting: equal`.

**Selective SAC export.** `seisforge ant export-sac` is a standalone
NetCDF4/HDF5 → SAC post-processing converter that filters pairs by station
list, distance range, and minimum window count. Output defaults to
`--layout by-source` (`{source}/{source}_{receiver}_{component}.sac`) to avoid
huge flat directories on large arrays, with `--layout flat` and
`--filename-template` overrides. Standard SAC headers are written:
`b`, `e`, `delta`, `dist`, `az`, `baz`, `evla`, `evlo`, `stla`, `stlo`,
`knetwk` (receiver network), `kstnm` (receiver station), `kevnm`
(`NET.STA` virtual source), and `kcmpnm`.

## 4. Assumed defaults

<div class="assumptions" markdown>

| parameter | default | note |
|---|---|---|
| `grid_tolerance_samples` | `1e-3` | assumed sample-alignment tolerance |
| `taper_fraction` | `0.05` | assumed Tukey taper fraction per window end |
| `whiten.water_level` | `1e-6` | assumed relative denominator floor |
| `ram` / `pcc2` / `pws` `water_level` | `1e-12` | assumed zero-amplitude guard |
| `block_size` | `64` | assumed source-station GEMM block size |
| `tf_pws_pair_chunk_size` | `64` | assumed pair chunk size for streaming `tf_pws` |
| `compression_level` | `4` | assumed NetCDF4 gzip level |

</div>

## 5. Verification

!!! verified "Verified"

    - `tests/test_ant_ccf_core.py`: checks lag sign against
      `np.correlate(u_B, u_A, "full")`, known-delay impulse recovery,
      wrap-around-free guard bands, whitening passband flatness and zero
      padding leakage, continuous RAM mask dilation, blocked upper-triangle
      GEMM across block sizes against an independent `float64` brute-force
      reference (`1e-11` for `complex128`, `1e-5` for `complex64`), two-real-channel
      PCC2 against direct complex Schimmel (1999), reverse-pair symmetry,
      unequal-window and zero-count linear stacking, and batch/streaming PWS
      equivalence.
    - `tests/test_ant_ccf_io.py` and `tests/test_ant_ccf_driver.py`: check
      YAML unknown-key and passband validation, mixed `response_state`
      rejection, synthetic cross-midnight SDS reading and grid checks,
      anti-alias `target_sampling_rate_hz` downsampling, atomic NetCDF4
      round-trips, triple-hash resume, CLI `stack-ccf` (including parallel
      `tf_pws`), and SAC export headers (`knetwk`, `kstnm`, `kevnm`) and
      layouts (`by-source`, `flat`).
    - Real-data end-to-end validation on a 6-station SEIS-S20 nodal subset
      (15 pairs, 9.8–130.3 km, 15 UTC days): `50 Hz -> 10 Hz` downsampling
      matched 50 Hz CCFs to `r = 1.0000` (0.1–0.5 Hz); `cc` vs `pcc2`
      matched to `r = 0.998–0.999`; linear and 2-D `tf_pws` stacks matched
      full-deployment ANCC v6.8 references with zero lag shift
      (`median |dt| = 0.00 s`, `median r = 0.92–0.93`, up to `r = 0.98` at
      130 km).

!!! unverified "Not yet verified"

    - Multi-process `max_workers > 1` and optional `backend: cupy` GPU
      execution have not yet been benchmarked on real large-N (300–500
      station) deployments.
    - Formal amplitude / relative-L2 benchmark against ANCC and NoisePy on
      identical day windows remains pending.

## 6. Usage

```bash
seisforge ant ccf -c ccf.yml                          # all configured dates
seisforge ant ccf -c ccf.yml --date 2026-01-01        # single UTC day (HPC job array)
seisforge ant stack-ccf -c ccf_stack.yml --max-workers 4
seisforge ant export-sac -i stacked_ZZ.nc -o sac_ZZ --min-distance-km 5 --layout by-source
```

Minimal `ccf.yml`:

```yaml
input:
  sds_root: ./SDS
  station_table: ./ccf_stations.csv
  sampling_rate_hz: 20.0
  response_state: velocity
processing:
  cc_type: cc
  window_length_s: 500.0
  max_lag_s: 60.0
  lag_guard_s: 20.0
  ram_half_width_s: 10.0
  whiten: {taper_min_hz: 0.1, freq_min_hz: 0.2, freq_max_hz: 5.0, taper_max_hz: 6.0}
output:
  dir: ./daily_ccf
run:
  start_date: "2026-01-01"
  end_date: "2026-01-03"
```

Runnable example and full annotated configs: `examples/07_ccf/` (`ccf.yml`,
`ccf_stack.yml`, `stations.csv`, `run.sh`).

## References

- Bensen, G. D., Ritzwoller, M. H., Barmin, M. P., Levshin, A. L., Lin, F.,
  Moschetti, M. P., Shapiro, N. M., & Yang, Y. (2007). Processing seismic
  ambient noise data to obtain reliable broad-band surface wave dispersion
  measurements. *Geophysical Journal International*, 169(3), 1239–1260.
  [doi:10.1111/j.1365-246X.2007.03374.x](https://doi.org/10.1111/j.1365-246X.2007.03374.x)
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
