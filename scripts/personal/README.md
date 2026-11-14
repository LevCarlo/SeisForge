# Personal processing scripts (backup)

These are the author's own processing scripts, kept here as a backup. They are
**not** part of the `seisforge` package and are **not** examples.

- Paths (data directories, cluster home directories, Slurm settings) are those
  of the author's machines and are left unchanged.
- They are not guaranteed to run anywhere else, and may drift out of step with
  the package API.
- They are not covered by the test suite or by `ruff` (CI lints only `src` and
  `tests`).

If a procedure here proves reusable, turn it into a documented example under
`examples/` or a maintained helper under `tools/` instead of building on these
files.

| File | Purpose |
|---|---|
| `request_earthquake_data.py` | download earthquake waveforms (ObsPy mass downloader) |
| `retrieve_noise_data.py` | download continuous ambient-noise data |
| `preprocess_p.py` | preprocess teleseismic P-wave event data to SAC (uses `utils.py`) |
| `utils.py` | stream QC and SNR helpers for `preprocess_p.py` |
| `run_disp_wide_array.sbatch` | Slurm array job: one `seisforge inv` dispersion run per station directory |
