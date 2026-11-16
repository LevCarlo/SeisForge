# AFTAN benchmark: SeisForge against Fortran AFTAN and PyAFTAN

**Validation material, not a tutorial.** For a runnable AFTAN example see
`examples/02_aftan`. This folder compares SeisForge's AFTAN with the Fortran
AFTAN (f77) and PyAFTAN results on one real cross-correlation. The files are
kept unchanged from the earlier `examples/AFTAN/`.

| file | what it is |
|---|---|
| `CCF/M07A/M07A_M15A/COR_M07A_M15A.SAC` | ZZ (LHZ-LHZ) cross-correlation, virtual source M07A, receiver M15A, 561.9 km, 1 Hz, lags -3000 to 3000 s |
| `FTAN_f77/` | Fortran AFTAN result (`FTAN/TA_M07A_TA_M15A.npz`) and its figure |
| `FTAN_python/` | PyAFTAN result and its figure |
| `FTAN_seisforge/` | SeisForge basic-FTAN dataset and figure from `aftan_seisforge.yml` |
| `aftan_seisforge.yml` | the SeisForge settings (stack branch, 1-60 s, PMF) |
| `compare.ipynb` | the comparison plots |

The notebook also reads `FTAN_seisforge/M07A/M07A_M15A/COR_M07A_M15A.stack.pmf.dat`,
which is not stored: from this folder run

```bash
seisforge ant aftan -c aftan_seisforge.yml
```

first. Note that `aftan_seisforge.yml` sets `io.overwrite: true`, so this
replaces the stored SeisForge outputs.

## Data source

Cross-correlation of continuous data from USArray Transportable Array
stations TA.M07A and TA.M15A (network code TA).

- Network: USArray Transportable Array (EarthScope_TA), operated by the IRIS
  Transportable Array.
- DOI: [10.7914/SN/TA](https://doi.org/10.7914/SN/TA), as listed on the FDSN
  network page <https://www.fdsn.org/networks/detail/TA/> (checked 2026-11-15).
- Acknowledgement requested there: "EarthScope Transportable Array is
  sponsored by the National Science Foundation and operated by the
  Incorporated Research Institutions for Seismology (IRIS)." Cite the DOI
  above when you use these data.

Time span of the correlated data and processing parameters (pre-processing,
window length, normalization, stacking): **not recorded**. The SAC header
carries only a placeholder reference time (2000-01-01) and `user0 = 14`,
`user2 = 69.8`, whose meaning is not recorded either.
