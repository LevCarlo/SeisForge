# Receiver-Function Tools: Overview

`seisforge.rf` provides three independent tools for teleseismic P receiver
functions (RFs). SeisForge does not compute RFs from earthquake records. QC
and H-k read RFs made by other software, and the synthetic RFs can be used
anywhere.

| tool | page | input | output |
|---|---|---|---|
| RF quality control (`rf qc`) | [Quality control](qc.md) | radial RFs of one station, two frequency bands | copies of the RFs that pass six tests |
| H-k stacking (`rf hkseq`) | [H-k stacking](hk.md) | radial RFs with ray parameters, assumed $V_P$ (or $V_S$) | thickness $H$ and $\kappa = V_P/V_S$ of one or two layers, with a robust spread |
| Synthetic P RFs (Python) | [Synthetic P receiver functions](synthetic.md) | layered model ($h$, $V_P$, $V_S$, $\rho$), ray parameter $p$, Gaussian $a$ | radial RF vs time |

QC output is a valid H-k input, but either tool takes RFs from elsewhere.
`seisforge.inv.predict_receiver_function` calls the synthetic RF to invert
RFs for $V_S(z)$.

## Conventions shared by the tools

- **Polarity.** Radial $R$ is positive away from the source and vertical $Z$
  positive up. The direct P pulse is positive. For a velocity increase with
  depth, Ps and PpPs are positive and PpSs + PsPs is negative.
- **Time.** Time is measured from direct P, which is at 0 s.
- **Gaussian.** `gauss_a` $= a$ in $e^{-\omega^2/4a^2}$, as in iterative
  time-domain deconvolution (Ligorría & Ammon 1999). The pulse has width
  $\sim 1/a$ ([synthetic RFs](synthetic.md#3-amplitude-and-time-conventions)).
  "High" and "low" frequency in the QC and H-k configs are only the names of
  two input folders, typically $a = 5$ and $a = 2.5$.
- **SAC headers read.** `b` (time of the first sample relative to direct P),
  `delta`, `user0` = ray parameter in s/km, `user2` = signal-to-noise ratio
  (both header names configurable in QC). QC requires one time axis
  (`b`, `delta`, `npts`) for all traces of a band and stops otherwise; H-k
  reads each trace's own.
- **Ray parameter.** $p$ in s/km; teleseismic P has $p \approx$ 0.04–0.08 s/km.

## Checks shared by all pages

Every figure on these pages is made by a script in `docs/rf/scripts/` with
the same name as the image, from package functions and synthetic data with
a known answer, and runs in seconds:

```bash
PYTHONPATH=src python docs/rf/scripts/<figure name>.py
```

`hk_derivation.py` and `synthetic_derivation.py` derive the boxed formulas
with SymPy, check them against the package, and write them to
`docs/rf/generated/`, from where the pages include them.

## References

- Ligorría, J. P., & Ammon, C. J. (1999). Iterative deconvolution and
  receiver-function estimation. *Bulletin of the Seismological Society of
  America*, 89(5), 1395–1400.
  [doi:10.1785/BSSA0890051395](https://doi.org/10.1785/BSSA0890051395)
