# Receiver-Function Quality Control

!!! overview "Method at a glance"

    A station's RFs include traces spoiled by noise, by a wrong
    deconvolution, or by events outside the useful distance range. Each
    test below removes one kind. The tests run in a fixed order, and each
    one sees only the traces that passed the previous ones:

    1. **Header tests.** Keep $\mathrm{SNR} \ge s_{\min}$ and
       $p_{\min} \le p \le p_{\max}$ (§1).
    2. **Direct P.** The largest pulse must be positive and close to 0 s (§2).
    3. **Variance outliers.** Robust z-score of each trace's variance (§3).
    4. **F-test.** Tests whether a trace departs from the mean of the others
       by more than their own scatter allows; gives a p-value (§4).
    5. **Coherence.** Correlation of each trace with the mean trace (§5).

    The variance, F and coherence tests compare a trace with the others, so
    their result depends on which traces are still in the set (§6). Each
    test can be switched off. The two frequency bands are screened
    separately.

![QC on synthetic RFs with known labels](img/qc_synthetic.svg#only-light)
![QC on synthetic RFs with known labels](img/qc_synthetic-dark.svg#only-dark)

*(a) 30 good synthetic RFs (35 km crust, $a$ = 2.5, 3 % noise) and 12 bad
ones: 3 each with SNR header 2, $p$ = 0.09 s/km, reversed polarity, and 60 %
noise. With the settings of `examples/04_rf_qc_hk` (F-test off), exactly
the 12 bad traces are removed. Two noisy traces fail the direct-P test
because a noise peak exceeds direct P. (b), (c) MAD and CC metrics of the
traces that enter those tests. (d) Cumulative F-test p-values (§4). For 30
noisy copies of one RF at one $p$, where the null hypothesis holds, they
are uniform (1 of 30 below 0.05). For the 30 good traces, whose $p$ spans
0.045–0.078 s/km, 9 fall below 0.05: their Ps and multiples move by up to
about 1 s, which 3 % noise cannot hide. Script:
`docs/rf/scripts/qc_synthetic.py`.*

## 1. Header tests

SNR and $p$ are read from SAC headers (`user2`, `user0` by default) as the
RF program wrote them. A trace is kept if

$$
\mathrm{SNR} \ge s_{\min},\qquad p_{\min} \le p \le p_{\max} .
$$

The $p$ range selects epicentral distances, roughly 30°–90° for
0.04–0.08 s/km. Closer events have triplicated P, and farther events
arrive near grazing incidence or as diffracted P.

## 2. Direct P

In a window $W$ (default $[-5, 30]$ s), let $t^\ast$ be the time of the
largest $|r(t)|$. The trace is kept if

$$
t_{\min} \le t^\ast \le t_{\max}\quad\text{and}\quad r(t^\ast) > 0 .
$$

In a radial P RF, direct P is the largest pulse and is positive
([conventions](index.md#conventions-shared-by-the-tools)). The test removes
reversed traces, traces whose largest peak is noise, and RFs whose direct P
is shifted far from 0 s. Over slow sediment, a sediment reverberation can
exceed direct P (panel a of the [two-layer figure](hk.md#5-sediment-then-crust)),
and the test removes good traces. Such stations need a wider
$[t_{\min}, t_{\max}]$ or no P test.

## 3. Variance outliers

Let $v_i$ be the variance of trace $i$ in $W$, $\tilde v$ the median of the
$v_i$, and $\mathrm{MAD} = \operatorname{median}_i |v_i - \tilde v|$. A
trace is kept if

$$
z_i = \frac{|v_i - \tilde v|}{1.4826\,\mathrm{MAD}} \le z_{\max}.
$$

The factor $1.4826 = 1/\Phi^{-1}(3/4)$ makes $1.4826\,\mathrm{MAD}$ equal to
the standard deviation for Gaussian data (Rousseeuw & Croux 1993). Then
$z_{\max} = 2.5$ would remove 1.2 % of the traces, two-sided. The $v_i$
are not Gaussian: a variance is positive and skewed, so the fraction is an
order of magnitude, not a probability. The test is two-sided and also
removes traces that are unusually quiet.

## 4. F-test

Let $r_j(t_n)$ be the $N$ traces sampled at $n$ times in $W$. The **null
hypothesis** for trace $i$ is that every trace is one common waveform plus
noise, $r_j = s + e_j$, with the $e_j$ independent, zero-mean, Gaussian,
stationary in time, and of equal variance $\sigma^2$. Let $m_{-i}$ be the
mean of the other $N - 1$ traces,

$$
d_i = r_i - m_{-i},\qquad
E_{-i} = \sum_{j \ne i}\sum_n \big(r_j(t_n) - m_{-i}(t_n)\big)^2 .
$$

Under the null hypothesis $d_i$ has variance $\sigma^2 N/(N-1)$ per sample
and is independent of $E_{-i}$, so

$$
F_i = \frac{\sum_n d_i(t_n)^2 \,\big/\, \big(\nu\,N/(N-1)\big)}{E_{-i} \,\big/\, \big((N-2)\,\nu\big)}
\;\sim\; F\big(\nu,\ (N-2)\,\nu\big),
\qquad
p_i = P\big(F \ge F_i\big),
\label{eq:ftest}
$$

and trace $i$ is kept if $p_i \ge p_{\min}$. The test is one-sided: a trace
closer to the others than expected is never rejected. At level $p_{\min}$
it rejects a fraction $p_{\min}$ of traces that satisfy the null
hypothesis, by design.

**Degrees of freedom.** With independent samples $\nu = n$. RF samples are
correlated over about one pulse width by the Gaussian filter, so $\nu$ is
the effective number (Satterthwaite 1946): a sum of squares of $n$
correlated stationary Gaussian samples is approximated by a scaled
$\chi^2_\nu$ with

$$
\nu = \frac{n^2}{n + 2\sum_{k\ge1}(n-k)\,\rho_k^2},
$$

where $\rho_k$ is the noise autocorrelation at lag $k$. It is estimated from
the residuals of all traces about their mean, pooled, and the sum stops at
the first lag where the estimate is not positive. For white noise through
$e^{-\omega^2/4a^2}$ the autocorrelation is $e^{-a^2\tau^2/2}$ and
$\nu \approx n\,\Delta t\,a/\sqrt\pi$: 49 instead of 351 samples for
$W = [-5, 30]$ s, $a = 2.5$, $\Delta t = 0.1$ s.

**What it does not model.** The null hypothesis needs one common waveform.
RFs over a range of $p$ have Ps and multiples at different times, and
back-azimuth variations from dipping or anisotropic structure change the
waveform too. With low noise such differences are significant, and good
traces get small p-values: 9 of 30 in panel (d), and 5–7 per band in
`examples/04_rf_qc_hk`, which therefore keeps the test off. The test suits
RFs from a narrow $p$ and back-azimuth range, or noisier data. Equal noise
variance across traces is also assumed; a single noisier trace is rejected
even when its signal is right.

Before SeisForge 0.2.0 this test returned
$1 - [F(f_i) - F(1/f_i)]$ with $f_i = (E/\nu)/(E_{-i}/\nu_{-i})$, $E$ the
residual of all traces, and $\nu = N n - 1$ counting every sample as
independent. That value is not a p-value and exceeds 1 when $f_i < 1$.

## 5. Coherence with the mean

With $\bar r = \frac1N\sum_j r_j$ over the traces that enter the test,
trace $i$ is kept if its Pearson correlation in $W$ satisfies

$$
\rho_i = \operatorname{corr}(r_i, \bar r) \ge \rho_{\min}.
$$

The mean includes $r_i$ itself. Let $m$ be the mean of the other traces.
A trace uncorrelated with $m$ still has

$$
\rho_i = \frac{\sigma_i}{\sqrt{\sigma_i^2 + (N-1)^2\,\sigma_m^2}},
$$

where $\sigma_i$ and $\sigma_m$ are the standard deviations of $r_i$ and $m$
in $W$. This is about $1/(N-1)$ for $\sigma_i = \sigma_m$, which matters
only for a few traces (derived here). The mean waveform depends on
back azimuth when the structure is dipping or anisotropic. Traces from an
under-sampled back-azimuth range then correlate poorly although they are
correct.

## 6. Order dependence

The variance, F and coherence tests use the median, the residuals and the
mean of the current set. Removing gross outliers first makes these
statistics represent the good traces. This is why the header and direct-P
tests come first. A set dominated by bad traces still defeats tests 3–5,
because they assume that most traces are good.

All traces must share one time axis: the same `b`, `delta` and `npts`
(relative tolerance $10^{-6}$, assumed for SAC's single precision). The
tests compare traces sample by sample, so QC stops with an error naming
the first trace that differs.

!!! verified "Verified"

    - `tests/test_rf_traces.py`: the header tests keep file names aligned
      with traces, `reverse` returns the complement, the direct-P test needs
      a positive peak in the time range, the F-test keeps fewer than three
      traces, and traces on different time axes are refused.
    - `tests/test_rf_qc_core.py`: under the null hypothesis the F-test
      p-values are uniform (Kolmogorov–Smirnov, 3000 p-values, for white
      noise and for noise through $a$ = 2.5 and 5), the rejection rate at
      0.05 is $0.05 \pm 0.015$, the effective $
u$ matches
      $n\,\Delta t\,a/\sqrt\pi$ within 5 % and is $n$ for white noise, and
      a reversed trace gets $p < 10^{-6}$.
    - `tests/test_rf_qc_workflow.py`: existing outputs stop the run unless
      overwrite is set, which empties the CLEAN folders first; the root
      logger is left alone; the per-band table records every file's
      metrics, fate and the software versions.
    - On the synthetic set of the figure (and of `examples/04_rf_qc_hk`),
      the tests remove exactly the 12 bad traces. This is checked on
      synthetic data only.

!!! unverified "Not verified"

    - The thresholds are not calibrated against a false-rejection rate. The
      values in the configs are choices.
    - The F-test assumes one common waveform and equal noise; it rejects
      good RFs that differ by moveout (§4), so the example config switches
      it off. No moveout correction is applied. The source of the earlier
      form of this test has not been traced.
    - The variance and coherence tests have no unit test beyond the figure
      script.

## 7. Usage

```bash
seisforge rf qc -c rf_qc.yml [--plot] [--overwrite]
```

One run screens both bands of one station. The keys this page refers to:

```yaml
IO:
  ROOT: .                          # other IO paths are relative to ROOT
  RAW_HighFreq_RF: data/Gauss_5.0  # input folders; files matching *<component>*
  RAW_LowFreq_RF: data/Gauss_2.5
  CLEAN_HighFreq_RF: output/Gauss_5.0_CLEAN   # passing RFs are copied here
  CLEAN_LowFreq_RF: output/Gauss_2.5_CLEAN
  FIGURE: output/figures
  LOG: output/logs
  OVERWRITE: false                 # optional; true (or --overwrite) replaces outputs
META: {station: RF01, network: XX, component: R}
QC:
  LowFreq:                         # and HighFreq, same keys
    SNR: {ENABLE: true, THRESHOLD: 3.0, SAC_HEADER: user2}           # §1
    SLOWNESS: {ENABLE: true, MIN: 0.04, MAX: 0.08, SAC_HEADER: user0} # §1
    PAMP: {ENABLE: true, TMIN: -2.0, TMAX: 2.0, WINDOW: [-5, 30]}   # §2
    MAD: {ENABLE: true, THRESHOLD: 2.5, WINDOW: [-5, 30]}            # §3
    F_TEST: {ENABLE: false, THRESHOLD: 0.05, WINDOW: [-5, 30]}       # §4
    CC: {ENABLE: true, THRESHOLD: 0.5, WINDOW: [-5, 30]}             # §5
```

Every key must be present except `IO.OVERWRITE` (false when absent);
unknown keys are refused. The SAC files need `b`, `delta` and `npts`, the
same for all traces of a band (§6). The passing files are copied unchanged
into the `CLEAN_*` folders, which must not overlap the input folders.
`LOG/NET.STA_RF_QC_<band>.dat` lists every input file with its metrics
(`SNR`, `slowness_s_per_km`, `P_time_s`, `P_amp`, `MAD_z`, `F_p`, `CC`;
`nan` where a test was off or not reached) and its fate (`kept` or
`removed:<TEST>`), under a header with the settings and the software
versions. The log `NET.STA_RF_QC.log` records the number of traces after
each test and the software versions. `--plot` writes
`NET.STA_RF_QC_<band>.png`.

Existing outputs (files in a `CLEAN_*` folder matching the component, the
tables, the log, and with `--plot` the figures) stop the run before
anything is written. With `--overwrite` or `IO.OVERWRITE: true` the
matching files in the `CLEAN_*` folders are deleted first, so the folders
hold exactly this run's survivors.

Example with a known answer: `examples/04_rf_qc_hk`; reference config:
`examples/reference_configs/rf_qc.yml`.

## References

- Satterthwaite, F. E. (1946). An approximate distribution of estimates of
  variance components. *Biometrics Bulletin*, 2(6), 110–114.
  [doi:10.2307/3002019](https://doi.org/10.2307/3002019)
- Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute
  deviation. *Journal of the American Statistical Association*, 88(424),
  1273–1283.
  [doi:10.1080/01621459.1993.10476408](https://doi.org/10.1080/01621459.1993.10476408)
