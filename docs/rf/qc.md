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
    4. **F-test.** Compares the residual variance of the set with and
       without each trace (§4).
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
traces that enter those tests. (d) The value the F-test compares with its
threshold, for the 30 good traces only (§4). Script:
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

Let $r_i(t_n)$ be the $N$ traces sampled at $n_W$ times in $W$. Let $E$ be
the summed squared residual of all traces about their mean, and $E_{-i}$
the same sum with trace $i$ left out (about the mean of the other
$N - 1$ traces). The test forms

$$
f_i = \frac{E/\nu}{E_{-i}/\nu_{-i}},\qquad \nu = N n_W - 1,\quad \nu_{-i} = (N-1)\,n_W - 1,
$$

and keeps trace $i$ if

$$
Q_i = 1 - \big[F_{\nu,\nu_{-i}}(f_i) - F_{\nu,\nu_{-i}}(1/f_i)\big] \ge Q_{\min},
$$

with $F_{\nu_1,\nu_2}$ the cumulative F distribution. A trace far from the
others raises $E$ above $E_{-i}$, so $f_i > 1$ and $Q_i$ is small. For
$f_i > 1$ and $\nu \approx \nu_{-i}$, $Q_i$ is the two-sided p-value
$2[1 - F(f_i)]$. For $f_i < 1$, $Q_i$ exceeds 1, which happens for most
good traces in panel (d). The degrees of freedom count every sample as
independent. A Gaussian-filtered RF has far fewer independent values, about
one per pulse width, so $Q_i$ is not calibrated as a probability. In panel
(d) three of 30 traces from one model fall below 0.05.

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

!!! verified "Verified"

    - `tests/test_rf_traces.py`: the header tests keep file names aligned
      with traces, `reverse` returns the complement, the direct-P test needs
      a positive peak in the time range, and the F-test keeps a single trace.
    - On the synthetic set of the figure (and of `examples/04_rf_qc_hk`),
      the tests remove exactly the 12 bad traces. This is checked on
      synthetic data only.

!!! unverified "Not verified"

    - The thresholds are not calibrated against a false-rejection rate. The
      values in the configs are choices.
    - The F-test value is not a p-value (§4). It rejects 3 of 30 good
      synthetic traces at 0.05, and the example config switches it off.
      The source of this test has not been traced.
    - The variance and coherence tests have no unit test beyond the figure
      script.

## 7. Usage

```bash
seisforge rf qc -c rf_qc.yml [--plot]
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

Every key must be present; there are no defaults. The SAC files need `b`,
`e` and `delta`; QC uses the time axis of the first trace for all of them.
The passing files are copied unchanged into the `CLEAN_*` folders, and a
file of the same name there is replaced. The log `NET.STA_RF_QC.log` in
`LOG` records the number of traces after each test and the software
versions. `--plot` writes `NET.STA_RF_QC_<band>.png`.

Example with a known answer: `examples/04_rf_qc_hk`; reference config:
`examples/reference_configs/rf_qc.yml`.

## References

- Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute
  deviation. *Journal of the American Statistical Association*, 88(424),
  1273–1283.
  [doi:10.1080/01621459.1993.10476408](https://doi.org/10.1080/01621459.1993.10476408)
