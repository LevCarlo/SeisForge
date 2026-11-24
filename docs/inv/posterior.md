# Posterior Ensembles and Convergence

!!! overview "Method at a glance"

    The result of a run is a set of stored draws $\theta^{(j)}$
    from several chains. Four things are read from it:

    1. **Vs ensemble.** Each draw is turned into Vs(z); depth-wise
       statistics summarize the ensemble (§1).
    2. **Predicted data.** Each draw is forwarded again and compared with
       the data (§2).
    3. **Convergence.** Split R-hat, effective sample size and lagging
       chains test whether the draws represent the posterior (§3).
    4. **Forward failures.** The fraction of models inside the prior that
       the solver rejected (§4).

    A posterior means something only next to its prior: where both are
    equally wide, the data say nothing.

## 1. What the Vs ensemble shows

The stored draws give profiles $V_s^{(j)}(z)$ on a depth grid (spacing
`discretization.dz`). At each depth the ensemble has a mean, a standard
deviation and quantiles (1, 5, 16, 50, 84, 95, 99 %). These are
**marginal** statistics, one depth at a time. A mean or median profile is
not a member of the ensemble and need not fit the data; a 5–95 % band does
not say which profiles inside it are allowed, because neighbouring depths
are correlated. The draw with the largest posterior density (`best_vs`)
is one model, but a single sample is a noisy estimate of the mode.

![Depth-wise statistics](img/quantile_profiles.svg#only-light)
![Depth-wise statistics](img/quantile_profiles-dark.svg#only-dark)

*Ensemble with a known answer: 2.0 km/s over 3.0 km/s, the interface depth
uniform on 4–6 km. (a) Every member has a step. (b) The mean is a linear
ramp (exact: $2 + (z-4)/2$; largest deviation 0.018 km/s for 2000 members)
that no member has; the $q$-quantile is a step at $6 - 2q$ km (found at
5.91, 5.02, 4.09 km for $q$ = 0.05, 0.5, 0.95). Script:
`docs/inv/scripts/quantile_profiles.py`.*

**Resolution.** Comparing the posterior band with the prior band of a
prior-only run, depth by depth, shows where the data constrain Vs. Where
the two are equally wide, the posterior repeats the prior; this happens
below the depths that the longest periods sense, and wherever a trade-off
lets the data be fitted equally well by different profiles. A posterior
that piles up at a parameter bound shows the bound, not the Earth.

## 2. Predicted data

Every stored draw is forwarded once more, and the predictions are stored
with their quantiles next to the observations. These are predictions
without added noise, the posterior of $g(\theta)$. The data
should lie within the prediction band given their error bars. A systematic
offset over several periods means the model space cannot fit the data (too
narrow bounds or constraints, too few coefficients, a wrong observable)
rather than a sampling problem.

## 3. Convergence

Chains started at different points that sample the same distribution are
the evidence that the stored draws represent the posterior. Three
diagnostics, per parameter, need at least two chains and four draws per
chain.

**Split R-hat** (Gelman et al. 2013). Each of the $m$ chains is cut in
half, giving $2m$ sequences of $n$ draws; $W$ is the mean within-sequence
variance and $B/n$ the variance of the sequence means:

$$
\boxed{\;\hat R = \sqrt{\frac{\frac{n-1}{n}W + \frac{B}{n}}{W}}\;}
$$

$\hat R \to 1$ when all sequences sample the same distribution; splitting
also catches a trend within one chain. The log warns above 1.1 (assumed;
1.01 is the usual level for final results, Vehtari et al. 2021).

**Effective sample size.** Correlated draws are worth fewer independent
ones. With the multi-chain autocorrelation
$\hat\rho_t = 1 - (W - \overline{\gamma_t})/\widehat{\operatorname{var}}^+$
($\overline{\gamma_t}$ the autocovariance at lag $t$ averaged over chains),

$$
\boxed{\;\mathrm{ESS} = \frac{mn}{\hat\tau},\qquad
\hat\tau = -1 + 2\sum_{k=0}^{K} \big(\hat\rho_{2k} + \hat\rho_{2k+1}\big)\;}
$$

where the sum stops before the first negative pair and the pairs are made
non-increasing (Geyer's initial monotone sequence, Geyer 1992), as in
Vehtari et al. (2021) but without rank normalization. For an AR(1) chain
with lag-1 correlation $\phi$, $\hat\tau \to (1+\phi)/(1-\phi)$. The log
warns below 100 (assumed); percentiles of the Vs ensemble need several
hundred.

**Lagging chains.** A chain whose log posterior lies wholly below that of
the best chain (highest median), its 95th percentile under the best
chain's 5th (assumed percentiles), is stuck in a lower mode. It is
reported, never dropped: deciding that a mode carries no posterior mass
needs more chains or [tempering](sampler.md#4-parallel-tempering).

![Convergence diagnostics](img/convergence_diagnostics.svg#only-light)
![Convergence diagnostics](img/convergence_diagnostics-dark.svg#only-dark)

*(a) Four AR(1) chains of 4000 draws: the package ESS against
$mn(1-\phi)/(1+\phi)$, within 11 % for $\phi$ = 0–0.98. (b) Four chains of
independent draws, one shifted by $\delta$: $\hat R$ = 1.001 at
$\delta = 0$, passes 1.1 near $\delta = 1$; the shifted chain is flagged as
lagging from $\delta \approx 3.75$ (filled). Script:
`docs/inv/scripts/convergence_diagnostics.py`.*

**Acceptance.** The log prints each chain's acceptance after burn-in,
where the proposal is fixed, with a note below 0.1 or above 0.6 (assumed
bounds around the optimum of about 0.23; Roberts et al. 1997).

## 4. Forward failures

A forward attempt is an evaluation inside the hard prior (the starting
point and every proposal that passed the prior); a failure is an attempt
whose solver found no solution (forward page,
[§2](forward.md#2-forward-observables)). Their ratio is recorded per chain;
with tempering, over all replicas and for the stored $T = 1$ replica alone.
A failed model is rejected, so a high rate means the posterior is the prior
truncated by the solver. The log warns above 1 % (assumed) and names the
two most frequent errors. Remedies are constraints that exclude the
failing models (for example Vs increasing into the half-space) or dropping
the periods or modes that fail.

## 5. A whole inversion with a known answer

![Synthetic recovery](img/recovery.svg#only-light)
![Synthetic recovery](img/recovery-dark.svg#only-dark)

*Three parameters (Vs at 0, 1 and 3 km, linear in between), phase velocity
and ellipticity at 0.5–5 s with Gaussian noise, three adaptive chains of
1200 steps. (a) The truth lies inside the posterior 5–95 % band at 98 % of
the depths; the band is on average 0.09 of the prior width; split R-hat
1.010–1.013, ESS 202–224. (b, c) Predicted data against the data. The truth
uses the inversion's own parameterization and layering, so this checks the
chain parameterization → forward → likelihood → sampler → ensemble, not how
well the parameterization represents real structure. Script:
`docs/inv/scripts/recovery.py`.*

!!! verified "Verified"

    R-hat of independent chains, ESS of AR(1) chains against the analytic
    value, a chain in another mode raising R-hat and being flagged
    (`tests/test_inv_diagnostics.py`); Vs ensembles and their statistics
    (`tests/test_inv_ensemble.py`); predicted-data ensembles
    (`tests/test_inv_predictive.py`); forward-failure counts
    (`tests/test_inv_forward_failures.py`); synthetic recovery of a known
    truth (`tests/test_inv_recovery.py`, the case of the figure). Recovery
    is tested on synthetic data only.

!!! unverified "Not verified"

    Passing diagnostics are necessary, not sufficient: chains that all miss
    the same mode pass them. Recovery with structure the parameterization
    cannot represent, and with real data, has not been tested.

## 6. Usage

```bash
seisforge inv disp-hv --inv inv.yaml --obs obs.yaml -o output
```

| file | content |
|---|---|
| `disp_hv_mcmc.nc` | `theta(chain, draw, parameter)`, `log_prob`, the prior and likelihood terms, `accepted`; `split_rhat`, `effective_sample_size`, attr `lagging_chains`; `forward_failure_rate`; with tempering `swap_acceptance`; with adaptation `proposal_covariance` |
| `disp_hv_vs.nc` | Vs(z) of every draw, `mean`, `std`, `p01`…`p99`, `median`, `best_vs` |
| `disp_hv_predictive.nc` | predicted data of every draw, their quantiles, the observations |
| `disp_hv.log` | settings, layering error, acceptance, diagnostics, warnings |
| `resolved_inv.yaml`, `resolved_obs.yaml` | snapshots of the inputs |

Prefixes `disp` and `hv` replace `disp_hv` for single data types; a
prior-only run writes `*_prior_mcmc.nc` and `*_prior_vs.nc`. A run with a
known answer and a plotting script is `examples/06_inversion`.

## References

- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., &
  Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.), §11.4. CRC
  Press.
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical
  Science*, 7(4), 473–483.
  [doi:10.1214/ss/1177011137](https://doi.org/10.1214/ss/1177011137)
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and
  optimal scaling of random walk Metropolis algorithms. *Annals of Applied
  Probability*, 7(1), 110–120.
  [doi:10.1214/aoap/1034625254](https://doi.org/10.1214/aoap/1034625254)
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
  (2021). Rank-normalization, folding, and localization: an improved
  $\widehat R$ for assessing convergence of MCMC. *Bayesian Analysis*,
  16(2), 667–718. [doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)
