# Sampling the Posterior

!!! overview "Method at a glance"

    The posterior $p(\theta\mid\mathbf d) \propto
    p(\theta)\,L(\theta)$ is sampled by random-walk
    Metropolis, with several independent chains:

    1. **Step.** Propose $\theta' = \theta +
       \epsilon$, $\epsilon \sim N(0, C)$; accept with
       probability $\min(1, p(\theta'\mid\mathbf d)/p(\theta\mid\mathbf d))$
       (§1).
    2. **Escape.** Optionally, every $N$-th step proposes a point uniformly
       in the parameter box (§2).
    3. **Adapt.** Optionally, $C$ is learned from the chain during burn-in
       and then frozen (§3).
    4. **Temper.** Optionally, each chain runs replicas at higher
       temperatures that exchange states, to cross between separated modes
       (§4).

    Only the stored draws after burn-in are used; how to judge them is on
    the [posterior page](posterior.md).

## 1. The Metropolis step

With a symmetric proposal, accepting $\theta'$ with probability

$$
\boxed{\;\alpha = \min\!\left(1,\ \frac{p(\theta')\,L(\theta')}{p(\theta)\,L(\theta)}\right)\;}
\label{eq:metropolis}
$$

leaves the posterior invariant (Metropolis et al. 1953): the normalizing
constant cancels, so the unnormalized prior and likelihood suffice. A
rejected step repeats the current state. $C$ starts diagonal with
`proposal.sigma` per parameter (default 5 % of its bound range). A proposal
outside the bounds, outside the hard prior, or with a failed forward
calculation has zero posterior density and is rejected; only proposals
inside the prior reach the solver.

The first `burn_in` steps are discarded, and every `thin`-th step after
them is stored. Each chain draws from its own generator, seeded from
`seed` by `numpy.random.SeedSequence.spawn`, so chains are independent and
the result does not depend on whether they run in sequence, in threads or
in processes.

Steps that are too small are almost always accepted but move slowly;
steps that are too large are almost always rejected. For a Gaussian target
in many dimensions the optimal acceptance is about 0.23 (Roberts et al.
1997).

## 2. Global jumps

Every `global_jump_interval`-th step proposes $\theta'$
uniformly in the parameter box instead. This independence proposal has a
constant density $q$, so its acceptance ratio
$p(\theta')q/[p(\theta)q]$ is eq.
$\eqref{eq:metropolis}$ again: the step leaves the posterior invariant and
lets a chain leave a local region in one move. It is rarely accepted once
the posterior is much narrower than the box, so it costs little and helps
mostly early.

## 3. Adaptive proposals

Parameters that trade off against each other, such as neighbouring
B-spline coefficients, make the posterior a narrow tilted ridge. A
diagonal proposal must then take steps as small as the ridge is narrow,
and the chain crawls along it. The adaptive Metropolis algorithm (Haario et
al. 2001) estimates the shape from the chain itself. With $d$ parameters
and starting sigmas $\sigma$, after $t_0$ =
`initial_diagonal_steps` steps and then every `update_interval_steps`,

$$
\boxed{\;C = \frac{2.38^2}{d}\Big(\operatorname{Cov}(\theta_k,\dots,\theta_n)
+ \varepsilon\,\operatorname{diag}(\sigma^2)\Big),
\qquad k = \lfloor f\,(n+1)\rfloor\;}
\label{eq:adaptive}
$$

- $2.38^2/d$ is the scaling that is optimal for a Gaussian target (Gelman
  et al. 1996); the proposal then has the target's shape.
- $\varepsilon$ = `regularization_fraction` ($10^{-4}$, assumed) keeps $C$
  positive definite when a parameter has not moved, without distorting
  sampled directions. It is relative to each parameter's own starting
  variance, so parameters with different units are treated alike.
- $f$ = `discard_early_history_fraction` (default 0, the whole history)
  leaves the earliest fraction of the states out. A chain that starts far
  away first drifts towards the posterior; that drift inflates the
  covariance along its direction, and the learned steps come out too wide.
  With $f = 0.5$ only the most recent half is used.
- **Adaptation stops at the end of burn-in.** $C$ is then fixed, and the
  stored draws come from an ordinary Metropolis chain with a fixed
  symmetric proposal, which leaves the posterior exactly invariant. No
  argument about continuously adapted chains is needed.

![Adaptive Metropolis](img/adaptive_metropolis.svg#only-light)
![Adaptive Metropolis](img/adaptive_metropolis-dark.svg#only-dark)

*2-D Gaussian with correlation 0.98, four chains of 6000 steps (burn-in
3000), started at the cross with diagonal sigma 0.25. (a) Stored draws;
the frozen proposal (chain 0, red) is within 10 % of
$(2.38^2/d)\,\Sigma$ (dotted). (b) After burn-in the diagonal chain still
crawls (ESS 46, split R-hat 1.06 over 12 000 draws); the adaptive chain
mixes (ESS 1540, R-hat 1.00, acceptance 0.33–0.37). Script:
`docs/inv/scripts/adaptive_metropolis.py`.*

The window at the first update must hold more than $d$ states, and
`burn_in` must reach at least $t_0$. The log warns (assumed thresholds)
when fewer steps are adapted than run diagonally, or the last window holds
fewer than $100d$ states.

## 4. Parallel tempering

A chain stuck in one of several separated modes cannot leave by small
steps. With tempering, each chain is a ladder of replicas at temperatures
$1 = T_1 < \dots < T_n$, replica $i$ sampling
$p(\theta)\,L(\theta)^{1/T_i}$ (Geyer 1991). Hot
replicas see a flattened likelihood and move between modes. Every
`swap_interval` steps, neighbours $i, i+1$ (even and odd pairs in turn)
exchange states with probability

$$
\boxed{\;\alpha_{\text{swap}} = \min\!\Big(1,\ \exp\big[(1/T_i - 1/T_{i+1})(\ln L_{i+1} - \ln L_i)\big]\Big)\;}
\label{eq:swap}
$$

which keeps the product of the tempered posteriors invariant
(Appendix A). The prior cancels, so it is not tempered. Only the $T = 1$
replica is stored; it samples the posterior. The default ladder is
geometric, $T_i = T_{\max}^{(i-1)/(n-1)}$, with 5 temperatures up to 30.
Replica $i$ proposes with $\sigma\sqrt{T_i}$ (an assumed
scaling: tempering a Gaussian likelihood by $1/T$ widens it by
$\sqrt T$) and, with adaptation, learns its own $C$.

![Parallel tempering](img/tempering_modes.svg#only-light)
![Parallel tempering](img/tempering_modes-dark.svg#only-dark)

*Two modes of weight 0.3 and 0.7, separated by a drop of about 128 in
$\ln L$; four chains started in the smaller mode. (a) A random walk never
leaves it. (b) A ladder of six temperatures up to 200 puts 0.68 of the
draws in the larger mode (exact 0.70); neighbouring replicas swap in 64–84 %
of attempts. Script: `docs/inv/scripts/tempering_modes.py`.*

A ladder of $n$ replicas costs about $n$ plain runs. Neighbouring
temperatures must be close enough that swaps are accepted (the
`swap_acceptance` output; a pair near 0 needs a temperature in between),
and $T_{\max}$ must flatten the barrier: roughly, comparable to the drop of
$\ln L$ between the modes. Tempering helps with separated modes, not with
a correlated ridge (§3). A prior-only run ignores it.

## 5. Starting points

`initial_strategy: theta0` starts every chain at the `initial` values;
`prior_uniform` draws starting points uniformly in the box and keeps those
with finite posterior density; `initial_thetas` gives them explicitly.
Starting points spread over the prior are what makes the between-chain
comparison of the [convergence diagnostics](posterior.md#3-convergence)
meaningful. When the hard constraints leave little of the box (10 % in the
[example prior](model.md#4-the-prior)), `prior_uniform` needs many draws
(`max_initial_attempts`, default 10 000).

!!! verified "Verified"

    Burn-in, thinning, bounds, global-jump proposals, multi-chain and
    thread execution, prior-only sampling (`tests/test_inv_samplers.py`);
    adaptation: running moments against `numpy.cov`, eq.
    $\eqref{eq:adaptive}$, freezing after burn-in, identical draws before the
    first update, the windowed history, recovery of a target covariance from
    a distant start, executor independence
    (`tests/test_inv_adaptive_proposal.py`); tempering recovers the weights
    of separated modes (`tests/test_inv_tempering.py`); bit-for-bit
    regression of the sampler against a stored baseline
    (`tests/test_inv_sampler_baseline.py`). Checks of correctness are on
    analytic targets; sampling efficiency on real posteriors is not
    guaranteed by them.

!!! unverified "Not verified"

    The $\sqrt{T}$ proposal scaling and the default ladder are assumed, not
    tuned on inversion posteriors. No sampler can show that a mode was
    never visited; independent chains and tempering make it less likely.

## 6. Usage

```yaml
sampler:
  n_steps: 16000
  burn_in: 5000
  thin: 10
  n_chains: 4
  executor: process          # serial | thread | process
  max_workers: 4
  seed: 20260702
  initial_strategy: prior_uniform
  global_jump_interval: 1000
  adaptation: {initial_diagonal_steps: 1500, update_interval_steps: 250,
               discard_early_history_fraction: 0.5}
  # tempering: {n_temperatures: 5, max_temperature: 30, swap_interval: 1}
inversion:
  parameters:
    - {name: c0, initial: 3.05, prior: {bounds: [2.5, 3.55]}, proposal: {sigma: 0.035}}
```

`--prior-only` samples $p(\theta)$ alone (no data, no solver);
`--with-prior` runs it before the posterior. A complete configuration is
`examples/reference_configs/inversion_inv.yaml`.

## Appendix A: the swap acceptance

Replica $i$ targets $\pi_i(\theta) \propto
p(\theta) L(\theta)^{\beta_i}$, $\beta_i = 1/T_i$,
and the joint target is $\prod_i \pi_i(\theta_i)$. Exchanging
$\theta_i$ and $\theta_{i+1}$ is a symmetric
proposal, so Metropolis accepts it with the ratio of joint densities,

$$
\frac{\pi_i(\theta_{i+1})\,\pi_{i+1}(\theta_i)}
{\pi_i(\theta_i)\,\pi_{i+1}(\theta_{i+1})}
= \exp\big[(\beta_i - \beta_{i+1})(\ln L_{i+1} - \ln L_i)\big],
$$

in which the priors cancel: eq. $\eqref{eq:swap}$. Within a replica the
step of §1 accepts with
$\Delta\ln p + \beta_i\,\Delta\ln L$. Every move leaves the joint target
invariant, so its first marginal, $\pi_1$, is the posterior (Earl & Deem
2005).

## References

- Earl, D. J., & Deem, M. W. (2005). Parallel tempering: theory,
  applications, and new perspectives. *Physical Chemistry Chemical
  Physics*, 7, 3910–3916.
  [doi:10.1039/B509983H](https://doi.org/10.1039/B509983H)
- Gelman, A., Roberts, G. O., & Gilks, W. R. (1996). Efficient Metropolis
  jumping rules. In J. M. Bernardo et al. (Eds.), *Bayesian Statistics 5*,
  599–607. Oxford University Press.
- Geyer, C. J. (1991). Markov chain Monte Carlo maximum likelihood. In
  *Computing Science and Statistics: Proceedings of the 23rd Symposium on
  the Interface*, 156–163.
- Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis
  algorithm. *Bernoulli*, 7(2), 223–242.
  [doi:10.2307/3318737](https://doi.org/10.2307/3318737)
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &
  Teller, E. (1953). Equation of state calculations by fast computing
  machines. *Journal of Chemical Physics*, 21(6), 1087–1092.
  [doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and
  optimal scaling of random walk Metropolis algorithms. *Annals of Applied
  Probability*, 7(1), 110–120.
  [doi:10.1214/aoap/1034625254](https://doi.org/10.1214/aoap/1034625254)
