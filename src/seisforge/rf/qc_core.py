"""Pure statistics for receiver-function quality control.

Leave-one-out F test
--------------------
The traces ``r_j(t_n)``, ``j = 1..N``, are sampled at the same ``n`` times in
the QC window. Null hypothesis for trace ``i``: every trace is a common
signal plus noise, ``r_j = s + e_j``, where the ``e_j`` are independent,
zero-mean Gaussian, stationary in time, and have the same variance
``sigma^2`` for every trace. With ``m_-i`` the mean of the other ``N - 1``
traces,

    d_i  = r_i - m_-i                            (misfit of trace i)
    E_-i = sum_{j != i} sum_t (r_j - m_-i)^2     (scatter of the others)

``d_i`` has variance ``sigma^2 N / (N - 1)`` per sample and is independent
of ``E_-i`` (the mean of Gaussian samples is independent of the residuals
about it). If the ``n`` samples were independent,

    F_i = [sum_t d_i^2 / (nu N / (N - 1))] / [E_-i / ((N - 2) nu)]

would follow ``F(nu, (N - 2) nu)`` with ``nu = n``. The p-value is the upper
tail ``P(F >= F_i)``: a small p-value means trace ``i`` is farther from the
others than their own scatter allows. The test is one-sided; a trace closer
to the mean than expected is never rejected.

Effective degrees of freedom. RF samples are not independent: the Gaussian
filter of the deconvolution correlates neighbours over about one pulse
width. A sum of squares of ``n`` correlated stationary Gaussian samples is
approximated by a scaled chi-square with (Satterthwaite 1946)

    nu_eff = n^2 / sum_{s,t} rho(s - t)^2 = n^2 / (n + 2 sum_{k>=1} (n - k) rho_k^2),

``rho_k`` the noise autocorrelation at lag ``k``. ``rho_k`` is estimated from
the residuals of all traces about their mean, pooled over traces, and the
sum is truncated at the first lag where the estimate is not positive (beyond
it the estimate is noise). For white noise ``nu_eff = n``; for white noise
through ``exp(-omega^2 / 4a^2)`` it is about ``n delta_t a / sqrt(pi)``.

The same ``nu_eff`` scales numerator and denominator; it sets only the
reference distribution. On synthetic traces that satisfy the null
hypothesis the p-values are uniform (``tests/test_rf_qc_core.py``). With
``N`` traces tested at level ``alpha``, about ``alpha N`` good traces are
rejected by design.

References: Satterthwaite, F. E. (1946), Biometrics Bulletin 2(6), 110-114.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import f as f_dist


def effective_dof(residuals: np.ndarray) -> float:
    """Satterthwaite effective number of independent samples per trace.

    ``residuals`` has shape ``(n_traces, n_samples)``: noise realisations
    (for example traces minus their mean). See the module docstring.
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] < 2:
        raise ValueError("residuals must have shape (n_traces, n_samples >= 2).")
    n = residuals.shape[1]
    autocov = np.array(
        [np.mean(residuals[:, : n - k] * residuals[:, k:]) for k in range(n)]
    )
    if autocov[0] <= 0:
        return float(n)
    rho = autocov / autocov[0]
    not_positive = np.flatnonzero(rho[1:] <= 0)
    n_lags = not_positive[0] if not_positive.size else n - 1
    lags = np.arange(1, n_lags + 1)
    return float(n**2 / (n + 2 * np.sum((n - lags) * rho[lags] ** 2)))


def leave_one_out_f_test(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One-sided leave-one-out F test of each trace against the others.

    Parameters
    ----------
    data
        ``(n_traces, n_samples)`` traces on a common time axis, ``n_traces >= 3``.

    Returns
    -------
    p_value, f_value, nu_eff
        Per-trace p-value ``P(F >= f_i)`` and statistic under
        ``F(nu_eff, (N - 2) nu_eff)``, and the effective degrees of freedom
        per trace (module docstring).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (n_traces, n_samples).")
    n_traces = data.shape[0]
    if n_traces < 3:
        raise ValueError("The leave-one-out F test needs at least three traces.")

    nu = effective_dof(data - data.mean(axis=0))
    dof_others = (n_traces - 2) * nu
    total = data.sum(axis=0)
    f_value = np.empty(n_traces)
    for i in range(n_traces):
        others = np.delete(data, i, axis=0)
        mean_others = (total - data[i]) / (n_traces - 1)
        misfit = np.sum((data[i] - mean_others) ** 2) / (nu * n_traces / (n_traces - 1))
        scatter = np.sum((others - mean_others) ** 2) / dof_others
        f_value[i] = misfit / scatter if scatter > 0 else np.inf
    p_value = f_dist.sf(f_value, nu, dof_others)
    return p_value, f_value, nu
