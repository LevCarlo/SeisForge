"""Fixed sampler runs whose output is pinned bit for bit.

`tools/make_sampler_baseline.py` runs these cases and stores the result in
`tests/data/sampler_baseline.npz`; `tests/test_inv_sampler_baseline.py`
reruns them and compares. The baseline was generated from commit 91b7cac
(before adaptive proposals existed), so it pins that the sampler draws the
same chains when adaptation is off. Regenerate it only with the script, and
only for a change listed under "Changed results" in `CHANGELOG.md`.

The targets are analytic and module-level, so they pickle for the process
executor.
"""

from __future__ import annotations

import numpy as np

from seisforge.inv.samplers import MetropolisConfig, run_metropolis

# Correlated three-dimensional Gaussian with unequal scales; precision matrix
# of covariance diag(s) R diag(s), R with off-diagonal 0.8 / 0.3 / 0.5.
_SCALES = np.array([1.0, 2.0, 0.2])
_CORRELATION = np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]])
GAUSSIAN_COVARIANCE = _SCALES[:, None] * _CORRELATION * _SCALES[None, :]
_PRECISION = np.linalg.inv(GAUSSIAN_COVARIANCE)


def gaussian_log_target(theta) -> float:
    theta = np.asarray(theta, dtype=float)
    return float(-0.5 * theta @ _PRECISION @ theta)


def bimodal_split_target(theta) -> tuple[float, float]:
    """``(log prior, log likelihood)``: flat prior, two modes at x = -2 and +2."""
    theta = np.asarray(theta, dtype=float)
    a = -0.5 * np.sum(((theta - np.array([-2.0, 0.0])) / 0.4) ** 2)
    b = -0.5 * np.sum(((theta - np.array([2.0, 0.0])) / 0.4) ** 2)
    top = max(a, b)
    return 0.0, float(top + np.log(np.exp(a - top) + np.exp(b - top)))


_BOUNDS = ([-5.0, -8.0, -1.0], [5.0, 8.0, 1.0])
_STARTS = np.array([[0.5, -1.0, 0.1], [-0.5, 1.0, -0.1]])

CASES = {
    "plain": dict(
        target=gaussian_log_target,
        starts=_STARTS,
        config=dict(n_steps=600, proposal_sigma=[0.6, 1.2, 0.15], burn_in=100, thin=2, bounds=_BOUNDS),
        seeds=11,
    ),
    "global_jump": dict(
        target=gaussian_log_target,
        starts=_STARTS,
        config=dict(
            n_steps=600,
            proposal_sigma=[0.6, 1.2, 0.15],
            burn_in=100,
            thin=2,
            bounds=_BOUNDS,
            global_jump_interval=9,
        ),
        seeds=12,
    ),
    "tempered": dict(
        target=bimodal_split_target,
        starts=np.array([[-2.0, 0.0], [2.0, 0.0]]),
        config=dict(
            n_steps=600,
            proposal_sigma=0.3,
            burn_in=100,
            thin=2,
            bounds=([-4.0, -3.0], [4.0, 3.0]),
            temperatures=(1.0, 3.0, 9.0),
            swap_interval=2,
        ),
        seeds=13,
    ),
}


def run_case(name: str, *, executor: str = "serial", **config_overrides):
    case = CASES[name]
    config = MetropolisConfig(**{**case["config"], **config_overrides})
    return run_metropolis(case["target"], case["starts"], config, seeds=case["seeds"], executor=executor)


def result_arrays(result) -> dict[str, np.ndarray]:
    arrays = {
        "samples": result.samples,
        "log_prob": result.log_prob,
        "accepted": result.accepted,
        "final_theta": np.stack([c.final_theta for c in result.chains]),
    }
    if result.chains[0].swap_acceptance is not None:
        arrays["swap_acceptance"] = np.stack([c.swap_acceptance for c in result.chains])
    return arrays
