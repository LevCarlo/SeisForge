"""Survey how disba fails, for the failure policy in `seisforge.inv.forward`.

Runs disba's phase-velocity solver on random layered models (2-5 layers,
Vs 0.1-4.5 km/s in any order, Vp/Vs 1.0001-3, five periods in 0.05-60 s) and
on models with Vs increasing downwards, and prints which exceptions occur.
Takes a few minutes (numba compilation on the first run).

    python tools/probe_disba_failures.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from disba import PhaseDispersion


def _solve(thickness, vp, vs, rho, periods) -> str:
    try:
        result = PhaseDispersion(thickness, vp, vs, rho)(periods)
    except Exception as exc:  # survey: record every type
        return type(exc).__name__
    return "ok" if len(result.velocity) == len(periods) else "truncated"


def _random_model(rng, *, increasing: bool):
    n_layers = int(rng.integers(2, 6))
    vs = rng.uniform(0.1, 4.5, n_layers)
    if increasing:
        vs = np.sort(vs)
        ratio = rng.uniform(1.5, 2.2, n_layers)
    else:
        ratio = rng.uniform(1.0001, 3.0, n_layers)
    thickness = np.r_[rng.uniform(0.001, 5.0, n_layers - 1), 0.0]
    rho = rng.uniform(1.5, 3.3, n_layers)
    periods = np.sort(rng.uniform(0.05, 60.0, 5))
    return thickness, vs * ratio, vs, rho, periods, ratio


def main() -> None:
    rng = np.random.default_rng(1)
    outcomes, slow_half_space, low_ratio = Counter(), 0, 0
    for _ in range(3000):
        thickness, vp, vs, rho, periods, ratio = _random_model(rng, increasing=False)
        outcome = _solve(thickness, vp, vs, rho, periods)
        outcomes[outcome] += 1
        if outcome == "DispersionError":
            slow_half_space += vs[-1] < vs[:-1].max()
            low_ratio += (vs[-1] >= vs[:-1].max()) and ratio.min() < 1.2
    print("any-order models:", dict(outcomes))
    print("  DispersionError with a half-space slower than a layer above:", slow_half_space)
    print("  DispersionError otherwise, with Vp/Vs < 1.2 somewhere:", low_ratio)

    rng = np.random.default_rng(2)
    outcomes = Counter(
        _solve(*_random_model(rng, increasing=True)[:5]) for _ in range(500)
    )
    print("Vs increasing downwards, Vp/Vs 1.5-2.2:", dict(outcomes))


if __name__ == "__main__":
    main()
