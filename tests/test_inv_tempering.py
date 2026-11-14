import math

import numpy as np
import pytest

from seisforge.inv import MetropolisConfig, geometric_temperatures, run_metropolis
from seisforge.inv.setup import _tempering_from_config

# Two narrow modes, weights 0.3 at -4 and 0.7 at +4 (sigma 0.25): the
# likelihood drops by ~128 between them, so a random walk with steps of 0.3
# never crosses, while a ladder up to T = 200 does.
WEIGHTS, MEANS, WIDTH = (0.3, 0.7), (-4.0, 4.0), 0.25


def _log_likelihood(x) -> float:
    terms = [math.log(w) - 0.5 * ((x[0] - m) / WIDTH) ** 2 for w, m in zip(WEIGHTS, MEANS, strict=True)]
    top = max(terms)
    return top + math.log(sum(math.exp(t - top) for t in terms))


def _config(**kwargs):
    return MetropolisConfig(
        n_steps=8000, proposal_sigma=0.3, burn_in=1000, bounds=([-8.0], [8.0]), **kwargs
    )


def test_tempering_recovers_the_weights_of_separated_modes():
    starts = np.full((2, 1), -4.0)
    plain = run_metropolis(_log_likelihood, starts, _config(), seeds=3)
    assert np.mean(plain.samples[..., 0] > 0) == 0.0

    ladder = geometric_temperatures(5, 200.0)
    tempered = run_metropolis(
        lambda x: (0.0, _log_likelihood(x)), starts, _config(temperatures=ladder), seeds=3
    )
    # 2 x 7000 stored draws with mode switches every few tens of steps: the
    # mode fraction scatters by about 0.02-0.03 around 0.7.
    assert np.mean(tempered.samples[..., 0] > 0) == pytest.approx(0.7, abs=0.08)
    for chain in tempered.chains:
        assert chain.swap_acceptance.shape == (4,)
        assert np.all(chain.swap_acceptance > 0.2)
    assert np.allclose(tempered.log_prob, [[_log_likelihood([x]) for x in c] for c in tempered.samples[..., 0]])


def test_geometric_ladder_and_validation():
    assert geometric_temperatures(3, 100.0) == pytest.approx((1.0, 10.0, 100.0))
    with pytest.raises(ValueError, match="start at 1"):
        _config(temperatures=(2.0, 4.0))
    with pytest.raises(ValueError, match="start at 1"):
        _config(temperatures=(1.0,))
    with pytest.raises(ValueError, match="swap_interval"):
        _config(temperatures=(1.0, 2.0), swap_interval=0)


def test_tempering_config():
    assert _tempering_from_config(None) == (None, 1)
    ladder, interval = _tempering_from_config({})
    assert ladder == pytest.approx(geometric_temperatures(5, 30.0)) and interval == 1
    assert _tempering_from_config({"temperatures": [1, 3], "swap_interval": 2}) == ((1.0, 3.0), 2)
    with pytest.raises(ValueError, match="not both"):
        _tempering_from_config({"temperatures": [1, 3], "n_temperatures": 4})
