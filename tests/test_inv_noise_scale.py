import copy

import numpy as np
import pytest

from seisforge.inv import gaussian_observable_fit
from seisforge.inv.setup import build_inversion_setup


def _config(noise_parameter=None, weight=1.0, model_uses_noise=False):
    rng = np.random.default_rng(0)
    periods = np.linspace(1.0, 5.0, 8)
    config = {
        "inversion": {
            "parameters": [
                {"name": "vs0", "initial": 1.5, "prior": {"bounds": [1.2, 1.8]}},
                {"name": "vs1", "initial": 2.5, "prior": {"bounds": [2.2, 2.8]}},
                {"name": "noise.dispersion", "initial": 0.0, "prior": {"bounds": [-1.0, 1.0]}},
            ]
        },
        "model": {
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 2.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "vs0"},
                        "bottom": {"parameter": "noise.dispersion" if model_uses_noise else "vs1"},
                    },
                },
                {"top_km": 2.0, "bottom_km": 5.0, "profile": {"type": "constant", "value": 3.2}},
            ]
        },
        "discretization": {"dz": 0.25, "zmax": 5.0},
        "observations": {
            "dispersion": {
                "periods": periods.tolist(),
                "velocity": (1.8 + 0.1 * periods + 0.02 * rng.standard_normal(8)).tolist(),
                "sigma": [0.02] * 8,
            }
        },
        "likelihood": {"dispersion_weight": weight},
    }
    if noise_parameter is not None:
        config["observations"]["dispersion"]["noise_parameter"] = noise_parameter
    return config


def _fit(noise_scale, weight=1.0):
    return gaussian_observable_fit(
        periods=[1.0, 2.0, 3.0, 4.0],
        observed=[1.0, 2.0, 3.0, 4.0],
        predicted=[1.1, 1.8, 3.05, 4.2],
        sigma=[0.1, 0.1, 0.2, 0.1],
        weight=weight,
        noise_scale=noise_scale,
    )


def test_unit_noise_scale_is_the_plain_gaussian_bit_for_bit():
    fit = _fit(1.0)
    assert fit.loglike == -0.5 * fit.chi2
    assert fit.weighted_chi2 == fit.chi2


def test_log_likelihood_peaks_at_h_squared_equal_chi2_over_n():
    # d/dh [-chi2 / (2 h^2) - N ln h] = 0  ->  h^2 = chi2 / N.
    chi2 = _fit(1.0).chi2
    best = np.sqrt(chi2 / 4)
    grid = best * np.array([0.8, 0.95, 1.0, 1.05, 1.25])
    values = [_fit(h).loglike for h in grid]
    assert int(np.argmax(values)) == 2
    expected = -0.5 * 4 - 4 * np.log(best)
    assert _fit(best).loglike == pytest.approx(expected, rel=1e-12)


def test_target_reads_the_noise_scale_from_theta():
    without = build_inversion_setup(_config()).target
    sampled = build_inversion_setup(_config("noise.dispersion")).target
    theta = np.array([1.5, 2.5, 0.0])
    assert sampled.log_posterior(theta) == without.log_posterior(theta)

    theta_scaled = np.array([1.5, 2.5, np.log10(2.0)])
    result = sampled.evaluate(theta_scaled)
    fit = result.likelihood.dispersion
    assert result.noise_scales == {"dispersion": pytest.approx(2.0)}
    assert fit.loglike == pytest.approx(-0.5 * fit.chi2 / 4.0 - fit.n_data * np.log(2.0))

    log_prior, log_like = sampled.log_prior_and_likelihood(theta_scaled)
    assert log_prior + log_like == pytest.approx(result.log_posterior)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"noise_parameter": "noise.missing"}, "not an inversion parameter"),
        ({"noise_parameter": "noise.dispersion", "weight": 2.0}, "leave its likelihood weight"),
        ({"noise_parameter": "noise.dispersion", "model_uses_noise": True}, "also used by the model"),
    ],
)
def test_noise_parameter_setup_errors(kwargs, message):
    config = copy.deepcopy(_config(**kwargs))
    with pytest.raises(ValueError, match=message):
        build_inversion_setup(config)
