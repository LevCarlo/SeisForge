import numpy as np

from seisforge.inv.scaling import (
    brocher_rho_from_vp,
    brocher_rho_from_vs,
    brocher_vp_from_vs,
    estimate_rho,
    estimate_vp,
)


def test_brocher_rho_from_vs_uses_vp_polynomial():
    vs = np.array([2.0, 3.0, 4.0])

    rho = brocher_rho_from_vs(vs)
    expected = brocher_rho_from_vp(brocher_vp_from_vs(vs))

    np.testing.assert_allclose(rho, expected)


def test_constant_vp_vs_scaling():
    vs = np.array([2.0, 3.0])

    vp = estimate_vp(vs, method="constant_vp_vs", vp_vs=1.8)

    np.testing.assert_allclose(vp, np.array([3.6, 5.4]))


def test_constant_density_scaling():
    vs = np.array([2.0, 3.0])

    rho = estimate_rho(vs, method="constant", rho=2.8)

    np.testing.assert_allclose(rho, np.array([2.8, 2.8]))
