from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayLike = float | np.ndarray


def brocher_vp_from_vs(vs: ArrayLike, *, check_range: bool = True) -> ArrayLike:
    """
    Calculate P-wave velocity (Vp) from S-wave velocity (Vs) using Brocher's empirical relationship.
    Only works for Vs from 0.0 to 4.5 km/s, for commomn crustal rock seismic velocities.

    Parameters
    ----------
    vs : float or np.ndarray
        S-wave velocity in km/s.

    Returns
    -------
    float or np.ndarray
        P-wave velocity in km/s.
    """
    vs = np.asarray(vs, dtype=float)
    if check_range and np.any((vs < 0.0) | (vs > 4.5)):
        raise ValueError("Brocher's empirical relationship is only valid for Vs from 0.0 to 4.5 km/s.")
    return 0.9409 + 2.0947 * vs - 0.8206 * vs**2 + 0.2683 * vs**3 - 0.0251 * vs**4


def brocher_rho_from_vp(vp: ArrayLike) -> ArrayLike:
    """
    Calculate density (rho) from P-wave velocity (Vp) using Brocher's empirical relationship.

    Parameters
    ----------
    vp : float or np.ndarray
        P-wave velocity in km/s.

    Returns
    -------
    float or np.ndarray
        Density in g/cm^3.
    """
    vp = np.asarray(vp, dtype=float)
    return 1.6612 * vp - 0.4721 * vp**2 + 0.0671 * vp**3 - 0.0043 * vp**4 + 0.000106 * vp**5


def brocher_rho_from_vs(vs: ArrayLike, *, check_range: bool = True) -> ArrayLike:
    """
    Calculate density (rho) from S-wave velocity (Vs) using Brocher's empirical relationship.
    Only works for Vs from 0.0 to 4.5 km/s, for commomn crustal rock seismic velocities.

    Parameters
    ----------
    vs : float or np.ndarray
        S-wave velocity in km/s.

    Returns
    -------
    float or np.ndarray
        Density in g/cm^3.
    """
    vp = brocher_vp_from_vs(vs, check_range=check_range)
    return brocher_rho_from_vp(vp)


def constant_vp_vs(vs: ArrayLike, vp_vs: float = 1.75) -> ArrayLike:
    return np.asarray(vs, dtype=float) * vp_vs


def constant_rho(vs: ArrayLike, rho: float = 2.7) -> ArrayLike:
    return np.full_like(np.asarray(vs, dtype=float), fill_value=rho, dtype=float)


VP_SCALINGS: dict[str, Callable[..., ArrayLike]] = {
    "brocher2005": brocher_vp_from_vs,
    "constant_vp_vs": constant_vp_vs,
}

RHO_SCALINGS: dict[str, Callable[..., ArrayLike]] = {
    "brocher2005": brocher_rho_from_vs,
    "constant": constant_rho,
}


def estimate_vp(vs: ArrayLike, method: str = "brocher2005", **kwargs) -> ArrayLike:
    key = method.lower()
    try:
        func = VP_SCALINGS[key]
    except KeyError:
        raise ValueError(f"Unknown method '{method}' for estimating Vp. Available methods: {list(VP_SCALINGS.keys())}")
    return func(vs, **kwargs)


def estimate_rho(vs: ArrayLike, method: str = "brocher2005", **kwargs) -> ArrayLike:
    key = method.lower()
    try:
        func = RHO_SCALINGS[key]
    except KeyError:
        raise ValueError(f"Unknown method '{method}' for estimating density. Available methods: {list(RHO_SCALINGS.keys())}")
    return func(vs, **kwargs)
