"""Stable plane-wave boundary solves for layered elastic media.

Finite-layer wave amplitudes are referenced to the boundary from which they
decay.  Consequently no exponentially growing column is ever formed.  This is
the scattering/boundary-system counterpart of multiplying transfer matrices
and avoids the classic Thomson--Haskell overflow.
"""

from __future__ import annotations

import numpy as np

from .models import ElasticLayerModel


def vertical_decay_wavenumber(
    horizontal_wavenumber_per_km: complex,
    omega_rad_s: float,
    velocity_km_s: float,
) -> complex:
    """Return the radiation-branch vertical decay wavenumber ``gamma``.

    The downward wave is proportional to ``exp(-gamma*z)`` for the
    ``exp(i*k*x-i*omega*t)`` convention.  The sign is selected explicitly so
    evanescent waves decay and propagating waves carry energy downwards.
    """

    value = np.sqrt(
        complex(horizontal_wavenumber_per_km) ** 2
        - (omega_rad_s / velocity_km_s) ** 2
        + 0j
    )
    if value.real < 0.0 or (np.isclose(value.real, 0.0) and value.imag > 0.0):
        value = -value
    return value


def _psv_column(
    horizontal_wavenumber_per_km: complex,
    omega_rad_s: float,
    vp_km_s: float,
    vs_km_s: float,
    density_g_cm3: float,
    *,
    wave: str,
    vertical_exponent_per_km: complex,
) -> np.ndarray:
    k = horizontal_wavenumber_per_km
    s = vertical_exponent_per_km
    mu = density_g_cm3 * vs_km_s**2
    lam = density_g_cm3 * vp_km_s**2 - 2.0 * mu

    if wave == "p":
        return np.array(
            [
                1j * k,
                s,
                2j * mu * k * s,
                (lam + 2.0 * mu) * s**2 - lam * k**2,
            ],
            dtype=np.complex128,
        )
    if wave == "sv":
        return np.array(
            [
                -s,
                1j * k,
                -mu * (s**2 + k**2),
                2j * mu * k * s,
            ],
            dtype=np.complex128,
        )
    raise ValueError("wave must be 'p' or 'sv'")


def _psv_layer_bases(
    model: ElasticLayerModel,
    layer: int,
    omega_rad_s: float,
    horizontal_wavenumber_per_km: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state bases at the top and bottom of one finite layer."""

    thickness = model.thickness_km[layer]
    gamma_p = vertical_decay_wavenumber(
        horizontal_wavenumber_per_km, omega_rad_s, model.vp_km_s[layer]
    )
    gamma_s = vertical_decay_wavenumber(
        horizontal_wavenumber_per_km, omega_rad_s, model.vs_km_s[layer]
    )
    exponents = (-gamma_p, -gamma_s, gamma_p, gamma_s)
    waves = ("p", "sv", "p", "sv")
    reference_depths = (0.0, 0.0, thickness, thickness)

    top = np.empty((4, 4), dtype=np.complex128)
    bottom = np.empty((4, 4), dtype=np.complex128)
    for column, (wave, exponent, reference) in enumerate(
        zip(waves, exponents, reference_depths, strict=True)
    ):
        state = _psv_column(
            horizontal_wavenumber_per_km,
            omega_rad_s,
            model.vp_km_s[layer],
            model.vs_km_s[layer],
            model.density_g_cm3[layer],
            wave=wave,
            vertical_exponent_per_km=exponent,
        )
        top[:, column] = state * np.exp(exponent * (0.0 - reference))
        bottom[:, column] = state * np.exp(
            exponent * (thickness - reference)
        )
    return top, bottom


def _psv_halfspace_basis(
    model: ElasticLayerModel,
    omega_rad_s: float,
    horizontal_wavenumber_per_km: complex,
) -> np.ndarray:
    layer = model.n_layers - 1
    gamma_p = vertical_decay_wavenumber(
        horizontal_wavenumber_per_km, omega_rad_s, model.vp_km_s[layer]
    )
    gamma_s = vertical_decay_wavenumber(
        horizontal_wavenumber_per_km, omega_rad_s, model.vs_km_s[layer]
    )
    return np.column_stack(
        [
            _psv_column(
                horizontal_wavenumber_per_km,
                omega_rad_s,
                model.vp_km_s[layer],
                model.vs_km_s[layer],
                model.density_g_cm3[layer],
                wave="p",
                vertical_exponent_per_km=-gamma_p,
            ),
            _psv_column(
                horizontal_wavenumber_per_km,
                omega_rad_s,
                model.vp_km_s[layer],
                model.vs_km_s[layer],
                model.density_g_cm3[layer],
                wave="sv",
                vertical_exponent_per_km=-gamma_s,
            ),
        ]
    )


def psv_surface_compliance(
    model: ElasticLayerModel,
    frequency_hz: float,
    horizontal_wavenumber_per_km: complex,
) -> np.ndarray:
    """Return the 2x2 surface displacement/traction compliance for P--SV.

    Rows are ``(u_x, u_z)`` and columns are unit ``(t_x, t_z)`` loads.
    """

    if frequency_hz <= 0.0 or not np.isfinite(frequency_hz):
        raise ValueError("frequency_hz must be finite and positive")
    omega = 2.0 * np.pi * frequency_hz
    finite_layers = model.n_layers - 1
    unknowns = 4 * finite_layers + 2
    system = np.zeros((unknowns, unknowns), dtype=np.complex128)
    rhs = np.zeros((unknowns, 2), dtype=np.complex128)

    halfspace = _psv_halfspace_basis(
        model, omega, horizontal_wavenumber_per_km
    )
    if finite_layers == 0:
        surface_basis = halfspace
    else:
        bases = [
            _psv_layer_bases(model, layer, omega, horizontal_wavenumber_per_km)
            for layer in range(finite_layers)
        ]
        surface_basis = bases[0][0]

    # Surface tractions are prescribed. State entries 2 and 3 are txz and tzz.
    system[:2, : surface_basis.shape[1]] = surface_basis[2:4, :]
    rhs[:2, :] = np.eye(2)

    row = 2
    for layer in range(finite_layers):
        left = bases[layer][1]
        left_slice = slice(4 * layer, 4 * layer + 4)
        if layer + 1 < finite_layers:
            right = bases[layer + 1][0]
            right_slice = slice(4 * (layer + 1), 4 * (layer + 2))
        else:
            right = halfspace
            right_slice = slice(4 * finite_layers, unknowns)
        system[row : row + 4, left_slice] = left
        system[row : row + 4, right_slice] = -right
        row += 4

    coefficients = np.linalg.solve(system, rhs)
    surface_coefficients = coefficients[: surface_basis.shape[1], :]
    return surface_basis[:2, :] @ surface_coefficients


def _sh_layer_bases(
    model: ElasticLayerModel,
    layer: int,
    omega_rad_s: float,
    horizontal_wavenumber_per_km: complex,
) -> tuple[np.ndarray, np.ndarray]:
    thickness = model.thickness_km[layer]
    gamma = vertical_decay_wavenumber(
        horizontal_wavenumber_per_km, omega_rad_s, model.vs_km_s[layer]
    )
    mu = model.density_g_cm3[layer] * model.vs_km_s[layer] ** 2
    down = np.array([1.0, -mu * gamma], dtype=np.complex128)
    up = np.array([1.0, mu * gamma], dtype=np.complex128)
    top = np.column_stack((down, up * np.exp(-gamma * thickness)))
    bottom = np.column_stack((down * np.exp(-gamma * thickness), up))
    return top, bottom


def sh_surface_compliance(
    model: ElasticLayerModel,
    frequency_hz: float,
    horizontal_wavenumber_per_km: complex,
) -> complex:
    """Return surface transverse displacement for a unit SH traction."""

    if frequency_hz <= 0.0 or not np.isfinite(frequency_hz):
        raise ValueError("frequency_hz must be finite and positive")
    omega = 2.0 * np.pi * frequency_hz
    finite_layers = model.n_layers - 1
    layer = model.n_layers - 1
    gamma = vertical_decay_wavenumber(
        horizontal_wavenumber_per_km, omega, model.vs_km_s[layer]
    )
    mu = model.density_g_cm3[layer] * model.vs_km_s[layer] ** 2
    halfspace = np.array([[1.0], [-mu * gamma]], dtype=np.complex128)

    unknowns = 2 * finite_layers + 1
    system = np.zeros((unknowns, unknowns), dtype=np.complex128)
    rhs = np.zeros(unknowns, dtype=np.complex128)
    if finite_layers == 0:
        surface_basis = halfspace
    else:
        bases = [
            _sh_layer_bases(model, index, omega, horizontal_wavenumber_per_km)
            for index in range(finite_layers)
        ]
        surface_basis = bases[0][0]

    system[0, : surface_basis.shape[1]] = surface_basis[1, :]
    rhs[0] = 1.0
    row = 1
    for index in range(finite_layers):
        left = bases[index][1]
        left_slice = slice(2 * index, 2 * index + 2)
        if index + 1 < finite_layers:
            right = bases[index + 1][0]
            right_slice = slice(2 * (index + 1), 2 * (index + 2))
        else:
            right = halfspace
            right_slice = slice(2 * finite_layers, unknowns)
        system[row : row + 2, left_slice] = left
        system[row : row + 2, right_slice] = -right
        row += 2

    coefficients = np.linalg.solve(system, rhs)
    return complex(surface_basis[0, :] @ coefficients[: surface_basis.shape[1]])
