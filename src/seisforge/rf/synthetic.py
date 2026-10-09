"""Synthetic P receiver functions for isotropic, flat-layered media.

A plane P wave with horizontal slowness ``p`` arrives from the half-space. The
free-surface radial and vertical displacements are found with the
Thomson-Haskell propagator for motion-stress vectors (Haskell 1962; Aki &
Richards 2002, ch. 7; Kennett 1983), and the receiver function is their
spectral ratio, Gaussian low-pass filtered:

    RF(w) = R(w) / Z(w) * exp(-w^2 / (4 a^2))

Conventions
-----------
* Depth ``z`` is positive downward, the wave travels toward ``+x``. Radial
  ``R`` is positive along ``+x`` (away from the source) and vertical ``Z`` is
  positive **up**, so the direct-P pulse of a P receiver function is positive.
* Plane waves are ``exp(i w (p x + q z - t))`` with vertical slowness
  ``q = +sqrt(1/v^2 - p^2)`` for down-going and ``-q`` for up-going waves.
* Motion-stress vector ``b = (u_x, u_z, s_xz / (i w), s_zz / (i w))``; with the
  traction scaled by ``i w`` the layer eigenvector matrix does not depend on
  frequency.
* The propagator is used only for real vertical slownesses (``p < 1 / Vp`` in
  every layer and the half-space). That always holds for teleseismic P
  (``p`` up to about 0.08 s/km means ``Vp < 12.5 km/s``), and keeps the
  Haskell product numerically stable; other cases raise ``ValueError``.

Each layer takes its own Vp, Vs, and density, so density from per-segment
scaling enters the reflection/transmission physics directly.
"""

from __future__ import annotations

import numpy as np


AMPLITUDE_CONVENTIONS = ("impulse", "gaussian_peak")
_GAUSSIAN_CUTOFF = 1e-12


def p_wave_surface_response(
    thickness_km: np.ndarray,
    vp_km_s: np.ndarray,
    vs_km_s: np.ndarray,
    rho_g_cm3: np.ndarray,
    ray_parameter_s_per_km: float,
    angular_frequency_rad_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-surface radial and vertical displacement for a unit incident P wave.

    Arrays describe layers from the surface down; the last entry is the
    half-space (its thickness is ignored). Returns complex ``(R, Z)`` spectra
    at the given angular frequencies, for an up-going P wave of unit
    potential-free amplitude at the top of the half-space.
    """
    thickness = np.asarray(thickness_km, dtype=float)
    vp = np.asarray(vp_km_s, dtype=float)
    vs = np.asarray(vs_km_s, dtype=float)
    rho = np.asarray(rho_g_cm3, dtype=float)
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    p = float(ray_parameter_s_per_km)
    _check_layers(thickness, vp, vs, rho, p)

    # Surface traction vanishes: b(0) = u_x * e_x + u_z * e_z. Propagate the
    # two unit columns down to the top of the half-space.
    state = np.zeros((omega.size, 4, 2), dtype=complex)
    state[:, 0, 0] = 1.0
    state[:, 1, 1] = 1.0
    for h, a, b, r in zip(thickness[:-1], vp[:-1], vs[:-1], rho[:-1]):
        eigenvectors, vertical_slowness = layer_eigensystem(a, b, r, p)
        phase = np.exp(1j * omega[:, None] * vertical_slowness[None, :] * h)
        # b(bottom) = E diag(exp(i w q h)) E^-1 b(top)
        state = eigenvectors @ (phase[:, :, None] * (np.linalg.inv(eigenvectors) @ state))

    eigenvectors, _ = layer_eigensystem(vp[-1], vs[-1], rho[-1], p)
    amplitudes = np.linalg.inv(eigenvectors) @ state
    # Half-space wave amplitudes [down P, down S, up P, up S]: the incident up
    # P has unit amplitude and no S wave comes up from below.
    up_p = amplitudes[:, 2, :]
    up_s = amplitudes[:, 3, :]
    determinant = up_p[:, 0] * up_s[:, 1] - up_p[:, 1] * up_s[:, 0]
    u_x = up_s[:, 1] / determinant
    u_z = -up_s[:, 0] / determinant
    return u_x, -u_z


def layer_eigensystem(
    vp_km_s: float,
    vs_km_s: float,
    rho_g_cm3: float,
    ray_parameter_s_per_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Motion-stress eigenvectors and vertical slownesses of one layer.

    Columns are [down P, down S, up P, up S], each a unit-amplitude plane wave
    evaluated at its reference depth. Returns ``(E, q)`` with ``E`` 4x4.
    """
    p = ray_parameter_s_per_km
    mu = rho_g_cm3 * vs_km_s**2
    lam = rho_g_cm3 * vp_km_s**2 - 2.0 * mu
    q_p = np.sqrt(1.0 / vp_km_s**2 - p**2)
    q_s = np.sqrt(1.0 / vs_km_s**2 - p**2)

    columns = []
    slownesses = []
    for direction in (1.0, -1.0):
        for kind in ("P", "S"):
            if kind == "P":
                s_z = direction * q_p
                # P motion is along the slowness vector (p, s_z).
                u = np.array([p, s_z]) * vp_km_s
            else:
                s_z = direction * q_s
                # SV motion is perpendicular to the slowness vector.
                u = np.array([s_z, -p]) * vs_km_s
            tau_xz = mu * (s_z * u[0] + p * u[1])
            tau_zz = lam * (p * u[0] + s_z * u[1]) + 2.0 * mu * s_z * u[1]
            columns.append([u[0], u[1], tau_xz, tau_zz])
            slownesses.append(s_z)
    eigenvectors = np.array(columns, dtype=float).T
    # Reorder from [down P, down S, up P, up S] (already in that order).
    return eigenvectors, np.array(slownesses, dtype=float)


def receiver_function(
    thickness_km: np.ndarray,
    vp_km_s: np.ndarray,
    vs_km_s: np.ndarray,
    rho_g_cm3: np.ndarray,
    ray_parameter_s_per_km: float,
    *,
    dt_s: float,
    n_samples: int,
    gauss_a: float,
    time_shift_s: float = 5.0,
    amplitude: str = "impulse",
    n_fft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-domain P receiver function ``(time_s, rf)``.

    ``time_s = n * dt - time_shift_s``, so the direct P arrival is at 0 s.

    ``gauss_a`` is the Gaussian parameter ``a`` of ``exp(-w^2 / (4 a^2))``
    (rad/s), the convention of iterative time-domain deconvolution (Ligorria
    & Ammon 1999) and of seispy. ``amplitude`` selects the time scaling:

    ``"impulse"``
        Continuous inverse Fourier transform of the filtered spectrum
        (units 1/s). The Gaussian has unit area, so a spike of amplitude ``A``
        peaks at ``A * a / sqrt(pi)``. Matches Zhu's ``hk`` synthetics.
    ``"gaussian_peak"``
        Divided by ``a / sqrt(pi)``: a spike of amplitude ``A`` peaks at
        ``A``, as in iterative-deconvolution outputs with a unit-peak
        Gaussian.

    ``n_fft`` defaults to the next power of two of ``2 * n_samples``; the
    padding keeps late reverberations from wrapping into the window.
    """
    if amplitude not in AMPLITUDE_CONVENTIONS:
        raise ValueError(f"amplitude must be one of {AMPLITUDE_CONVENTIONS}.")
    if dt_s <= 0 or n_samples < 2 or gauss_a <= 0:
        raise ValueError("dt_s, gauss_a must be positive and n_samples >= 2.")
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(2 * n_samples)))
    if n_fft < n_samples:
        raise ValueError("n_fft must be at least n_samples.")

    omega = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt_s)
    gaussian = gaussian_filter(omega, gauss_a)
    # Skip frequencies the Gaussian has already removed (< 1e-12 of DC).
    passband = gaussian > _GAUSSIAN_CUTOFF
    spectrum = np.zeros(omega.size, dtype=complex)
    radial, vertical = p_wave_surface_response(
        thickness_km,
        vp_km_s,
        vs_km_s,
        rho_g_cm3,
        ray_parameter_s_per_km,
        omega[passband],
    )
    spectrum[passband] = radial / vertical * gaussian[passband]
    # The physics uses exp(-i w t); numpy's inverse FFT uses exp(+i w t), so
    # conjugate, then delay by time_shift_s.
    spectrum = np.conj(spectrum) * np.exp(-1j * omega * time_shift_s)
    rf = np.fft.irfft(spectrum, n=n_fft)[:n_samples] / dt_s
    if amplitude == "gaussian_peak":
        rf = rf / (gauss_a / np.sqrt(np.pi))
    time = np.arange(n_samples) * dt_s - time_shift_s
    return time, rf


def gaussian_filter(angular_frequency_rad_s: np.ndarray, gauss_a: float) -> np.ndarray:
    """``exp(-w^2 / (4 a^2))``; unit DC gain, i.e. a unit-area time-domain pulse."""
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    return np.exp(-(omega**2) / (4.0 * gauss_a**2))


def _check_layers(thickness, vp, vs, rho, p) -> None:
    n = vs.size
    if not (thickness.size == vp.size == rho.size == n) or n < 1:
        raise ValueError("thickness, vp, vs, and rho must have the same non-zero length.")
    if np.any(vs <= 0) or np.any(vp <= vs) or np.any(rho <= 0):
        raise ValueError("Requires 0 < Vs < Vp and positive density in every layer.")
    if n > 1 and np.any(thickness[:-1] <= 0):
        raise ValueError("Finite-layer thicknesses must be positive.")
    if not 0 <= p < 1.0 / np.max(vp):
        raise ValueError(
            "Ray parameter must satisfy 0 <= p < 1/max(Vp) (no evanescent P waves)."
        )
