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
from numba import njit
from scipy.fft import next_fast_len


AMPLITUDE_CONVENTIONS = ("impulse", "gaussian_peak")
_GAUSSIAN_CUTOFF = 1e-12
# Minimum time after the window for reverberations to decay before wrapping.
_MIN_DECAY_S = 60.0
_MAX_FFT_GROWTH = 16


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
    at the given angular frequencies, for an up-going P wave of unit amplitude
    at the top of the half-space.
    """
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    transfer, start, tau_p, tau_s = _wave_basis(
        thickness_km, vp_km_s, vs_km_s, rho_g_cm3, ray_parameter_s_per_km
    )
    radial = np.empty(omega.size, dtype=complex)
    vertical = np.empty(omega.size, dtype=complex)
    _surface_response(transfer, start, tau_p, tau_s, omega, 0.0, False, radial, vertical)
    return radial, vertical


def _wave_basis(thickness_km, vp_km_s, vs_km_s, rho_g_cm3, ray_parameter_s_per_km):
    """Interface transfer matrices in the up/down wave basis of each layer.

    With wave amplitudes ``c_k`` (columns [down P, down S, up P, up S]) at the
    top of layer ``k``, propagation through the layer is the diagonal phase
    ``diag(exp(i w q h))`` and the interface below maps
    ``c_{k+1} = E_{k+1}^-1 E_k diag(...) c_k``. For real vertical slownesses
    the transfer matrices ``E_{k+1}^-1 E_k`` are real and frequency-free.
    Returns ``(transfer, start, tau_p, tau_s)`` with ``start = E_0^-1[:, :2]``
    (the traction-free surface states) and ``tau = q * h`` per finite layer.
    """
    thickness = np.asarray(thickness_km, dtype=float)
    vp = np.asarray(vp_km_s, dtype=float)
    vs = np.asarray(vs_km_s, dtype=float)
    rho = np.asarray(rho_g_cm3, dtype=float)
    p = float(ray_parameter_s_per_km)
    _check_layers(thickness, vp, vs, rho, p)

    eigenvectors, vertical_slowness = layer_eigensystem(vp, vs, rho, p)
    inverse = np.linalg.inv(eigenvectors)
    transfer = inverse[1:] @ eigenvectors[:-1]
    start = np.ascontiguousarray(inverse[0][:, :2])
    tau_p = vertical_slowness[:-1, 0] * thickness[:-1]
    tau_s = vertical_slowness[:-1, 1] * thickness[:-1]
    return transfer, start, tau_p, tau_s


# Safe fast-math subset: reassociation and FMA contraction speed up the
# inner loop about 1.4x, while NaN/Inf semantics are kept (no 'nnan'/'ninf').
@njit(cache=True, nogil=True, fastmath={"reassoc", "contract", "arcp", "nsz"})
def _surface_response(transfer, start, tau_p, tau_s, omega, d_omega, uniform, radial, vertical):
    """Propagate the two surface states to the half-space for every frequency.

    In the half-space the incident up-going P has unit amplitude and no S
    wave comes up, which fixes the surface displacement (u_x, u_z); radial is
    u_x and vertical is -u_z (positive up). On a uniform grid
    ``omega[j] = omega[0] + j * d_omega`` the layer phases follow the recurrence
    ``exp(i w_j tau) = exp(i w_{j-1} tau) * exp(i d_omega tau)``, refreshed by
    a direct exponential every 64 frequencies to bound rounding drift.
    """
    n_layers = tau_p.size
    zp = np.empty(n_layers, dtype=np.complex128)
    zs = np.empty(n_layers, dtype=np.complex128)
    dp = np.empty(n_layers, dtype=np.complex128)
    ds = np.empty(n_layers, dtype=np.complex128)
    if uniform:
        for k in range(n_layers):
            dp[k] = np.exp(1j * d_omega * tau_p[k])
            ds[k] = np.exp(1j * d_omega * tau_s[k])
    for j in range(omega.size):
        w = omega[j]
        if not uniform or j % 64 == 0:
            for k in range(n_layers):
                zp[k] = np.exp(1j * w * tau_p[k])
                zs[k] = np.exp(1j * w * tau_s[k])
        # Two surface states (columns x and z) as scalars.
        x0 = start[0, 0] + 0j
        x1 = start[1, 0] + 0j
        x2 = start[2, 0] + 0j
        x3 = start[3, 0] + 0j
        z0 = start[0, 1] + 0j
        z1 = start[1, 1] + 0j
        z2 = start[2, 1] + 0j
        z3 = start[3, 1] + 0j
        for k in range(n_layers):
            a = zp[k]
            b = zs[k]
            ac = a.conjugate()
            bc = b.conjugate()
            x0 *= a
            x1 *= b
            x2 *= ac
            x3 *= bc
            z0 *= a
            z1 *= b
            z2 *= ac
            z3 *= bc
            m = transfer[k]
            y0 = m[0, 0] * x0 + m[0, 1] * x1 + m[0, 2] * x2 + m[0, 3] * x3
            y1 = m[1, 0] * x0 + m[1, 1] * x1 + m[1, 2] * x2 + m[1, 3] * x3
            y2 = m[2, 0] * x0 + m[2, 1] * x1 + m[2, 2] * x2 + m[2, 3] * x3
            y3 = m[3, 0] * x0 + m[3, 1] * x1 + m[3, 2] * x2 + m[3, 3] * x3
            x0, x1, x2, x3 = y0, y1, y2, y3
            y0 = m[0, 0] * z0 + m[0, 1] * z1 + m[0, 2] * z2 + m[0, 3] * z3
            y1 = m[1, 0] * z0 + m[1, 1] * z1 + m[1, 2] * z2 + m[1, 3] * z3
            y2 = m[2, 0] * z0 + m[2, 1] * z1 + m[2, 2] * z2 + m[2, 3] * z3
            y3 = m[3, 0] * z0 + m[3, 1] * z1 + m[3, 2] * z2 + m[3, 3] * z3
            z0, z1, z2, z3 = y0, y1, y2, y3
            if uniform:
                zp[k] = a * dp[k]
                zs[k] = b * ds[k]
        # Half-space rows: up P (index 2) = 1, up S (index 3) = 0.
        determinant = x2 * z3 - z2 * x3
        if determinant == 0:
            radial[j] = np.nan
            vertical[j] = np.nan
        else:
            radial[j] = z3 / determinant
            vertical[j] = x3 / determinant


def layer_eigensystem(
    vp_km_s,
    vs_km_s,
    rho_g_cm3,
    ray_parameter_s_per_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Motion-stress eigenvectors and vertical slownesses of layers.

    Columns are [down P, down S, up P, up S], each a unit-amplitude plane wave
    at its reference depth. Scalars give ``E`` of shape (4, 4) and ``q`` of
    shape (4,); arrays of n layers give (n, 4, 4) and (n, 4).

    P motion is along the slowness vector ``(p, q)`` (scaled by Vp), SV motion
    is perpendicular to it (``(q, -p)`` scaled by Vs), and the scaled
    tractions are ``tau_xz = mu (q u_x + p u_z)`` and
    ``tau_zz = lambda (p u_x + q u_z) + 2 mu q u_z``.
    """
    vp = np.asarray(vp_km_s, dtype=float)
    vs = np.asarray(vs_km_s, dtype=float)
    rho = np.asarray(rho_g_cm3, dtype=float)
    p = ray_parameter_s_per_km
    mu = rho * vs**2
    lam = rho * vp**2 - 2.0 * mu
    q_p = np.sqrt(1.0 / vp**2 - p**2)
    q_s = np.sqrt(1.0 / vs**2 - p**2)

    q = np.stack([q_p, q_s, -q_p, -q_s], axis=-1)
    is_p = np.array([True, False, True, False])
    velocity = np.where(is_p, vp[..., None], vs[..., None])
    u_x = np.where(is_p, p, q) * velocity
    u_z = np.where(is_p, q, -p) * velocity
    tau_xz = mu[..., None] * (q * u_x + p * u_z)
    tau_zz = lam[..., None] * (p * u_x + q * u_z) + 2.0 * mu[..., None] * q * u_z
    eigenvectors = np.stack([u_x, u_z, tau_xz, tau_zz], axis=-2)
    return eigenvectors, q


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
    tail_tolerance: float = 1e-3,
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

    FFT length. The frequency-domain result is periodic, so reverberations
    arriving after the FFT period wrap into the window. By default the FFT
    covers ``max(2 * window, window + 60 s)``; if the trace has not decayed
    to ``tail_tolerance`` of its peak by the end of the padding, the length is
    doubled (up to 16 times). A trace that still has not decayed (a near-zero
    vertical spectrum makes ``R/Z`` ring for very long) raises ``ValueError``
    rather than returning a wrapped result. An explicit ``n_fft`` is used as
    given, without the check.
    """
    if amplitude not in AMPLITUDE_CONVENTIONS:
        raise ValueError(f"amplitude must be one of {AMPLITUDE_CONVENTIONS}.")
    if dt_s <= 0 or n_samples < 2 or gauss_a <= 0:
        raise ValueError("dt_s, gauss_a must be positive and n_samples >= 2.")
    adaptive = n_fft is None
    if adaptive:
        n_fft = default_fft_length(n_samples, dt_s)
    if n_fft < n_samples:
        raise ValueError("n_fft must be at least n_samples.")

    basis = _wave_basis(
        thickness_km, vp_km_s, vs_km_s, rho_g_cm3, ray_parameter_s_per_km
    )
    limit = _MAX_FFT_GROWTH * n_fft
    d_omega = 2.0 * np.pi / (n_fft * dt_s)
    ratio = _spectral_ratio(basis, 0.0, d_omega, _passband_size(n_fft, dt_s, gauss_a))
    while True:
        series = _rf_series(ratio, n_fft, dt_s, gauss_a, time_shift_s)
        if not adaptive:
            break
        tail = _tail_ratio(series, n_samples, dt_s, gauss_a)
        if tail <= tail_tolerance:
            break
        if 2 * n_fft > limit:
            raise ValueError(
                f"Receiver function has not decayed after {n_fft * dt_s:.0f} s "
                f"(tail {tail:.1e} of peak): the vertical spectrum is nearly zero "
                "and R/Z rings."
            )
        # Doubling the FFT halves the frequency step: the even points are the
        # current grid, so only the new odd points are computed.
        n_fft *= 2
        d_omega /= 2.0
        n_pass = _passband_size(n_fft, dt_s, gauss_a)
        refined = np.empty(n_pass, dtype=complex)
        n_even = (n_pass + 1) // 2
        refined[0::2] = ratio[:n_even]
        refined[1::2] = _spectral_ratio(basis, d_omega, 2.0 * d_omega, n_pass // 2)
        ratio = refined

    rf = series[:n_samples]
    if amplitude == "gaussian_peak":
        rf = rf / (gauss_a / np.sqrt(np.pi))
    time = np.arange(n_samples) * dt_s - time_shift_s
    return time, rf


def default_fft_length(n_samples: int, dt_s: float) -> int:
    """FFT length covering ``max(2 * window, window + 60 s)``, FFT-friendly."""
    n_decay = int(np.ceil(_MIN_DECAY_S / dt_s))
    return next_fast_len(max(2 * n_samples, n_samples + n_decay), real=True)


def _passband_size(n_fft, dt_s, gauss_a):
    """Number of leading rfft frequencies where the Gaussian exceeds the cutoff."""
    omega = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt_s)
    return int(np.count_nonzero(gaussian_filter(omega, gauss_a) > _GAUSSIAN_CUTOFF))


def _spectral_ratio(basis, omega_start, d_omega, count):
    """``R/Z`` at ``omega_start + j * d_omega`` for ``j < count``."""
    transfer, start, tau_p, tau_s = basis
    omega = omega_start + d_omega * np.arange(count)
    radial = np.empty(count, dtype=complex)
    vertical = np.empty(count, dtype=complex)
    _surface_response(transfer, start, tau_p, tau_s, omega, d_omega, True, radial, vertical)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = radial / vertical
    if not np.all(np.isfinite(ratio)):
        raise ValueError("Surface response is singular (zero vertical motion).")
    return ratio


def _rf_series(ratio, n_fft, dt_s, gauss_a, time_shift_s):
    """Full periodic RF series (impulse amplitude) of length ``n_fft``.

    ``ratio`` holds ``R/Z`` on the leading passband frequencies of the grid;
    the remaining frequencies are zero.
    """
    omega = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt_s)
    spectrum = np.zeros(omega.size, dtype=complex)
    n_pass = ratio.size
    spectrum[:n_pass] = ratio * gaussian_filter(omega[:n_pass], gauss_a)
    # The physics uses exp(-i w t); numpy's inverse FFT uses exp(+i w t), so
    # conjugate, then delay by time_shift_s.
    spectrum = np.conj(spectrum) * np.exp(-1j * omega * time_shift_s)
    return np.fft.irfft(spectrum, n=n_fft) / dt_s


def _tail_ratio(series, n_samples, dt_s, gauss_a):
    """Peak of the last quarter of the padding relative to the window peak.

    The last few Gaussian widths are skipped: they hold the wrapped acausal
    flank of the direct-P pulse when the time shift is small.
    """
    guard = int(np.ceil(4.0 / gauss_a / dt_s))
    end = series.size - guard
    begin = max(n_samples, end - (series.size - n_samples) // 4)
    if end <= begin:
        return np.inf
    peak = np.max(np.abs(series[:n_samples]))
    return float(np.max(np.abs(series[begin:end])) / peak) if peak > 0 else 0.0


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
