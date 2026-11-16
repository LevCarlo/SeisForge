"""Rotation of nine-component ambient-noise CCFs between ZNE and ZRT.

Conventions
-----------
- CCF component ``ab``: first letter = component at the virtual-source station
  A, second letter = component at the receiver station B
  (``C_ab(t) = sum_tau u_a^A(tau) u_b^B(tau + t)``; positive lag is
  propagation A -> B).
- Angles: ``theta`` = azimuth from A to B, ``psi`` = back azimuth (from B to
  A), clockwise from north. Unit vectors are written as (N, E).
- Radial/transverse axes, ``radial_convention="lin2008"`` (default): at *both*
  stations R points from A towards B (along the propagation of the positive
  lag) and T is R turned 90 degrees clockwise (seen from above), as drawn in
  Fig. 2 of Lin, Moschetti & Ritzwoller (2008):

      source A:   R = ( cos theta,  sin theta),  T = (-sin theta,  cos theta)
      receiver B: R = (-cos psi,   -sin psi),    T = ( sin psi,   -cos psi)

  In matrix form ``C_ZRT = M_s C_ZNE M_r^T`` with
  ``M_s = [[1,0,0],[0,cos theta,sin theta],[0,-sin theta,cos theta]]`` and
  ``M_r = [[1,0,0],[0,-cos psi,-sin psi],[0,sin psi,-cos psi]]``.
  The coefficients of TT, RR, TR and RT agree with Lin et al. (2008), eq. 1.
- ``radial_convention="legacy"``: the convention of SeisForge before the
  change (CHANGELOG, "Changed results"): R and T reversed at both stations
  (R points from B towards A). RR, RT, TR, TT are the same in both
  conventions (two sign flips); ZR, RZ, ZT and TZ change sign. ZZ never
  changes. Kept so that older results can be reproduced.

`seisforge ant rotate-ccf` writes the convention used into SAC ``kuser0``
(`RADIAL_CONVENTION_SAC_HEADER`), so that AFTAN can pick the right phase
constants for ZR/RZ.

Reference: Lin, F.-C., Moschetti, M. P., Ritzwoller, M. H. (2008). Surface
wave tomography of the western United States from ambient seismic noise:
Rayleigh and Love wave phase velocity maps. Geophys. J. Int. 173, 281-298,
doi:10.1111/j.1365-246X.2008.03720.x.
"""

import numpy as np

__all__ = [
    "CONVENTION_DEPENDENT_COMPONENTS",
    "DEFAULT_RADIAL_CONVENTION",
    "RADIAL_CONVENTIONS",
    "RADIAL_CONVENTION_SAC_HEADER",
    "REQUIRED_ZNE_BY_ZRT_COMPONENT",
    "ZRT_COMPONENTS",
    "horizontal_axes",
    "required_zne_components",
    "rotate_zne_ccf_to_zrt",
    "rotate_zrt_ccf_to_zne",
    "validate_radial_convention",
]

RADIAL_CONVENTIONS = ("lin2008", "legacy")
DEFAULT_RADIAL_CONVENTION = "lin2008"
RADIAL_CONVENTION_SAC_HEADER = "kuser0"
# Components whose sign depends on the radial convention (one R or T axis).
CONVENTION_DEPENDENT_COMPONENTS = frozenset({"ZR", "RZ", "ZT", "TZ"})

ZRT_COMPONENTS = ("ZZ", "ZR", "ZT", "RZ", "RR", "RT", "TZ", "TR", "TT")

REQUIRED_ZNE_BY_ZRT_COMPONENT = {
    "ZZ": ("ZZ",),
    "ZR": ("ZN", "ZE"),
    "ZT": ("ZN", "ZE"),
    "RZ": ("NZ", "EZ"),
    "TZ": ("NZ", "EZ"),
    "RR": ("NN", "NE", "EN", "EE"),
    "RT": ("NN", "NE", "EN", "EE"),
    "TR": ("NN", "NE", "EN", "EE"),
    "TT": ("NN", "NE", "EN", "EE"),
}
_ZRT_COMPONENTS = ZRT_COMPONENTS
_REQUIRED_BY_COMPONENT = REQUIRED_ZNE_BY_ZRT_COMPONENT


def validate_radial_convention(radial_convention: str) -> str:
    """Return the normalized radial convention name or raise ValueError."""
    normalized = str(radial_convention).strip().lower()
    if normalized not in RADIAL_CONVENTIONS:
        raise ValueError(
            f"radial_convention must be one of {RADIAL_CONVENTIONS}; got {radial_convention!r}."
        )
    return normalized


def horizontal_axes(azimuth, back_azimuth, radial_convention=DEFAULT_RADIAL_CONVENTION, degrees=True):
    """Unit vectors (N, E) of R and T at the source and receiver stations.

    Returns ``(source, receiver)``, each ``{"R": (n, e), "T": (n, e)}``; see the
    module docstring for the conventions.
    """
    sign = 1.0 if validate_radial_convention(radial_convention) == "lin2008" else -1.0
    theta = np.deg2rad(azimuth) if degrees else azimuth
    psi = np.deg2rad(back_azimuth) if degrees else back_azimuth
    source = {
        "R": (sign * np.cos(theta), sign * np.sin(theta)),
        "T": (-sign * np.sin(theta), sign * np.cos(theta)),
    }
    receiver = {
        "R": (-sign * np.cos(psi), -sign * np.sin(psi)),
        "T": (sign * np.sin(psi), -sign * np.cos(psi)),
    }
    return source, receiver


def required_zne_components(components=None):
    """Return the ZNE input components needed for requested ZRT outputs."""
    if components is None:
        components = ZRT_COMPONENTS
    else:
        components = tuple(components)

    invalid_components = [comp for comp in components if comp not in ZRT_COMPONENTS]
    if invalid_components:
        raise ValueError(
            "Invalid components requested for rotation: "
            f"{invalid_components}. Valid components are: {ZRT_COMPONENTS}"
        )

    return tuple(
        dict.fromkeys(
            req
            for comp in components
            for req in REQUIRED_ZNE_BY_ZRT_COMPONENT[comp]
        )
    )


def rotate_zne_ccf_to_zrt(
    ccf,
    azimuth,
    back_azimuth,
    components=None,
    degrees=True,
    check=True,
    radial_convention=DEFAULT_RADIAL_CONVENTION,
):
    """Rotate nine-component cross-correlation functions from ZNE to ZRT.

    The input CCFs are assumed to be organized with the first letter denoting
    the virtual-source station component and the second letter denoting the
    receiver-station component. For example, ``NZ`` means the correlation
    between the N component at the virtual-source station and the Z component
    at the receiver station. Axis conventions are in the module docstring.

    Parameters
    ----------
    ccf : dict
        Cross-correlation functions in the ZNE coordinate system. Required keys
        depend on the requested output components and may include ``ZZ``, ``ZN``,
        ``ZE``, ``NZ``, ``NN``, ``NE``, ``EZ``, ``EN``, and ``EE``. Values can be
        scalars or NumPy-compatible arrays.
    azimuth : float
        Interstation azimuth from the virtual-source station to the receiver
        station.
    back_azimuth : float
        Back azimuth from the receiver station to the virtual-source station.
    components : sequence of str, optional
        Output ZRT components to compute. If None, all nine ZRT components are
        returned: ``ZZ``, ``ZR``, ``ZT``, ``RZ``, ``RR``, ``RT``, ``TZ``, ``TR``,
        and ``TT``.
    degrees : bool, optional
        If True, ``azimuth`` and ``back_azimuth`` are interpreted in degrees.
        Otherwise, they are interpreted in radians.
    check : bool, optional
        If True, check missing input components and shape consistency among the
        input components required for the requested outputs.
    radial_convention : {"lin2008", "legacy"}, optional
        Direction of the R and T axes (module docstring). ``"lin2008"``
        (default): R along the propagation A -> B at both stations, as in
        Lin et al. (2008), Fig. 2. ``"legacy"``: the pre-change SeisForge
        convention (R and T reversed); ZR, RZ, ZT, TZ change sign, the other
        components are bit for bit the same.

    Returns
    -------
    rotated_ccf : dict
        Rotated cross-correlation functions in the ZRT coordinate system.
    """
    radial_convention = validate_radial_convention(radial_convention)
    if components is None:
        components = ZRT_COMPONENTS
    else:
        components = tuple(components)

    invalid_components = [comp for comp in components if comp not in ZRT_COMPONENTS]
    if invalid_components:
        raise ValueError(
            "Invalid components requested for rotation: "
            f"{invalid_components}. Valid components are: {ZRT_COMPONENTS}"
        )

    required_components = required_zne_components(components)

    if check:
        missing_components = [comp for comp in required_components if comp not in ccf]
        if missing_components:
            raise KeyError(
                "Missing input ZNE components required for rotation: "
                f"{missing_components}"
            )

        shapes = {comp: np.asarray(ccf[comp]).shape for comp in required_components}
        unique_shapes = set(shapes.values())
        if len(unique_shapes) > 1:
            raise ValueError(f"Inconsistent input CCF shapes: {shapes}")

    theta = np.deg2rad(azimuth) if degrees else azimuth
    psi = np.deg2rad(back_azimuth) if degrees else back_azimuth

    cth = np.cos(theta)
    sth = np.sin(theta)
    cps = np.cos(psi)
    sps = np.sin(psi)

    rotated_ccf = {}

    # lin2008 axes (module docstring): source R = (cth, sth), T = (-sth, cth);
    # receiver R = (-cps, -sps), T = (sps, -cps).
    if "ZZ" in components:
        rotated_ccf["ZZ"] = np.asarray(ccf["ZZ"])

    if "ZR" in components:
        rotated_ccf["ZR"] = -cps * np.asarray(ccf["ZN"]) - sps * np.asarray(ccf["ZE"])

    if "ZT" in components:
        rotated_ccf["ZT"] = sps * np.asarray(ccf["ZN"]) - cps * np.asarray(ccf["ZE"])

    if "RZ" in components:
        rotated_ccf["RZ"] = cth * np.asarray(ccf["NZ"]) + sth * np.asarray(ccf["EZ"])

    if "TZ" in components:
        rotated_ccf["TZ"] = -sth * np.asarray(ccf["NZ"]) + cth * np.asarray(ccf["EZ"])

    # Two axes each: the same expressions (and floating-point operations) in
    # both conventions.
    if "RR" in components:
        rotated_ccf["RR"] = (
            -cth * cps * np.asarray(ccf["NN"])
            - cth * sps * np.asarray(ccf["NE"])
            - sth * cps * np.asarray(ccf["EN"])
            - sth * sps * np.asarray(ccf["EE"])
        )

    if "RT" in components:
        rotated_ccf["RT"] = (
            cth * sps * np.asarray(ccf["NN"])
            - cth * cps * np.asarray(ccf["NE"])
            + sth * sps * np.asarray(ccf["EN"])
            - sth * cps * np.asarray(ccf["EE"])
        )

    if "TR" in components:
        rotated_ccf["TR"] = (
            sth * cps * np.asarray(ccf["NN"])
            + sth * sps * np.asarray(ccf["NE"])
            - cth * cps * np.asarray(ccf["EN"])
            - cth * sps * np.asarray(ccf["EE"])
        )

    if "TT" in components:
        rotated_ccf["TT"] = (
            -sth * sps * np.asarray(ccf["NN"])
            + sth * cps * np.asarray(ccf["NE"])
            + cth * sps * np.asarray(ccf["EN"])
            - cth * cps * np.asarray(ccf["EE"])
        )

    if radial_convention == "legacy":
        # Negation is exact in floating point, so these equal the old
        # formulas bit for bit.
        for comp in CONVENTION_DEPENDENT_COMPONENTS.intersection(rotated_ccf):
            rotated_ccf[comp] = -rotated_ccf[comp]

    return rotated_ccf


def rotate_zrt_ccf_to_zne(
    ccf,
    azimuth,
    back_azimuth,
    degrees=True,
    radial_convention=DEFAULT_RADIAL_CONVENTION,
):
    """Rotate nine-component CCFs from ZRT back to ZNE (inverse of `rotate_zne_ccf_to_zrt`).

    Same conventions as `rotate_zne_ccf_to_zrt` (module docstring), including
    ``radial_convention``; the unit vectors come from `horizontal_axes`. Both
    bases are orthonormal, so the inverse is the transpose:
    ``C_ij = sum_ab S_a[i] R_b[j] X_ab`` with ``i, j`` in N/E and ``a, b`` in
    R/T (Z passes through). All nine ZRT components are required; all nine
    ZNE components are returned. Used to build synthetic ZNE input; checked
    by a round trip in the tests.
    """
    missing = [comp for comp in ZRT_COMPONENTS if comp not in ccf]
    if missing:
        raise KeyError(f"Missing ZRT components: {missing}")
    source_axes, receiver_axes = horizontal_axes(
        azimuth, back_azimuth, radial_convention, degrees=degrees
    )
    source = {"Z": None, **source_axes}
    receiver = {"Z": None, **receiver_axes}
    horizontal = {"N": 0, "E": 1}

    def weight(basis, rotated, physical):
        if rotated == "Z" or physical == "Z":
            return 1.0 if rotated == physical else 0.0
        return basis[rotated][horizontal[physical]]

    result = {}
    for first in "ZNE":
        for second in "ZNE":
            result[first + second] = sum(
                weight(source, comp[0], first) * weight(receiver, comp[1], second) * np.asarray(ccf[comp])
                for comp in ZRT_COMPONENTS
            )
    return result
