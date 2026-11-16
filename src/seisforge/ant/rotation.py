"""Rotation of nine-component ambient-noise CCFs between ZNE and ZRT.

Conventions
-----------
- CCF component ``ab``: first letter = component at the virtual-source station
  A, second letter = component at the receiver station B
  (``C_ab(t) = sum_tau u_a^A(tau) u_b^B(tau + t)``; positive lag is
  propagation A -> B).
- Angles: ``theta`` = azimuth from A to B, ``psi`` = back azimuth (from B to
  A), clockwise from north. Unit vectors are written as (N, E).
- Radial/transverse axes: at *both* stations R points from A towards B (along the propagation of the positive
  lag) and T is R turned 90 degrees clockwise (seen from above), as drawn in
  Fig. 2 of Lin, Moschetti & Ritzwoller (2008):

      source A:   R = ( cos theta,  sin theta),  T = (-sin theta,  cos theta)
      receiver B: R = (-cos psi,   -sin psi),    T = ( sin psi,   -cos psi)

  In matrix form ``C_ZRT = M_s C_ZNE M_r^T`` with
  ``M_s = [[1,0,0],[0,cos theta,sin theta],[0,-sin theta,cos theta]]`` and
  ``M_r = [[1,0,0],[0,-cos psi,-sin psi],[0,sin psi,-cos psi]]``.
  The coefficients of TT, RR, TR and RT agree with Lin et al. (2008), eq. 1.
- SeisForge before 0.2.0 reversed R and T at both stations; its ZR, RZ, ZT
  and TZ have the opposite sign (RR, RT, TR, TT and ZZ are the same). Such
  files are not recognised: rotate them again (CHANGELOG, "Changed results").

`seisforge ant rotate-ccf` writes ``lin2008`` into SAC ``kuser0``
(`RADIAL_CONVENTION_SAC_HEADER`) as a record of how the file was made; no
code reads it.

Reference: Lin, F.-C., Moschetti, M. P., Ritzwoller, M. H. (2008). Surface
wave tomography of the western United States from ambient seismic noise:
Rayleigh and Love wave phase velocity maps. Geophys. J. Int. 173, 281-298,
doi:10.1111/j.1365-246X.2008.03720.x.
"""

import numpy as np

__all__ = [
    "RADIAL_CONVENTION",
    "RADIAL_CONVENTION_SAC_HEADER",
    "REQUIRED_ZNE_BY_ZRT_COMPONENT",
    "ZRT_COMPONENTS",
    "interstation_azimuths",
    "horizontal_axes",
    "required_zne_components",
    "rotate_zne_ccf_to_zrt",
    "rotate_zrt_ccf_to_zne",
]

# Provenance label written by rotate-ccf (SAC kuser0, 8 characters max).
RADIAL_CONVENTION = "lin2008"
RADIAL_CONVENTION_SAC_HEADER = "kuser0"

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


def interstation_azimuths(
    *,
    az=None,
    baz=None,
    evla=None,
    evlo=None,
    stla=None,
    stlo=None,
):
    """Azimuth (A to B) and back azimuth (B to A) in degrees, and where they came from.

    ``az``/``baz`` (SAC header convention) are used when both are given;
    otherwise they are computed on the WGS84 ellipsoid from the virtual
    source ``evla/evlo`` and the receiver ``stla/stlo`` (degrees) with
    `obspy.geodetics.gps2dist_azimuth`. Missing values are None. Returns
    ``(azimuth, back_azimuth, source)``; raises ValueError when neither set
    is complete.
    """
    if az is not None and baz is not None:
        return float(az), float(baz), "SAC headers az/baz"
    if None in (evla, evlo, stla, stlo):
        raise ValueError(
            "azimuth/back_azimuth need SAC az and baz, or evla, evlo, stla and stlo."
        )
    from obspy.geodetics.base import gps2dist_azimuth

    _, azimuth, back_azimuth = gps2dist_azimuth(evla, evlo, stla, stlo)
    return azimuth, back_azimuth, "computed from SAC evla/evlo/stla/stlo"


def horizontal_axes(azimuth, back_azimuth, degrees=True):
    """Unit vectors (N, E) of R and T at the source and receiver stations.

    Returns ``(source, receiver)``, each ``{"R": (n, e), "T": (n, e)}``; see the
    module docstring for the convention.
    """
    theta = np.deg2rad(azimuth) if degrees else azimuth
    psi = np.deg2rad(back_azimuth) if degrees else back_azimuth
    source = {
        "R": (np.cos(theta), np.sin(theta)),
        "T": (-np.sin(theta), np.cos(theta)),
    }
    receiver = {
        "R": (-np.cos(psi), -np.sin(psi)),
        "T": (np.sin(psi), -np.cos(psi)),
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
    Returns
    -------
    rotated_ccf : dict
        Rotated cross-correlation functions in the ZRT coordinate system.
    """
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

    # Axes (module docstring): source R = (cth, sth), T = (-sth, cth);
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

    # Two horizontal axes each: the same expressions (and floating-point
    # operations) as before 0.2.0, so these are bit for bit unchanged.
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

    return rotated_ccf


def rotate_zrt_ccf_to_zne(
    ccf,
    azimuth,
    back_azimuth,
    degrees=True,
):
    """Rotate nine-component CCFs from ZRT back to ZNE (inverse of `rotate_zne_ccf_to_zrt`).

    Same conventions as `rotate_zne_ccf_to_zrt` (module docstring); the unit
    vectors come from `horizontal_axes`. Both
    bases are orthonormal, so the inverse is the transpose:
    ``C_ij = sum_ab S_a[i] R_b[j] X_ab`` with ``i, j`` in N/E and ``a, b`` in
    R/T (Z passes through). All nine ZRT components are required; all nine
    ZNE components are returned. Used to build synthetic ZNE input; checked
    by a round trip in the tests.
    """
    missing = [comp for comp in ZRT_COMPONENTS if comp not in ccf]
    if missing:
        raise KeyError(f"Missing ZRT components: {missing}")
    source_axes, receiver_axes = horizontal_axes(azimuth, back_azimuth, degrees=degrees)
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
