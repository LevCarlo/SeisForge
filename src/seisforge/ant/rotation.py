import numpy as np


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
    at the receiver station.

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

    if "ZZ" in components:
        rotated_ccf["ZZ"] = np.asarray(ccf["ZZ"])

    if "ZR" in components:
        rotated_ccf["ZR"] = cps * np.asarray(ccf["ZN"]) + sps * np.asarray(ccf["ZE"])

    if "ZT" in components:
        rotated_ccf["ZT"] = -sps * np.asarray(ccf["ZN"]) + cps * np.asarray(ccf["ZE"])

    if "RZ" in components:
        rotated_ccf["RZ"] = -cth * np.asarray(ccf["NZ"]) - sth * np.asarray(ccf["EZ"])

    if "TZ" in components:
        rotated_ccf["TZ"] = sth * np.asarray(ccf["NZ"]) - cth * np.asarray(ccf["EZ"])

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


def rotate_zrt_ccf_to_zne(ccf, azimuth, back_azimuth, degrees=True):
    """Rotate nine-component CCFs from ZRT back to ZNE (inverse of `rotate_zne_ccf_to_zrt`).

    Same conventions as `rotate_zne_ccf_to_zrt`: the first letter is the
    virtual-source component, the second the receiver component. With
    theta = azimuth and psi = back azimuth, the unit vectors in (N, E) are

        source:   R = (-cos theta, -sin theta),  T = (sin theta, -cos theta)
        receiver: R = (cos psi, sin psi),        T = (-sin psi, cos psi)

    (read off the forward rotation). Both bases are orthonormal, so the
    inverse is the transpose: ``C_ij = sum_ab S_a[i] R_b[j] X_ab`` with
    ``i, j`` in N/E and ``a, b`` in R/T (Z passes through). All nine ZRT
    components are required; all nine ZNE components are returned. Used to
    build synthetic ZNE input; checked by a round trip in the tests.
    """
    missing = [comp for comp in ZRT_COMPONENTS if comp not in ccf]
    if missing:
        raise KeyError(f"Missing ZRT components: {missing}")
    theta = np.deg2rad(azimuth) if degrees else azimuth
    psi = np.deg2rad(back_azimuth) if degrees else back_azimuth
    source = {"Z": None, "R": (-np.cos(theta), -np.sin(theta)), "T": (np.sin(theta), -np.cos(theta))}
    receiver = {"Z": None, "R": (np.cos(psi), np.sin(psi)), "T": (-np.sin(psi), np.cos(psi))}
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
