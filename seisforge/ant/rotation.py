import numpy as np


_ZRT_COMPONENTS = ("ZZ", "ZR", "ZT", "RZ", "RR", "RT", "TZ", "TR", "TT")

_REQUIRED_BY_COMPONENT = {
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
        components = _ZRT_COMPONENTS
    else:
        components = tuple(components)

    invalid_components = [comp for comp in components if comp not in _ZRT_COMPONENTS]
    if invalid_components:
        raise ValueError(
            "Invalid components requested for rotation: "
            f"{invalid_components}. Valid components are: {_ZRT_COMPONENTS}"
        )

    required_components = tuple(
        dict.fromkeys(
            req
            for comp in components
            for req in _REQUIRED_BY_COMPONENT[comp]
        )
    )

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
