import numpy as np

from seisforge.ant.rotation import required_zne_components, rotate_zne_ccf_to_zrt


def test_rotate_zne_ccf_to_zrt_identity_angles():
    ccf = {
        "ZZ": np.array([1.0, 2.0]),
        "ZN": np.array([3.0, 4.0]),
        "ZE": np.array([5.0, 6.0]),
        "NZ": np.array([7.0, 8.0]),
        "EZ": np.array([9.0, 10.0]),
        "NN": np.array([1.0, 2.0]),
        "NE": np.array([3.0, 4.0]),
        "EN": np.array([5.0, 6.0]),
        "EE": np.array([7.0, 8.0]),
    }

    rotated = rotate_zne_ccf_to_zrt(ccf, azimuth=0.0, back_azimuth=0.0)

    np.testing.assert_allclose(rotated["ZZ"], ccf["ZZ"])
    np.testing.assert_allclose(rotated["ZR"], -ccf["ZN"])
    np.testing.assert_allclose(rotated["ZT"], -ccf["ZE"])
    np.testing.assert_allclose(rotated["RZ"], ccf["NZ"])
    np.testing.assert_allclose(rotated["TZ"], ccf["EZ"])
    np.testing.assert_allclose(rotated["RR"], -ccf["NN"])
    np.testing.assert_allclose(rotated["RT"], -ccf["NE"])
    np.testing.assert_allclose(rotated["TR"], -ccf["EN"])
    np.testing.assert_allclose(rotated["TT"], -ccf["EE"])


def test_rotate_zne_ccf_to_zrt_validates_missing_components():
    try:
        rotate_zne_ccf_to_zrt({"ZZ": np.array([1.0])}, 0.0, 0.0, components=["ZR"])
    except KeyError as exc:
        assert "Missing input ZNE components" in str(exc)
    else:
        raise AssertionError("missing input components should raise KeyError")


def test_required_zne_components_for_selected_outputs():
    assert required_zne_components(["ZR", "RR"]) == (
        "ZN",
        "ZE",
        "NN",
        "NE",
        "EN",
        "EE",
    )


def test_zrt_to_zne_is_the_inverse_of_zne_to_zrt():
    """Round trip both ways for arbitrary angles (orthonormal bases: exact inverse)."""
    import numpy as np

    from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zne_ccf_to_zrt, rotate_zrt_ccf_to_zne

    rng = np.random.default_rng(3)
    zrt = {comp: rng.normal(size=5) for comp in ZRT_COMPONENTS}
    for azimuth, back_azimuth in ((0.0, 180.0), (37.0, 221.5), (300.0, 115.0)):
        zne = rotate_zrt_ccf_to_zne(zrt, azimuth, back_azimuth)
        back = rotate_zne_ccf_to_zrt(zne, azimuth, back_azimuth)
        for comp in ZRT_COMPONENTS:
            np.testing.assert_allclose(back[comp], zrt[comp], atol=1e-12)
        again = rotate_zrt_ccf_to_zne(rotate_zne_ccf_to_zrt(zne, azimuth, back_azimuth), azimuth, back_azimuth)
        for comp in zne:
            np.testing.assert_allclose(again[comp], zne[comp], atol=1e-12)


def test_zrt_to_zne_for_north_pointing_path():
    """Analytic case: azimuth 0, back azimuth 180 -> source R = +N, receiver R = +N."""
    import numpy as np

    from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zrt_ccf_to_zne

    zrt = {comp: np.zeros(1) for comp in ZRT_COMPONENTS}
    zrt["RR"] = np.ones(1)
    zne = rotate_zrt_ccf_to_zne(zrt, 0.0, 180.0)
    np.testing.assert_allclose(zne["NN"], 1.0, atol=1e-15)  # (+1) * (+1)
    for comp in ("ZZ", "NE", "EN", "EE", "ZN", "NZ"):
        np.testing.assert_allclose(zne[comp], 0.0, atol=1e-15)


ZNE_KEYS = ("ZZ", "ZN", "ZE", "NZ", "NN", "NE", "EZ", "EN", "EE")


def _legacy_rotation(ccf, azimuth, back_azimuth):
    """The formulas of `rotate_zne_ccf_to_zrt` before the lin2008 change (commit 8865b26)."""
    th, ps = np.deg2rad(azimuth), np.deg2rad(back_azimuth)
    cth, sth, cps, sps = np.cos(th), np.sin(th), np.cos(ps), np.sin(ps)
    c = {k: np.asarray(v) for k, v in ccf.items()}
    return {
        "ZZ": c["ZZ"],
        "ZR": cps * c["ZN"] + sps * c["ZE"],
        "ZT": -sps * c["ZN"] + cps * c["ZE"],
        "RZ": -cth * c["NZ"] - sth * c["EZ"],
        "TZ": sth * c["NZ"] - cth * c["EZ"],
        "RR": -cth * cps * c["NN"] - cth * sps * c["NE"] - sth * cps * c["EN"] - sth * sps * c["EE"],
        "RT": cth * sps * c["NN"] - cth * cps * c["NE"] + sth * sps * c["EN"] - sth * cps * c["EE"],
        "TR": sth * cps * c["NN"] + sth * sps * c["NE"] - cth * cps * c["EN"] - cth * sps * c["EE"],
        "TT": -sth * sps * c["NN"] + sth * cps * c["NE"] + cth * sps * c["EN"] - cth * cps * c["EE"],
    }


def test_conventions_differ_only_in_the_sign_of_zr_rz_zt_tz_bit_for_bit():
    """legacy reproduces the old formulas exactly; lin2008 flips exactly four components."""
    from seisforge.ant.rotation import ZRT_COMPONENTS

    rng = np.random.default_rng(11)
    ccf = {k: rng.normal(size=4001) for k in ZNE_KEYS}
    for azimuth, back_azimuth in ((37.0, 217.0), (123.4, 301.9), (359.1, 181.3), (250.0, 69.2)):
        old = _legacy_rotation(ccf, azimuth, back_azimuth)
        legacy = rotate_zne_ccf_to_zrt(ccf, azimuth, back_azimuth, radial_convention="legacy")
        lin = rotate_zne_ccf_to_zrt(ccf, azimuth, back_azimuth)
        for comp in ZRT_COMPONENTS:
            np.testing.assert_array_equal(legacy[comp], old[comp], err_msg=comp)
            expected = -old[comp] if comp in ("ZR", "RZ", "ZT", "TZ") else old[comp]
            np.testing.assert_array_equal(lin[comp], expected, err_msg=comp)


def test_lin2008_axes_point_along_the_propagation_at_both_stations():
    """Lin et al. (2008) Fig. 2: R from station 1 towards station 2 at both ends, T = R turned 90 deg clockwise."""
    from seisforge.ant.rotation import horizontal_axes

    for azimuth in (0.0, 37.0, 145.0, 260.0):
        x = np.array([np.cos(np.deg2rad(azimuth)), np.sin(np.deg2rad(azimuth))])  # A -> B, (N, E)
        source, receiver = horizontal_axes(azimuth, azimuth + 180.0)
        for axes in (source, receiver):
            np.testing.assert_allclose(axes["R"], x, atol=1e-15)
            # Clockwise by 90 deg in (N, E): (n, e) -> (-e, n).
            np.testing.assert_allclose(axes["T"], (-axes["R"][1], axes["R"][0]), atol=1e-15)
        legacy_source, legacy_receiver = horizontal_axes(azimuth, azimuth + 180.0, "legacy")
        np.testing.assert_allclose(legacy_source["R"], -x, atol=1e-15)
        np.testing.assert_allclose(legacy_receiver["R"], -x, atol=1e-15)


def test_horizontal_components_match_lin_2008_equation_1():
    """Lin et al. (2008) eq. 1 with theta = azimuth, psi = back azimuth.

    The matrix is eq. 1 as compared term by term with the paper when the
    convention change was planned (the code then matched it); it pins that
    the horizontal-horizontal components stay on eq. 1 in both conventions.
    """
    theta, psi = np.deg2rad(52.0), np.deg2rad(236.5)
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    # Rows TT, RR, TR, RT; columns EE, EN, NN, NE.
    lin_eq1 = np.array(
        [
            [-ct * cp, ct * sp, -st * sp, st * cp],
            [-st * sp, -st * cp, -ct * cp, -ct * sp],
            [-ct * sp, -ct * cp, st * cp, st * sp],
            [-st * cp, st * sp, ct * sp, -ct * cp],
        ]
    )
    columns = ("EE", "EN", "NN", "NE")
    for convention in ("lin2008", "legacy"):
        for j, name in enumerate(columns):
            ccf = {k: np.zeros(1) for k in ZNE_KEYS}
            ccf[name] = np.ones(1)
            rotated = rotate_zne_ccf_to_zrt(ccf, 52.0, 236.5, radial_convention=convention)
            for i, comp in enumerate(("TT", "RR", "TR", "RT")):
                np.testing.assert_allclose(rotated[comp], lin_eq1[i, j], atol=1e-15, err_msg=f"{comp} {name}")


def test_unknown_radial_convention_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="radial_convention"):
        rotate_zne_ccf_to_zrt({k: np.zeros(1) for k in ZNE_KEYS}, 0.0, 180.0, radial_convention="aki")


def test_inverse_round_trip_in_the_legacy_convention():
    from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zrt_ccf_to_zne

    rng = np.random.default_rng(5)
    zrt = {comp: rng.normal(size=5) for comp in ZRT_COMPONENTS}
    zne = rotate_zrt_ccf_to_zne(zrt, 37.0, 221.5, radial_convention="legacy")
    back = rotate_zne_ccf_to_zrt(zne, 37.0, 221.5, radial_convention="legacy")
    for comp in ZRT_COMPONENTS:
        np.testing.assert_allclose(back[comp], zrt[comp], atol=1e-12)
