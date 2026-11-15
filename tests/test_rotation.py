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
    np.testing.assert_allclose(rotated["ZR"], ccf["ZN"])
    np.testing.assert_allclose(rotated["ZT"], ccf["ZE"])
    np.testing.assert_allclose(rotated["RZ"], -ccf["NZ"])
    np.testing.assert_allclose(rotated["TZ"], -ccf["EZ"])
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
    """Analytic case: azimuth 0, back azimuth 180 -> source R = -N, receiver R = -N."""
    import numpy as np

    from seisforge.ant.rotation import ZRT_COMPONENTS, rotate_zrt_ccf_to_zne

    zrt = {comp: np.zeros(1) for comp in ZRT_COMPONENTS}
    zrt["RR"] = np.ones(1)
    zne = rotate_zrt_ccf_to_zne(zrt, 0.0, 180.0)
    np.testing.assert_allclose(zne["NN"], 1.0, atol=1e-15)  # (-1) * (-1)
    for comp in ("ZZ", "NE", "EN", "EE", "ZN", "NZ"):
        np.testing.assert_allclose(zne[comp], 0.0, atol=1e-15)
