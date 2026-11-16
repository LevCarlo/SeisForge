"""AFTAN component handling: phase constants per component, branch, run mode and R/T convention.

Inputs and the known answer come from `aftan_component_cases` (diffuse-field
Rayleigh CCFs rotated with the package rotation; see that module and
`seisforge.ant.synthetic`). The synthetic does not read the AFTAN constant
table, so the accuracy tests are an independent check of it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aftan_component_cases import RESULT_KEYS, baseline_cases, run, write_ccfs

BASELINE = Path(__file__).parent / "data" / "aftan_component_baseline.npz"

# (mode, component) whose numbers must stay bit for bit as in the baseline.
# ZZ, RR and TT: the lin2008 change of the rotation and of the phase
# constants leaves them untouched. ZR and RZ changed (CHANGELOG "Changed
# results"); their baseline arrays were regenerated with the fix and are
# checked by `test_changed_components_match_the_regenerated_baseline`.
UNCHANGED = {
    (mode, component)
    for mode in ("pair", "station")
    for component in ("ZZ", "RR", "TT")
}
CHANGED = {(mode, component) for mode in ("pair", "station") for component in ("ZR", "RZ")}


@pytest.fixture(scope="module")
def ccf_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ccf")
    write_ccfs(directory)
    return directory


@pytest.fixture(scope="module")
def legacy_dir(tmp_path_factory):
    """CCFs rotated with the legacy axes, convention recorded in SAC kuser0."""
    directory = tmp_path_factory.mktemp("ccf_legacy")
    write_ccfs(directory, radial_convention="legacy", record_convention=True)
    return directory


@pytest.fixture(scope="module")
def legacy_unrecorded_dir(tmp_path_factory):
    """Legacy-axes CCFs without kuser0, like files rotated before the change."""
    directory = tmp_path_factory.mktemp("ccf_legacy_old")
    write_ccfs(directory, radial_convention="legacy")
    return directory


@pytest.mark.parametrize(
    "mode, component, branch",
    [case for case in baseline_cases() if case[:2] in UNCHANGED | CHANGED],
)
def test_components_match_baseline_bitwise(mode, component, branch, ccf_dir, tmp_path):
    baseline = np.load(BASELINE)
    results = run(mode, component, branch, ccf_dir, tmp_path)
    assert results
    for result in results:
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(
                np.asarray(getattr(result, key)),
                baseline[f"{mode}/{component}/{result.branch}/{key}"],
                err_msg=f"{mode} {component} {result.branch} {key}",
            )


# Largest |c_measured - c_true| (km/s) accepted with the correct phase
# constant. Measured with the right constant: <= 0.010 km/s (Gaussian filter
# and instantaneous-period bias); a pi/2 error in the constant gives about
# 0.25 km/s at this distance and a pi error about 0.6 km/s. 0.03 km/s
# (about 1 %) separates the two with margin on both sides.
PHASE_VELOCITY_TOLERANCE_KM_S = 0.03
COMPONENTS = ("ZZ", "ZR", "RZ", "RR", "TT")


@pytest.mark.parametrize("mode", ["pair", "station"])
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("branch", ["positive", "negative"])
def test_phase_velocity_matches_the_true_curve(mode, component, branch, ccf_dir, tmp_path):
    from aftan_component_cases import phase_velocity_error

    (result,) = run(mode, component, branch, ccf_dir, tmp_path)
    assert result.branch == branch
    assert result.pi_over_4_mode == "auto"
    assert phase_velocity_error(result) < PHASE_VELOCITY_TOLERANCE_KM_S


@pytest.mark.parametrize("mode", ["pair", "station"])
@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("branch", ["positive", "negative"])
def test_legacy_rotation_recorded_in_the_file_gets_legacy_constants(
    mode, component, branch, legacy_dir, tmp_path
):
    from aftan_component_cases import phase_velocity_error

    (result,) = run(mode, component, branch, legacy_dir, tmp_path)
    assert result.radial_convention == "legacy"
    assert result.radial_convention_source == "sac_kuser0"
    assert phase_velocity_error(result) < PHASE_VELOCITY_TOLERANCE_KM_S
    assert not any("pi_over_4" in w or "convention" in w for w in result.warnings)


@pytest.mark.parametrize("component, expected_pi_over_4", [("ZR", 1.0), ("RZ", -3.0)])
def test_unrecorded_legacy_files_need_the_config_option(component, expected_pi_over_4, legacy_unrecorded_dir, tmp_path):
    """Old (pre-change) ZR/RZ files carry no kuser0: auto assumes lin2008 and warns; the option fixes it."""
    from aftan_component_cases import phase_velocity_error

    (assumed,) = run("station", component, "positive", legacy_unrecorded_dir, tmp_path / "auto")
    assert assumed.radial_convention_source == "assumed_default"
    assert any("convention not recorded" in w for w in assumed.warnings)
    assert phase_velocity_error(assumed) > 0.4  # pi off: the case the warning is about
    (fixed,) = run(
        "station", component, "positive", legacy_unrecorded_dir, tmp_path / "legacy",
        aftan={"radial_convention": "legacy"},
    )
    assert fixed.pi_over_4 == expected_pi_over_4
    assert fixed.radial_convention_source == "config"
    assert not any("convention" in w for w in fixed.warnings)
    assert phase_velocity_error(fixed) < PHASE_VELOCITY_TOLERANCE_KM_S


def test_configured_convention_contradicting_the_file_stops(legacy_dir, tmp_path):
    with pytest.raises(ValueError, match="records legacy"):
        run("station", "ZR", "positive", legacy_dir, tmp_path, aftan={"radial_convention": "lin2008"})


def test_station_auto_mode_stops_when_the_component_is_unknown(ccf_dir, tmp_path):
    from aftan_component_cases import sac_name

    directory = tmp_path / "anonymous"
    directory.mkdir()
    (directory / "pair.SAC").write_bytes((ccf_dir / sac_name("ZR")).read_bytes())
    (directory / "prediction.dat").write_bytes((ccf_dir / "prediction.dat").read_bytes())
    with pytest.raises(ValueError, match="needs the stored component"):
        run("station", "ZR", "positive", directory, tmp_path / "out", pattern="pair.SAC")
    # Manual mode still runs and says the component is unknown.
    (result,) = run(
        "station", "ZR", "positive", directory, tmp_path / "manual", pattern="pair.SAC",
        aftan={"pi_over_4": -3.0},
    )
    assert any("component unknown" in w for w in result.warnings)


def test_pair_component_contradicting_the_file_stops(tmp_path):
    from aftan_component_cases import sac_name

    write_ccfs(tmp_path / "kc", kcmpnm=True)
    directory = tmp_path / "mixed"
    directory.mkdir()
    # A ZR file (kcmpnm ZR) under the RZ name of the pair layout.
    (directory / sac_name("RZ")).write_bytes((tmp_path / "kc" / sac_name("ZR")).read_bytes())
    (directory / "prediction.dat").write_bytes((tmp_path / "kc" / "prediction.dat").read_bytes())
    with pytest.raises(ValueError, match="configured component RZ"):
        run("pair", "RZ", "positive", directory, tmp_path / "out")


def test_empirical_greens_function_needs_the_pi_over_2_shifted_constant(ccf_dir, tmp_path):
    """AFTAN measures C(t); G = -dC/dt (Lin et al. 2008, sec. 2) is pi/2 later in phase.

    Positive branch only: -dC/dt of a symmetric CCF is antisymmetric, so its
    time-reversed negative branch has the opposite shift.
    """
    from obspy import read

    from aftan_component_cases import phase_velocity_error, sac_name

    directory = tmp_path / "egf"
    directory.mkdir()
    trace = read(str(ccf_dir / sac_name("ZZ")))[0]
    trace.data = (-np.gradient(trace.data.astype(float), trace.stats.delta)).astype(np.float32)
    trace.write(str(directory / sac_name("ZZ")), format="SAC")
    (directory / "prediction.dat").write_bytes((ccf_dir / "prediction.dat").read_bytes())
    (on_ccf,) = run("station", "ZZ", "positive", ccf_dir, tmp_path / "ccf")
    (egf_auto,) = run("station", "ZZ", "positive", directory, tmp_path / "auto")
    (egf_shifted,) = run(
        "station", "ZZ", "positive", directory, tmp_path / "shifted", aftan={"pi_over_4": 1.0}
    )
    assert on_ccf.pi_over_4 == -1.0 and phase_velocity_error(on_ccf) < PHASE_VELOCITY_TOLERANCE_KM_S
    assert phase_velocity_error(egf_auto) > 0.2  # the CCF constant is pi/2 off on G
    assert phase_velocity_error(egf_shifted) < PHASE_VELOCITY_TOLERANCE_KM_S


@pytest.mark.parametrize("component", COMPONENTS)
def test_pair_and_station_mode_agree_on_the_same_file(component, ccf_dir, tmp_path):
    pair = run("pair", component, "both", ccf_dir, tmp_path / "pair")
    station = run("station", component, "both", ccf_dir, tmp_path / "station")
    assert [r.branch for r in pair] == [r.branch for r in station] == ["positive", "negative"]
    for p, s in zip(pair, station):
        assert p.pi_over_4 == s.pi_over_4
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(getattr(p, key), getattr(s, key), err_msg=key)


@pytest.mark.parametrize(
    "mode, component, aftan, expected",
    [
        ("station", "ZZ", None, None),
        ("pair", "RR", None, None),
        ("pair", "ZR", None, "convention not recorded"),  # no kuser0 in these files
        ("station", "ZR", {"radial_convention": "lin2008"}, None),
        ("station", "RZ", {"pi_over_4": -1.0}, "phase velocity is wrong"),
    ],
)
def test_phase_constant_warning_reaches_log_and_dat(mode, component, aftan, expected, ccf_dir, tmp_path):
    (result,) = run(mode, component, "positive", ccf_dir, tmp_path, aftan=aftan)
    dat = result.output_dat.read_text()
    log = (result.output_dat.parent / "aftan.log").read_text()
    if expected is None:
        assert not result.warnings or not any("pi_over_4" in w or "convention" in w for w in result.warnings)
        assert "phase_velocity_warning" not in dat
    else:
        assert any(expected in w for w in result.warnings)
        assert "# phase_velocity_warning: " in dat and expected in dat
        assert expected in log


def test_phase_constant_warning_rules():
    from seisforge.ant.aftan.phase import phase_constant_warning

    assert phase_constant_warning("ZZ", -1.0) is None
    assert phase_constant_warning("TT", 7.0) is None  # -1 + 8: same phase
    assert phase_constant_warning("ZR", -3.0) is None
    assert phase_constant_warning("RZ", 1.0, radial_convention="lin2008", convention_source="sac_kuser0") is None
    assert phase_constant_warning("RZ", -3.0, radial_convention="legacy") is None
    assert "wrong" in phase_constant_warning("RZ", -1.0)
    assert "wrong" in phase_constant_warning("RZ", -3.0)  # legacy value on lin2008 axes
    assert "wrong" in phase_constant_warning("ZZ", 1.0)  # manual value checked too
    assert "not recorded" in phase_constant_warning("ZR", 5.0, convention_source="assumed_default")
    assert phase_constant_warning("ZZ", -1.0, convention_source="assumed_default") is None
    assert "not been checked" in phase_constant_warning("ZT", -1.0)
    assert "component unknown (why)" in phase_constant_warning(None, -1.0, component_note="why")


def test_resolve_radial_convention_rules():
    from seisforge.ant.aftan.phase import resolve_radial_convention

    assert resolve_radial_convention("auto", None) == ("lin2008", "assumed_default")
    assert resolve_radial_convention("auto", "legacy") == ("legacy", "sac_kuser0")
    assert resolve_radial_convention("legacy", None) == ("legacy", "config")
    assert resolve_radial_convention("legacy", "legacy") == ("legacy", "sac_kuser0")
    with pytest.raises(ValueError, match="records lin2008"):
        resolve_radial_convention("legacy", "lin2008")
    with pytest.raises(ValueError, match="radial_convention"):
        resolve_radial_convention("aki", None)


def test_component_from_sac_reads_the_rotation_header(tmp_path):
    from obspy import read

    from aftan_component_cases import sac_name
    from seisforge.ant.aftan.io import _component_from_sac

    write_ccfs(tmp_path, kcmpnm=True)
    renamed = tmp_path / "pair_one.SAC"
    (tmp_path / sac_name("RZ")).rename(renamed)
    assert _component_from_sac(renamed, read(str(renamed))[0]) == ("RZ", "sac_kcmpnm")


def test_component_from_sac_rules():
    from obspy import Trace

    from seisforge.ant.aftan.io import _component_from_sac

    def trace(kcmpnm=None):
        tr = Trace(data=np.zeros(3))
        tr.stats.sac = {} if kcmpnm is None else {"kcmpnm": kcmpnm}
        return tr

    assert _component_from_sac(Path("A_B_zr.SAC"), trace()) == ("ZR", "file_name")
    assert _component_from_sac(Path("A_B_ZR_pws.SAC"), trace()) == ("ZR", "file_name")
    assert _component_from_sac(Path("COR_A_B.SAC"), trace("RZ")) == ("RZ", "sac_kcmpnm")
    assert _component_from_sac(Path("COR_A_B.SAC"), trace("BHZ"))[0] is None
    assert _component_from_sac(Path("XX.RR_XX.B_ZZ.SAC"), trace())[0] is None  # ambiguous
    assert _component_from_sac(Path("XX.RR_XX.B_ZZ.SAC"), trace("ZZ")) == ("ZZ", "sac_kcmpnm")
    with pytest.raises(ValueError, match="disagrees"):
        _component_from_sac(Path("A_B_ZZ.SAC"), trace("ZR"))
