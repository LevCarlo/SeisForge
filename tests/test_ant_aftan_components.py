"""AFTAN component handling: phase constants per component, branch and run mode.

Inputs and the known answer come from `aftan_component_cases` (diffuse-field
Rayleigh CCFs rotated with the package rotation; see that module and
`seisforge.ant.synthetic`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aftan_component_cases import RESULT_KEYS, baseline_cases, run, write_ccfs

BASELINE = Path(__file__).parent / "data" / "aftan_component_baseline.npz"

# (mode, component) whose numbers must stay bit for bit as in the baseline.
# All of them for now: T19 changes no AFTAN number. The ZR/RZ fix (follow-up
# task, docs/aftan_phase_constants.md) removes its cases from this set.
UNCHANGED = {
    (mode, component)
    for mode in ("pair", "station")
    for component in ("ZZ", "ZR", "RZ", "RR", "TT")
}


@pytest.fixture(scope="module")
def ccf_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ccf")
    write_ccfs(directory)
    return directory


@pytest.mark.parametrize(
    "mode, component, branch",
    [case for case in baseline_cases() if case[:2] in UNCHANGED],
)
def test_unchanged_components_match_baseline_bitwise(mode, component, branch, ccf_dir, tmp_path):
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

# Known wrong phase constants (docs/aftan_phase_constants.md). strict=True:
# the follow-up fix makes these pass, and pytest then asks to drop the mark.
_KNOWN_WRONG = {
    ("pair", "ZR", "negative"): "pair mode swaps ZR->RZ on the negative branch and uses -1; needs +1",
    ("pair", "RZ", "positive"): "table RZ=-1; needs -3",
    ("pair", "RZ", "negative"): "pair mode swaps RZ->ZR on the negative branch and uses +1; needs -3",
    ("station", "ZR", "positive"): "station mode uses -1 for every file; ZR needs +1",
    ("station", "ZR", "negative"): "station mode uses -1 for every file; ZR needs +1",
    ("station", "RZ", "positive"): "station mode uses -1 for every file; RZ needs -3",
    ("station", "RZ", "negative"): "station mode uses -1 for every file; RZ needs -3",
}


def _accuracy_cases():
    for mode in ("pair", "station"):
        for component in ("ZZ", "ZR", "RZ", "RR", "TT"):
            for branch in ("positive", "negative"):
                reason = _KNOWN_WRONG.get((mode, component, branch))
                marks = [pytest.mark.xfail(strict=True, reason=f"known issue, follow-up task: {reason}")] if reason else []
                yield pytest.param(mode, component, branch, marks=marks, id=f"{mode}-{component}-{branch}")


@pytest.mark.parametrize("mode, component, branch", list(_accuracy_cases()))
def test_phase_velocity_matches_the_true_curve(mode, component, branch, ccf_dir, tmp_path):
    from aftan_component_cases import phase_velocity_error

    (result,) = run(mode, component, branch, ccf_dir, tmp_path)
    assert result.branch == branch
    assert phase_velocity_error(result) < PHASE_VELOCITY_TOLERANCE_KM_S


@pytest.mark.parametrize(
    "component",
    [
        "ZZ",
        "RR",
        "TT",
        pytest.param("ZR", marks=pytest.mark.xfail(strict=True, reason="known issue: station mode uses -1 for ZR")),
        pytest.param("RZ", marks=pytest.mark.xfail(strict=True, reason="known issue: modes differ on the RZ negative branch")),
    ],
)
def test_pair_and_station_mode_agree_on_the_same_file(component, ccf_dir, tmp_path):
    pair = run("pair", component, "both", ccf_dir, tmp_path / "pair")
    station = run("station", component, "both", ccf_dir, tmp_path / "station")
    assert [r.branch for r in pair] == [r.branch for r in station] == ["positive", "negative"]
    for p, s in zip(pair, station):
        assert p.pi_over_4 == s.pi_over_4
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(getattr(p, key), getattr(s, key), err_msg=key)


@pytest.mark.parametrize(
    "mode, component, expected",
    [
        ("station", "ZZ", None),
        ("pair", "RR", None),
        ("pair", "ZR", "under revision"),  # positive: +1, agrees with the check
        ("station", "ZR", "phase velocity is wrong"),
        ("station", "RZ", "phase velocity is wrong"),
    ],
)
def test_phase_constant_warning_reaches_log_and_dat(mode, component, expected, ccf_dir, tmp_path):
    (result,) = run(mode, component, "positive", ccf_dir, tmp_path)
    dat = result.output_dat.read_text()
    log = (result.output_dat.parent / "aftan.log").read_text()
    if expected is None:
        assert not any("pi_over_4" in w for w in result.warnings)
        assert "phase_velocity_warning" not in dat
    else:
        assert any(expected in w for w in result.warnings)
        assert "# phase_velocity_warning: " in dat and expected in dat
        assert expected in log


def test_phase_constant_warning_rules():
    from seisforge.ant.aftan.phase import phase_constant_warning

    assert phase_constant_warning("ZZ", -1.0) is None
    assert phase_constant_warning("TT", 7.0) is None  # -1 + 8: same phase
    assert "wrong" in phase_constant_warning("RZ", -1.0)
    assert "wrong" in phase_constant_warning("ZZ", 1.0)  # manual value checked too
    assert "under revision" in phase_constant_warning("RZ", 5.0)
    assert "not been checked" in phase_constant_warning("ZT", -1.0)
    assert "component unknown (why)" in phase_constant_warning(None, -1.0, component_note="why")


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
