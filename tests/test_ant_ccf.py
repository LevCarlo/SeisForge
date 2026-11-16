"""`seisforge ant rotate-ccf`: one station pair, path templates, angles from config or SAC headers."""

from __future__ import annotations

import numpy as np
from obspy import Trace, read
import pytest
import yaml

from seisforge.ant.ccf_rotate import build_rotate_ccf_job, run_rotate_ccf_config
from seisforge.ant.rotation import interstation_azimuths


def _write_sac(path, data, sac=None):
    trace = Trace(data=np.asarray(data, dtype=np.float32))
    trace.stats.delta = 1.0
    if sac is not None:
        trace.stats.sac = sac
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.write(str(path), format="SAC")


def _config(tmp_path, **rotation):
    config = {
        "io": {"input_template": "in/S1_S2_{component}_pws.SAC", "output_template": "out/S1_S2_{component}.SAC"},
        "rotation": {"components": ["ZR", "ZT"], **rotation},
    }
    path = tmp_path / "rotate.yml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_writes_requested_components_with_configured_angles(tmp_path, assert_provenance):
    _write_sac(tmp_path / "in" / "S1_S2_ZN_pws.SAC", [1.0, 2.0])
    _write_sac(tmp_path / "in" / "S1_S2_ZE_pws.SAC", [3.0, 4.0])
    _write_sac(tmp_path / "in" / "S1_S2_ZN_ls.SAC", [10.0, 20.0])  # other stack product: not read

    result = run_rotate_ccf_config(_config(tmp_path, azimuth=0.0, back_azimuth=0.0))

    assert set(result.written) == {"ZR", "ZT"}
    assert result.log_file == tmp_path / "out" / "rotate_ccf.log"
    zr = read(str(tmp_path / "out" / "S1_S2_ZR.SAC"))[0]
    # azimuth = back azimuth = 0: receiver R = -N, T = -E (Lin et al. 2008 axes).
    np.testing.assert_allclose(zr.data, [-1.0, -2.0])
    np.testing.assert_allclose(read(str(tmp_path / "out" / "S1_S2_ZT.SAC"))[0].data, [-3.0, -4.0])
    assert zr.stats.sac.kcmpnm == "ZR"
    assert zr.stats.sac.kuser0.strip() == "lin2008"
    log = result.log_file.read_text()
    assert "source=config or command line" in log
    assert "radial_convention: lin2008 (SAC kuser0)" in log
    assert "read ZN:" in log and "read NN:" not in log
    assert_provenance(log)


def test_angles_from_sac_az_baz(tmp_path):
    # lcalda False: ObsPy would otherwise recompute az/baz from the coordinates on write.
    sac = {"az": 0.0, "baz": 180.0, "evla": 0.0, "evlo": 0.0, "stla": 0.0, "stlo": 1.0, "lcalda": False}
    _write_sac(tmp_path / "in" / "S1_S2_ZN_pws.SAC", [1.0, 2.0], sac)
    _write_sac(tmp_path / "in" / "S1_S2_ZE_pws.SAC", [3.0, 4.0], sac)
    result = run_rotate_ccf_config(_config(tmp_path))
    # az/baz win over the coordinates (which would give 90/270).
    np.testing.assert_allclose(read(str(result.written["ZR"]))[0].data, [1.0, 2.0])
    assert "source=SAC headers az/baz" in result.log_file.read_text()


def test_angles_from_sac_coordinates(tmp_path):
    sac = {"evla": 0.0, "evlo": 0.0, "stla": 1.0, "stlo": 0.0, "lcalda": False}
    _write_sac(tmp_path / "in" / "S1_S2_ZN_pws.SAC", [1.0, 2.0], sac)
    _write_sac(tmp_path / "in" / "S1_S2_ZE_pws.SAC", [3.0, 4.0], sac)
    result = run_rotate_ccf_config(_config(tmp_path))
    # Receiver due north of the source: azimuth 0, back azimuth 180, receiver R = +N, T = +E.
    np.testing.assert_allclose(read(str(result.written["ZR"]))[0].data, [1.0, 2.0])
    np.testing.assert_allclose(read(str(result.written["ZT"]))[0].data, [3.0, 4.0])
    assert "azimuth=0, back_azimuth=180, source=computed from SAC evla/evlo/stla/stlo" in result.log_file.read_text()


def test_interstation_azimuths_rules():
    assert interstation_azimuths(az=10.0, baz=190.0, evla=0.0, evlo=0.0, stla=1.0, stlo=0.0)[:2] == (10.0, 190.0)
    azimuth, back_azimuth, source = interstation_azimuths(evla=0.0, evlo=0.0, stla=0.0, stlo=1.0)
    assert azimuth == pytest.approx(90.0) and back_azimuth == pytest.approx(270.0)
    assert source.startswith("computed")
    with pytest.raises(ValueError, match="az and baz"):
        interstation_azimuths(az=10.0, evla=0.0)


def test_existing_output_is_not_replaced_without_overwrite(tmp_path):
    _write_sac(tmp_path / "in" / "S1_S2_ZN_pws.SAC", [1.0, 2.0])
    _write_sac(tmp_path / "in" / "S1_S2_ZE_pws.SAC", [3.0, 4.0])
    config = _config(tmp_path, azimuth=0.0, back_azimuth=0.0)
    run_rotate_ccf_config(config)
    before = (tmp_path / "out" / "S1_S2_ZR.SAC").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        run_rotate_ccf_config(config, azimuth=90.0, back_azimuth=270.0)
    assert (tmp_path / "out" / "S1_S2_ZR.SAC").read_bytes() == before
    run_rotate_ccf_config(config, azimuth=90.0, back_azimuth=270.0, overwrite=True)
    assert (tmp_path / "out" / "S1_S2_ZR.SAC").read_bytes() != before


def test_wildcards_must_match_exactly_one_file(tmp_path):
    _write_sac(tmp_path / "in" / "S1_S2_ZN_pws.SAC", [1.0, 2.0])
    _write_sac(tmp_path / "in" / "S1_S2_ZE_pws.SAC", [3.0, 4.0])
    config = _config(tmp_path, azimuth=0.0, back_azimuth=0.0)
    result = run_rotate_ccf_config(config, input_template="in/*_{component}_pws.SAC")
    assert set(result.written) == {"ZR", "ZT"}
    _write_sac(tmp_path / "in" / "S9_S2_ZN_pws.SAC", [1.0, 2.0])
    with pytest.raises(FileNotFoundError, match="expected one file"):
        run_rotate_ccf_config(config, input_template="in/*_{component}_pws.SAC", overwrite=True)


def test_missing_input_is_an_error_naming_the_file(tmp_path):
    _write_sac(tmp_path / "in" / "S1_S2_ZN_pws.SAC", [1.0, 2.0])
    with pytest.raises(FileNotFoundError, match="S1_S2_ZE_pws.SAC"):
        run_rotate_ccf_config(_config(tmp_path, azimuth=0.0, back_azimuth=0.0))


@pytest.mark.parametrize(
    "config, match",
    [
        ({"io": {"input_template": "a.SAC", "output_template": "b_{component}.SAC"}}, "input_template"),
        ({"io": {"input_template": "a_{component}.SAC", "output_template": "b_{component}.SAC"},
          "rotation": {"azimuth": 10.0}}, "both azimuth and back_azimuth"),
        ({"io": {"input_template": "a_{component}.SAC", "output_template": "b_{component}.SAC",
                 "root_datadir": "x"}}, r"unknown keys: io.root_datadir"),
        ({"io": {"input_template": "a_{component}.SAC", "output_template": "b_{component}.SAC"},
          "rotation": {"components": ["ZX"]}}, "Invalid components"),
    ],
)
def test_config_errors(config, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        build_rotate_ccf_job(config, base_dir=tmp_path)
