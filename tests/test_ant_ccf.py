from __future__ import annotations

import numpy as np
from obspy import Trace
import pytest

from seisforge.ant.ccf_rotate import (
    build_rotate_ccf_jobs,
    run_rotate_ccf_config,
)


def _write_sac(path, data, sac=None):
    trace = Trace(data=np.asarray(data, dtype=np.float32))
    trace.stats.delta = 1.0
    if sac is not None:
        trace.stats.sac = sac
    path.parent.mkdir(parents=True, exist_ok=True)
    trace.write(str(path), format="SAC")


def _base_config(**extra):
    config = {
        "io": {
            "root_datadir": "CC_ZNE",
            "output_root_datadir": "CC_ZRT",
            "source_station": "WT.2001",
            "receiver_station": "WT.2100",
            "input_template": "*_{component}_pws.SAC",
        },
        "rotation": {"obj_components": "ZR,ZT"},
        "azimuth": 0.0,
        "back_azimuth": 0.0,
    }
    for key, value in extra.items():
        config[key] = value
    return config


def test_build_rotate_ccf_jobs_resolves_station_pair_paths(tmp_path):
    job = build_rotate_ccf_jobs(_base_config(), base_dir=tmp_path)[0]

    assert job.input_dir == tmp_path / "CC_ZNE" / "WT.2001" / "WT.2001_WT.2100"
    assert job.output_dir == tmp_path / "CC_ZRT" / "WT.2001" / "WT.2001_WT.2100"
    assert job.input_template == "*_{component}_pws.SAC"
    assert job.obj_components == ("ZR", "ZT")
    assert job.name == "WT.2001_WT.2100"


def test_build_rotate_ccf_jobs_allows_header_inferred_azimuths(tmp_path):
    config = _base_config(azimuth=None, back_azimuth=None)
    config.pop("azimuth")
    config.pop("back_azimuth")

    job = build_rotate_ccf_jobs(config, base_dir=tmp_path)[0]

    assert job.azimuth is None
    assert job.back_azimuth is None


def test_build_rotate_ccf_jobs_accepts_components_key(tmp_path):
    config = _base_config()
    config["rotation"] = {"components": "ZR,ZT"}

    job = build_rotate_ccf_jobs(config, base_dir=tmp_path)[0]

    assert job.obj_components == ("ZR", "ZT")


def test_run_rotate_ccf_config_writes_requested_sac_outputs(tmp_path):
    input_dir = tmp_path / "CC_ZNE" / "S1" / "S1_S2"
    _write_sac(input_dir / "S1_S2_ZN_pws.SAC", [1.0, 2.0])
    _write_sac(input_dir / "S1_S2_ZE_pws.SAC", [3.0, 4.0])
    _write_sac(input_dir / "S1_S2_ZN_ls.SAC", [10.0, 20.0])
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  root_datadir: CC_ZNE",
                "  output_root_datadir: CC_ZRT",
                "  source_station: S1",
                "  receiver_station: S2",
                "  input_template: '*_{component}_pws.SAC'",
                "rotation:",
                "  obj_components: [ZR, ZT]",
                "azimuth: 0.0",
                "back_azimuth: 0.0",
            ]
        )
    )

    results = run_rotate_ccf_config(config_file)

    assert len(results) == 1
    assert set(results[0].written) == {"ZR", "ZT"}
    output_dir = tmp_path / "CC_ZRT" / "S1" / "S1_S2"
    assert results[0].log_file == output_dir / "rotate_ccf.log"
    log_text = results[0].log_file.read_text()
    assert "required_input_components inferred from obj_components: ZN,ZE" in log_text
    assert "specified in config or CLI; skip SAC HEADER" in log_text
    from obspy import read

    np.testing.assert_allclose(
        read(str(output_dir / "S1_S2_ZR_pws.SAC"))[0].data,
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        read(str(output_dir / "S1_S2_ZT_pws.SAC"))[0].data,
        [3.0, 4.0],
    )
    assert not (output_dir / "S1_S2_ZR_ls.SAC").exists()


def test_run_rotate_ccf_config_prefers_sac_az_baz_headers(tmp_path):
    input_dir = tmp_path / "CC_ZNE" / "S1" / "S1_S2"
    sac = {"az": 0.0, "baz": 0.0}
    _write_sac(input_dir / "S1_S2_ZN_pws.SAC", [1.0, 2.0], sac=sac)
    _write_sac(input_dir / "S1_S2_ZE_pws.SAC", [3.0, 4.0], sac=sac)
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  root_datadir: CC_ZNE",
                "  output_root_datadir: CC_ZRT",
                "  source_station: S1",
                "  receiver_station: S2",
                "  input_template: '*_{component}_pws.SAC'",
                "rotation:",
                "  obj_components: [ZR, ZT]",
            ]
        )
    )

    result = run_rotate_ccf_config(config_file)[0]

    from obspy import read

    output_dir = tmp_path / "CC_ZRT" / "S1" / "S1_S2"
    np.testing.assert_allclose(
        read(str(output_dir / "S1_S2_ZR_pws.SAC"))[0].data,
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        read(str(output_dir / "S1_S2_ZT_pws.SAC"))[0].data,
        [3.0, 4.0],
    )
    assert "source=SAC headers az/baz" in result.log_file.read_text()
    assert "SAC HEADER az/baz: success" in result.log_file.read_text()
    assert "coordinate fallback: skipped" in result.log_file.read_text()


def test_run_rotate_ccf_config_infers_azimuths_from_sac_headers(tmp_path):
    input_dir = tmp_path / "CC_ZNE" / "S1" / "S1_S2"
    sac = {
        "evla": 0.0,
        "evlo": 0.0,
        "stla": 1.0,
        "stlo": 0.0,
        "lcalda": False,
    }
    _write_sac(input_dir / "S1_S2_ZN_pws.SAC", [1.0, 2.0], sac=sac)
    _write_sac(input_dir / "S1_S2_ZE_pws.SAC", [3.0, 4.0], sac=sac)
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  root_datadir: CC_ZNE",
                "  output_root_datadir: CC_ZRT",
                "  source_station: S1",
                "  receiver_station: S2",
                "  input_template: '*_{component}_pws.SAC'",
                "rotation:",
                "  obj_components: [ZR, ZT]",
            ]
        )
    )

    run_rotate_ccf_config(config_file)

    from obspy import read

    output_dir = tmp_path / "CC_ZRT" / "S1" / "S1_S2"
    np.testing.assert_allclose(
        read(str(output_dir / "S1_S2_ZR_pws.SAC"))[0].data,
        [-1.0, -2.0],
    )
    np.testing.assert_allclose(
        read(str(output_dir / "S1_S2_ZT_pws.SAC"))[0].data,
        [-3.0, -4.0],
    )
    log_text = (output_dir / "rotate_ccf.log").read_text()
    assert "azimuth/back_azimuth: not fully specified" in log_text
    assert "failed to read from SAC HEADER" in log_text
    assert "coordinate fallback: success" in log_text
    assert "source=computed from SAC evla/evlo/stla/stlo" in log_text


def test_run_rotate_ccf_config_preserves_input_naming_by_replacing_component(tmp_path):
    input_dir = tmp_path / "CC_ZNE" / "WT.2001" / "WT.2001_WT.2100"
    name = "WT.2001_WT.2100"
    inputs = {
        "ZZ": [0.0, 0.0],
        "ZN": [1.0, 2.0],
        "ZE": [3.0, 4.0],
        "NN": [5.0, 6.0],
        "NE": [7.0, 8.0],
        "EN": [9.0, 10.0],
        "EE": [11.0, 12.0],
    }
    for component, data in inputs.items():
        _write_sac(input_dir / f"{name}_{component}_pws.SAC", data)
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  root_datadir: CC_ZNE",
                "  output_root_datadir: CC_ZRT",
                "  source_station: WT.2001",
                "  receiver_station: WT.2100",
                "  input_template: '*_{component}_pws.SAC'",
                "rotation:",
                "  obj_components: [ZZ, ZR, RR]",
                "azimuth: 0.0",
                "back_azimuth: 0.0",
            ]
        )
    )

    result = run_rotate_ccf_config(config_file)[0]

    output_dir = tmp_path / "CC_ZRT" / "WT.2001" / "WT.2001_WT.2100"
    assert result.written["ZZ"] == output_dir / f"{name}_ZZ_pws.SAC"
    assert result.written["ZR"] == output_dir / f"{name}_ZR_pws.SAC"
    assert result.written["RR"] == output_dir / f"{name}_RR_pws.SAC"
    assert not (output_dir / "ZR.SAC").exists()


def test_run_rotate_ccf_config_does_not_read_diagonal_from_reciprocal_pair(tmp_path):
    reciprocal_dir = tmp_path / "CC_ZNE" / "WT.2001" / "WT.2001_WT.2009"
    reciprocal_name = "WT.2001_WT.2009"
    for component, data in {
        "ZZ": [7.0, 7.0],
        "NN": [8.0, 8.0],
        "EE": [9.0, 9.0],
    }.items():
        _write_sac(reciprocal_dir / f"{reciprocal_name}_{component}_pws.SAC", data)

    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  root_datadir: CC_ZNE",
                "  output_root_datadir: CC_ZRT",
                "  source_station: WT.2009",
                "  receiver_station: WT.2001",
                "  input_template: '*_{component}_pws.SAC'",
                "rotation:",
                "  obj_components: [ZZ, RR]",
                "azimuth: 0.0",
                "back_azimuth: 0.0",
            ]
        )
    )

    with pytest.raises(FileNotFoundError, match="required input component 'ZZ'"):
        run_rotate_ccf_config(config_file)


def test_run_rotate_ccf_config_does_not_read_offdiagonal_from_reciprocal_pair(tmp_path):
    reciprocal_dir = tmp_path / "CC_ZNE" / "WT.2001" / "WT.2001_WT.2009"
    _write_sac(reciprocal_dir / "WT.2001_WT.2009_ZN_pws.SAC", [1.0, 1.0])
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  root_datadir: CC_ZNE",
                "  output_root_datadir: CC_ZRT",
                "  source_station: WT.2009",
                "  receiver_station: WT.2001",
                "  input_template: '*_{component}_pws.SAC'",
                "rotation:",
                "  obj_components: [ZR]",
                "azimuth: 0.0",
                "back_azimuth: 0.0",
            ]
        )
    )

    with pytest.raises(FileNotFoundError, match="required input component 'ZN'"):
        run_rotate_ccf_config(config_file)
