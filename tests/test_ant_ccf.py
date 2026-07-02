from __future__ import annotations

import numpy as np
from obspy import Trace

from seisforge.ant.ccf import (
    build_rotate_ccf_jobs,
    run_rotate_ccf_config,
)


def _write_sac(path, data, sac=None):
    trace = Trace(data=np.asarray(data, dtype=np.float32))
    trace.stats.delta = 1.0
    if sac is not None:
        trace.stats.sac = sac
    trace.write(str(path), format="SAC")


def test_build_rotate_ccf_jobs_resolves_relative_paths(tmp_path):
    config = {
        "io": {
            "input_dir": "zne",
            "output_dir": "zrt",
            "input_template": "{name}.{component}.SAC",
        },
        "rotation": {"obj_components": "ZR,ZT"},
        "jobs": [
            {"name": "pair01", "azimuth": 10.0, "back_azimuth": 190.0},
        ],
    }

    job = build_rotate_ccf_jobs(config, base_dir=tmp_path)[0]

    assert job.input_dir == tmp_path / "zne"
    assert job.output_dir == tmp_path / "zrt"
    assert job.input_template == "{name}.{component}.SAC"
    assert job.obj_components == ("ZR", "ZT")
    assert job.name == "pair01"


def test_build_rotate_ccf_jobs_allows_header_inferred_azimuths(tmp_path):
    config = {
        "io": {"input_dir": "zne", "output_dir": "zrt"},
        "rotation": {"obj_components": "ZR,ZT"},
    }

    job = build_rotate_ccf_jobs(config, base_dir=tmp_path)[0]

    assert job.azimuth is None
    assert job.back_azimuth is None


def test_build_rotate_ccf_jobs_accepts_legacy_components_key(tmp_path):
    config = {
        "io": {"input_dir": "zne", "output_dir": "zrt"},
        "rotation": {"components": "ZR,ZT"},
    }

    job = build_rotate_ccf_jobs(config, base_dir=tmp_path)[0]

    assert job.obj_components == ("ZR", "ZT")


def test_run_rotate_ccf_config_writes_requested_sac_outputs(tmp_path):
    input_dir = tmp_path / "zne"
    input_dir.mkdir()
    _write_sac(input_dir / "ZN.SAC", [1.0, 2.0])
    _write_sac(input_dir / "ZE.SAC", [3.0, 4.0])
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  input_dir: zne",
                "  output_dir: zrt",
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
    assert results[0].log_file == tmp_path / "zrt" / "rotate_ccf.log"
    assert results[0].log_file.exists()
    log_text = results[0].log_file.read_text()
    assert "obj_components: ZR,ZT" in log_text
    assert "required_input_components inferred from obj_components: ZN,ZE" in log_text
    assert "specified in config or CLI; skip SAC HEADER" in log_text
    from obspy import read

    np.testing.assert_allclose(
        read(str(tmp_path / "zrt" / "ZR.SAC"))[0].data,
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        read(str(tmp_path / "zrt" / "ZT.SAC"))[0].data,
        [3.0, 4.0],
    )


def test_run_rotate_ccf_config_prefers_sac_az_baz_headers(tmp_path):
    input_dir = tmp_path / "zne"
    input_dir.mkdir()
    sac = {"az": 0.0, "baz": 0.0}
    _write_sac(input_dir / "ZN.SAC", [1.0, 2.0], sac=sac)
    _write_sac(input_dir / "ZE.SAC", [3.0, 4.0], sac=sac)
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  input_dir: zne",
                "  output_dir: zrt",
                "rotation:",
                "  obj_components: [ZR, ZT]",
            ]
        )
    )

    result = run_rotate_ccf_config(config_file)[0]

    from obspy import read

    np.testing.assert_allclose(
        read(str(tmp_path / "zrt" / "ZR.SAC"))[0].data,
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        read(str(tmp_path / "zrt" / "ZT.SAC"))[0].data,
        [3.0, 4.0],
    )
    assert "source=SAC headers az/baz" in result.log_file.read_text()
    assert "SAC HEADER az/baz: success" in result.log_file.read_text()
    assert "coordinate fallback: skipped" in result.log_file.read_text()


def test_run_rotate_ccf_config_infers_azimuths_from_sac_headers(tmp_path):
    input_dir = tmp_path / "zne"
    input_dir.mkdir()
    sac = {
        "evla": 0.0,
        "evlo": 0.0,
        "stla": 1.0,
        "stlo": 0.0,
        "lcalda": False,
    }
    _write_sac(input_dir / "ZN.SAC", [1.0, 2.0], sac=sac)
    _write_sac(input_dir / "ZE.SAC", [3.0, 4.0], sac=sac)
    config_file = tmp_path / "rotate.yml"
    config_file.write_text(
        "\n".join(
            [
                "io:",
                "  input_dir: zne",
                "  output_dir: zrt",
                "rotation:",
                "  obj_components: [ZR, ZT]",
            ]
        )
    )

    run_rotate_ccf_config(config_file)

    from obspy import read

    np.testing.assert_allclose(
        read(str(tmp_path / "zrt" / "ZR.SAC"))[0].data,
        [-1.0, -2.0],
    )
    np.testing.assert_allclose(
        read(str(tmp_path / "zrt" / "ZT.SAC"))[0].data,
        [-3.0, -4.0],
    )
    log_text = (tmp_path / "zrt" / "rotate_ccf.log").read_text()
    assert "azimuth/back_azimuth: not fully specified" in log_text
    assert "failed to read from SAC HEADER" in log_text
    assert "coordinate fallback: success" in log_text
    assert "source=computed from SAC evla/evlo/stla/stlo" in log_text


def test_run_rotate_ccf_config_preserves_input_naming_by_replacing_component(tmp_path):
    input_dir = tmp_path / "zne"
    input_dir.mkdir()
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
                "  input_dir: zne",
                "  output_dir: zrt",
                f"  input_template: '{name}_{{component}}_pws.SAC'",
                "  output_template: '{component}.SAC'",
                "rotation:",
                "  obj_components: [ZZ, ZR, RR]",
                "azimuth: 0.0",
                "back_azimuth: 0.0",
            ]
        )
    )

    result = run_rotate_ccf_config(config_file)[0]

    assert result.written["ZZ"] == tmp_path / "zrt" / f"{name}_ZZ_pws.SAC"
    assert result.written["ZR"] == tmp_path / "zrt" / f"{name}_ZR_pws.SAC"
    assert result.written["RR"] == tmp_path / "zrt" / f"{name}_RR_pws.SAC"
    assert not (tmp_path / "zrt" / "ZR.SAC").exists()
