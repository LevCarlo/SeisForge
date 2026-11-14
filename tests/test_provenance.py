import logging
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

import seisforge
from seisforge import provenance
from seisforge.provenance import DEPENDENCIES, UNKNOWN, software_provenance


@pytest.fixture(autouse=True)
def _fresh_cache():
    provenance._cached_provenance.cache_clear()
    yield
    provenance._cached_provenance.cache_clear()


def test_software_provenance_keys_and_string_values():
    record = software_provenance()

    expected = {
        "seisforge_version",
        "seisforge_git_commit",
        "seisforge_git_dirty",
        "python_version",
        *(f"{name}_version" for name in DEPENDENCIES),
    }
    assert set(record) == expected
    assert all(isinstance(value, str) and value for value in record.values())
    assert record["seisforge_version"] == seisforge.__version__
    assert record["numpy_version"] == metadata.version("numpy")
    assert record["seisforge_git_dirty"] in {"true", "false", UNKNOWN}


def test_software_provenance_is_cached_and_returns_copies(monkeypatch):
    first = software_provenance()
    first["seisforge_version"] = "edited"

    def fail(*args, **kwargs):
        raise AssertionError("git must not run again after the first call")

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    second = software_provenance()
    assert second["seisforge_version"] == seisforge.__version__


def test_git_fields_unknown_when_git_cannot_run(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", missing_git)

    record = software_provenance()

    assert record["seisforge_git_commit"] == UNKNOWN
    assert record["seisforge_git_dirty"] == UNKNOWN
    assert record["seisforge_version"] == seisforge.__version__


def test_git_fields_unknown_when_git_fails(monkeypatch):
    def failing(args, **kwargs):
        return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal")

    monkeypatch.setattr(provenance.subprocess, "run", failing)

    assert software_provenance()["seisforge_git_commit"] == UNKNOWN


def test_git_fields_unknown_outside_a_git_checkout(tmp_path):
    # A copy of the package outside any repository, as in a wheel install.
    assert provenance._git_state(tmp_path.resolve()) == (UNKNOWN, UNKNOWN)


def test_git_fields_ignore_an_unrelated_enclosing_repository(tmp_path):
    # An installed package inside some other git repository (for example a
    # project-local virtualenv) must not report that repository's commit.
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git executable not available")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package_dir = tmp_path / "site-packages" / "seisforge"
    package_dir.mkdir(parents=True)

    assert provenance._git_state(package_dir.resolve()) == (UNKNOWN, UNKNOWN)


def test_git_commit_recorded_for_own_checkout(tmp_path):
    if subprocess.run(["git", "--version"], capture_output=True).returncode != 0:
        pytest.skip("git executable not available")
    package_dir = tmp_path / "src" / "seisforge"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    assert provenance._git_state(package_dir.resolve()) == (head, "false")
    (package_dir / "__init__.py").write_text("x = 1\n")
    assert provenance._git_state(package_dir.resolve()) == (head, "true")


def test_version_falls_back_when_package_not_installed(monkeypatch):
    import importlib

    def not_found(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", not_found)
    try:
        reloaded = importlib.reload(seisforge)
        assert reloaded.__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(seisforge)
    assert seisforge.__version__ == metadata.version("seisforge")


def test_rf_qc_log_records_provenance(tmp_path, assert_provenance):
    from seisforge.rf.qc import run_rf_qc_config
    import yaml

    root = tmp_path / "NET.STA"
    for name in ("raw_high", "raw_low"):
        (root / name).mkdir(parents=True)
    step = {"ENABLE": False}
    band = {
        key: step for key in ("SNR", "SLOWNESS", "PAMP", "MAD", "F_TEST", "CC")
    }
    config = {
        "IO": {
            "ROOT": str(root),
            "RAW_HighFreq_RF": "raw_high",
            "RAW_LowFreq_RF": "raw_low",
            "CLEAN_HighFreq_RF": "clean_high",
            "CLEAN_LowFreq_RF": "clean_low",
            "FIGURE": "figures",
            "LOG": "logs",
        },
        "META": {"station": "STA", "network": "NET"},
        "QC": {"HighFreq": band, "LowFreq": band},
    }
    config_file = tmp_path / "qc.yaml"
    config_file.write_text(yaml.safe_dump(config))

    root_logger = logging.getLogger()
    saved = (root_logger.level, list(root_logger.handlers))
    try:
        run_rf_qc_config(str(config_file))
    finally:
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers[:] = saved[1]
        root_logger.setLevel(saved[0])

    assert_provenance((root / "logs" / "NET.STA_RF_QC.log").read_text())


def test_package_dir_matches_imported_package():
    assert provenance._PACKAGE_DIR == Path(seisforge.__file__).resolve().parent
