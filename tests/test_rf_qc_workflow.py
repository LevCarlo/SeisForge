"""`seisforge rf qc` driver: overwrite policy, logging, provenance, plotting state."""

from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import pytest
import yaml
from obspy import Trace
from obspy.core.util import AttribDict

from seisforge.rf.qc import run_rf_qc_config


def _write_rfs(folder, n=5, bad=()):
    folder.mkdir(parents=True)
    time_s = np.arange(-5.0, 20.0 + 0.05, 0.1)
    rng = np.random.default_rng(0)
    for i in range(n):
        data = np.exp(-(2.5 * time_s) ** 2) + 0.01 * rng.normal(size=time_s.size)
        trace = Trace(data)
        trace.stats.delta = 0.1
        trace.stats.sac = AttribDict(b=-5.0, e=float(time_s[-1]), delta=0.1, npts=time_s.size,
                                     user0=0.06, user2=1.0 if i in bad else 10.0)
        trace.write(str(folder / f"EV{i}.XX.STA.R.sac"), format="SAC")


def _config(tmp_path, **io):
    root = tmp_path / "XX.STA"
    for band in ("high", "low"):
        _write_rfs(root / band, bad=(1,))
    off = {"ENABLE": False}
    band = {"SNR": {"ENABLE": True, "THRESHOLD": 3.0, "SAC_HEADER": "user2"},
            "SLOWNESS": off, "PAMP": off, "MAD": off, "F_TEST": off, "CC": off}
    config = {
        "IO": {"ROOT": str(root), "RAW_HighFreq_RF": "high", "RAW_LowFreq_RF": "low",
               "CLEAN_HighFreq_RF": "out/high", "CLEAN_LowFreq_RF": "out/low",
               "FIGURE": "out/fig", "LOG": "out/log", **io},
        "META": {"station": "STA", "network": "XX", "component": "R"},
        "QC": {"HighFreq": band, "LowFreq": band},
    }
    path = tmp_path / "qc.yml"
    path.write_text(yaml.safe_dump(config))
    return root, path


def test_qc_refuses_to_overwrite_and_replaces_with_overwrite(tmp_path):
    # Before 0.2.0 a second run copied over the CLEAN folder silently and
    # left files of the earlier run that no longer passed.
    root, path = _config(tmp_path)
    run_rf_qc_config(str(path))
    clean = root / "out" / "low"
    assert sorted(p.name for p in clean.iterdir()) == [
        f"EV{i}.XX.STA.R.sac" for i in (0, 2, 3, 4)
    ]
    (clean / "EV1.XX.STA.R.sac").write_bytes((root / "low" / "EV1.XX.STA.R.sac").read_bytes())
    with pytest.raises(FileExistsError, match="overwrite"):
        run_rf_qc_config(str(path))
    run_rf_qc_config(str(path), overwrite=True)
    assert not (clean / "EV1.XX.STA.R.sac").exists()


def test_qc_overwrite_from_config(tmp_path):
    root, path = _config(tmp_path, OVERWRITE=True)
    run_rf_qc_config(str(path))
    run_rf_qc_config(str(path))


def test_qc_leaves_root_logger_alone(tmp_path):
    root_logger = logging.getLogger()
    before = (root_logger.level, list(root_logger.handlers))
    _, path = _config(tmp_path)
    run_rf_qc_config(str(path))
    assert (root_logger.level, list(root_logger.handlers)) == before
    assert not logging.getLogger("seisforge.rf").handlers


def test_qc_table_records_metrics_fate_and_provenance(tmp_path, assert_provenance):
    root, path = _config(tmp_path)
    run_rf_qc_config(str(path))
    text = (root / "out" / "log" / "XX.STA_RF_QC_LowFreq.dat").read_text()
    assert_provenance(text)
    assert "# QC.LowFreq.SNR: enable=True" in text
    rows = {line.split()[0]: line.split() for line in text.splitlines() if not line.startswith("#")}
    assert rows["EV1.XX.STA.R.sac"][1] == "removed:SNR"
    assert rows["EV0.XX.STA.R.sac"][1:3] == ["kept", "10"]
    log = (root / "out" / "log" / "XX.STA_RF_QC.log").read_text()
    assert_provenance(log)
    assert "Number of RF traces after SNR selection: 4" in log


def test_clean_dir_inside_input_dir_is_refused(tmp_path):
    _, path = _config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["IO"]["CLEAN_LowFreq_RF"] = "low/clean"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="overlaps"):
        run_rf_qc_config(str(path))


def test_importing_rf_plotting_keeps_rcparams():
    code = (
        "import matplotlib as mpl; before = dict(mpl.rcParams); "
        "import seisforge.rf.plotting; "
        "assert dict(mpl.rcParams) == before"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
