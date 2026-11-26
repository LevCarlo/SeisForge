"""Smoke tests of the examples: each `run.sh` in a temporary copy, checked against its README.

The numbers and ranges asserted here are those of each README's section
"How to tell the result is right". `06_inversion` and `full_chain` run with
short chains (files and chain-independent numbers only); their full runs,
which check the README's convergence numbers, and the short `full_chain`
run are marked `slow` (`python -m pytest -m slow`, a separate CI step), so
the default suite stays near 60 s.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def run_example(name: str, tmp_path: Path, edit=None) -> str:
    """Copy an example (without its outputs), optionally edit it, run `bash run.sh`; stdout."""
    folder = tmp_path / name
    shutil.copytree(EXAMPLES / name, folder, ignore=shutil.ignore_patterns("output", "result.png"))
    if edit is not None:
        edit(folder)
    env = {
        **os.environ,
        "SEISFORGE": f"{sys.executable} -m seisforge.cli.main",
        "PYTHON": sys.executable,
        "MPLBACKEND": "Agg",
    }
    completed = subprocess.run(
        ["bash", str(folder / "run.sh")], env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert (folder / "result.png").is_file()
    return completed.stdout


def numbers(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def shorten_chains(path: Path) -> None:
    """Short sampler run: smoke test only (files and the logged layering error)."""
    config = yaml.safe_load(path.read_text())
    config["sampler"].update(n_steps=600, burn_in=200, global_jump_interval=200)
    config["sampler"]["adaptation"].update(initial_diagonal_steps=100, update_interval_steps=50)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def test_01_ccf_rotate(tmp_path):
    out = run_example("01_ccf_rotate", tmp_path)
    errors = numbers(r"max \|rotated - truth\| = ([\d.]+)", out)
    assert len(errors) == 9 and max(errors) < 0.05


def test_02_aftan(tmp_path):
    out = run_example("02_aftan", tmp_path)
    phase = numbers(r"max \|phase - truth\| ([\d.]+)", out)
    group = numbers(r"max \|group - truth\| ([\d.]+)", out)
    assert len(phase) == 6 and max(phase) < 0.03
    assert max(group[2:]) < 0.03  # 80 and 120 km; 40 km is short (README)


def test_03_eikonal(tmp_path):
    out = run_example("03_eikonal", tmp_path)
    errors = numbers(r"max \|map - truth\| ([\d.]+)", out)
    assert len(errors) == 7 and max(errors) < 0.03


def test_04_rf_qc_hk(tmp_path):
    out = run_example("04_rf_qc_hk", tmp_path)
    assert "RFs kept by QC: 30 of 42" in out
    assert not list((tmp_path / "04_rf_qc_hk" / "output" / "Gauss_2.5_CLEAN").glob("BAD_*"))
    h_km, k = numbers(r"H = ([\d.]+) \+-", out)[0], numbers(r"k = ([\d.]+) \+-", out)[0]
    assert 34.5 <= h_km <= 35.5 and 1.73 <= k <= 1.77


def test_05_hvsr_forward(tmp_path):
    out = run_example("05_hvsr_forward", tmp_path)
    (peak_hz,) = numbers(r"H/V peak: [\d.]+ at ([\d.]+) Hz", out)
    assert 0.95 <= peak_hz <= 1.05
    assert "0 frequencies failed" in out


def _layering_sigma_ratios(log: Path) -> list[float]:
    return numbers(r"layering_error_sigma_ratio: ([\d.eE+-]+)", log.read_text())


def test_06_inversion_short(tmp_path):
    out = run_example("06_inversion", tmp_path, edit=lambda folder: shorten_chains(folder / "inv.yaml"))
    posterior = tmp_path / "06_inversion" / "output" / "posterior"
    for name in ("disp_hv_mcmc.nc", "disp_hv_vs.nc", "disp_hv_predictive.nc", "disp_hv.log"):
        assert (posterior / name).is_file()
    assert max(_layering_sigma_ratios(posterior / "disp_hv.log")) < 0.1
    assert "max split R-hat" in out


@pytest.mark.slow  # 16 s; keeps the default suite near 60 s
def test_full_chain_short(tmp_path):
    out = run_example("full_chain", tmp_path, edit=lambda folder: shorten_chains(folder / "inv.yaml"))
    differences = numbers(r"([+-][\d.]+)", out.split("truth (km/s) at")[1].splitlines()[0])
    assert len(differences) == 4 and max(abs(d) for d in differences) <= 0.05
    assert (tmp_path / "full_chain" / "output" / "inversion" / "disp_hv_vs.nc").is_file()


@pytest.mark.slow
def test_06_inversion_full(tmp_path):
    out = run_example("06_inversion", tmp_path)
    assert numbers(r"max split R-hat: ([\d.]+)", out)[0] < 1.01
    assert numbers(r"min effective sample size: ([\d.]+)", out)[0] > 400
    inside, total = (int(v) for v in re.search(r"band at (\d+) of (\d+)", out).groups())
    # Nominal 5-95 % band; README records 238/241 (98.8 %) after continuous depth priors.
    assert inside >= 0.95 * total


@pytest.mark.slow
def test_full_chain_full(tmp_path):
    out = run_example("full_chain", tmp_path)
    assert numbers(r"max split R-hat: ([\d.]+)", out)[0] < 1.05
    # README records min ESS 378 (crust.c5; all others > 560) and 218/241 (90.5 %) inside 5-95 %.
    assert numbers(r"min effective sample size: ([\d.]+)", out)[0] > 300
    inside, total = (int(v) for v in re.search(r"band at (\d+) of (\d+)", out).groups())
    assert inside >= 0.88 * total
