"""tools/aftan_cycle_set.py: manual phase-cycle override of saved AFTAN outputs (stand-alone tool)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from seisforge.ant.aftan import apply_phase_cycle_shift

TOOL = Path(__file__).resolve().parents[1] / "tools" / "aftan_cycle_set.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("aftan_cycle_set", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look the module up while it executes
    spec.loader.exec_module(module)
    return module


def test_manual_phase_cycle_override_previews_then_updates_nc_and_dat(tmp_path):
    tool = _load_tool()
    period = np.array([2.0, 3.0, 4.0])
    uncorrected = np.array([2.8, 2.9, 3.0])
    reference = np.array([3.0, 3.1, 3.2])
    distance_km = 80.0
    phase_map = tmp_path / "pair.negative.pmf.vph.nc"
    dat_file = tmp_path / "pair.negative.pmf.dat"
    dataset = xr.Dataset(
        data_vars={
            "picked_phase_velocity": (("target_period",), uncorrected),
            "picked_phase_velocity_uncorrected": (
                ("target_period",),
                uncorrected,
            ),
            "reference_phase_velocity": (("target_period",), reference),
            "picked_instant_period": (("target_period",), period),
            "phase_cycle_shift": np.asarray(0, dtype=np.int32),
            "phase_cycle_reference_misfit": np.asarray(0.5),
            "phase_cycle_reference_misfit_gap": np.asarray(0.1),
            "phase_cycle_search_limit_reached": np.asarray(0, dtype=np.int8),
        },
        coords={"target_period": period},
        attrs={"distance_km": distance_km},
    )
    dataset.to_netcdf(phase_map, engine="scipy")
    table = np.column_stack(
        [period, period, np.full(3, 2.7), uncorrected, np.ones(3), np.full(3, 20.0)]
    )
    np.savetxt(
        dat_file,
        table,
        header=(
            "target_period_s instant_period_s group_velocity_km_s "
            "phase_velocity_km_s amplitude snr"
        ),
    )

    preview = tool.override_phase_cycle(phase_map, -1)
    assert not preview.written
    with xr.open_dataset(phase_map) as unchanged:
        assert unchanged["phase_cycle_shift"].item() == 0

    result = tool.override_phase_cycle(
        phase_map,
        -1,
        write=True,
        reason="visual cycle inspection",
    )
    expected = apply_phase_cycle_shift(period, uncorrected, distance_km, -1)
    assert result.written
    with xr.open_dataset(phase_map) as updated:
        np.testing.assert_allclose(updated["picked_phase_velocity"], expected)
        np.testing.assert_allclose(
            updated["picked_phase_velocity_automatic"], uncorrected
        )
        assert updated["phase_cycle_shift"].item() == -1
        assert updated.attrs["phase_cycle_selection_source"] == "manual_override"
        assert updated.attrs["phase_cycle_manual_reason"] == "visual cycle inspection"
    np.testing.assert_allclose(np.loadtxt(dat_file)[:, 3], expected, rtol=2e-4)


def test_command_line_preview_changes_no_file(tmp_path):
    """Default (no --in-place) is a preview: both files stay byte for byte as they were."""
    period = np.array([2.0, 3.0, 4.0])
    velocity = np.array([2.8, 2.9, 3.0])
    phase_map = tmp_path / "pair.positive.vph.nc"
    dat_file = tmp_path / "pair.positive.dat"
    xr.Dataset(
        data_vars={
            "picked_phase_velocity": (("target_period",), velocity),
            "picked_phase_velocity_uncorrected": (("target_period",), velocity),
            "reference_phase_velocity": (("target_period",), velocity + 0.2),
            "picked_instant_period": (("target_period",), period),
            "phase_cycle_shift": np.asarray(0, dtype=np.int32),
        },
        coords={"target_period": period},
        attrs={"distance_km": 80.0},
    ).to_netcdf(phase_map, engine="scipy")
    np.savetxt(dat_file, np.column_stack([period, period, np.full(3, 2.7), velocity, np.ones(3), np.full(3, 20.0)]))
    before = (phase_map.read_bytes(), dat_file.read_bytes())
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1] / "src"), os.environ.get("PYTHONPATH", "")]))}
    completed = subprocess.run(
        [sys.executable, str(TOOL), str(phase_map), "--shift", "1"],
        capture_output=True, text=True, env=env, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "No files were changed" in completed.stdout
    assert (phase_map.read_bytes(), dat_file.read_bytes()) == before
