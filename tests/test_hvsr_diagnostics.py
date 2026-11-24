"""Settings that a solver cannot use, modes it leaves out, and the branch rule.

Each test here failed before the fix it guards (see CHANGELOG, HVSR entries).
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

import seisforge.hvsr.modes as modes_module
from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    ForwardStatus,
    HVDFAReferenceSolver,
    NativeDiffuseFieldSolver,
)
from seisforge.hvsr.propagator import vertical_decay_wavenumber


def _layer_model() -> ElasticLayerModel:
    return ElasticLayerModel([0.05], [0.6, 2.0], [0.2, 1.0], [1.8, 2.2])


@pytest.mark.parametrize(
    "name, value",
    [("precision_percent", 5.0), ("sh_damping", 0.05), ("psv_damping", 0.05)],
)
def test_hvdfa_only_controls_are_not_accepted_by_shared_settings(name, value) -> None:
    # Before: accepted and silently ignored by the native and Fortran solvers.
    with pytest.raises(TypeError):
        DiffuseFieldSettings(**{name: value})


def test_body_wave_sample_count_is_not_accepted_by_shared_settings() -> None:
    # Before: a quadrature-order hint for native/Fortran; with adaptive
    # quadrature it would have no effect there, so it is an HV-DFA control.
    with pytest.raises(TypeError):
        DiffuseFieldSettings(body_wave_wavenumbers=100)


def test_hvdfa_controls_reach_the_command_line(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "HV"
    executable.touch(mode=0o755)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, stdout="1 2\n", stderr="")

    monkeypatch.setattr("seisforge.hvsr.reference.subprocess.run", fake_run)
    solver = HVDFAReferenceSolver(
        executable,
        body_wave_wavenumbers=300,
        precision_percent=0.5,
        sh_damping=0.01,
        psv_damping=0.02,
    )
    solver(_layer_model(), [1.0], DiffuseFieldSettings(include_body_waves=False))
    command = captured["command"]
    options = dict(zip(command[1::2], command[2::2], strict=False))
    assert options["-prec"] == "0.5"
    assert options["-ash"] == "0.01"
    assert options["-apsv"] == "0.02"
    assert options["-nks"] == "0"  # include_body_waves=False wins over the count


def test_skipped_surface_mode_is_recorded_and_not_ok(monkeypatch) -> None:
    original = modes_module.refine_surface_wavenumber_per_km

    def fail_love_mode_0(model, frequency_hz, velocity, wave, **kwargs):
        if wave == "love" and frequency_hz == 1.0:
            raise RuntimeError("forced failure")
        return original(model, frequency_hz, velocity, wave, **kwargs)

    monkeypatch.setattr(modes_module, "refine_surface_wavenumber_per_km", fail_love_mode_0)
    settings = DiffuseFieldSettings(
        max_rayleigh_modes=1, max_love_modes=1, include_body_waves=False
    )
    result = NativeDiffuseFieldSolver()(_layer_model(), [0.5, 1.0], settings)

    # Before: status OK and no record of the missing Love mode.
    assert result.diagnostics[0].status is ForwardStatus.OK
    assert result.diagnostics[1].status is ForwardStatus.INCOMPLETE
    (issue,) = result.diagnostics[1].issues
    assert (issue.frequency_hz, issue.term, issue.mode) == (1.0, "love", 0)
    assert "root" in issue.reason
    assert "love mode 0" in result.diagnostics[1].message
    assert result.contributions.issues == (issue,)
    assert np.isfinite(result.hvsr[1])  # the value is kept, but labelled


def test_rejected_residue_is_recorded(monkeypatch) -> None:
    def reject(value, *, label):
        if label.startswith("Rayleigh"):
            raise RuntimeError(f"{label} pole residue has negative spectral energy")
        return max(float(np.real(value)), 0.0)

    monkeypatch.setattr(modes_module, "_positive_residue", reject)
    settings = DiffuseFieldSettings(
        max_rayleigh_modes=1, max_love_modes=0, include_body_waves=True
    )
    result = NativeDiffuseFieldSolver()(_layer_model(), [2.0], settings)
    (issue,) = result.diagnostics[0].issues
    assert (issue.term, issue.mode) == ("rayleigh", 0)
    assert "residue" in issue.reason
    assert result.diagnostics[0].status is ForwardStatus.INCOMPLETE


def test_radiation_branch_keeps_decay_off_the_cut() -> None:
    # k slightly above the real axis: k^2 - (omega/v)^2 = -3 + 2e-10 i. The
    # principal root has Re = 5.8e-11 > 0, far above the round-off of the
    # argument (~1e-15), so it is not on the cut and must not be flipped to a
    # growing (Re < 0) wave. Before: an absolute 1e-8 test flipped it.
    gamma = vertical_decay_wavenumber(1.0 + 1.0e-10j, 2.0, 1.0)
    assert gamma.real > 0.0
    # Exactly on the cut (real k inside the cone): outgoing, Im < 0.
    on_cut = vertical_decay_wavenumber(1.0, 2.0, 1.0)
    assert on_cut.real == 0.0 and on_cut.imag < 0.0
    # Scale invariance: the same geometry in metres (k, omega/v x 1e-3).
    scaled = vertical_decay_wavenumber(1.0e-3 + 1.0e-13j, 2.0e-3, 1.0)
    assert scaled.real > 0.0
