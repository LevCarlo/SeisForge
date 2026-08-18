from __future__ import annotations

from pathlib import Path
import os
import subprocess

import numpy as np
import pytest

from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    GreenFunctionContributions,
    HVDFAExecutionError,
    HVDFAReferenceSolver,
    combine_green_function_contributions,
    predict_diffuse_field_hvsr,
    predict_with_hvdfa,
)


@pytest.fixture
def model() -> ElasticLayerModel:
    return ElasticLayerModel(
        thickness_km=[0.5, 1.0, 3.0],
        vp_km_s=[1.8, 2.2, 3.2, 5.5],
        vs_km_s=[0.5, 1.0, 1.8, 3.2],
        density_g_cm3=[1.9, 2.1, 2.4, 2.7],
    )


def test_elastic_model_validates_halfspace_contract() -> None:
    with pytest.raises(ValueError, match="one fewer"):
        ElasticLayerModel([1.0], [2.0], [1.0], [2.0])
    with pytest.raises(ValueError, match="greater than"):
        ElasticLayerModel([], [1.0], [1.0], [2.0])


def test_green_function_terms_reproduce_dfa_formula() -> None:
    terms = GreenFunctionContributions(
        rayleigh_horizontal=[1.0, 2.0],
        rayleigh_vertical=[4.0, 5.0],
        love_horizontal=[2.0, 3.0],
        psv_horizontal=[3.0, 4.0],
        psv_vertical=[6.0, 7.0],
        sh_horizontal=[4.0, 5.0],
    )
    result = combine_green_function_contributions([1.0, 2.0], terms)

    horizontal = 2.0 * np.array([10.0, 14.0])
    vertical = np.array([10.0, 12.0])
    np.testing.assert_allclose(result.horizontal_energy, horizontal)
    np.testing.assert_allclose(result.vertical_energy, vertical)
    np.testing.assert_allclose(result.hvsr, np.sqrt(horizontal / vertical))

    mean = combine_green_function_contributions(
        [1.0, 2.0], terms, horizontal_definition="mean"
    )
    np.testing.assert_allclose(mean.hvsr, result.hvsr / np.sqrt(2.0))


def test_invalid_energy_is_reported_without_crashing() -> None:
    terms = GreenFunctionContributions(
        rayleigh_horizontal=[1.0],
        rayleigh_vertical=[0.0],
        love_horizontal=[0.0],
        psv_horizontal=[0.0],
        psv_vertical=[0.0],
        sh_horizontal=[0.0],
    )
    result = combine_green_function_contributions([1.0], terms)
    assert np.isnan(result.hvsr[0])
    assert result.diagnostics[0].status.value == "invalid"


def test_reference_adapter_converts_units_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: ElasticLayerModel,
) -> None:
    executable = tmp_path / "HV"
    executable.touch(mode=0o755)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        workdir = Path(kwargs["cwd"])
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        captured["model"] = (workdir / "model.txt").read_text()
        captured["frequencies"] = np.loadtxt(workdir / "frequencies.txt")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="1 10 2 20 3 30\n",
            stderr="",
        )

    monkeypatch.setattr("seisforge.hvsr.reference.subprocess.run", fake_run)
    solver = HVDFAReferenceSolver(executable, omp_threads=1)
    result = predict_diffuse_field_hvsr(
        model,
        [3.0, 1.0, 2.0],
        solver=solver,
        settings=DiffuseFieldSettings(horizontal_definition="mean"),
    )

    np.testing.assert_allclose(captured["frequencies"], [1.0, 2.0, 3.0])
    model_lines = str(captured["model"]).splitlines()
    assert model_lines[0] == "4"
    np.testing.assert_allclose(
        np.fromstring(model_lines[1], sep=" "), [500.0, 1800.0, 500.0, 1900.0]
    )
    np.testing.assert_allclose(
        np.fromstring(model_lines[-1], sep=" "), [0.0, 5500.0, 3200.0, 2700.0]
    )
    assert captured["environment"]["OMP_NUM_THREADS"] == "1"
    np.testing.assert_allclose(result.frequency_hz, [3.0, 1.0, 2.0])
    np.testing.assert_allclose(result.hvsr, np.array([30.0, 10.0, 20.0]) / np.sqrt(2.0))


def test_reference_adapter_rejects_error_text_even_with_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: ElasticLayerModel,
) -> None:
    executable = tmp_path / "HV"
    executable.touch(mode=0o755)

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="Error in model", stderr="")

    monkeypatch.setattr("seisforge.hvsr.reference.subprocess.run", fake_run)
    with pytest.raises(HVDFAExecutionError, match="non-numeric|expected"):
        predict_with_hvdfa(executable, model, [1.0, 2.0])


def test_reference_settings_reject_empty_physics() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DiffuseFieldSettings(
            max_rayleigh_modes=0,
            max_love_modes=0,
            body_wave_wavenumbers=0,
        )


@pytest.mark.skipif(
    "SEISFORGE_HVDFA_EXECUTABLE" not in os.environ,
    reason="set SEISFORGE_HVDFA_EXECUTABLE to run the upstream regression",
)
def test_upstream_hvdfa_regression(model: ElasticLayerModel) -> None:
    solver = HVDFAReferenceSolver(os.environ["SEISFORGE_HVDFA_EXECUTABLE"])
    result = solver(model, [3.0, 0.2, 1.0, 5.0])

    assert result.backend == "hvdfa_reference"
    assert result.approximation == "full_dfa"
    assert np.all(np.isfinite(result.hvsr))
    assert np.all(result.hvsr > 0.0)
