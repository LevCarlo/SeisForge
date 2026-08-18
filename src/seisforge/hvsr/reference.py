"""Safe adapter for García-Jerez et al.'s standalone HV-DFA executable.

HV-DFA is an independently distributed CC BY-NC program.  SeisForge neither
vendors nor builds it; this adapter only executes a user-supplied binary.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .models import (
    DiffuseFieldPrediction,
    DiffuseFieldSettings,
    ElasticLayerModel,
    ForwardStatus,
    FrequencyDiagnostic,
    HorizontalDefinition,
)


class HVDFAExecutionError(RuntimeError):
    """Raised when HV-DFA fails or returns an invalid result."""


@dataclass(frozen=True)
class HVDFAReferenceSolver:
    """Callable exact-reference backend using a local HV-DFA executable."""

    executable: str | os.PathLike[str]
    timeout_s: float = 120.0
    omp_threads: int = 1

    def __post_init__(self) -> None:
        executable = Path(self.executable).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"HV-DFA executable does not exist: {executable}")
        if not os.access(executable, os.X_OK):
            raise PermissionError(f"HV-DFA executable is not executable: {executable}")
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if self.omp_threads < 1:
            raise ValueError("omp_threads must be at least one")
        object.__setattr__(self, "executable", executable)

    def __call__(
        self,
        model: ElasticLayerModel,
        frequencies_hz,
        settings: DiffuseFieldSettings | None = None,
    ) -> DiffuseFieldPrediction:
        return predict_with_hvdfa(
            self.executable,
            model,
            frequencies_hz,
            settings=settings,
            timeout_s=self.timeout_s,
            omp_threads=self.omp_threads,
        )


def _validated_frequencies(values) -> np.ndarray:
    frequencies = np.asarray(values, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must contain finite positive values")
    if np.unique(frequencies).size != frequencies.size:
        raise ValueError("frequencies_hz cannot contain duplicates")
    return frequencies


def _write_model(path: Path, model: ElasticLayerModel) -> None:
    # HV-DFA expects SI units and encodes the half-space with zero thickness.
    thickness_m = np.concatenate((model.thickness_km * 1000.0, [0.0]))
    table = np.column_stack(
        (
            thickness_m,
            model.vp_km_s * 1000.0,
            model.vs_km_s * 1000.0,
            model.density_g_cm3 * 1000.0,
        )
    )
    with path.open("w", encoding="ascii") as stream:
        stream.write(f"{model.n_layers}\n")
        np.savetxt(stream, table, fmt="%.17g")


def _parse_hvdfa_output(stdout: str, expected_frequencies: np.ndarray) -> np.ndarray:
    try:
        values = np.array([float(token) for token in stdout.split()], dtype=np.float64)
    except ValueError as exc:
        raise HVDFAExecutionError("HV-DFA returned non-numeric output") from exc
    expected_size = 2 * expected_frequencies.size
    if values.size != expected_size:
        excerpt = " ".join(stdout.strip().split())[:300]
        raise HVDFAExecutionError(
            f"HV-DFA returned {values.size} numeric values; expected {expected_size}. "
            f"Output: {excerpt!r}"
        )
    pairs = values.reshape(-1, 2)
    if not np.allclose(pairs[:, 0], expected_frequencies, rtol=5e-6, atol=1e-10):
        raise HVDFAExecutionError("HV-DFA output frequencies do not match its input")
    hvsr = pairs[:, 1]
    if not np.all(np.isfinite(hvsr)) or np.any(hvsr < 0.0):
        raise HVDFAExecutionError("HV-DFA returned non-finite or negative H/V values")
    return hvsr


def predict_with_hvdfa(
    executable: str | os.PathLike[str],
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    settings: DiffuseFieldSettings | None = None,
    timeout_s: float = 120.0,
    omp_threads: int = 1,
) -> DiffuseFieldPrediction:
    """Evaluate full DFA H/V using a user-provided HV-DFA binary.

    A private temporary working directory makes concurrent calls safe even
    though the original program uses fixed filenames for some outputs.
    """

    solver_path = Path(executable).expanduser().resolve()
    if not solver_path.is_file():
        raise FileNotFoundError(f"HV-DFA executable does not exist: {solver_path}")
    if not os.access(solver_path, os.X_OK):
        raise PermissionError(f"HV-DFA executable is not executable: {solver_path}")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("timeout_s must be finite and positive")
    if omp_threads < 1:
        raise ValueError("omp_threads must be at least one")
    if not isinstance(model, ElasticLayerModel):
        raise TypeError("model must be an ElasticLayerModel")
    settings = settings or DiffuseFieldSettings()
    frequencies = _validated_frequencies(frequencies_hz)
    order = np.argsort(frequencies, kind="stable")
    sorted_frequencies = frequencies[order]

    with tempfile.TemporaryDirectory(prefix="seisforge-hvdfa-") as directory:
        workdir = Path(directory)
        model_path = workdir / "model.txt"
        frequency_path = workdir / "frequencies.txt"
        _write_model(model_path, model)
        np.savetxt(frequency_path, sorted_frequencies, fmt="%.17g")

        command = [
            str(solver_path),
            "-f",
            model_path.name,
            "-ff",
            frequency_path.name,
            "-nf",
            str(sorted_frequencies.size),
            "-nmr",
            str(settings.max_rayleigh_modes),
            "-nml",
            str(settings.max_love_modes),
            "-nks",
            str(settings.body_wave_wavenumbers),
            "-prec",
            f"{settings.precision_percent:.17g}",
            "-ash",
            f"{settings.sh_damping:.17g}",
            "-apsv",
            f"{settings.psv_damping:.17g}",
            "-hv",
        ]
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(omp_threads)
        try:
            completed = subprocess.run(
                command,
                cwd=workdir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HVDFAExecutionError(
                f"HV-DFA exceeded the {timeout_s:g} s timeout"
            ) from exc

    if completed.returncode != 0:
        raise HVDFAExecutionError(
            f"HV-DFA exited with status {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    sorted_hvsr = _parse_hvdfa_output(completed.stdout, sorted_frequencies)
    hvsr = np.empty_like(sorted_hvsr)
    hvsr[order] = sorted_hvsr
    if settings.horizontal_definition is HorizontalDefinition.MEAN:
        hvsr = hvsr / np.sqrt(2.0)

    diagnostics = tuple(
        FrequencyDiagnostic(float(value), ForwardStatus.OK) for value in frequencies
    )
    approximation = (
        "full_dfa" if settings.body_wave_wavenumbers > 0 else "surface_waves_only"
    )
    return DiffuseFieldPrediction(
        frequency_hz=frequencies,
        hvsr=hvsr,
        backend="hvdfa_reference",
        approximation=approximation,
        horizontal_definition=settings.horizontal_definition,
        diagnostics=diagnostics,
    )
