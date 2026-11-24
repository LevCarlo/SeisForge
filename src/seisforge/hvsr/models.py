"""Data contracts for diffuse-field H/V forward calculations.

The public model uses the units already used by SeisForge: kilometres,
kilometres per second, and grams per cubic centimetre.  Backend-specific unit
conversion belongs in the backend adapter, not in the scientific model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _immutable_vector(
    values: ArrayLike,
    name: str,
    *,
    allow_empty: bool = False,
    allow_nonfinite: bool = False,
) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1 or (array.size == 0 and not allow_empty):
        qualifier = "" if allow_empty else "non-empty "
        raise ValueError(f"{name} must be a {qualifier}one-dimensional array")
    if not allow_nonfinite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ElasticLayerModel:
    """Isotropic elastic stack terminated by a half-space.

    Parameters are ordered from the free surface downwards. ``thickness_km``
    contains finite-layer thicknesses only, so it has one fewer entry than the
    velocity and density arrays.
    """

    thickness_km: ArrayLike
    vp_km_s: ArrayLike
    vs_km_s: ArrayLike
    density_g_cm3: ArrayLike

    def __post_init__(self) -> None:
        thickness = _immutable_vector(
            self.thickness_km, "thickness_km", allow_empty=True
        )
        vp = _immutable_vector(self.vp_km_s, "vp_km_s")
        vs = _immutable_vector(self.vs_km_s, "vs_km_s")
        density = _immutable_vector(self.density_g_cm3, "density_g_cm3")

        if not (vp.size == vs.size == density.size):
            raise ValueError("vp_km_s, vs_km_s, and density_g_cm3 must match")
        if thickness.size != vp.size - 1:
            raise ValueError(
                "thickness_km must contain one value per finite layer "
                "(one fewer than the property arrays)"
            )
        if np.any(thickness <= 0.0):
            raise ValueError("finite-layer thicknesses must be positive")
        if np.any(vp <= 0.0) or np.any(vs <= 0.0):
            raise ValueError("velocities must be positive")
        if np.any(vp <= vs):
            raise ValueError("vp_km_s must be greater than vs_km_s in every layer")
        if np.any(density <= 0.0):
            raise ValueError("densities must be positive")

        object.__setattr__(self, "thickness_km", thickness)
        object.__setattr__(self, "vp_km_s", vp)
        object.__setattr__(self, "vs_km_s", vs)
        object.__setattr__(self, "density_g_cm3", density)

    @property
    def n_layers(self) -> int:
        """Number of layers including the bottom half-space."""

        return int(self.vp_km_s.size)


class HorizontalDefinition(str, Enum):
    """Convention used for the horizontal component of H/V."""

    SUM = "sum"
    MEAN = "mean"


@dataclass(frozen=True)
class DiffuseFieldSettings:
    """Which terms of the diffuse-field H/V to compute; used by every solver.

    These are choices about the physics, not about numerical accuracy:

    - ``max_rayleigh_modes``, ``max_love_modes``: surface-wave modes
      ``0, 1, ...`` whose pole residues are added (0 leaves that wave out).
    - ``include_body_waves``: add the body-wave integral over
      ``0 < k < omega / Vs_halfspace``.
    - ``horizontal_definition``: ``sum`` or ``mean`` of the two horizontal
      powers (see `HorizontalDefinition`).

    Accuracy controls belong to the solver that uses them:
    `NativeDiffuseFieldSolver` and `FortranDiffuseFieldSolver` (quadrature
    tolerance, contour shift, residue step) and `HVDFAReferenceSolver`
    (``body_wave_wavenumbers``, ``precision_percent``, ``sh_damping``,
    ``psv_damping``). A control that a solver would ignore therefore cannot be
    set on it.
    """

    max_rayleigh_modes: int = 20
    max_love_modes: int = 20
    include_body_waves: bool = True
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM

    def __post_init__(self) -> None:
        horizontal = HorizontalDefinition(self.horizontal_definition)
        object.__setattr__(self, "horizontal_definition", horizontal)
        if self.max_rayleigh_modes < 0 or self.max_love_modes < 0:
            raise ValueError("mode limits cannot be negative")
        if not isinstance(self.include_body_waves, bool):
            raise TypeError("include_body_waves must be true or false")
        if (
            self.max_rayleigh_modes == 0
            and self.max_love_modes == 0
            and not self.include_body_waves
        ):
            raise ValueError("at least one surface- or body-wave contribution is required")


class ForwardStatus(str, Enum):
    """Per-frequency status for a forward prediction.

    ``OK``: every requested term was computed. ``INCOMPLETE``: H/V is given,
    but at least one term was left out or did not reach its accuracy target;
    the diagnostic lists them. ``INVALID``: no H/V (NaN).
    """

    OK = "ok"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True)
class ContributionIssue:
    """A term missing from, or not converged in, one frequency's Green function.

    ``term`` is ``"rayleigh"`` or ``"love"`` (one mode, ``mode`` its index,
    left out) or ``"body_waves"`` (``mode`` is None; the integral is included
    but its error estimate exceeds the requested tolerance).
    """

    frequency_hz: float
    term: str
    mode: int | None
    reason: str

    def describe(self) -> str:
        name = self.term if self.mode is None else f"{self.term} mode {self.mode}"
        return f"{name}: {self.reason}"


@dataclass(frozen=True)
class FrequencyDiagnostic:
    frequency_hz: float
    status: ForwardStatus
    message: str = ""
    issues: tuple[ContributionIssue, ...] = ()


GREEN_FUNCTION_TERMS = (
    "rayleigh_horizontal",
    "rayleigh_vertical",
    "love_horizontal",
    "psv_horizontal",
    "psv_vertical",
    "sh_horizontal",
)


@dataclass(frozen=True)
class GreenFunctionContributions:
    """Imaginary Green-function terms entering the DFA H/V ratio.

    ``issues`` records the terms a solver left out or could not converge at
    a frequency; an empty tuple means every requested term is complete.
    """

    rayleigh_horizontal: ArrayLike
    rayleigh_vertical: ArrayLike
    love_horizontal: ArrayLike
    psv_horizontal: ArrayLike
    psv_vertical: ArrayLike
    sh_horizontal: ArrayLike
    issues: tuple[ContributionIssue, ...] = ()

    def __post_init__(self) -> None:
        arrays = []
        size = None
        for name in GREEN_FUNCTION_TERMS:
            array = _immutable_vector(getattr(self, name), name)
            if size is None:
                size = array.size
            elif array.size != size:
                raise ValueError("all Green-function contributions must have equal length")
            arrays.append((name, array))
        for name, array in arrays:
            object.__setattr__(self, name, array)
        object.__setattr__(self, "issues", tuple(self.issues))

    @classmethod
    def zeros(cls, size: int) -> GreenFunctionContributions:
        """All six terms zero (a part that was not requested)."""

        zeros = np.zeros(size)
        return cls(**{name: zeros for name in GREEN_FUNCTION_TERMS})


@dataclass(frozen=True)
class DiffuseFieldPrediction:
    """Result of a diffuse-field H/V forward calculation."""

    frequency_hz: ArrayLike
    hvsr: ArrayLike
    backend: str
    approximation: str = "full_dfa"
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM
    horizontal_energy: ArrayLike | None = None
    vertical_energy: ArrayLike | None = None
    contributions: GreenFunctionContributions | None = None
    diagnostics: tuple[FrequencyDiagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        frequency = _immutable_vector(self.frequency_hz, "frequency_hz")
        hvsr = _immutable_vector(self.hvsr, "hvsr", allow_nonfinite=True)
        if hvsr.size != frequency.size:
            raise ValueError("frequency_hz and hvsr must have equal length")
        if np.any(frequency <= 0.0):
            raise ValueError("frequencies must be positive")
        horizontal = HorizontalDefinition(self.horizontal_definition)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "hvsr", hvsr)
        object.__setattr__(self, "horizontal_definition", horizontal)

        for name in ("horizontal_energy", "vertical_energy"):
            values = getattr(self, name)
            if values is not None:
                array = _immutable_vector(values, name, allow_nonfinite=True)
                if array.size != frequency.size:
                    raise ValueError(f"{name} must match frequency_hz")
                object.__setattr__(self, name, array)

        if self.contributions is not None:
            first = self.contributions.rayleigh_horizontal
            if first.size != frequency.size:
                raise ValueError("contributions must match frequency_hz")
        if self.diagnostics and len(self.diagnostics) != frequency.size:
            raise ValueError("diagnostics must match frequency_hz when provided")
