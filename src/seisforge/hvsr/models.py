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
    """Numerical controls shared by the HV-DFA reference adapter."""

    max_rayleigh_modes: int = 20
    max_love_modes: int = 20
    body_wave_wavenumbers: int = 100
    precision_percent: float = 0.1
    sh_damping: float = 0.0
    psv_damping: float = 0.0
    horizontal_definition: HorizontalDefinition | str = HorizontalDefinition.SUM

    def __post_init__(self) -> None:
        horizontal = HorizontalDefinition(self.horizontal_definition)
        object.__setattr__(self, "horizontal_definition", horizontal)
        if self.max_rayleigh_modes < 0 or self.max_love_modes < 0:
            raise ValueError("mode limits cannot be negative")
        if self.body_wave_wavenumbers < 0:
            raise ValueError("body_wave_wavenumbers cannot be negative")
        if (
            self.max_rayleigh_modes == 0
            and self.max_love_modes == 0
            and self.body_wave_wavenumbers == 0
        ):
            raise ValueError("at least one surface- or body-wave contribution is required")
        if not np.isfinite(self.precision_percent) or self.precision_percent <= 0.0:
            raise ValueError("precision_percent must be finite and positive")
        for name in ("sh_damping", "psv_damping"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


class ForwardStatus(str, Enum):
    """Per-frequency status for a forward prediction."""

    OK = "ok"
    INVALID = "invalid"


@dataclass(frozen=True)
class FrequencyDiagnostic:
    frequency_hz: float
    status: ForwardStatus
    message: str = ""


@dataclass(frozen=True)
class GreenFunctionContributions:
    """Imaginary Green-function terms entering the DFA H/V ratio."""

    rayleigh_horizontal: ArrayLike
    rayleigh_vertical: ArrayLike
    love_horizontal: ArrayLike
    psv_horizontal: ArrayLike
    psv_vertical: ArrayLike
    sh_horizontal: ArrayLike

    def __post_init__(self) -> None:
        arrays = []
        size = None
        for name in self.__dataclass_fields__:
            array = _immutable_vector(getattr(self, name), name)
            if size is None:
                size = array.size
            elif array.size != size:
                raise ValueError("all Green-function contributions must have equal length")
            arrays.append((name, array))
        for name, array in arrays:
            object.__setattr__(self, name, array)


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
