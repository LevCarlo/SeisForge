"""Model objects for single-station inversion workflows.

The important split is:

parameterized Vs model -> feature-preserving discretization -> layered model

`LayeredModel` is the final format for forward solvers such as disba. The
profile/segment classes are the parameterization layer used by BayesBay now and
custom MCMC later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from seisforge.inv.scaling import estimate_rho, estimate_vp


ArrayLike = float | np.ndarray


def _as_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class LayeredModel:
    """Elastic layered model ready for forward modeling.

    Units are km, km/s, and g/cm^3. The last layer is a half-space and should
    have thickness 0, matching common surface-wave forward-model conventions.
    """

    thickness: ArrayLike
    vp: ArrayLike
    vs: ArrayLike
    rho: ArrayLike

    def __post_init__(self):
        thickness = _as_array(self.thickness, name="thickness")
        vp = _as_array(self.vp, name="vp")
        vs = _as_array(self.vs, name="vs")
        rho = _as_array(self.rho, name="rho")

        n_layers = len(vs)
        if not (len(thickness) == len(vp) == len(rho) == n_layers):
            raise ValueError("thickness, vp, vs, and rho must have the same length.")
        if n_layers == 0:
            raise ValueError("LayeredModel cannot be empty.")
        if np.any(thickness[:-1] <= 0):
            raise ValueError("All finite-layer thicknesses must be positive.")
        if not np.isclose(thickness[-1], 0.0):
            raise ValueError("The last layer must be a half-space with thickness 0.")
        if np.any(vs <= 0) or np.any(vp <= 0) or np.any(rho <= 0):
            raise ValueError("vp, vs, and rho must be positive.")
        if np.any(vp <= vs):
            raise ValueError("Vp must be greater than Vs in every layer.")

        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "vp", vp)
        object.__setattr__(self, "vs", vs)
        object.__setattr__(self, "rho", rho)

    @property
    def n_layers(self) -> int:
        return len(self.vs)

    @property
    def interfaces(self) -> np.ndarray:
        """Interface depths below the surface, excluding the half-space marker."""
        return np.cumsum(self.thickness[:-1])

    def as_disba(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays in disba order: thickness, vp, vs, rho."""
        return self.thickness, self.vp, self.vs, self.rho


class VsProfile(Protocol):
    """Layer-internal Vs parameterization.

    A profile evaluates local depth inside one segment. Future B-spline classes
    should implement this same method, so samplers do not care how Vs is stored.
    """

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ConstantVs:
    value: float

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        z_local = np.asarray(z_local, dtype=float)
        return np.full_like(z_local, self.value, dtype=float)


@dataclass(frozen=True)
class GradientVs:
    top: float
    bottom: float

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        if thickness <= 0:
            raise ValueError("GradientVs requires a positive segment thickness.")
        z_local = np.asarray(z_local, dtype=float)
        fraction = z_local / thickness
        return self.top + (self.bottom - self.top) * fraction


@dataclass(frozen=True)
class BSplineVs:
    """Placeholder interface for future B-spline Vs parameterization."""

    control_depths: ArrayLike
    control_vs: ArrayLike
    degree: int = 3

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        raise NotImplementedError("BSplineVs will be implemented after the BayesBay MVP.")


@dataclass(frozen=True)
class VsSegment:
    """Depth interval with its own Vs profile.

    Segment boundaries are treated as model features. Discretization never
    creates a layer that crosses from one segment into another.
    """

    top: float
    bottom: float
    profile: VsProfile

    def __post_init__(self):
        if self.top < 0:
            raise ValueError("Segment top depth must be non-negative.")
        if self.bottom <= self.top:
            raise ValueError("Segment bottom depth must be greater than top depth.")

    @property
    def thickness(self) -> float:
        return self.bottom - self.top

    def evaluate(self, z: ArrayLike) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return self.profile.evaluate(z - self.top, thickness=self.thickness)


@dataclass(frozen=True)
class DiscretizationConfig:
    """Controls conversion from parameterized model to forward-model layers."""

    dz: float = 0.5
    zmax: float | None = None
    force_depths: tuple[float, ...] = ()

    def __post_init__(self):
        if self.dz <= 0:
            raise ValueError("Discretization dz must be positive.")
        if self.zmax is not None and self.zmax <= 0:
            raise ValueError("zmax must be positive when provided.")


@dataclass(frozen=True)
class ScalingConfig:
    """Vp/rho scaling recipe applied when building a LayeredModel."""

    vp_method: str = "brocher2005"
    rho_method: str = "brocher2005"
    vp_kwargs: dict | None = None
    rho_kwargs: dict | None = None

    def apply(self, vs: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        vs = np.asarray(vs, dtype=float)
        vp = estimate_vp(vs, method=self.vp_method, **(self.vp_kwargs or {}))
        rho = estimate_rho(vs, method=self.rho_method, **(self.rho_kwargs or {}))
        return np.asarray(vp, dtype=float), np.asarray(rho, dtype=float)


@dataclass(frozen=True)
class LayeredVsModel:
    """Vs-primary layered parameterization.

    This is the simplest BayesBay-friendly model: the sampler can perturb layer
    Vs values directly, while Vp and rho are derived by scaling.
    """

    thickness: ArrayLike
    vs: ArrayLike
    scaling: ScalingConfig = ScalingConfig()

    def __post_init__(self):
        thickness = _as_array(self.thickness, name="thickness")
        vs = _as_array(self.vs, name="vs")
        if len(thickness) != len(vs):
            raise ValueError("thickness and vs must have the same length.")
        if not np.isclose(thickness[-1], 0.0):
            raise ValueError("The last layer must be a half-space with thickness 0.")
        if np.any(thickness[:-1] <= 0):
            raise ValueError("All finite-layer thicknesses must be positive.")
        if np.any(vs <= 0):
            raise ValueError("Vs must be positive.")

        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "vs", vs)

    @property
    def interfaces(self) -> np.ndarray:
        return np.cumsum(self.thickness[:-1])

    def evaluate(self, z: ArrayLike) -> np.ndarray:
        """Piecewise-constant Vs at depth z."""
        z = np.asarray(z, dtype=float)
        interfaces = self.interfaces
        indices = np.searchsorted(interfaces, z, side="right")
        indices = np.clip(indices, 0, len(self.vs) - 1)
        return self.vs[indices]

    def discontinuities(self) -> np.ndarray:
        return self.interfaces

    def to_layered_model(self) -> LayeredModel:
        vp, rho = self.scaling.apply(self.vs)
        return LayeredModel(self.thickness, vp, self.vs, rho)


@dataclass(frozen=True)
class ParameterizedVsModel:
    """Feature-preserving Vs model composed of depth segments."""

    segments: tuple[VsSegment, ...]
    scaling: ScalingConfig = ScalingConfig()

    def __post_init__(self):
        if len(self.segments) == 0:
            raise ValueError("ParameterizedVsModel requires at least one segment.")
        if not np.isclose(self.segments[0].top, 0.0):
            raise ValueError("The first segment must start at 0 km.")

        for prev, current in zip(self.segments[:-1], self.segments[1:]):
            if not np.isclose(prev.bottom, current.top):
                raise ValueError("Segments must be contiguous and ordered.")

    @property
    def zmax(self) -> float:
        return self.segments[-1].bottom

    def discontinuities(self) -> np.ndarray:
        return np.array([segment.top for segment in self.segments[1:]], dtype=float)

    def evaluate(self, z: ArrayLike) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z, dtype=float)
        filled = np.zeros_like(z, dtype=bool)

        for i, segment in enumerate(self.segments):
            if i == len(self.segments) - 1:
                mask = (z >= segment.top) & (z <= segment.bottom)
            else:
                mask = (z >= segment.top) & (z < segment.bottom)
            if np.any(mask):
                result[mask] = segment.evaluate(z[mask])
                filled[mask] = True

        if not np.all(filled):
            raise ValueError("Requested depths fall outside the model range.")
        return result

    def to_layered_model(self, config: DiscretizationConfig | None = None) -> LayeredModel:
        config = config or DiscretizationConfig()
        zmax = self.zmax if config.zmax is None else min(config.zmax, self.zmax)
        if zmax <= 0:
            raise ValueError("Discretization zmax must be positive.")

        layer_tops = []
        layer_bottoms = []
        for segment in self.segments:
            top = segment.top
            bottom = min(segment.bottom, zmax)
            if bottom <= top:
                continue
            boundaries = _segment_boundaries(
                top,
                bottom,
                dz=config.dz,
                force_depths=config.force_depths,
            )
            layer_tops.extend(boundaries[:-1])
            layer_bottoms.extend(boundaries[1:])
            if np.isclose(bottom, zmax):
                break

        tops = np.asarray(layer_tops, dtype=float)
        bottoms = np.asarray(layer_bottoms, dtype=float)
        if len(tops) == 0:
            raise ValueError("Discretization produced no finite layers.")

        thickness = (bottoms - tops).tolist()
        z_mid = 0.5 * (tops + bottoms)
        vs = self.evaluate(z_mid).tolist()

        # Append a half-space sampled just below the deepest finite boundary.
        thickness.append(0.0)
        vs.append(float(self.evaluate(np.array([zmax]))[0]))

        vs = np.asarray(vs, dtype=float)
        vp, rho = self.scaling.apply(vs)
        return LayeredModel(thickness, vp, vs, rho)


def _segment_boundaries(
    top: float,
    bottom: float,
    *,
    dz: float,
    force_depths: tuple[float, ...] = (),
) -> np.ndarray:
    n = max(1, int(np.ceil((bottom - top) / dz)))
    boundaries = np.linspace(top, bottom, n + 1)
    forced = [depth for depth in force_depths if top < depth < bottom]
    if forced:
        boundaries = np.unique(np.concatenate([boundaries, np.asarray(forced)]))
    return boundaries
