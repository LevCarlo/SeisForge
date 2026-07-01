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

from seisforge.inv.bspline import evaluate_bspline
from seisforge.inv.scaling import estimate_rho, estimate_vp


ArrayLike = float | np.ndarray


def _as_array(values, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_depth_profile(
    z,
    values,
    *,
    value_name: str,
    min_points: int,
    positive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    z = _as_array(z, name="z")
    values = _as_array(values, name=value_name)
    if len(z) != len(values):
        raise ValueError(f"z and {value_name} must have the same length.")
    if len(z) < min_points:
        raise ValueError(f"z and {value_name} must contain at least {min_points} points.")
    if not np.isclose(z[0], 0.0):
        raise ValueError("The first depth must be 0 km.")
    if np.any(np.diff(z) <= 0):
        raise ValueError("Depth values must be strictly increasing.")
    if positive and np.any(values <= 0):
        raise ValueError(f"{value_name} must be positive.")
    return z, values


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
    """B-spline Vs profile inside one depth segment.

    Coefficients are basis-expansion weights on the segment-local normalized
    coordinate. They are not point velocities except at the open-clamped segment
    endpoints.
    """

    coefficients: ArrayLike
    degree: int = 3
    knot_spacing: str = "geometric"
    knot_alpha: float = 2.0

    def __post_init__(self):
        coefficients = _as_array(self.coefficients, name="coefficients")
        if np.any(coefficients <= 0):
            raise ValueError("B-spline Vs coefficients must be positive.")
        if self.degree < 0:
            raise ValueError("B-spline degree must be non-negative.")
        if self.knot_alpha <= 0:
            raise ValueError("B-spline knot_alpha must be positive.")
        object.__setattr__(self, "coefficients", coefficients)

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        if thickness <= 0:
            raise ValueError("BSplineVs requires a positive segment thickness.")
        return evaluate_bspline(
            z_local,
            self.coefficients,
            degree=self.degree,
            domain=(0.0, float(thickness)),
            spacing=self.knot_spacing,
            alpha=self.knot_alpha,
        )


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


def layered_vs_model_from_layer_top_depths(
    z: ArrayLike,
    vs: ArrayLike,
    *,
    scaling: ScalingConfig = ScalingConfig(),
) -> LayeredVsModel:
    """Build a piecewise-constant model when depths are layer tops.

    The final depth marks the top of the half-space. For example,
    z=[0, 2, 5] and vs=[1.5, 2.5, 3.5] becomes 0-2 km, 2-5 km, and a
    5+ km half-space. Use this only when a reference model is already layered,
    not for point-sampled continuous profiles.
    """
    z, vs = _validate_depth_profile(z, vs, value_name="vs", min_points=1)
    thickness = np.r_[np.diff(z), 0.0]
    return LayeredVsModel(thickness=thickness, vs=vs, scaling=scaling)


def layered_model_from_layer_top_depths(
    z: ArrayLike,
    *,
    vs: ArrayLike,
    vp: ArrayLike | None = None,
    rho: ArrayLike | None = None,
    scaling: ScalingConfig = ScalingConfig(),
) -> LayeredModel:
    """Build a full elastic `LayeredModel` when depths are layer tops.

    `vs` is required. If `vp` or `rho` are omitted, they are estimated from
    `vs` using `scaling`. Explicit `vp`/`rho` arrays must use the same layer-top
    depths as `vs`.
    """
    vs_model = layered_vs_model_from_layer_top_depths(z, vs, scaling=scaling)
    scaled_vp, scaled_rho = scaling.apply(vs_model.vs)
    if vp is None:
        layer_vp = scaled_vp
    else:
        _, layer_vp = _validate_depth_profile(z, vp, value_name="vp", min_points=1)
    if rho is None:
        layer_rho = scaled_rho
    else:
        _, layer_rho = _validate_depth_profile(z, rho, value_name="rho", min_points=1)
    return LayeredModel(
        thickness=vs_model.thickness,
        vp=layer_vp,
        vs=vs_model.vs,
        rho=layer_rho,
    )


def layer_values_from_depth_profile(
    z: ArrayLike,
    values: ArrayLike,
    *,
    config: DiscretizationConfig | None = None,
    boundaries: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a point-valued depth profile into layer thicknesses and values.

    Here `(z, values)` means "property value measured/evaluated at this depth",
    not "layer-top value". The profile is linearly interpolated, and each output
    layer is assigned the midpoint value. The returned arrays both include the
    final half-space marker/value.

    Pass `boundaries` for variable layer spacing, for example dense shallow
    layers and coarser deep layers. If omitted, `config.dz` builds a regular
    grid from 0 to `config.zmax` or the deepest input depth.
    """
    z, values = _validate_depth_profile(
        z,
        values,
        value_name="values",
        min_points=2,
    )
    config = config or DiscretizationConfig()
    zmax = z[-1] if config.zmax is None else min(config.zmax, z[-1])

    if boundaries is None:
        boundaries = _segment_boundaries(
            0.0,
            float(zmax),
            dz=config.dz,
            force_depths=config.force_depths,
        )
    else:
        boundaries = _as_array(boundaries, name="boundaries")
        if len(boundaries) < 2:
            raise ValueError("boundaries must contain at least two depths.")
        if not np.isclose(boundaries[0], 0.0):
            raise ValueError("The first boundary must be 0 km.")
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("boundaries must be strictly increasing.")
        if boundaries[-1] > z[-1] and not np.isclose(boundaries[-1], z[-1]):
            raise ValueError("The deepest boundary cannot exceed the input profile.")
        zmax = float(boundaries[-1])

    layer_tops = boundaries[:-1]
    layer_bottoms = boundaries[1:]
    z_mid = 0.5 * (layer_tops + layer_bottoms)
    layer_values = np.interp(z_mid, z, values)
    halfspace_value = float(np.interp(zmax, z, values))
    thickness = np.r_[np.diff(boundaries), 0.0]
    layer_values = np.r_[layer_values, halfspace_value]
    return thickness, layer_values


def layered_vs_model_from_depth_profile(
    z: ArrayLike,
    vs: ArrayLike,
    *,
    config: DiscretizationConfig | None = None,
    boundaries: ArrayLike | None = None,
    scaling: ScalingConfig = ScalingConfig(),
) -> LayeredVsModel:
    """Build a `LayeredVsModel` from a point-sampled `(z, Vs)` profile."""
    thickness, layer_vs = layer_values_from_depth_profile(
        z,
        vs,
        config=config,
        boundaries=boundaries,
    )
    return LayeredVsModel(thickness=thickness, vs=layer_vs, scaling=scaling)


def layered_model_from_depth_profiles(
    z: ArrayLike,
    *,
    vs: ArrayLike,
    vp: ArrayLike | None = None,
    rho: ArrayLike | None = None,
    config: DiscretizationConfig | None = None,
    boundaries: ArrayLike | None = None,
    scaling: ScalingConfig = ScalingConfig(),
) -> LayeredModel:
    """Build a full elastic `LayeredModel` from sampled depth profiles.

    `vs` is required. If `vp` or `rho` are omitted, they are estimated from
    `vs` using `scaling`. Explicit `vp`/`rho` profiles use the same layer
    boundaries and midpoint sampling as `vs`.
    """
    thickness, layer_vs = layer_values_from_depth_profile(
        z,
        vs,
        config=config,
        boundaries=boundaries,
    )
    scaled_vp, scaled_rho = scaling.apply(layer_vs)
    if vp is None:
        layer_vp = scaled_vp
    else:
        _, layer_vp = layer_values_from_depth_profile(
            z,
            vp,
            config=config,
            boundaries=boundaries,
        )
    if rho is None:
        layer_rho = scaled_rho
    else:
        _, layer_rho = layer_values_from_depth_profile(
            z,
            rho,
            config=config,
            boundaries=boundaries,
        )
    return LayeredModel(thickness=thickness, vp=layer_vp, vs=layer_vs, rho=layer_rho)


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


def parameterized_vs_model_from_depth_profile(
    z: ArrayLike,
    vs: ArrayLike,
    *,
    scaling: ScalingConfig = ScalingConfig(),
) -> ParameterizedVsModel:
    """Build a linearly interpolated Vs model from sampled depth-Vs pairs.

    This is the usual helper for continuous reference profiles. Each adjacent
    pair of depth samples becomes one `GradientVs` segment. Later,
    `to_layered_model(DiscretizationConfig(dz=...))` controls how finely those
    gradient segments are approximated by constant-velocity layers.
    """
    z, vs = _validate_depth_profile(z, vs, value_name="vs", min_points=2)
    segments = tuple(
        VsSegment(
            top=float(z[i]),
            bottom=float(z[i + 1]),
            profile=GradientVs(top=float(vs[i]), bottom=float(vs[i + 1])),
        )
        for i in range(len(z) - 1)
    )
    return ParameterizedVsModel(segments=segments, scaling=scaling)


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
