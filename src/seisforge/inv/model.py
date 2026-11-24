"""Model objects for single-station inversion workflows.

The important split is:

parameterized Vs model -> feature-preserving discretization -> layered model

`LayeredModel` is the final format for forward solvers such as disba. The
profile/segment classes are the parameterization layer used by BayesBay now and
custom MCMC later.

A `ParameterizedVsModel` is an ordered stack of `VsSegment`s separated by
`Interface`s. Each interface has an absolute depth and an optional gradational
width; each segment has its own Vs profile (in segment-local coordinates that
stretch with the interfaces) and optionally its own Vp/rho scaling. The
conversion to finite layers lives in `seisforge.inv.layering`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from seisforge.inv.bspline import bspline_knots, evaluate_bspline, evaluate_bspline_derivative
from seisforge.inv.scaling import RHO_FROM_VP_SCALINGS, estimate_rho, estimate_vp
from seisforge.inv.transitions import (
    TRANSITION_SHAPES,
    transition_half_support_km,
    transition_weight,
    transition_weight_derivative_per_km,
)


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
    # Optional provenance: index of the parameterized segment each layer was
    # sampled from (the half-space included). Not used by forward solvers.
    segment_index: ArrayLike | None = field(default=None, compare=False)

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

        if self.segment_index is not None:
            segment_index = np.asarray(self.segment_index, dtype=int)
            if segment_index.shape != vs.shape:
                raise ValueError("segment_index must have one entry per layer.")
            object.__setattr__(self, "segment_index", segment_index)

        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "vp", vp)
        object.__setattr__(self, "vs", vs)
        object.__setattr__(self, "rho", rho)

    @property
    def n_layers(self) -> int:
        return len(self.vs)

    @property
    def layer_tops(self) -> np.ndarray:
        """Top depth of every layer, the half-space included."""
        return np.r_[0.0, self.interfaces]

    @property
    def interfaces(self) -> np.ndarray:
        """Interface depths below the surface, excluding the half-space marker."""
        return np.cumsum(self.thickness[:-1])

    def as_disba(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays in disba order: thickness, vp, vs, rho."""
        return self.thickness, self.vp, self.vs, self.rho

    def to_hvsr_model(self):
        """Convert to `seisforge.hvsr.ElasticLayerModel` (finite thicknesses only)."""
        from seisforge.hvsr.models import ElasticLayerModel

        return ElasticLayerModel(self.thickness[:-1], self.vp, self.vs, self.rho)


class VsProfile(Protocol):
    """Layer-internal Vs parameterization.

    A profile evaluates local depth inside one segment, so samplers do not
    care how Vs is stored. ``derivative`` is ``dVs/dz`` (km/s per km) and
    ``breakpoints`` the local depths inside the segment where the profile is
    not smooth (B-spline knots); prior constraints use both to analyse the
    continuous profile without a depth grid.
    """

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        ...

    def derivative(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        ...

    def breakpoints(self, *, thickness: float) -> np.ndarray:
        ...


@dataclass(frozen=True)
class ConstantVs:
    value: float

    def evaluate(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        z_local = np.asarray(z_local, dtype=float)
        return np.full_like(z_local, self.value, dtype=float)

    def derivative(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        return np.zeros_like(np.asarray(z_local, dtype=float))

    def breakpoints(self, *, thickness: float) -> np.ndarray:
        return np.array([], dtype=float)


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

    def derivative(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        if thickness <= 0:
            raise ValueError("GradientVs requires a positive segment thickness.")
        z_local = np.asarray(z_local, dtype=float)
        return np.full_like(z_local, (self.bottom - self.top) / thickness, dtype=float)

    def breakpoints(self, *, thickness: float) -> np.ndarray:
        return np.array([], dtype=float)


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

    def derivative(self, z_local: ArrayLike, *, thickness: float) -> np.ndarray:
        if thickness <= 0:
            raise ValueError("BSplineVs requires a positive segment thickness.")
        return evaluate_bspline_derivative(
            z_local,
            self.coefficients,
            degree=self.degree,
            domain=(0.0, float(thickness)),
            spacing=self.knot_spacing,
            alpha=self.knot_alpha,
        )

    def breakpoints(self, *, thickness: float) -> np.ndarray:
        """Distinct interior knots, as local depths in km."""
        knots = bspline_knots(
            len(self.coefficients),
            degree=self.degree,
            spacing=self.knot_spacing,
            alpha=self.knot_alpha,
        )
        interior = np.unique(knots[(knots > 0.0) & (knots < 1.0)])
        return interior * float(thickness)


@dataclass(frozen=True)
class VsSegment:
    """Depth interval with its own Vs profile.

    Segment boundaries are treated as model features. Discretization never
    creates a layer that crosses from one segment into another.
    """

    top: float
    bottom: float
    profile: VsProfile
    name: str | None = None
    # Vp/rho scaling for this segment; None uses the model-level scaling.
    scaling: ScalingConfig | None = None

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

    def evaluate_clamped(self, z: ArrayLike) -> np.ndarray:
        """Evaluate with depths clamped to the segment (constant extension).

        Gradational interfaces evaluate each neighbouring segment slightly
        outside its own interval; clamping keeps that extension bounded for
        every profile type, including B-splines.
        """
        z = np.clip(np.asarray(z, dtype=float), self.top, self.bottom)
        return self.evaluate(z)

    def derivative(self, z: ArrayLike) -> np.ndarray:
        """``dVs/dz`` (km/s per km) inside the segment."""
        z = np.asarray(z, dtype=float)
        return self.profile.derivative(z - self.top, thickness=self.thickness)

    def derivative_clamped(self, z: ArrayLike) -> np.ndarray:
        """Derivative of `evaluate_clamped`: 0 outside the segment."""
        z = np.asarray(z, dtype=float)
        inside = (z >= self.top) & (z <= self.bottom)
        clamped = np.clip(z, self.top, self.bottom)
        return np.where(inside, self.derivative(clamped), 0.0)

    def breakpoints(self) -> np.ndarray:
        """Absolute depths inside the segment where the profile is not smooth."""
        return self.top + self.profile.breakpoints(thickness=self.thickness)


@dataclass(frozen=True)
class Interface:
    """Boundary between two segments.

    ``depth_km`` is the absolute depth of the interface (the 50 % point of a
    gradational transition). ``width_km`` is the 10-90 % transition thickness;
    0 is a hard step. ``shape`` is one of `seisforge.inv.transitions.TRANSITION_SHAPES`.
    """

    depth_km: float
    width_km: float = 0.0
    shape: str = "sigmoid"
    name: str | None = None

    def __post_init__(self):
        if not np.isfinite(self.depth_km) or self.depth_km <= 0:
            raise ValueError("Interface depth must be finite and positive.")
        if not np.isfinite(self.width_km) or self.width_km < 0:
            raise ValueError("Interface width must be finite and non-negative.")
        if self.shape not in TRANSITION_SHAPES:
            raise ValueError(f"Interface shape must be one of {TRANSITION_SHAPES}.")

    @property
    def is_hard(self) -> bool:
        return self.width_km == 0

    @property
    def half_support_km(self) -> float:
        """Half-thickness of the depth range the transition affects."""
        return transition_half_support_km(self.width_km, self.shape)

    @property
    def support_km(self) -> tuple[float, float]:
        half = self.half_support_km
        return self.depth_km - half, self.depth_km + half

    def weight(self, z: ArrayLike) -> np.ndarray:
        """Weight of the segment below: 0 above, 1 below the transition."""
        return transition_weight(z, self.depth_km, self.width_km, self.shape)

    def weight_derivative(self, z: ArrayLike) -> np.ndarray:
        """``d weight / dz`` (1/km); 0 for a hard step."""
        return transition_weight_derivative_per_km(z, self.depth_km, self.width_km, self.shape)


@dataclass(frozen=True)
class DiscretizationConfig:
    """Controls conversion from parameterized model to forward-model layers."""

    dz: float = 0.5
    zmax: float | None = None
    force_depths: tuple[float, ...] = ()
    # Fixed layer count per segment (int, or mapping segment name -> int).
    # Unlike dz, a fixed count keeps layer geometry continuous when interface
    # depths are sampled, so the likelihood has no steps in depth.
    layers_per_segment: int | Mapping[str, int] | None = None
    # Thin layers placed across the support of each gradational interface.
    transition_layers: int = 16
    # How a layer takes its value from the continuous model:
    # "midpoint" (value at layer centre) or "mean" (depth average).
    layer_value: str = "midpoint"
    # Depth-dependent layering (see `seisforge.inv.layering`): target layer
    # thickness h(z) = min(dz_max, dz + (dz_growth_ratio - 1) * z), z the
    # absolute depth in km. 1 keeps the uniform dz layering bit-for-bit.
    dz_growth_ratio: float = 1.0
    # Largest target thickness in km for the graded layering; None: no cap.
    dz_max: float | None = None

    def __post_init__(self):
        if self.dz <= 0:
            raise ValueError("Discretization dz must be positive.")
        if not np.isfinite(self.dz_growth_ratio) or self.dz_growth_ratio < 1.0:
            raise ValueError("dz_growth_ratio must be finite and at least 1.")
        if self.dz_max is not None and not self.dz_max >= self.dz:
            raise ValueError("dz_max must be at least dz when provided.")
        if self.zmax is not None and self.zmax <= 0:
            raise ValueError("zmax must be positive when provided.")
        if self.transition_layers < 2:
            raise ValueError("transition_layers must be at least 2.")
        if self.layer_value not in ("midpoint", "mean"):
            raise ValueError("layer_value must be 'midpoint' or 'mean'.")
        counts = self.layers_per_segment
        if counts is not None:
            values = counts.values() if isinstance(counts, Mapping) else [counts]
            if any(int(value) != value or value < 1 for value in values):
                raise ValueError("layers_per_segment must be positive integers.")


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
        rho_from_vp = RHO_FROM_VP_SCALINGS.get(self.rho_method.lower())
        if rho_from_vp is not None:
            rho = rho_from_vp(vp, **(self.rho_kwargs or {}))
        else:
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
    grid from 0 to `config.zmax` or the deepest input depth (graded when
    `config.dz_growth_ratio` > 1, see `seisforge.inv.layering`).
    """
    z, values = _validate_depth_profile(
        z,
        values,
        value_name="values",
        min_points=2,
    )
    config = config or DiscretizationConfig()
    zmax = z[-1] if config.zmax is None else min(config.zmax, z[-1])

    if boundaries is None and config.dz_growth_ratio != 1.0:
        from seisforge.inv.layering import graded_layer_boundaries

        boundaries = graded_layer_boundaries(
            0.0,
            float(zmax),
            dz_km=config.dz,
            growth_ratio=config.dz_growth_ratio,
            max_thickness_km=config.dz_max,
            force_depths=config.force_depths,
        )
    elif boundaries is None:
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
    """Feature-preserving Vs model: segments separated by interfaces.

    ``interfaces[i]`` separates ``segments[i]`` and ``segments[i + 1]`` and its
    depth must equal their shared boundary. Without explicit interfaces every
    boundary is a hard step (``width_km = 0``), which is the classic segmented
    model. Gradational interfaces blend the clamped profiles of both
    neighbours; their supports must not overlap each other, the surface, or
    the model bottom.
    """

    segments: tuple[VsSegment, ...]
    scaling: ScalingConfig = ScalingConfig()
    interfaces: tuple[Interface, ...] | None = None

    def __post_init__(self):
        if len(self.segments) == 0:
            raise ValueError("ParameterizedVsModel requires at least one segment.")
        if not np.isclose(self.segments[0].top, 0.0):
            raise ValueError("The first segment must start at 0 km.")

        for prev, current in zip(self.segments[:-1], self.segments[1:]):
            if not np.isclose(prev.bottom, current.top):
                raise ValueError("Segments must be contiguous and ordered.")

        if self.interfaces is None:
            interfaces = tuple(Interface(depth_km=segment.top) for segment in self.segments[1:])
        else:
            interfaces = tuple(self.interfaces)
            if len(interfaces) != len(self.segments) - 1:
                raise ValueError("A model needs exactly one interface between each pair of segments.")
            for interface, segment in zip(interfaces, self.segments[1:]):
                if not np.isclose(interface.depth_km, segment.top):
                    raise ValueError("Interface depths must equal the segment boundaries.")
        object.__setattr__(self, "interfaces", interfaces)
        _validate_transition_supports(interfaces, self.zmax)

    @property
    def zmax(self) -> float:
        return self.segments[-1].bottom

    def interface_depths(self) -> np.ndarray:
        return np.array([interface.depth_km for interface in self.interfaces], dtype=float)

    def discontinuities(self) -> np.ndarray:
        """Depths of hard (zero-width) interfaces."""
        return np.array(
            [interface.depth_km for interface in self.interfaces if interface.is_hard],
            dtype=float,
        )

    def segment_scaling(self, index: int) -> ScalingConfig:
        return self.segments[index].scaling or self.scaling

    def segment_index(self, z: ArrayLike) -> np.ndarray:
        """Segment that owns each depth when every interface is hard."""
        z = np.asarray(z, dtype=float)
        if np.any(z < 0) or np.any(z > self.zmax):
            raise ValueError("Requested depths fall outside the model range.")
        return np.searchsorted(self.interface_depths(), z, side="right")

    def evaluate(self, z: ArrayLike) -> np.ndarray:
        """Vs at depth ``z`` (km/s), including gradational interfaces."""
        return self._blend(z, lambda index, depth: (self.segments[index].evaluate_clamped(depth),))[0]

    def evaluate_properties(self, z: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vs, Vp, and rho at depth ``z``.

        Vp and rho come from each segment's own scaling applied to that
        segment's Vs, and are blended with the same transition weights as Vs.
        Scaling a blended Vs with one relation would apply e.g. a sediment
        relation to crustal velocities inside the transition.
        """
        return self._blend(z, self._segment_properties)

    def _segment_properties(self, index: int, z: np.ndarray):
        vs = self.segments[index].evaluate_clamped(z)
        try:
            vp, rho = self.segment_scaling(index).apply(vs)
        except ValueError as exc:
            label = self.segments[index].name or f"segment {index}"
            raise ValueError(f"{label}: {exc}") from exc
        return vs, vp, rho

    def _blend(self, z: ArrayLike, values) -> tuple[np.ndarray, ...]:
        z = np.asarray(z, dtype=float)
        owner = self.segment_index(z)
        result = None
        for index in range(len(self.segments)):
            mask = owner == index
            if not np.any(mask):
                continue
            parts = values(index, z[mask])
            if result is None:
                result = tuple(np.empty(z.shape, dtype=float) for _ in parts)
            for out, part in zip(result, parts):
                out[mask] = part
        for index, interface in enumerate(self.interfaces):
            if interface.is_hard:
                continue
            top, bottom = interface.support_km
            mask = (z > top) & (z < bottom)
            if not np.any(mask):
                continue
            weight = interface.weight(z[mask])
            upper = values(index, z[mask])
            lower = values(index + 1, z[mask])
            for out, above, below in zip(result, upper, lower):
                out[mask] = (1.0 - weight) * above + weight * below
        return result

    def to_layered_model(self, config: DiscretizationConfig | None = None) -> LayeredModel:
        """Convert to finite layers; see `seisforge.inv.layering`."""
        from seisforge.inv.layering import layered_model_from_parameterized

        return layered_model_from_parameterized(self, config or DiscretizationConfig())


def _validate_transition_supports(interfaces: tuple[Interface, ...], zmax: float) -> None:
    previous_bottom = 0.0
    for interface in interfaces:
        top, bottom = interface.support_km
        label = interface.name or f"interface at {interface.depth_km:g} km"
        if top < previous_bottom and not np.isclose(top, previous_bottom):
            raise ValueError(
                f"{label}: transition support overlaps the surface or the previous interface."
            )
        previous_bottom = max(previous_bottom, bottom)
    if previous_bottom > zmax and not np.isclose(previous_bottom, zmax):
        raise ValueError("Interface transition support extends below the model bottom.")


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
