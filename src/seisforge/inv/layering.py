"""Conversion of a parameterized Vs model to finite layers, and its diagnostics.

Forward solvers (disba, future RF and compliance kernels) see only the layered
model, so this step decides whether a well-designed continuous model actually
reaches the physics. The rules:

1. A hard interface (``width_km = 0``) is always a layer boundary, so no layer
   straddles a velocity jump.
2. Each segment between interface supports is split into layers either by a
   target thickness ``dz`` or by a fixed count (``layers_per_segment``). A
   fixed count keeps layer geometry continuous when interface depths are
   sampled; ``dz`` changes the layer count in steps.
3. The support of every gradational interface gets ``transition_layers``
   equal layers, so the transition is resolved whatever its width.
4. Each layer takes Vs, Vp, and rho from the continuous model, either at the
   layer midpoint or as the depth average (``layer_value``). Vp and rho come
   from the owning segment's scaling, blended inside transitions.
5. The half-space takes the model properties at the bottom depth.

`layering_report` quantifies what the conversion lost: velocity misfit inside
layers, vertical travel-time error (the quantity receiver-function delays
depend on), interface preservation, and transition sampling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from seisforge.inv.model import (
    DiscretizationConfig,
    LayeredModel,
    ParameterizedVsModel,
    _segment_boundaries,
)


@dataclass(frozen=True)
class LayerGeometry:
    """Finite-layer boundaries and the bottom (half-space top) depth."""

    tops_km: np.ndarray
    bottoms_km: np.ndarray
    zmax_km: float

    @property
    def thickness_km(self) -> np.ndarray:
        return self.bottoms_km - self.tops_km


def layer_geometry(
    model: ParameterizedVsModel,
    config: DiscretizationConfig,
) -> LayerGeometry:
    """Finite-layer boundaries that respect interfaces and transition supports."""
    zmax = model.zmax if config.zmax is None else min(config.zmax, model.zmax)
    if zmax <= 0:
        raise ValueError("Discretization zmax must be positive.")

    tops: list[float] = []
    bottoms: list[float] = []
    for top, bottom, n_layers in _regions(model, config):
        bottom = min(bottom, zmax)
        if bottom <= top:
            continue
        if n_layers is None:
            boundaries = _segment_boundaries(
                top, bottom, dz=config.dz, force_depths=config.force_depths
            )
        else:
            boundaries = _fixed_count_boundaries(
                top, bottom, n_layers, force_depths=config.force_depths
            )
        tops.extend(boundaries[:-1])
        bottoms.extend(boundaries[1:])
        if np.isclose(bottom, zmax):
            break

    if not tops:
        raise ValueError("Discretization produced no finite layers.")
    return LayerGeometry(
        tops_km=np.asarray(tops, dtype=float),
        bottoms_km=np.asarray(bottoms, dtype=float),
        zmax_km=float(zmax),
    )


def layered_model_from_parameterized(
    model: ParameterizedVsModel,
    config: DiscretizationConfig,
) -> LayeredModel:
    """Discretize a parameterized model into a forward-ready `LayeredModel`."""
    geometry = layer_geometry(model, config)
    if config.layer_value == "mean":
        vs, vp, rho = _layer_means(model, geometry.tops_km, geometry.bottoms_km)
    else:
        z_mid = 0.5 * (geometry.tops_km + geometry.bottoms_km)
        vs, vp, rho = model.evaluate_properties(z_mid)

    bottom = np.array([geometry.zmax_km])
    vs_half, vp_half, rho_half = model.evaluate_properties(bottom)
    segment_index = np.r_[
        model.segment_index(0.5 * (geometry.tops_km + geometry.bottoms_km)),
        model.segment_index(bottom),
    ]
    return LayeredModel(
        thickness=np.r_[geometry.thickness_km, 0.0],
        vp=np.r_[vp, vp_half],
        vs=np.r_[vs, vs_half],
        rho=np.r_[rho, rho_half],
        segment_index=segment_index,
    )


@dataclass(frozen=True)
class LayeringReport:
    """How faithfully a `LayeredModel` represents its parameterized model.

    Errors are measured over the finite layers (0 to the half-space top).
    Vertical travel time is ``integral dz / v``; its relative error bounds the
    error in vertical delay times such as receiver-function Ps arrivals.
    """

    n_layers: int
    min_thickness_km: float
    hard_interfaces_preserved: bool
    max_interface_offset_km: float
    transition_layer_counts: tuple[int, ...]
    max_abs_vs_error_km_s: float
    vs_vertical_time_relative_error: float
    vp_vertical_time_relative_error: float

    def summary(self) -> str:
        return (
            f"{self.n_layers} layers (min {self.min_thickness_km:.4g} km); "
            f"hard interfaces preserved: {self.hard_interfaces_preserved} "
            f"(max offset {self.max_interface_offset_km:.2g} km); "
            f"transition layers: {list(self.transition_layer_counts)}; "
            f"max |dVs| {self.max_abs_vs_error_km_s:.4g} km/s; "
            f"vertical time error Vs {self.vs_vertical_time_relative_error:+.3e}, "
            f"Vp {self.vp_vertical_time_relative_error:+.3e}"
        )


def layering_report(
    model: ParameterizedVsModel,
    layered: LayeredModel,
    *,
    quadrature_points: int = 16,
) -> LayeringReport:
    """Compare a layered model with the continuous model it came from."""
    thickness = layered.thickness[:-1]
    tops = layered.layer_tops[:-1]
    bottoms = tops + thickness
    zmax = float(bottoms[-1])

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    z = 0.5 * (tops + bottoms)[:, None] + 0.5 * thickness[:, None] * nodes[None, :]
    vs_c, vp_c, _ = model.evaluate_properties(z.ravel())
    vs_c = vs_c.reshape(z.shape)
    vp_c = vp_c.reshape(z.shape)
    half = 0.5 * thickness[:, None] * weights[None, :]

    vs_time = float(np.sum(half / vs_c))
    vp_time = float(np.sum(half / vp_c))
    vs_time_layered = float(np.sum(thickness / layered.vs[:-1]))
    vp_time_layered = float(np.sum(thickness / layered.vp[:-1]))

    hard = [depth for depth in model.discontinuities() if depth < zmax]
    boundaries = layered.layer_tops
    offsets = [float(np.min(np.abs(boundaries - depth))) for depth in hard]
    max_offset = max(offsets, default=0.0)

    mid = 0.5 * (tops + bottoms)
    counts = []
    for interface in model.interfaces:
        if interface.is_hard:
            continue
        top, bottom = interface.support_km
        counts.append(int(np.count_nonzero((mid > top) & (mid < bottom))))

    return LayeringReport(
        n_layers=int(layered.n_layers),
        min_thickness_km=float(np.min(thickness)),
        hard_interfaces_preserved=bool(max_offset <= 1e-9),
        max_interface_offset_km=max_offset,
        transition_layer_counts=tuple(counts),
        max_abs_vs_error_km_s=float(np.max(np.abs(vs_c - layered.vs[:-1, None]))),
        vs_vertical_time_relative_error=(vs_time_layered - vs_time) / vs_time,
        vp_vertical_time_relative_error=(vp_time_layered - vp_time) / vp_time,
    )


def _regions(
    model: ParameterizedVsModel,
    config: DiscretizationConfig,
) -> list[tuple[float, float, int | None]]:
    """Ordered (top, bottom, layer count or None for dz) depth regions."""
    interfaces = model.interfaces
    regions: list[tuple[float, float, int | None]] = []
    for index, segment in enumerate(model.segments):
        top = segment.top
        if index > 0:
            top += interfaces[index - 1].half_support_km
        bottom = segment.bottom
        if index < len(interfaces):
            bottom -= interfaces[index].half_support_km
        if bottom > top and not np.isclose(bottom, top):
            regions.append((top, bottom, _segment_layer_count(segment.name, config)))
        if index < len(interfaces) and not interfaces[index].is_hard:
            support_top, support_bottom = interfaces[index].support_km
            regions.append((support_top, support_bottom, config.transition_layers))
    return regions


def _segment_layer_count(name: str | None, config: DiscretizationConfig) -> int | None:
    counts = config.layers_per_segment
    if counts is None:
        return None
    if isinstance(counts, Mapping):
        if name is None or name not in counts:
            return None
        return int(counts[name])
    return int(counts)


def _fixed_count_boundaries(
    top: float,
    bottom: float,
    n_layers: int,
    *,
    force_depths: tuple[float, ...] = (),
) -> np.ndarray:
    boundaries = np.linspace(top, bottom, n_layers + 1)
    forced = [depth for depth in force_depths if top < depth < bottom]
    if forced:
        boundaries = np.unique(np.concatenate([boundaries, np.asarray(forced)]))
    return boundaries


def _layer_means(
    model: ParameterizedVsModel,
    tops: np.ndarray,
    bottoms: np.ndarray,
    quadrature_points: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
    z = 0.5 * (tops + bottoms)[:, None] + 0.5 * (bottoms - tops)[:, None] * nodes[None, :]
    properties = model.evaluate_properties(z.ravel())
    return tuple(
        0.5 * np.sum(values.reshape(z.shape) * weights[None, :], axis=1)
        for values in properties
    )
