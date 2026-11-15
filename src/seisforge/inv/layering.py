"""Conversion of a parameterized Vs model to finite layers, and its diagnostics.

Forward solvers (disba, future RF and compliance kernels) see only the layered
model, so this step decides whether a well-designed continuous model actually
reaches the physics. The rules:

1. A hard interface (``width_km = 0``) is always a layer boundary, so no layer
   straddles a velocity jump.
2. Each segment between interface supports is split into layers either by a
   target thickness ``dz`` or by a fixed count (``layers_per_segment``). A
   fixed count keeps layer geometry continuous when interface depths are
   sampled; ``dz`` changes the layer count in steps. The target thickness is
   uniform by default, or grows with depth (``dz_growth_ratio`` > 1, see
   *Graded layering* below).
3. The support of every gradational interface gets ``transition_layers``
   equal layers, so the transition is resolved whatever its width.
4. Each layer takes Vs, Vp, and rho from the continuous model, either at the
   layer midpoint or as the depth average (``layer_value``). Vp and rho come
   from the owning segment's scaling, blended inside transitions.
5. The half-space takes the model properties at the bottom depth.

`layering_report` quantifies what the conversion lost: velocity misfit inside
layers, vertical travel-time error (the quantity receiver-function delays
depend on), interface preservation, and transition sampling.
`layering_observable_error` measures what matters for the fit: the change of
the predicted observables against a fine uniform reference layering.

Graded layering
---------------
With ``dz_growth_ratio`` r > 1 the target layer thickness at absolute depth z
(km, from the surface, not from the segment top) is

    h(z) = min(dz_max, dz + (r - 1) z),

so layers stacked from the surface without other constraints have thickness
``dz * r**k`` (k = 0, 1, ...) until they reach ``dz_max``. Inside a region
[top, bottom] the layers are equally spaced in the stretched coordinate

    s(z) = integral_0^z dz' / h(z')
         = ln(1 + (r - 1) z / dz) / (r - 1)             for z <= z_c,
         = s(z_c) + (z - z_c) / dz_max                  for z >  z_c,

with z_c = (dz_max - dz) / (r - 1) the depth where the cap is reached
(z_c = infinity without ``dz_max``). The region gets
n = ceil(s(bottom) - s(top)) layers with boundaries at s^-1 of n + 1 equal
steps from s(top) to s(bottom). Each step is at most 1 in s, so no layer is
thicker than h at its bottom; region boundaries (segment ends, transition
supports, ``zmax``) and ``force_depths`` are kept exactly as with uniform
``dz``. Transition supports (``transition_layers``) and segments with a
fixed ``layers_per_segment`` count are not affected. ``dz_growth_ratio = 1``
uses the uniform code path, so default results are bit-for-bit unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from seisforge.inv.forward import predict_dispersion, predict_rayleigh_hv
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
        if n_layers is None and config.dz_growth_ratio != 1.0:
            boundaries = graded_layer_boundaries(
                top,
                bottom,
                dz_km=config.dz,
                growth_ratio=config.dz_growth_ratio,
                max_thickness_km=config.dz_max,
                force_depths=config.force_depths,
            )
        elif n_layers is None:
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


def target_layer_thickness_km(
    depth_km,
    *,
    dz_km: float,
    growth_ratio: float,
    max_thickness_km: float | None = None,
) -> np.ndarray:
    """Graded target thickness ``h(z) = min(dz_max, dz + (r - 1) z)`` in km."""
    depth_km = np.asarray(depth_km, dtype=float)
    _check_graded(dz_km, growth_ratio, max_thickness_km)
    h = dz_km + (growth_ratio - 1.0) * depth_km
    if max_thickness_km is not None:
        h = np.minimum(h, max_thickness_km)
    return h


def graded_layer_boundaries(
    top_km: float,
    bottom_km: float,
    *,
    dz_km: float,
    growth_ratio: float,
    max_thickness_km: float | None = None,
    force_depths: tuple[float, ...] = (),
) -> np.ndarray:
    """Layer boundaries of one region for the graded layering (module docstring).

    Returns increasing depths in km from ``top_km`` to ``bottom_km`` (both
    exact), with every ``force_depths`` entry strictly inside the region added.
    """
    _check_graded(dz_km, growth_ratio, max_thickness_km)
    if not 0.0 <= top_km < bottom_km:
        raise ValueError("Region must satisfy 0 <= top_km < bottom_km.")
    g = growth_ratio - 1.0
    z_cap = np.inf if max_thickness_km is None or g == 0 else (max_thickness_km - dz_km) / g

    def stretched(z):
        z = np.asarray(z, dtype=float)
        if g == 0:
            return z / dz_km
        below = np.log1p(g * np.minimum(z, z_cap) / dz_km) / g
        if np.isfinite(z_cap):
            below = below + np.maximum(z - z_cap, 0.0) / max_thickness_km
        return below

    def depth(s):
        s = np.asarray(s, dtype=float)
        if g == 0:
            return s * dz_km
        s_cap = float(stretched(z_cap)) if np.isfinite(z_cap) else np.inf
        z = dz_km * np.expm1(g * np.minimum(s, s_cap)) / g
        if np.isfinite(s_cap):
            z = z + np.maximum(s - s_cap, 0.0) * max_thickness_km
        return z

    s_top, s_bottom = float(stretched(top_km)), float(stretched(bottom_km))
    # The 1e-9 keeps a region that is an exact number of target layers from
    # gaining a sliver layer through rounding in log/exp.
    n = max(1, int(np.ceil(s_bottom - s_top - 1e-9)))
    boundaries = depth(np.linspace(s_top, s_bottom, n + 1))
    boundaries[0], boundaries[-1] = top_km, bottom_km
    forced = [d for d in force_depths if top_km < d < bottom_km]
    if forced:
        boundaries = np.unique(np.concatenate([boundaries, np.asarray(forced)]))
    return boundaries


def _check_graded(dz_km: float, growth_ratio: float, max_thickness_km: float | None) -> None:
    if not dz_km > 0:
        raise ValueError("dz_km must be positive.")
    if not np.isfinite(growth_ratio) or growth_ratio < 1.0:
        raise ValueError("growth_ratio must be finite and at least 1.")
    if max_thickness_km is not None and not max_thickness_km >= dz_km:
        raise ValueError("max_thickness_km must be at least dz_km.")


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


# Reference layering for `layering_observable_error` when none is given:
# uniform with a quarter of the configured dz. Assumed value: four times finer
# than the finest configured layers is well into the converged range for the
# smooth models tested (tests/test_inv_layering_convergence.py).
REFERENCE_DZ_FRACTION = 0.25


@dataclass(frozen=True)
class ObservableDifference:
    """Configured minus reference prediction of one observable.

    ``sigma_ratio`` is ``max |difference| / sigma`` when uncertainties were
    given, else None. Only periods both layerings solved are compared.
    """

    periods_s: np.ndarray
    difference: np.ndarray
    max_abs_difference: float
    sigma_ratio: float | None


@dataclass(frozen=True)
class LayeringObservableError:
    """Change of predicted observables from a fine uniform reference layering."""

    n_layers: int
    reference_n_layers: int
    reference_dz_km: float
    dispersion: ObservableDifference | None
    hv: ObservableDifference | None

    def summary(self) -> str:
        parts = [
            f"{self.n_layers} layers vs reference {self.reference_n_layers} "
            f"(uniform dz {self.reference_dz_km:.4g} km)"
        ]
        for label, unit, item in (
            ("phase velocity", " km/s", self.dispersion),
            ("H/V", "", self.hv),
        ):
            if item is None:
                continue
            text = f"max |d {label}| {item.max_abs_difference:.3g}{unit}"
            if item.sigma_ratio is not None:
                text += f" ({item.sigma_ratio:.3g} sigma)"
            parts.append(text)
        return "; ".join(parts)


def reference_discretization(
    config: DiscretizationConfig,
    *,
    reference_dz_km: float | None = None,
) -> DiscretizationConfig:
    """Fine uniform layering with the same zmax, interfaces and force_depths.

    ``reference_dz_km`` defaults to ``REFERENCE_DZ_FRACTION * config.dz``.
    Fixed segment counts are replaced by the reference dz; transition
    supports keep ``transition_layers``.
    """
    dz = config.dz * REFERENCE_DZ_FRACTION if reference_dz_km is None else reference_dz_km
    return replace(
        config, dz=float(dz), dz_growth_ratio=1.0, dz_max=None, layers_per_segment=None
    )


def layering_observable_error(
    model: ParameterizedVsModel,
    config: DiscretizationConfig,
    *,
    dispersion=None,
    hv=None,
    reference_dz_km: float | None = None,
) -> LayeringObservableError:
    """Predicted-observable error of a layering, against a fine reference.

    ``dispersion`` and ``hv`` are a `DispersionRequest` / `RayleighHVRequest`,
    or observations with a ``request()`` method and ``sigma`` (as in
    `seisforge.inv.likelihood`), in which case the largest difference is also
    given in units of the data uncertainty. Differences are configured minus
    reference. `ForwardError` from either layering propagates.
    """
    reference = reference_discretization(config, reference_dz_km=reference_dz_km)
    layered = layered_model_from_parameterized(model, config)
    layered_ref = layered_model_from_parameterized(model, reference)

    def compare(item, predict, value_name):
        if item is None:
            return None
        request = item.request() if hasattr(item, "request") else item
        sigma = getattr(item, "sigma", None)
        test = predict(layered, request)
        ref = predict(layered_ref, request)
        periods, i_test, i_ref = np.intersect1d(
            test.periods, ref.periods, return_indices=True
        )
        if periods.size == 0:
            raise ValueError(f"No common periods to compare for {value_name}.")
        difference = getattr(test, value_name)[i_test] - getattr(ref, value_name)[i_ref]
        ratio = None
        if sigma is not None:
            requested = np.asarray(request.periods, dtype=float)
            sigma = np.broadcast_to(np.asarray(sigma, dtype=float), requested.shape)
            index = [int(np.flatnonzero(requested == p)[0]) for p in periods]
            ratio = float(np.max(np.abs(difference) / sigma[index]))
        return ObservableDifference(
            periods_s=periods,
            difference=difference,
            max_abs_difference=float(np.max(np.abs(difference))),
            sigma_ratio=ratio,
        )

    return LayeringObservableError(
        n_layers=int(layered.n_layers),
        reference_n_layers=int(layered_ref.n_layers),
        reference_dz_km=float(reference.dz),
        dispersion=compare(dispersion, predict_dispersion, "velocity"),
        hv=compare(hv, predict_rayleigh_hv, "hv"),
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
