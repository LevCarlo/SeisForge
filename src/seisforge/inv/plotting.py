"""Plotting helpers for inversion models."""

from __future__ import annotations

import numpy as np

from seisforge.inv.model import LayeredModel, LayeredVsModel


def layer_stairs(
    thickness,
    values,
    *,
    zmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return layer values and depth edges for `Axes.stairs`.

    This is intentionally property-agnostic: pass the same `thickness` with
    `vs`, `vp`, `rho`, or any other layer attribute. A final zero-thickness
    value is treated as the half-space.
    """
    thickness = np.asarray(thickness, dtype=float)
    values = np.asarray(values, dtype=float)
    if thickness.ndim != 1 or values.ndim != 1:
        raise ValueError("thickness and values must be one-dimensional arrays.")
    if len(thickness) != len(values):
        raise ValueError("thickness and values must have the same length.")
    if len(thickness) == 0:
        raise ValueError("At least one layer is required.")
    if not np.all(np.isfinite(thickness)) or not np.all(np.isfinite(values)):
        raise ValueError("thickness and values must contain only finite values.")

    has_halfspace = np.isclose(thickness[-1], 0.0)
    finite_thickness = thickness[:-1] if has_halfspace else thickness
    if np.any(finite_thickness <= 0):
        raise ValueError("Finite layer thicknesses must be positive.")

    finite_edges = np.r_[0.0, np.cumsum(finite_thickness)]
    finite_bottom = float(finite_edges[-1])
    if zmax is None:
        if has_halfspace and np.isclose(finite_bottom, 0.0):
            raise ValueError("zmax is required for a pure half-space model.")
        zmax = finite_bottom
    if zmax <= 0:
        raise ValueError("zmax must be positive.")
    if not has_halfspace and zmax > finite_bottom:
        raise ValueError("zmax cannot exceed the finite model depth without a half-space value.")

    edges = finite_edges[finite_edges < zmax]
    if len(edges) == 0:
        edges = np.array([0.0])
    if not np.isclose(edges[-1], zmax):
        edges = np.r_[edges, zmax]

    layer_values = values[: len(edges) - 1]
    if has_halfspace and zmax > finite_bottom:
        layer_values = np.r_[values[: len(finite_edges) - 1], values[-1]]
    return layer_values, edges


def model_property_stairs(
    model: LayeredModel | LayeredVsModel,
    values,
    *,
    zmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stairs arrays for one property attached to a layered model."""
    return layer_stairs(model.thickness, values, zmax=zmax)


def plot_vs_layering(
    z,
    vs,
    layered_model: LayeredModel | LayeredVsModel,
    *,
    ax=None,
    zmax: float | None = None,
    profile_label: str = "sampled profile",
    layered_label: str = "layered model",
):
    """Compare a sampled `(z, vs)` profile with its layered approximation."""
    import matplotlib.pyplot as plt

    z = np.asarray(z, dtype=float)
    vs = np.asarray(vs, dtype=float)
    if zmax is None:
        zmax = float(z[-1])

    if ax is None:
        _, ax = plt.subplots()

    layer_vs, layer_edges = model_property_stairs(
        layered_model,
        layered_model.vs,
        zmax=zmax,
    )
    ax.plot(vs, z, color="tab:blue", linewidth=2.0, label=profile_label)
    ax.stairs(
        layer_vs,
        layer_edges,
        orientation="horizontal",
        baseline=None,
        color="black",
        linewidth=1.5,
        label=layered_label,
    )
    ax.set_xlabel("Vs (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_ylim(zmax, 0.0)
    ax.legend()
    return ax
