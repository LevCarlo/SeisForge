"""Extremes of the continuous Vs(z) profile of a `ParameterizedVsModel`.

The hard prior constraints are statements about the continuous model, so they
must not depend on a depth grid. This module cuts a depth range into *smooth
pieces* and reports values and slopes from them:

- a piece ends at every segment boundary, every edge of a transition support,
  and every B-spline knot, so Vs is smooth (polynomial, or a smooth blend)
  inside each piece;
- inside a segment, away from transitions, Vs and ``dVs/dz`` are the segment
  profile and its analytic derivative;
- inside the support of a gradational interface (`seisforge.inv.transitions`),
  ``Vs = (1 - S) v_u + S v_l`` with both neighbours clamped to their own
  intervals, and

      dVs/dz = S'(z) (v_l - v_u) + (1 - S) v_u' + S v_l'

  (``v'`` = 0 where a neighbour is clamped);
- a hard interface (width 0) is a jump: the piece above ends with the upper
  segment's value, the piece below starts with the lower segment's value, and
  no slope is defined across it.

Extremes inside a piece are located where the derivative changes sign: for
Vs the analytic slope, for the slope a centred difference of the analytic
slope (step `CURVATURE_STEP_KM`). The derivative is sampled at points fixed to
the piece (equally spaced from its top to its bottom), every sign change is
solved for its root (Illinois regula falsi, `ROOT_ITERATIONS`), and the
function is evaluated exactly at the roots and the piece ends. Every reported
value is attained by the profile, and the samples move with the interfaces and
knots, so the result does not depend on any global depth spacing or on where
an interface sits relative to one. Sample counts are assumed values
(`SEGMENT_PIECE_SAMPLES`, `TRANSITION_PIECE_SAMPLES`): between knots a cubic
B-spline has a quadratic slope (at most one sign change per sample interval is
missed only when two roots fall within one interval, i.e. an extreme of size
~ v'' dz^2 / 8), and a transition blend of such pieces has a handful of
extremes; `tests/test_inv_constraints_continuous.py` compares the result with
dense sampling.

Depth ranges are closed, ``[top_km, bottom_km]``, clipped to the model bottom.
A jump exactly at an end of the range contributes only the side inside it.
Units: km, km/s, km/s per km.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from seisforge.inv.model import ParameterizedVsModel

#: Samples per piece inside one segment (between knots). Assumed value.
SEGMENT_PIECE_SAMPLES = 17
#: Samples per piece inside a transition support (each side of the interface).
#: Assumed value.
TRANSITION_PIECE_SAMPLES = 65
#: Regula falsi iterations per root (Illinois variant, superlinear); assumed
#: value, far more than needed for brackets of a few hundred metres.
ROOT_ITERATIONS = 30
#: Step (km) of the centred difference that gives the curvature d2Vs/dz2 from
#: the analytic slope. Assumed value: rounding error ~1e-16 / 1e-6 = 1e-10.
CURVATURE_STEP_KM = 1e-6


@dataclass(frozen=True)
class SmoothPiece:
    """Depth interval on which Vs is smooth, with its value and slope functions."""

    top_km: float
    bottom_km: float
    vs: Callable[[np.ndarray], np.ndarray]
    slope: Callable[[np.ndarray], np.ndarray]
    n_samples: int

    def samples_km(self) -> np.ndarray:
        z = np.linspace(self.top_km, self.bottom_km, self.n_samples)
        # The bottom sample is seen from inside the piece; this matters only
        # for a degree-0 B-spline, whose value jumps at a knot.
        z[-1] = np.nextafter(self.bottom_km, self.top_km)
        return z


@dataclass(frozen=True)
class TurningPoints:
    """Vs at the ends of every smooth piece and at every local extreme inside.

    Depth-ordered; Vs is monotone between consecutive points of one piece. A
    jump appears as two points at the same depth.
    """

    depth_km: np.ndarray
    vs_km_s: np.ndarray

    def total_decrease_km_s(self) -> float:
        """Sum of all decreases of Vs with depth, jumps included."""
        drops = -np.diff(self.vs_km_s)
        return float(np.sum(drops[drops > 0.0]))

    def max_drop_km_s(self) -> float:
        """Largest ``Vs(z1) - Vs(z2)`` over ``z1 < z2``: the deepest velocity inversion."""
        if len(self.vs_km_s) == 0:
            return 0.0
        return float(np.max(np.maximum.accumulate(self.vs_km_s) - self.vs_km_s))


def smooth_pieces(model: ParameterizedVsModel, top_km: float, bottom_km: float) -> tuple[SmoothPiece, ...]:
    """Cut ``[top_km, bottom_km]`` into pieces on which Vs is smooth."""
    top_km, bottom_km = _check_range(model, top_km, bottom_km)
    if bottom_km <= top_km:
        return ()
    edges = [top_km, bottom_km]
    for segment in model.segments:
        edges.extend([segment.top, segment.bottom])
        edges.extend(segment.breakpoints())
    for interface in model.interfaces:
        if not interface.is_hard:
            edges.extend(interface.support_km)
    edges = np.unique(np.asarray(edges, dtype=float))
    edges = edges[(edges >= top_km) & (edges <= bottom_km)]

    pieces = []
    for upper, lower in zip(edges[:-1], edges[1:]):
        if not lower > upper:
            continue
        pieces.append(_piece(model, float(upper), float(lower)))
    return tuple(pieces)


def vs_turning_points(model: ParameterizedVsModel, top_km: float, bottom_km: float) -> TurningPoints:
    """`TurningPoints` of Vs over ``[top_km, bottom_km]``."""
    depths, values = [], []
    for piece in smooth_pieces(model, top_km, bottom_km):
        z, v = _local_extremes(piece.vs, piece.slope, piece.samples_km())
        depths.append(z)
        values.append(v)
    if not depths:
        z = np.array([top_km], dtype=float)
        return TurningPoints(z, model.evaluate(z))
    return TurningPoints(np.concatenate(depths), np.concatenate(values))


def vs_extremes_km_s(model: ParameterizedVsModel, top_km: float, bottom_km: float) -> tuple[float, float]:
    """Smallest and largest Vs over the range, both sides of every jump included."""
    points = vs_turning_points(model, top_km, bottom_km)
    return float(np.min(points.vs_km_s)), float(np.max(points.vs_km_s))


def vs_slope_extremes_km_s_per_km(
    model: ParameterizedVsModel, top_km: float, bottom_km: float
) -> tuple[float, float]:
    """Smallest and largest ``dVs/dz`` over the smooth pieces; jumps excluded.

    ``(0, 0)`` for an empty range.
    """
    lowest, highest = 0.0, 0.0
    first = True
    for piece in smooth_pieces(model, top_km, bottom_km):
        _, slope = _local_extremes(piece.slope, _curvature(piece.slope), piece.samples_km())
        low, high = float(np.min(slope)), float(np.max(slope))
        lowest, highest = (low, high) if first else (min(lowest, low), max(highest, high))
        first = False
    return lowest, highest


def _piece(model: ParameterizedVsModel, top_km: float, bottom_km: float) -> SmoothPiece:
    middle = 0.5 * (top_km + bottom_km)
    for index, interface in enumerate(model.interfaces):
        if interface.is_hard:
            continue
        support_top, support_bottom = interface.support_km
        if support_top <= middle <= support_bottom:
            upper, lower = model.segments[index], model.segments[index + 1]

            def vs(z, upper=upper, lower=lower, interface=interface):
                weight = interface.weight(z)
                return (1.0 - weight) * upper.evaluate_clamped(z) + weight * lower.evaluate_clamped(z)

            def slope(z, upper=upper, lower=lower, interface=interface):
                weight = interface.weight(z)
                contrast = lower.evaluate_clamped(z) - upper.evaluate_clamped(z)
                return (
                    interface.weight_derivative(z) * contrast
                    + (1.0 - weight) * upper.derivative_clamped(z)
                    + weight * lower.derivative_clamped(z)
                )

            return SmoothPiece(top_km, bottom_km, vs, slope, TRANSITION_PIECE_SAMPLES)
    segment = model.segments[int(model.segment_index(np.array([middle]))[0])]
    return SmoothPiece(
        top_km,
        bottom_km,
        segment.evaluate_clamped,
        lambda z, segment=segment: segment.derivative(np.clip(z, segment.top, segment.bottom)),
        SEGMENT_PIECE_SAMPLES,
    )


def _curvature(slope):
    def curvature(z):
        z = np.asarray(z, dtype=float)
        return (slope(z + CURVATURE_STEP_KM) - slope(z - CURVATURE_STEP_KM)) / (2.0 * CURVATURE_STEP_KM)

    return curvature


def _local_extremes(function, derivative, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``function`` at the piece ends and at every root of ``derivative`` between them."""
    d = np.asarray(derivative(z), dtype=float)
    roots = [z[1:-1][d[1:-1] == 0.0]]
    change = np.flatnonzero(d[:-1] * d[1:] < 0.0)
    if len(change):
        roots.append(_illinois_roots(derivative, z[change], z[change + 1], d[change], d[change + 1]))
    roots = np.sort(np.concatenate(roots))
    points = np.concatenate([z[:1], roots, z[-1:]])
    return points, np.asarray(function(points), dtype=float)


def _illinois_roots(function, a, b, fa, fb) -> np.ndarray:
    """Roots of ``function`` in brackets ``[a, b]`` with ``fa * fb < 0``, vectorized."""
    a, b, fa, fb = (np.array(x, dtype=float) for x in (a, b, fa, fb))
    for _ in range(ROOT_ITERATIONS):
        with np.errstate(divide="ignore", invalid="ignore"):
            c = b - fb * (b - a) / (fb - fa)
        c = np.where(np.isfinite(c), c, 0.5 * (a + b))
        # Converged: the new point moves by less than 1e-12 km (1 nm).
        if np.all(np.abs(c - b) < 1e-12):
            return c
        fc = np.asarray(function(c), dtype=float)
        crossed = fc * fb < 0.0
        # Root between b and c: the old b becomes a. Otherwise keep a and
        # halve its value (Illinois), so a stale end cannot stall convergence.
        a = np.where(crossed, b, a)
        fa = np.where(crossed, fb, 0.5 * fa)
        b, fb = c, fc
        done = fc == 0.0
        a = np.where(done, c, a)
        fa = np.where(done, 0.0, fa)
    return b


def _check_range(model: ParameterizedVsModel, top_km: float, bottom_km: float) -> tuple[float, float]:
    top_km, bottom_km = float(top_km), float(bottom_km)
    if not (np.isfinite(top_km) and np.isfinite(bottom_km)) or top_km < 0 or bottom_km < top_km:
        raise ValueError("Invalid constraint depth_range.")
    return top_km, min(bottom_km, model.zmax)
