"""Public forward entry point, independent of a particular numerical backend."""

from __future__ import annotations

from typing import Protocol

from .models import DiffuseFieldPrediction, DiffuseFieldSettings, ElasticLayerModel


class DiffuseFieldSolver(Protocol):
    def __call__(
        self,
        model: ElasticLayerModel,
        frequencies_hz,
        settings: DiffuseFieldSettings | None = None,
    ) -> DiffuseFieldPrediction: ...


def predict_diffuse_field_hvsr(
    model: ElasticLayerModel,
    frequencies_hz,
    *,
    solver: DiffuseFieldSolver,
    settings: DiffuseFieldSettings | None = None,
) -> DiffuseFieldPrediction:
    """Predict DFA H/V through an explicitly selected numerical solver."""

    return solver(model, frequencies_hz, settings)
