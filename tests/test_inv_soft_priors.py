import numpy as np
import pytest

from seisforge.inv.model import BSplineVs, ConstantVs, GradientVs, ParameterizedVsModel, VsSegment
from seisforge.inv.soft_priors import VsCurvaturePrior, soft_priors_from_config


def _model(profile):
    return ParameterizedVsModel(segments=(VsSegment(0.0, 4.0, profile),))


def test_vs_curvature_prior_is_neutral_for_linear_gradient():
    prior = VsCurvaturePrior(sigma=0.2, dz=0.05)

    evaluation = prior.evaluate(_model(GradientVs(top=2.0, bottom=3.0)))

    assert evaluation.diagnostics["vs_curvature_rms"] < 1.0e-10
    np.testing.assert_allclose(evaluation.log_prior, 0.0, atol=1.0e-18)


def test_vs_curvature_prior_penalizes_bent_bspline_profile():
    prior = VsCurvaturePrior(sigma=0.2, dz=0.02)
    model = _model(BSplineVs(coefficients=[2.0, 2.0, 4.0, 2.0, 3.0]))

    evaluation = prior.evaluate(model)

    rms = evaluation.diagnostics["vs_curvature_rms"]
    assert rms > 0.0
    np.testing.assert_allclose(evaluation.log_prior, -0.5 * (rms / 0.2) ** 2)


def test_vs_curvature_does_not_difference_across_segment_boundaries():
    model = ParameterizedVsModel(
        segments=(
            VsSegment(0.0, 2.0, ConstantVs(2.0)),
            VsSegment(2.0, 4.0, ConstantVs(3.0)),
        )
    )

    evaluation = VsCurvaturePrior(sigma=0.2, dz=0.05).evaluate(model)

    assert evaluation.diagnostics["vs_curvature_rms"] < 1.0e-10


def test_soft_prior_config_requires_known_type_and_positive_sigma():
    suite = soft_priors_from_config(
        {
            "SoftPriors": {
                "priors": [
                    {
                        "type": "vs_curvature",
                        "depth_range": [0.0, 4.0],
                        "dz": 0.1,
                        "sigma": 0.3,
                    }
                ]
            }
        }
    )

    assert len(suite.priors) == 1
    assert suite.priors[0].sigma == 0.3
    with pytest.raises(ValueError, match="Unknown soft prior type"):
        soft_priors_from_config({"SoftPriors": {"priors": [{"type": "unknown"}]}})
    with pytest.raises(ValueError, match="positive"):
        soft_priors_from_config(
            {"SoftPriors": {"priors": [{"type": "vs_curvature", "sigma": 0.0}]}}
        )
