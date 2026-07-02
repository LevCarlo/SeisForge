import numpy as np

from seisforge.inv.constraints import (
    BoundaryNonDecreasingVsConstraint,
    ConstraintSuite,
    MonotonicVsConstraint,
    VpVsRangeConstraint,
    VsRangeConstraint,
    constraints_from_config,
)
from seisforge.inv.parameterization import (
    ModelParameterization,
    ParameterSpec,
    UniformPrior,
    parameterization_from_config,
)
from seisforge.inv.prior import GeophysicalPrior


def _parameterization():
    return ModelParameterization(
        parameters=(
            ParameterSpec("sed.vs_top", 1.0, UniformPrior(0.5, 2.0), proposal_sigma=0.05),
            ParameterSpec("sed.vs_bottom", 2.0, UniformPrior(1.0, 3.0), proposal_sigma=0.05),
            ParameterSpec("crust.c0", 2.2, UniformPrior(1.8, 3.0), proposal_sigma=0.04),
            ParameterSpec("crust.c1", 2.6, UniformPrior(2.0, 3.4), proposal_sigma=0.04),
            ParameterSpec("crust.c2", 3.0, UniformPrior(2.4, 3.8), proposal_sigma=0.04),
            ParameterSpec("crust.c3", 3.4, UniformPrior(2.8, 4.2), proposal_sigma=0.04),
        ),
        model_config={
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 2.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "sed.vs_top"},
                        "bottom": {"parameter": "sed.vs_bottom"},
                    },
                },
                {
                    "top_km": 2.0,
                    "bottom_km": 10.0,
                    "profile": {
                        "type": "bspline",
                        "coefficients": [
                            {"parameter": "crust.c0"},
                            {"parameter": "crust.c1"},
                            {"parameter": "crust.c2"},
                            {"parameter": "crust.c3"},
                        ],
                        "degree": 3,
                        "knot_spacing": "uniform",
                    },
                },
            ],
        },
    )


def test_parameterization_maps_theta_to_segmented_model():
    parameterization = _parameterization()
    model = parameterization.vector_to_model(parameterization.theta0())

    np.testing.assert_allclose(model.evaluate(np.array([0.0, 2.0, 10.0])), [1.0, 2.2, 3.4])
    assert parameterization.names == (
        "sed.vs_top",
        "sed.vs_bottom",
        "crust.c0",
        "crust.c1",
        "crust.c2",
        "crust.c3",
    )


def test_parameterization_separates_prior_bounds_from_proposal_sigma():
    parameterization = _parameterization()
    lower, upper = parameterization.bounds()

    np.testing.assert_allclose(lower[:2], [0.5, 1.0])
    np.testing.assert_allclose(upper[:2], [2.0, 3.0])
    np.testing.assert_allclose(parameterization.proposal_sigmas()[:2], [0.05, 0.05])
    assert parameterization.log_parameter_prior(parameterization.theta0()) == 0.0

    outside = parameterization.theta0()
    outside[0] = 0.4
    assert parameterization.log_parameter_prior(outside) == -np.inf


def test_parameterization_from_config_accepts_yaml_like_mapping():
    config = {
        "Inversion": {
            "parameters": [
                {
                    "name": "vs0",
                    "initial": 2.0,
                    "prior": {"type": "uniform", "bounds": [1.0, 3.0]},
                    "proposal": {"sigma": 0.1},
                },
                {
                    "name": "vs1",
                    "initial": 3.0,
                    "prior": {"type": "uniform", "bounds": [2.0, 4.0]},
                    "proposal": {"sigma": 0.1},
                },
            ],
        },
        "Model": {
            "segments": [
                {
                    "top_km": 0.0,
                    "bottom_km": 5.0,
                    "profile": {
                        "type": "gradient",
                        "top": {"parameter": "vs0"},
                        "bottom": {"parameter": "vs1"},
                    },
                }
            ]
        },
    }

    parameterization = parameterization_from_config(config)
    model = parameterization.vector_to_model(parameterization.theta0())

    np.testing.assert_allclose(model.evaluate(np.array([0.0, 5.0])), [2.0, 3.0])


def test_constraint_suite_reports_physical_prior_support():
    parameterization = _parameterization()
    model = parameterization.vector_to_model(parameterization.theta0())
    suite = ConstraintSuite(
        constraints=(
            BoundaryNonDecreasingVsConstraint(),
            MonotonicVsConstraint(depth_range=(0.0, 10.0), dz=0.5),
            VsRangeConstraint(bounds=(0.5, 4.5), dz=0.5),
            VpVsRangeConstraint(bounds=(1.4, 2.7)),
        )
    )

    evaluation = suite.evaluate(model)

    assert evaluation.valid
    assert suite.log_physical_prior(model) == 0.0


def test_boundary_constraint_rejects_velocity_drop_between_segments():
    parameterization = _parameterization()
    theta = parameterization.theta0()
    theta[1] = 2.8
    theta[2] = 2.0
    model = parameterization.vector_to_model(theta)
    suite = ConstraintSuite(constraints=(BoundaryNonDecreasingVsConstraint(),))

    evaluation = suite.evaluate(model)

    assert not evaluation.valid
    assert evaluation.failed == ("boundary_non_decreasing_vs",)
    assert suite.log_physical_prior(model) == -np.inf


def test_constraints_from_config_builds_yaml_declared_rules():
    suite = constraints_from_config(
        {
            "constraints": [
                {"type": "positive_velocity"},
                {"type": "vp_gt_vs", "margin": 0.01},
                {"type": "vp_vs_range", "bounds": [1.4, 2.7]},
                {"type": "boundary_non_decreasing_vs"},
                {"type": "vs_range", "bounds": [0.5, 4.5], "depth_range": [0.0, 10.0]},
            ]
        }
    )

    assert len(suite.constraints) == 5


def test_geophysical_prior_is_prior_only_mcmc_target():
    parameterization = _parameterization()
    prior = GeophysicalPrior(
        parameterization=parameterization,
        constraints=ConstraintSuite(constraints=(BoundaryNonDecreasingVsConstraint(),)),
    )

    valid = prior.evaluate(parameterization.theta0())

    assert valid.success
    assert valid.log_prior == 0.0
    assert valid.model is not None
    assert valid.layered_model is not None

    invalid = parameterization.theta0()
    invalid[0] = 0.1
    result = prior.evaluate(invalid)

    assert not result.success
    assert result.log_prior == -np.inf
    assert result.error == "parameter_prior"
