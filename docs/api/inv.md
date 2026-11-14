# Inversion API

## Models

::: seisforge.inv.model
    options:
      members:
        - ParameterizedVsModel
        - VsSegment
        - Interface
        - ConstantVs
        - GradientVs
        - BSplineVs
        - ScalingConfig
        - DiscretizationConfig
        - LayeredModel

## Transitions

::: seisforge.inv.transitions

## Layering

::: seisforge.inv.layering

## Parameterization and priors

::: seisforge.inv.parameterization
    options:
      members:
        - ModelParameterization
        - ParameterSpec
        - UniformPrior
        - NormalPrior
        - HalfNormalPrior
        - parameterization_from_config
        - prior_from_config

::: seisforge.inv.model_config
    options:
      members:
        - parameterized_model_from_config
        - parameterized_model_to_config
        - validate_model_layout

## Forward models

::: seisforge.inv.forward

## Likelihood

::: seisforge.inv.likelihood
    options:
      members:
        - DispersionObservation
        - RayleighHVObservation
        - evaluate_joint_likelihood
        - gaussian_observable_fit

## Sampler

::: seisforge.inv.samplers
    options:
      members:
        - MetropolisConfig
        - run_metropolis
        - run_metropolis_chain
        - draw_valid_initial_thetas
        - ChainResult
        - MCMCResult

## Convergence diagnostics

::: seisforge.inv.diagnostics

## Running an inversion

::: seisforge.inv.setup
    options:
      members:
        - build_inversion_setup
        - InversionSetup

::: seisforge.inv.driver
    options:
      members:
        - run_inversion
        - InversionRunResult
