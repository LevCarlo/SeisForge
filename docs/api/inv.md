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

::: seisforge.inv.config_schema
    options:
      members:
        - prepare_inv_config
        - prepare_obs_config
        - validate_inv_config
        - validate_obs_config
        - INV_SECTION_RENAMES

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

## Posterior target

::: seisforge.inv.posterior
    options:
      members:
        - JointInversionTarget
        - JointPosteriorResult
        - CountingTarget
        - ForwardFailureCounter
        - ForwardFailureReport
        - ForwardFailureSummary
        - forward_failure_summary
        - FORWARD_FAILURE_WARNING_RATE

## Sampler

::: seisforge.inv.samplers
    options:
      members:
        - MetropolisConfig
        - AdaptationConfig
        - update_running_moments
        - running_covariance
        - adaptive_proposal_covariance
        - run_metropolis
        - run_metropolis_chain
        - run_tempered_chain
        - geometric_temperatures
        - draw_valid_initial_thetas
        - ChainLocalTarget
        - ChainResult
        - MCMCResult

## Convergence diagnostics

::: seisforge.inv.io
    options:
      members:
        - mcmc_result_to_xarray
        - adaptation_attrs

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
