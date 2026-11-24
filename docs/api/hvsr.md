# HVSR API

::: seisforge.hvsr.forward

::: seisforge.hvsr.models
    options:
      members:
        - ElasticLayerModel
        - DiffuseFieldSettings
        - DiffuseFieldPrediction
        - HorizontalDefinition
        - ForwardStatus
        - FrequencyDiagnostic
        - ContributionIssue
        - GreenFunctionContributions

::: seisforge.hvsr.core
    options:
      members:
        - combine_green_function_contributions
        - assemble_diffuse_field_prediction

::: seisforge.hvsr.native
    options:
      members:
        - NativeDiffuseFieldSolver

::: seisforge.hvsr.fortran
    options:
      members:
        - FortranDiffuseFieldSolver
        - fortran_extension_available

::: seisforge.hvsr.reference
    options:
      members:
        - HVDFAReferenceSolver
        - predict_with_hvdfa
