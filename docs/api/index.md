# API reference

These pages are generated from the docstrings in `src/seisforge`, so they
always match the installed code. They cover the public Python interface of
each workflow; the workflow pages explain methods and configuration.

| page | modules |
|---|---|
| [Inversion](inv.md) | `seisforge.inv`: models, interfaces, layering, priors, forward wrappers, sampler, run driver |
| [Receiver functions](rf.md) | `seisforge.rf.synthetic`, `seisforge.rf.hk_core` |
| [Ambient noise](ant.md) | AFTAN taper and SNR helpers, Eikonal/Helmholtz core formulas, run entry points |
| [HVSR](hvsr.md) | `seisforge.hvsr` diffuse-field forward modelling |

Names that start with an underscore are internal and not listed.

## Shared configuration helpers

::: seisforge.config_keys
    options:
      members:
        - check_keys
        - rename_legacy_sections
        - warn_legacy_sections
        - ConfigDeprecationWarning
