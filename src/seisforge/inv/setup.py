"""Assembled inversion setup for prior and posterior sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.config import likelihood_weights_from_config, observations_from_config
from seisforge.inv.constraints import ConstraintSuite, constraints_from_config
from seisforge.inv.io import discretization_from_config
from seisforge.inv.parameterization import ModelParameterization, parameterization_from_config
from seisforge.inv.posterior import JointInversionTarget
from seisforge.inv.prior import GeophysicalPrior
from seisforge.inv.soft_priors import SoftPriorSuite, soft_priors_from_config
from seisforge.inv.samplers import (
    ExecutorKind,
    MCMCResult,
    MetropolisConfig,
    ProgressCallback,
    draw_valid_initial_thetas,
    run_metropolis,
)


@dataclass(frozen=True)
class SamplerSettings:
    """Run settings around a `MetropolisConfig`."""

    config: MetropolisConfig
    n_chains: int = 1
    executor: ExecutorKind = "serial"
    max_workers: int | None = None
    seed: int | None = None
    initial_strategy: str = "theta0"
    initial_thetas: np.ndarray | None = None
    max_initial_attempts: int = 10000

    def __post_init__(self):
        if self.n_chains <= 0:
            raise ValueError("n_chains must be positive.")
        if self.initial_strategy not in {"theta0", "prior_uniform"}:
            raise ValueError("initial_strategy must be 'theta0' or 'prior_uniform'.")
        if self.max_initial_attempts <= 0:
            raise ValueError("max_initial_attempts must be positive.")
        if self.initial_thetas is not None:
            object.__setattr__(
                self,
                "initial_thetas",
                np.asarray(self.initial_thetas, dtype=float),
            )


@dataclass(frozen=True)
class InversionSetup:
    """Scientific objects needed to sample one inversion target."""

    parameterization: ModelParameterization
    constraints: ConstraintSuite
    soft_priors: SoftPriorSuite
    prior: GeophysicalPrior
    sampler: SamplerSettings
    target: JointInversionTarget | None = None

    def initial_thetas(self, *, target: str = "prior") -> np.ndarray:
        log_target = self._log_target(target)
        if self.sampler.initial_thetas is not None:
            return _normalize_initial_thetas(
                self.sampler.initial_thetas,
                n_chains=self.sampler.n_chains,
            )
        if self.sampler.initial_strategy == "theta0":
            return np.tile(self.parameterization.theta0(), (self.sampler.n_chains, 1))
        if self.sampler.config.bounds is None:
            raise ValueError("bounds are required for prior_uniform initialization.")
        return draw_valid_initial_thetas(
            log_target,
            self.sampler.config.bounds,
            n_chains=self.sampler.n_chains,
            seed=self.sampler.seed,
            max_attempts=self.sampler.max_initial_attempts,
        )

    def run_prior(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_every: int | None = None,
    ) -> MCMCResult:
        return self._run(
            "prior",
            progress_callback=progress_callback,
            progress_every=progress_every,
        )

    def run_posterior(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_every: int | None = None,
    ) -> MCMCResult:
        return self._run(
            "posterior",
            progress_callback=progress_callback,
            progress_every=progress_every,
        )

    def depth_grid(self) -> np.ndarray:
        discretization = self.prior.discretization
        dz = 0.5 if discretization is None else discretization.dz
        zmax = None if discretization is None else discretization.zmax
        if zmax is None:
            zmax = self.parameterization.vector_to_model(
                self.parameterization.theta0()
            ).zmax
        return np.arange(0.0, float(zmax) + 0.5 * dz, dz)

    def _run(
        self,
        target: str,
        *,
        progress_callback: ProgressCallback | None = None,
        progress_every: int | None = None,
    ) -> MCMCResult:
        return run_metropolis(
            self._log_target(target),
            self.initial_thetas(target=target),
            self.sampler.config,
            seeds=self.sampler.seed,
            executor=self.sampler.executor,
            max_workers=self.sampler.max_workers,
            progress_callback=progress_callback,
            progress_every=progress_every,
        )

    def _log_target(self, target: str):
        if target == "prior":
            return self.prior.log_prior
        if target == "posterior":
            if self.target is None:
                raise ValueError("Posterior target is not configured.")
            return self.target.log_posterior
        raise ValueError("target must be 'prior' or 'posterior'.")


def build_inversion_setup(config: dict) -> InversionSetup:
    """Assemble an inversion setup from a validated run configuration."""

    parameterization = parameterization_from_config(config)
    discretization = discretization_from_config(config.get("Discretization"))
    constraints = constraints_from_config(config)
    soft_priors = soft_priors_from_config(config)
    prior = GeophysicalPrior(
        parameterization=parameterization,
        constraints=constraints,
        soft_priors=soft_priors,
        discretization=discretization,
    )
    sampler = sampler_settings_from_config(config, parameterization)
    dispersion, hv = observations_from_config(config.get("Observations"))
    target = None
    if dispersion is not None or hv is not None:
        dispersion_weight, hv_weight = likelihood_weights_from_config(config)
        target = JointInversionTarget(
            prior=prior,
            dispersion=dispersion,
            hv=hv,
            dispersion_weight=dispersion_weight,
            hv_weight=hv_weight,
        )
    return InversionSetup(
        parameterization=parameterization,
        constraints=constraints,
        soft_priors=soft_priors,
        prior=prior,
        target=target,
        sampler=sampler,
    )


def sampler_settings_from_config(
    config: dict,
    parameterization: ModelParameterization,
) -> SamplerSettings:
    sampler_cfg = config.get("Sampler", {})
    proposal_sigma = sampler_cfg.get("proposal_sigma")
    if proposal_sigma is None or proposal_sigma == "from_parameters":
        proposal_sigma = parameterization.proposal_sigmas()
    bounds = sampler_cfg.get("bounds")
    if bounds is None and sampler_cfg.get("use_parameter_bounds", True):
        bounds = parameterization.bounds()

    metropolis = MetropolisConfig(
        n_steps=int(sampler_cfg.get("n_steps", 1000)),
        proposal_sigma=proposal_sigma,
        burn_in=int(sampler_cfg.get("burn_in", 0)),
        thin=int(sampler_cfg.get("thin", 1)),
        bounds=bounds,
        global_jump_interval=sampler_cfg.get("global_jump_interval"),
    )
    initial_thetas = sampler_cfg.get("initial_thetas")
    return SamplerSettings(
        config=metropolis,
        n_chains=int(sampler_cfg.get("n_chains", 1)),
        executor=sampler_cfg.get("executor", "serial"),
        max_workers=sampler_cfg.get("max_workers"),
        seed=sampler_cfg.get("seed"),
        initial_strategy=sampler_cfg.get("initial_strategy", "theta0"),
        initial_thetas=initial_thetas,
        max_initial_attempts=int(sampler_cfg.get("max_initial_attempts", 10000)),
    )


def _normalize_initial_thetas(initial_thetas, *, n_chains: int) -> np.ndarray:
    initial_thetas = np.asarray(initial_thetas, dtype=float)
    if initial_thetas.ndim == 1:
        return np.tile(initial_thetas, (n_chains, 1))
    if initial_thetas.ndim == 2 and len(initial_thetas) == n_chains:
        return initial_thetas
    raise ValueError("initial_thetas must be one theta vector or one vector per chain.")
