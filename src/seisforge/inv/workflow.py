"""High-level assembly helpers for joint inversion workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seisforge.inv.constraints import ConstraintSuite, constraints_from_config
from seisforge.inv.io import discretization_from_config
from seisforge.inv.likelihood import DispersionObservation, RayleighHVObservation
from seisforge.inv.parameterization import ModelParameterization, parameterization_from_config
from seisforge.inv.posterior import JointInversionTarget
from seisforge.inv.prior import GeophysicalPrior
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
class JointInversionSetup:
    """Assembled objects for prior-only and posterior MCMC."""

    parameterization: ModelParameterization
    constraints: ConstraintSuite
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


def joint_inversion_setup_from_config(config: dict) -> JointInversionSetup:
    """Assemble a joint inversion workflow from a YAML-like dictionary."""

    parameterization = parameterization_from_config(config)
    discretization = discretization_from_config(config.get("Discretization"))
    constraints = constraints_from_config(config)
    prior = GeophysicalPrior(
        parameterization=parameterization,
        constraints=constraints,
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
    return JointInversionSetup(
        parameterization=parameterization,
        constraints=constraints,
        prior=prior,
        target=target,
        sampler=sampler,
    )


def sampler_settings_from_config(
    config: dict,
    parameterization: ModelParameterization,
) -> SamplerSettings:
    sampler_cfg = config.get("Sampler", config.get("MCMC", {}))
    proposal_sigma = sampler_cfg.get("proposal_sigma")
    if proposal_sigma is None or proposal_sigma == "from_parameters":
        proposal_sigma = parameterization.proposal_sigmas()
    bounds = sampler_cfg.get("bounds")
    if bounds is None and sampler_cfg.get("use_parameter_bounds", True):
        bounds = parameterization.bounds()

    metropolis = MetropolisConfig(
        n_steps=int(sampler_cfg.get("n_steps", sampler_cfg.get("draws", 1000))),
        proposal_sigma=proposal_sigma,
        burn_in=int(sampler_cfg.get("burn_in", sampler_cfg.get("nburn", 0))),
        thin=int(sampler_cfg.get("thin", 1)),
        bounds=bounds,
        global_jump_interval=sampler_cfg.get("global_jump_interval"),
    )
    initial_thetas = sampler_cfg.get("initial_thetas", sampler_cfg.get("initial_theta"))
    return SamplerSettings(
        config=metropolis,
        n_chains=int(sampler_cfg.get("n_chains", sampler_cfg.get("chains", 1))),
        executor=sampler_cfg.get("executor", "serial"),
        max_workers=sampler_cfg.get("max_workers"),
        seed=sampler_cfg.get("seed"),
        initial_strategy=sampler_cfg.get("initial_strategy", "theta0"),
        initial_thetas=initial_thetas,
        max_initial_attempts=int(sampler_cfg.get("max_initial_attempts", 10000)),
    )


def observations_from_config(
    config: dict | None,
) -> tuple[DispersionObservation | None, RayleighHVObservation | None]:
    config = config or {}
    dispersion_cfg = config.get("dispersion")
    hv_cfg = config.get("hv", config.get("rayleigh_hv"))
    dispersion = (
        dispersion_observation_from_config(dispersion_cfg)
        if dispersion_cfg is not None
        else None
    )
    hv = rayleigh_hv_observation_from_config(hv_cfg) if hv_cfg is not None else None
    return dispersion, hv


def dispersion_observation_from_config(config: dict) -> DispersionObservation:
    return DispersionObservation(
        periods=config["periods"],
        velocity=config.get("velocity", config.get("values")),
        sigma=config["sigma"],
        mode=config.get("mode", 0),
        wave=config.get("wave", "rayleigh"),
        kind=config.get("kind", "phase"),
        algorithm=config.get("algorithm", "dunkin"),
        dc=config.get("dc", 0.005),
        dt=config.get("dt", 0.025),
    )


def rayleigh_hv_observation_from_config(config: dict) -> RayleighHVObservation:
    return RayleighHVObservation(
        periods=config["periods"],
        hv=config.get("hv", config.get("values")),
        sigma=config["sigma"],
        mode=config.get("mode", 0),
        wave=config.get("wave", "rayleigh"),
        algorithm=config.get("algorithm", "dunkin"),
        dc=config.get("dc", 0.005),
    )


def likelihood_weights_from_config(config: dict) -> tuple[float, float]:
    likelihood_cfg = config.get("Likelihood", {})
    if likelihood_cfg:
        return (
            float(likelihood_cfg.get("dispersion_weight", 1.0)),
            float(likelihood_cfg.get("hv_weight", 1.0)),
        )

    joint_cfg = config.get("Joint_Inversion", {})
    if "weight" in joint_cfg:
        weights = joint_cfg["weight"]
        if len(weights) != 2:
            raise ValueError("Joint_Inversion.weight must contain two values.")
        return float(weights[0]), float(weights[1])
    return 1.0, 1.0


def _normalize_initial_thetas(initial_thetas, *, n_chains: int) -> np.ndarray:
    initial_thetas = np.asarray(initial_thetas, dtype=float)
    if initial_thetas.ndim == 1:
        return np.tile(initial_thetas, (n_chains, 1))
    if initial_thetas.ndim == 2 and len(initial_thetas) == n_chains:
        return initial_thetas
    raise ValueError("initial_thetas must be one theta vector or one vector per chain.")
