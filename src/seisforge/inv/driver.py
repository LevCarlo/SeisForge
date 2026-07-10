"""Run-level driver for single-station inversion.

The driver coordinates configuration loading, sampling stages, output writing,
logging, and provenance. Scientific equations and target definitions live in
the lower-level inversion modules.
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from seisforge.inv.config import InversionInput, InversionKind, load_inversion_input
from seisforge.inv.ensemble import extract_vs_ensemble_from_result
from seisforge.inv.io import (
    dump_yaml,
    mcmc_result_to_xarray,
    prediction_ensemble_to_xarray,
    vs_ensemble_to_xarray,
)
from seisforge.inv.predictive import extract_prediction_ensemble_from_result
from seisforge.inv.prior import evaluate_prior_samples
from seisforge.inv.setup import InversionSetup, build_inversion_setup


@dataclass(frozen=True)
class InversionRunResult:
    """Output paths and high-level run diagnostics."""

    output_dir: Path
    prefix: str
    posterior_mcmc: Path | None = None
    posterior_vs: Path | None = None
    posterior_predictive: Path | None = None
    prior_mcmc: Path | None = None
    prior_vs: Path | None = None
    log_file: Path | None = None


def run_inversion(
    *,
    kind: InversionKind,
    inv_file: str | Path,
    output: str | Path,
    obs_file: str | Path | None = None,
    prior_only: bool = False,
    with_prior: bool = False,
    overwrite: bool = False,
    progress: bool | None = None,
    progress_every: int | None = None,
    log_level: str = "INFO",
) -> InversionRunResult:
    """Run one inversion command and write structured outputs."""

    if prior_only and with_prior:
        raise ValueError("--prior-only and --with-prior cannot both be used.")
    if not prior_only and obs_file is None:
        raise ValueError("--obs is required for posterior inversion runs.")

    inversion_input = load_inversion_input(
        kind=kind,
        inv_file=inv_file,
        obs_file=obs_file,
        require_observations=not prior_only,
    )
    setup = build_inversion_setup(inversion_input.run_config)

    output_dir = _prepare_output_dir(output, overwrite=overwrite)
    prefix = _prefix(kind)
    log_file = output_dir / f"{prefix}.log"
    logger = _configure_logger(log_file, level=log_level)
    attrs = _run_attrs(kind=kind, inversion_input=inversion_input, setup=setup)

    _write_resolved_config(
        inversion_input.run_config,
        output_dir / "resolved_config.yaml",
        attrs=attrs,
    )
    _write_input_snapshots(output_dir, inversion_input=inversion_input, attrs=attrs)
    _log_run_header(logger, inversion_input=inversion_input, setup=setup, output_dir=output_dir)

    show_progress = _should_show_progress(progress)
    if show_progress and setup.sampler.executor == "process":
        logger.warning("progress display is disabled for process executor")
        show_progress = False
    progress_every = _progress_every(progress_every, setup)
    callback = _progress_callback(logger, setup) if show_progress else None

    result = InversionRunResult(output_dir=output_dir, prefix=prefix, log_file=log_file)
    if prior_only or with_prior:
        logger.info("running hard-prior MCMC")
        prior_result = setup.run_prior(
            progress_callback=callback,
            progress_every=progress_every if show_progress else None,
        )
        result = _write_prior_outputs(
            result,
            setup=setup,
            prior_result=prior_result,
            attrs=attrs | {"run_stage": "prior"},
            logger=logger,
        )
        if prior_only:
            return result

    if setup.target is None:
        raise ValueError("Posterior inversion requires observation data.")
    logger.info("running posterior MCMC")
    posterior_result = setup.run_posterior(
        progress_callback=callback,
        progress_every=progress_every if show_progress else None,
    )
    return _write_posterior_outputs(
        result,
        setup=setup,
        posterior_result=posterior_result,
        attrs=attrs | {"run_stage": "posterior"},
        logger=logger,
    )


def _write_prior_outputs(
    result: InversionRunResult,
    *,
    setup: InversionSetup,
    prior_result,
    attrs: dict,
    logger: logging.Logger,
) -> InversionRunResult:
    prefix = result.prefix
    mcmc_file = result.output_dir / f"{prefix}_prior_mcmc.nc"
    vs_file = result.output_dir / f"{prefix}_prior_vs.nc"
    mcmc_result_to_xarray(
        prior_result,
        parameter_names=setup.parameterization.names,
        prior_diagnostics=evaluate_prior_samples(prior_result.samples, setup.prior),
        attrs=attrs,
    ).to_netcdf(mcmc_file)
    vs = extract_vs_ensemble_from_result(
        prior_result,
        parameterization=setup.parameterization,
        z=setup.depth_grid(),
    )
    vs_ensemble_to_xarray(vs, attrs=attrs).to_netcdf(vs_file)
    logger.info("prior acceptance rates: %s", prior_result.acceptance_rates)
    logger.info("prior mean acceptance: %.6g", prior_result.mean_acceptance_rate)
    logger.info("wrote %s", mcmc_file)
    logger.info("wrote %s", vs_file)
    return InversionRunResult(
        output_dir=result.output_dir,
        prefix=result.prefix,
        posterior_mcmc=result.posterior_mcmc,
        posterior_vs=result.posterior_vs,
        posterior_predictive=result.posterior_predictive,
        prior_mcmc=mcmc_file,
        prior_vs=vs_file,
        log_file=result.log_file,
    )


def _write_posterior_outputs(
    result: InversionRunResult,
    *,
    setup: InversionSetup,
    posterior_result,
    attrs: dict,
    logger: logging.Logger,
) -> InversionRunResult:
    prefix = result.prefix
    mcmc_file = result.output_dir / f"{prefix}_mcmc.nc"
    vs_file = result.output_dir / f"{prefix}_vs.nc"
    predictive_file = result.output_dir / f"{prefix}_predictive.nc"
    prior_diagnostics = evaluate_prior_samples(posterior_result.samples, setup.prior)
    mcmc_result_to_xarray(
        posterior_result,
        parameter_names=setup.parameterization.names,
        prior_diagnostics=prior_diagnostics,
        log_likelihood=posterior_result.log_prob - prior_diagnostics.log_prior,
        attrs=attrs,
    ).to_netcdf(mcmc_file)
    vs = extract_vs_ensemble_from_result(
        posterior_result,
        parameterization=setup.parameterization,
        z=setup.depth_grid(),
    )
    vs_ensemble_to_xarray(vs, attrs=attrs).to_netcdf(vs_file)
    prediction = extract_prediction_ensemble_from_result(
        posterior_result,
        target=setup.target,
    )
    prediction_ensemble_to_xarray(prediction, attrs=attrs).to_netcdf(predictive_file)
    best_theta = vs.best_theta
    logger.info("posterior acceptance rates: %s", posterior_result.acceptance_rates)
    logger.info("posterior mean acceptance: %.6g", posterior_result.mean_acceptance_rate)
    if best_theta is not None:
        logger.info("best theta: %s", best_theta)
    logger.info("wrote %s", mcmc_file)
    logger.info("wrote %s", vs_file)
    logger.info("wrote %s", predictive_file)
    return InversionRunResult(
        output_dir=result.output_dir,
        prefix=result.prefix,
        posterior_mcmc=mcmc_file,
        posterior_vs=vs_file,
        posterior_predictive=predictive_file,
        prior_mcmc=result.prior_mcmc,
        prior_vs=result.prior_vs,
        log_file=result.log_file,
    )


def _prepare_output_dir(output: str | Path, *, overwrite: bool) -> Path:
    output_dir = Path(output)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_resolved_config(config: dict, path: Path, *, attrs: dict) -> None:
    resolved = copy.deepcopy(config)
    resolved.setdefault("Run", {}).update(attrs)
    observations = resolved.get("Observations")
    if isinstance(observations, dict):
        for item in observations.values():
            if isinstance(item, dict):
                for key in ("periods", "velocity", "hv", "values", "sigma"):
                    item.pop(key, None)
    dump_yaml(resolved, path)


def _write_input_snapshots(
    output_dir: Path,
    *,
    inversion_input: InversionInput,
    attrs: dict,
) -> None:
    inv_snapshot = copy.deepcopy(inversion_input.inv_config)
    inv_snapshot.setdefault("Run", {}).update(attrs)
    dump_yaml(inv_snapshot, output_dir / "resolved_inv.yaml")
    if inversion_input.obs_config is not None:
        obs_snapshot = copy.deepcopy(inversion_input.obs_config)
        obs_snapshot.setdefault("Run", {}).update(attrs)
        dump_yaml(obs_snapshot, output_dir / "resolved_obs.yaml")


def _configure_logger(path: Path, *, level: str) -> logging.Logger:
    logger = logging.getLogger(f"seisforge.inv.{path}")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _progress_callback(logger: logging.Logger, setup: InversionSetup):
    n_chains = setup.sampler.n_chains
    n_steps = setup.sampler.config.n_steps

    def callback(chain_id: int, step: int, acceptance: float, log_prob: float) -> None:
        message = (
            f"progress chain={chain_id + 1}/{n_chains} "
            f"step={step}/{n_steps} acceptance={acceptance:.3f} "
            f"log_prob={log_prob:.6g}"
        )
        print(message, file=sys.stderr)
        logger.info(
            "progress chain=%s/%s step=%s/%s acceptance=%.3f log_prob=%.6g",
            chain_id + 1,
            n_chains,
            step,
            n_steps,
            acceptance,
            log_prob,
        )

    return callback


def _should_show_progress(progress: bool | None) -> bool:
    if progress is not None:
        return progress
    return sys.stderr.isatty()


def _progress_every(
    progress_every: int | None,
    setup: InversionSetup,
) -> int:
    if progress_every is not None:
        if progress_every <= 0:
            raise ValueError("progress_every must be positive.")
        return progress_every
    return max(1, setup.sampler.config.n_steps // 20)


def _run_attrs(
    *,
    kind: InversionKind,
    inversion_input: InversionInput,
    setup: InversionSetup,
) -> dict:
    lower, upper = setup.parameterization.bounds()
    attrs = {
        "inversion_kind": kind,
        "inv_file": str(inversion_input.inv_file),
        "obs_file": None if inversion_input.obs_file is None else str(inversion_input.obs_file),
        "dispersion_file": (
            None
            if inversion_input.dispersion_file is None
            else str(inversion_input.dispersion_file)
        ),
        "hv_file": None if inversion_input.hv_file is None else str(inversion_input.hv_file),
        "parameter_names": ", ".join(setup.parameterization.names),
        "parameter_lower_bounds": ", ".join(f"{value:.8g}" for value in lower),
        "parameter_upper_bounds": ", ".join(f"{value:.8g}" for value in upper),
        "parameter_proposal_sigma": ", ".join(
            f"{value:.8g}" for value in setup.parameterization.proposal_sigmas()
        ),
        "constraints": _constraint_names(setup),
        "soft_priors": _soft_prior_names(setup),
        "sampler_n_chains": int(setup.sampler.n_chains),
        "sampler_n_steps": int(setup.sampler.config.n_steps),
        "sampler_burn_in": int(setup.sampler.config.burn_in),
        "sampler_thin": int(setup.sampler.config.thin),
        "sampler_executor": setup.sampler.executor,
        "sampler_initial_strategy": setup.sampler.initial_strategy,
        "sampler_global_jump_interval": setup.sampler.config.global_jump_interval,
        "sampler_seed": setup.sampler.seed,
    }
    if setup.target is not None:
        attrs["dispersion_weight"] = float(setup.target.dispersion_weight)
        attrs["hv_weight"] = float(setup.target.hv_weight)
    if setup.target is not None and setup.target.dispersion is not None:
        dispersion = setup.target.dispersion
        attrs.update(
            {
                "dispersion_wave": dispersion.wave,
                "dispersion_kind": dispersion.kind,
                "dispersion_mode": int(dispersion.mode),
                "dispersion_algorithm": dispersion.algorithm,
                "dispersion_dc": float(dispersion.dc),
                "dispersion_dt": float(dispersion.dt),
            }
        )
    if setup.target is not None and setup.target.hv is not None:
        hv = setup.target.hv
        attrs.update(
            {
                "hv_wave": hv.wave,
                "hv_mode": int(hv.mode),
                "hv_algorithm": hv.algorithm,
                "hv_dc": float(hv.dc),
            }
        )
    return attrs


def _log_run_header(
    logger: logging.Logger,
    *,
    inversion_input: InversionInput,
    setup: InversionSetup,
    output_dir: Path,
) -> None:
    logger.info("inversion kind: %s", inversion_input.kind)
    logger.info("inv: %s", inversion_input.inv_file)
    if inversion_input.obs_file is not None:
        logger.info("obs: %s", inversion_input.obs_file)
    if inversion_input.dispersion_file is not None:
        logger.info("dispersion file: %s", inversion_input.dispersion_file)
    if inversion_input.hv_file is not None:
        logger.info("hv file: %s", inversion_input.hv_file)
    logger.info("output: %s", output_dir)
    logger.info("parameters: %s", ", ".join(setup.parameterization.names))
    logger.info("constraints: %s", _constraint_names(setup))
    logger.info("soft priors: %s", _soft_prior_names(setup))
    logger.info(
        "sampler: n_chains=%s n_steps=%s burn_in=%s thin=%s executor=%s",
        setup.sampler.n_chains,
        setup.sampler.config.n_steps,
        setup.sampler.config.burn_in,
        setup.sampler.config.thin,
        setup.sampler.executor,
    )


def _constraint_names(setup: InversionSetup) -> str:
    return ", ".join(
        getattr(constraint, "name", constraint.__class__.__name__)
        for constraint in setup.constraints.constraints
    )


def _soft_prior_names(setup: InversionSetup) -> str:
    return ", ".join(
        getattr(prior, "name", prior.__class__.__name__)
        for prior in setup.soft_priors.priors
    )


def _prefix(kind: InversionKind) -> str:
    return kind
