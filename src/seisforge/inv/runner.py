"""Command-line runners for single-station inversion workflows."""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from seisforge.inv.ensemble import extract_vs_ensemble_from_result
from seisforge.inv.io import (
    dump_yaml,
    load_yaml,
    mcmc_result_to_xarray,
    prediction_ensemble_to_xarray,
    read_dispersion_dat,
    read_hv_dat,
    vs_ensemble_to_xarray,
)
from seisforge.inv.predictive import extract_prediction_ensemble_from_result
from seisforge.inv.workflow import JointInversionSetup, joint_inversion_setup_from_config


InversionKind = Literal["disp", "hv", "disp_hv"]


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


def run_dispersion_inversion(
    *,
    config_file: str | Path,
    dispersion_file: str | Path | None,
    output: str | Path,
    prior_only: bool = False,
    with_prior: bool = False,
    overwrite: bool = False,
    progress: bool | None = None,
    progress_every: int | None = None,
    log_level: str = "INFO",
) -> InversionRunResult:
    return run_inversion(
        kind="disp",
        config_file=config_file,
        dispersion_file=dispersion_file,
        hv_file=None,
        output=output,
        prior_only=prior_only,
        with_prior=with_prior,
        overwrite=overwrite,
        progress=progress,
        progress_every=progress_every,
        log_level=log_level,
    )


def run_hv_inversion(
    *,
    config_file: str | Path,
    hv_file: str | Path | None,
    output: str | Path,
    prior_only: bool = False,
    with_prior: bool = False,
    overwrite: bool = False,
    progress: bool | None = None,
    progress_every: int | None = None,
    log_level: str = "INFO",
) -> InversionRunResult:
    return run_inversion(
        kind="hv",
        config_file=config_file,
        dispersion_file=None,
        hv_file=hv_file,
        output=output,
        prior_only=prior_only,
        with_prior=with_prior,
        overwrite=overwrite,
        progress=progress,
        progress_every=progress_every,
        log_level=log_level,
    )


def run_dispersion_hv_inversion(
    *,
    config_file: str | Path,
    dispersion_file: str | Path | None,
    hv_file: str | Path | None,
    output: str | Path,
    prior_only: bool = False,
    with_prior: bool = False,
    overwrite: bool = False,
    progress: bool | None = None,
    progress_every: int | None = None,
    log_level: str = "INFO",
) -> InversionRunResult:
    return run_inversion(
        kind="disp_hv",
        config_file=config_file,
        dispersion_file=dispersion_file,
        hv_file=hv_file,
        output=output,
        prior_only=prior_only,
        with_prior=with_prior,
        overwrite=overwrite,
        progress=progress,
        progress_every=progress_every,
        log_level=log_level,
    )


def run_inversion(
    *,
    kind: InversionKind,
    config_file: str | Path,
    dispersion_file: str | Path | None,
    hv_file: str | Path | None,
    output: str | Path,
    prior_only: bool = False,
    with_prior: bool = False,
    overwrite: bool = False,
    progress: bool | None = None,
    progress_every: int | None = None,
    log_level: str = "INFO",
) -> InversionRunResult:
    """Run one CLI inversion workflow and write NetCDF/log outputs."""

    if prior_only and with_prior:
        raise ValueError("--prior-only and --with-prior cannot both be used.")
    output_dir = _prepare_output_dir(output, overwrite=overwrite)
    prefix = _prefix(kind)
    log_file = output_dir / f"{prefix}.log"
    logger = _configure_logger(log_file, level=log_level)

    config_file = Path(config_file)
    config = load_yaml(config_file)
    run_config = _config_with_observations(
        config,
        kind=kind,
        dispersion_file=dispersion_file,
        hv_file=hv_file,
        require_data=not prior_only,
    )
    setup = joint_inversion_setup_from_config(run_config)
    attrs = _run_attrs(
        kind=kind,
        config_file=config_file,
        dispersion_file=dispersion_file,
        hv_file=hv_file,
        setup=setup,
    )
    _write_resolved_config(
        config,
        output_dir / "resolved_config.yaml",
        attrs=attrs,
    )

    logger.info("inversion kind: %s", kind)
    logger.info("config: %s", config_file)
    if dispersion_file is not None:
        logger.info("dispersion file: %s", dispersion_file)
    if hv_file is not None:
        logger.info("hv file: %s", hv_file)
    logger.info("output: %s", output_dir)
    logger.info("parameters: %s", ", ".join(setup.parameterization.names))
    logger.info("constraints: %s", _constraint_names(setup))
    logger.info(
        "sampler: n_chains=%s n_steps=%s burn_in=%s thin=%s executor=%s",
        setup.sampler.n_chains,
        setup.sampler.config.n_steps,
        setup.sampler.config.burn_in,
        setup.sampler.config.thin,
        setup.sampler.executor,
    )

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
    result = _write_posterior_outputs(
        result,
        setup=setup,
        posterior_result=posterior_result,
        attrs=attrs | {"run_stage": "posterior"},
        logger=logger,
    )
    return result


def _config_with_observations(
    config: dict,
    *,
    kind: InversionKind,
    dispersion_file: str | Path | None,
    hv_file: str | Path | None,
    require_data: bool,
) -> dict:
    run_config = copy.deepcopy(config)
    obs_meta = copy.deepcopy(run_config.get("Observations", {}))
    observations = {}
    if kind in {"disp", "disp_hv"}:
        if dispersion_file is None:
            if require_data:
                raise ValueError("--disp is required for dispersion posterior runs.")
        else:
            dispersion = obs_meta.get("dispersion", {})
            dispersion.update(read_dispersion_dat(dispersion_file))
            observations["dispersion"] = dispersion
    if kind in {"hv", "disp_hv"}:
        if hv_file is None:
            if require_data:
                raise ValueError("--hv is required for H/V posterior runs.")
        else:
            hv = obs_meta.get("hv", obs_meta.get("rayleigh_hv", {}))
            hv.update(read_hv_dat(hv_file))
            observations["hv"] = hv
    if observations:
        run_config["Observations"] = observations
    else:
        run_config.pop("Observations", None)
    return run_config


def _write_prior_outputs(
    result: InversionRunResult,
    *,
    setup: JointInversionSetup,
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
        attrs=attrs,
    ).to_netcdf(mcmc_file)
    vs = extract_vs_ensemble_from_result(
        prior_result,
        parameterization=setup.parameterization,
        z=_depth_grid(setup),
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
    setup: JointInversionSetup,
    posterior_result,
    attrs: dict,
    logger: logging.Logger,
) -> InversionRunResult:
    prefix = result.prefix
    mcmc_file = result.output_dir / f"{prefix}_mcmc.nc"
    vs_file = result.output_dir / f"{prefix}_vs.nc"
    predictive_file = result.output_dir / f"{prefix}_predictive.nc"
    mcmc_result_to_xarray(
        posterior_result,
        parameter_names=setup.parameterization.names,
        attrs=attrs,
    ).to_netcdf(mcmc_file)
    vs = extract_vs_ensemble_from_result(
        posterior_result,
        parameterization=setup.parameterization,
        z=_depth_grid(setup),
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


def _depth_grid(setup: JointInversionSetup) -> np.ndarray:
    discretization = setup.prior.discretization
    dz = 0.5 if discretization is None else discretization.dz
    zmax = None if discretization is None else discretization.zmax
    if zmax is None:
        zmax = setup.parameterization.vector_to_model(
            setup.parameterization.theta0()
        ).zmax
    return np.arange(0.0, float(zmax) + 0.5 * dz, dz)


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


def _progress_callback(logger: logging.Logger, setup: JointInversionSetup):
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
    setup: JointInversionSetup,
) -> int:
    if progress_every is not None:
        if progress_every <= 0:
            raise ValueError("progress_every must be positive.")
        return progress_every
    return max(1, setup.sampler.config.n_steps // 20)


def _run_attrs(
    *,
    kind: InversionKind,
    config_file: Path,
    dispersion_file: str | Path | None,
    hv_file: str | Path | None,
    setup: JointInversionSetup,
) -> dict:
    attrs = {
        "inversion_kind": kind,
        "config_file": str(config_file),
        "dispersion_file": None if dispersion_file is None else str(dispersion_file),
        "hv_file": None if hv_file is None else str(hv_file),
        "parameter_names": ", ".join(setup.parameterization.names),
        "constraints": _constraint_names(setup),
    }
    if setup.target is not None and setup.target.dispersion is not None:
        dispersion = setup.target.dispersion
        attrs.update(
            {
                "dispersion_wave": dispersion.wave,
                "dispersion_kind": dispersion.kind,
                "dispersion_mode": int(dispersion.mode),
            }
        )
    if setup.target is not None and setup.target.hv is not None:
        attrs["hv_wave"] = setup.target.hv.wave
        attrs["hv_mode"] = int(setup.target.hv.mode)
    return attrs


def _constraint_names(setup: JointInversionSetup) -> str:
    return ", ".join(
        getattr(constraint, "name", constraint.__class__.__name__)
        for constraint in setup.constraints.constraints
    )


def _prefix(kind: InversionKind) -> str:
    return kind
