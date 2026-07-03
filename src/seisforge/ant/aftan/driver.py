"""Station/file driver for SeisForge AFTAN workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from obspy import read

from .branch import _prepare_branch_traces
from .config import (
    build_station_aftan_config,
    _override_energy_map_config,
)
from .core import AFTANMeasurement
from .io import (
    _find_sac_files,
    _read_yaml,
    _snr_column_name,
    _write_dispersion_dat,
)
from .models import AFTANResult, BranchTrace, StationAFTANConfig
from .plot import _write_aftan_summary_plot
from .qc import (
    _format_qc_log_lines,
    _period_qc,
    _qc_log_lines,
    _qc_warnings,
    _raise_short_branch_if_requested,
)


def run_aftan_config(
    config_file,
    *,
    write_energy_map: bool | None = None,
    plot_energy_map: bool | None = None,
) -> list[AFTANResult]:
    """Run one station-level AFTAN config."""
    config_path = Path(config_file)
    config = build_station_aftan_config(_read_yaml(config_path), config_path.parent)
    config = _override_energy_map_config(config, write_energy_map, plot_energy_map)
    return run_station_aftan(config)


def run_station_aftan(config: StationAFTANConfig) -> list[AFTANResult]:
    """Measure dispersion for all SAC files in one station config."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.output_dir / "aftan.log"
    log_lines = _start_log(config)
    try:
        files = _find_sac_files(config.input_dir, config.sac_pattern)
        log_lines.append(f"matched_sac_files: {len(files)}")
        for path in files:
            log_lines.append(f"input: {path}")
        if not files:
            raise FileNotFoundError(
                f"No SAC files matched {config.input_dir / config.sac_pattern}"
            )

        results = []
        for path in files:
            path_results = run_aftan_file(path, config)
            for result in path_results:
                log_lines.extend(_result_log_lines(result))
            results.extend(path_results)
        log_lines.append(f"status: ok, measured {len(results)} files")
        _append_log(log_file, log_lines)
        return results
    except Exception as exc:
        log_lines.append(f"status: error, {type(exc).__name__}: {exc}")
        _append_log(log_file, log_lines)
        raise


def run_aftan_file(path: str | Path, config: StationAFTANConfig) -> list[AFTANResult]:
    """Measure dispersion for one SAC file."""
    path = Path(path)
    trace = read(str(path))[0]
    branch_traces = _prepare_branch_traces(trace, config.aftan.branch)

    results = []
    for branch_trace in branch_traces:
        results.append(_run_aftan_branch(path, branch_trace, config))
    return results


def _run_aftan_branch(
    path: Path,
    branch_trace: BranchTrace,
    config: StationAFTANConfig,
) -> AFTANResult:
    output_dat = config.output_dir / f"{path.stem}.{branch_trace.name}.dat"
    output_energy_map = config.output_dir / f"{path.stem}.{branch_trace.name}.basic_ftan.nc"
    output_energy_plot = config.output_dir / f"{path.stem}.{branch_trace.name}.basic_ftan.png"
    output_phase_map = config.output_dir / f"{path.stem}.{branch_trace.name}.phase_velocity.nc"
    output_pmf_dat = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf.dat"
    output_pmf_energy_map = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf_ftan.nc"
    output_pmf_energy_plot = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf_ftan.png"
    output_pmf_phase_map = config.output_dir / f"{path.stem}.{branch_trace.name}.pmf_phase_velocity.nc"
    output_paths = [output_dat]
    if config.aftan.pmf.enabled:
        output_paths.append(output_pmf_dat)
    if config.aftan.energy_map.enabled:
        output_paths.append(output_energy_map)
        if config.aftan.energy_map.phase_velocity:
            output_paths.append(output_phase_map)
        if config.aftan.pmf.enabled:
            output_paths.append(output_pmf_energy_map)
            if config.aftan.energy_map.phase_velocity:
                output_paths.append(output_pmf_phase_map)
    if config.aftan.energy_map.plot:
        output_paths.append(output_energy_plot)
        if config.aftan.pmf.enabled:
            output_paths.append(output_pmf_energy_plot)
    if any(item.exists() for item in output_paths) and not config.overwrite:
        raise FileExistsError(
            f"One or more output files for {path.name} already exist. Enable io.overwrite."
        )

    measurement = AFTANMeasurement(branch_trace.trace, path, config.aftan)
    measured = measurement.measure()
    qc = _period_qc(measured)
    warnings = branch_trace.warnings + _qc_warnings(qc, config.aftan.qc)
    _raise_short_branch_if_requested(qc, config.aftan.qc, f"{path.name} [{branch_trace.name}]")

    output_dat.parent.mkdir(parents=True, exist_ok=True)
    _write_dispersion_dat(
        output_dat,
        measured,
        debug=config.aftan.debug,
        snr_label=_snr_column_name(config.aftan.snr),
    )

    written_energy_map = None
    written_energy_plot = None
    written_phase_map = None
    if config.aftan.energy_map.enabled or config.aftan.energy_map.plot:
        dataset = measurement.energy_map_dataset(
            measured,
            stage="basic",
            source_file=path,
            branch=branch_trace.name,
            velocity_count=config.aftan.energy_map.velocity_count,
            normalize=config.aftan.energy_map.normalize,
        )
        if config.aftan.energy_map.enabled:
            dataset.to_netcdf(output_energy_map, engine="scipy")
            written_energy_map = output_energy_map
        phase_dataset = None
        if config.aftan.energy_map.phase_velocity:
            phase_dataset = measurement.phase_velocity_map_dataset(
                measured,
                stage="basic",
                source_file=path,
                branch=branch_trace.name,
                velocity_count=config.aftan.energy_map.velocity_count,
                normalize=config.aftan.energy_map.normalize,
                cycle_count=config.aftan.energy_map.phase_cycle_count,
            )
            if config.aftan.energy_map.enabled:
                phase_dataset.to_netcdf(output_phase_map, engine="scipy")
                written_phase_map = output_phase_map
        if config.aftan.energy_map.plot:
            _write_aftan_summary_plot(
                dataset,
                output_energy_plot,
                phase_dataset=phase_dataset,
                snr_label=_snr_column_name(config.aftan.snr),
            )
            written_energy_plot = output_energy_plot

    written_pmf_dat = None
    written_pmf_energy_map = None
    written_pmf_energy_plot = None
    written_pmf_phase_map = None
    pmf_qc = None
    pmf_warnings: tuple[str, ...] = ()
    pmf_measured = None
    if config.aftan.pmf.enabled:
        pmf_measured = measurement.measure_pmf(measured)
        pmf_qc = _period_qc(pmf_measured)
        pmf_warnings = _qc_warnings(pmf_qc, config.aftan.qc)
        _raise_short_branch_if_requested(
            pmf_qc,
            config.aftan.qc,
            f"PMF {path.name} [{branch_trace.name}]",
        )
        _write_dispersion_dat(
            output_pmf_dat,
            pmf_measured,
            debug=config.aftan.debug,
            snr_label=_snr_column_name(config.aftan.snr),
        )
        written_pmf_dat = output_pmf_dat
        if config.aftan.energy_map.enabled or config.aftan.energy_map.plot:
            pmf_dataset = measurement.energy_map_dataset(
                pmf_measured,
                stage="pmf",
                source_file=path,
                branch=branch_trace.name,
                velocity_count=config.aftan.energy_map.velocity_count,
                normalize=config.aftan.energy_map.normalize,
                signal_data=pmf_measured["signal"],
            )
            if config.aftan.energy_map.enabled:
                pmf_dataset.to_netcdf(output_pmf_energy_map, engine="scipy")
                written_pmf_energy_map = output_pmf_energy_map
            pmf_phase_dataset = None
            if config.aftan.energy_map.phase_velocity:
                pmf_phase_dataset = measurement.phase_velocity_map_dataset(
                    pmf_measured,
                    stage="pmf",
                    source_file=path,
                    branch=branch_trace.name,
                    velocity_count=config.aftan.energy_map.velocity_count,
                    normalize=config.aftan.energy_map.normalize,
                    cycle_count=config.aftan.energy_map.phase_cycle_count,
                )
                if config.aftan.energy_map.enabled:
                    pmf_phase_dataset.to_netcdf(output_pmf_phase_map, engine="scipy")
                    written_pmf_phase_map = output_pmf_phase_map
            if config.aftan.energy_map.plot:
                _write_aftan_summary_plot(
                    pmf_dataset,
                    output_pmf_energy_plot,
                    phase_dataset=pmf_phase_dataset,
                    snr_label=_snr_column_name(config.aftan.snr),
                )
                written_pmf_energy_plot = output_pmf_energy_plot

    return AFTANResult(
        input_file=path,
        branch=branch_trace.name,
        output_dat=output_dat,
        output_energy_map=written_energy_map,
        output_energy_plot=written_energy_plot,
        target_period=measured["target_period"],
        period=measured["period"],
        group_velocity=measured["group_velocity"],
        phase_velocity=measured["phase_velocity"],
        amplitude=measured["amplitude"],
        snr=measured["snr"],
        distance_km=measurement.distance_km,
        alpha=measurement.basic_alpha,
        alpha_mode=measurement.basic_alpha_mode,
        qc=qc,
        warnings=warnings,
        output_npz=None,
        pmf_alpha=measurement.pmf_alpha if config.aftan.pmf.enabled else None,
        pmf_alpha_mode=(
            measurement.pmf_alpha_mode if config.aftan.pmf.enabled else None
        ),
        output_phase_map=written_phase_map,
        output_phase_plot=None,
        output_pmf_npz=None,
        output_pmf_dat=written_pmf_dat,
        output_pmf_energy_map=written_pmf_energy_map,
        output_pmf_energy_plot=written_pmf_energy_plot,
        output_pmf_phase_map=written_pmf_phase_map,
        output_pmf_phase_plot=None,
        pmf_qc=pmf_qc,
        pmf_warnings=pmf_warnings,
        pmf_period_min=(
            float(pmf_measured["pmf_period_min"]) if pmf_measured is not None else None
        ),
        pmf_period_max=(
            float(pmf_measured["pmf_period_max"]) if pmf_measured is not None else None
        ),
        pmf_raw_period_min=(
            float(pmf_measured["pmf_raw_period_min"]) if pmf_measured is not None else None
        ),
        pmf_raw_period_max=(
            float(pmf_measured["pmf_raw_period_max"]) if pmf_measured is not None else None
        ),
    )


def _start_log(config: StationAFTANConfig) -> list[str]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    return [
        "",
        f"[{timestamp}] aftan start",
        f"station: {config.station}",
        f"input_dir: {config.input_dir}",
        f"sac_pattern: {config.sac_pattern}",
        f"output_dir: {config.output_dir}",
        f"requested_branch: {config.aftan.branch}",
        f"debug: {config.aftan.debug}",
        f"snr_output: {_snr_column_name(config.aftan.snr)}",
    ]


def _result_log_lines(result: AFTANResult) -> list[str]:
    lines = [
        (
            f"basic_result {result.input_file.name} [{result.branch}]: "
            f"distance={result.distance_km:.3f} km, "
            f"alpha={result.alpha:.6g} ({result.alpha_mode}), "
            f"period_grid={_period_range(result.target_period)}, "
            f"retained={result.qc['qc_retained_period_count']}/"
            f"{result.qc['qc_initial_period_count']}"
        ),
        (
            f"outputs {result.input_file.name} [{result.branch}] basic: "
            f"dat={result.output_dat}"
        ),
    ]
    lines.extend(_optional_output_lines(result))
    for warning in result.warnings:
        lines.append(f"warning {result.input_file.name} [{result.branch}]: {warning}")
    lines.extend(_qc_log_lines(result))

    if result.output_pmf_dat is not None:
        lines.append(
            f"pmf_result {result.input_file.name} [{result.branch}]: "
            f"alpha={result.pmf_alpha:.6g} ({result.pmf_alpha_mode}), "
            f"raw_period_bounds={_optional_range(result.pmf_raw_period_min, result.pmf_raw_period_max)}, "
            f"used_period_grid={_optional_range(result.pmf_period_min, result.pmf_period_max)}, "
            f"retained={result.pmf_qc['qc_retained_period_count']}/"
            f"{result.pmf_qc['qc_initial_period_count']}"
        )
        lines.append(
            f"outputs {result.input_file.name} [{result.branch}] pmf: "
            f"dat={result.output_pmf_dat}"
        )
        for warning in result.pmf_warnings:
            lines.append(f"warning PMF {result.input_file.name} [{result.branch}]: {warning}")
        if result.pmf_qc is not None:
            lines.extend(
                _format_qc_log_lines(
                    result.input_file.name,
                    result.branch,
                    result.pmf_qc,
                    prefix="qc PMF",
                )
            )
        lines.extend(_optional_pmf_output_lines(result))
    return lines


def _optional_output_lines(result: AFTANResult) -> list[str]:
    outputs = []
    if result.output_energy_map is not None:
        outputs.append(f"energy_map={result.output_energy_map}")
    if result.output_energy_plot is not None:
        outputs.append(f"summary_plot={result.output_energy_plot}")
    if result.output_phase_map is not None:
        outputs.append(f"phase_velocity_map={result.output_phase_map}")
    if not outputs:
        return []
    return [
        (
            f"outputs {result.input_file.name} [{result.branch}] diagnostics: "
            + ", ".join(outputs)
        )
    ]


def _optional_pmf_output_lines(result: AFTANResult) -> list[str]:
    outputs = []
    if result.output_pmf_energy_map is not None:
        outputs.append(f"energy_map={result.output_pmf_energy_map}")
    if result.output_pmf_energy_plot is not None:
        outputs.append(f"summary_plot={result.output_pmf_energy_plot}")
    if result.output_pmf_phase_map is not None:
        outputs.append(f"phase_velocity_map={result.output_pmf_phase_map}")
    if not outputs:
        return []
    return [
        (
            f"outputs {result.input_file.name} [{result.branch}] pmf_diagnostics: "
            + ", ".join(outputs)
        )
    ]


def _period_range(periods: np.ndarray) -> str:
    if periods.size == 0:
        return "empty"
    return f"{np.nanmin(periods):.6g}-{np.nanmax(periods):.6g} s ({periods.size} samples)"


def _optional_range(min_value: float | None, max_value: float | None) -> str:
    if min_value is None or max_value is None:
        return "n/a"
    return f"{min_value:.6g}-{max_value:.6g} s"


def _append_log(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")
