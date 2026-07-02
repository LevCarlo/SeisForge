"""Ambient-noise cross-correlation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import glob
from math import radians
from pathlib import Path
import re
from typing import Any

import yaml

from seisforge.ant.rotation import (
    ZRT_COMPONENTS,
    required_zne_components,
    rotate_zne_ccf_to_zrt,
)


_LEGACY_COMPONENT_OUTPUT_TEMPLATE = "{component}.SAC"


@dataclass(frozen=True)
class RotateCCFJob:
    """One ZNE-to-ZRT cross-correlation rotation job."""

    input_dir: Path
    output_dir: Path
    azimuth: float | None = None
    back_azimuth: float | None = None
    input_template: str = "{component}.SAC"
    output_template: str | None = None
    obj_components: tuple[str, ...] = ZRT_COMPONENTS
    degrees: bool = True
    overwrite: bool = False
    name: str | None = None


@dataclass(frozen=True)
class RotateCCFResult:
    """Files written by one rotation job."""

    job: RotateCCFJob
    written: dict[str, Path]
    log_file: Path


def run_rotate_ccf_config(config_file, **overrides) -> list[RotateCCFResult]:
    """Run ZNE-to-ZRT CCF rotation jobs from a YAML configuration file."""
    config_path = Path(config_file)
    config = _read_yaml(config_path)
    jobs = build_rotate_ccf_jobs(config, base_dir=config_path.parent, **overrides)
    return [rotate_ccf_job(job) for job in jobs]


def build_rotate_ccf_jobs(
    config: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    **overrides,
) -> list[RotateCCFJob]:
    """Build rotation jobs from a config dictionary and optional CLI overrides."""
    base_path = Path(base_dir) if base_dir is not None else None
    defaults = _collect_defaults(config)
    defaults.update({key: value for key, value in overrides.items() if value is not None})

    raw_jobs = config.get("jobs") or [config.get("job", {})]
    jobs = []
    for raw_job in raw_jobs:
        params = dict(defaults)
        params.update(raw_job or {})
        jobs.append(_make_job(params, base_path))
    return jobs


def rotate_ccf_job(job: RotateCCFJob) -> RotateCCFResult:
    """Read ZNE SAC CCFs, rotate them, and write ZRT SAC CCFs."""
    from obspy import read

    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = job.output_dir / "rotate_ccf.log"
    log_lines = _start_log(job)

    try:
        input_components = required_zne_components(job.obj_components)
        log_lines.append(f"obj_components: {','.join(job.obj_components)}")
        log_lines.append(
            "required_input_components inferred from obj_components: "
            f"{','.join(input_components)}"
        )
        traces = {}
        input_paths = {}
        ccf = {}
        for component in input_components:
            pattern = _format_path(job.input_dir, job.input_template, component, job.name)
            log_lines.append(f"input candidate {component}: {pattern}")
            try:
                path = _resolve_input_file(pattern)
            except Exception as exc:
                raise FileNotFoundError(
                    "Failed to resolve required input component "
                    f"{component!r}; obj_components={job.obj_components}; "
                    f"required_input_components={input_components}; "
                    f"attempted pattern/path={pattern}"
                ) from exc
            log_lines.append(f"read {component}: {path}")
            try:
                stream = read(str(path))
            except Exception as exc:
                raise OSError(
                    "Failed to read required input component "
                    f"{component!r}; obj_components={job.obj_components}; "
                    f"required_input_components={input_components}; path={path}"
                ) from exc
            if len(stream) != 1:
                raise ValueError(f"Expected one trace in {path}, got {len(stream)}")
            trace = stream[0]
            traces[component] = trace
            input_paths[component] = path
            ccf[component] = trace.data

        azimuth, back_azimuth, azimuth_source, azimuth_log = _resolve_azimuths(
            job,
            next(iter(traces.values())),
        )
        log_lines.extend(azimuth_log)
        log_lines.append(
            "azimuths: "
            f"azimuth={azimuth:.6g}, back_azimuth={back_azimuth:.6g}, "
            f"degrees={job.degrees}, source={azimuth_source}"
        )

        rotated = rotate_zne_ccf_to_zrt(
            ccf,
            azimuth=azimuth,
            back_azimuth=back_azimuth,
            components=job.obj_components,
            degrees=job.degrees,
        )

        written = {}
        for component, data in rotated.items():
            source_component = required_zne_components([component])[0]
            output_path = _format_output_path(
                job,
                component,
                input_paths[source_component],
                source_component,
            )
            if output_path.exists() and not job.overwrite:
                raise FileExistsError(
                    f"{output_path} already exists. Use --overwrite to replace it."
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_trace = traces[source_component].copy()
            output_trace.data = data
            _set_component_header(output_trace, component)
            output_trace.write(str(output_path), format="SAC")
            log_lines.append(f"wrote {component}: {output_path}")
            written[component] = output_path

        log_lines.append(f"status: ok, wrote {len(written)} files")
        _append_log(log_file, log_lines)
        return RotateCCFResult(job=job, written=written, log_file=log_file)
    except Exception as exc:
        log_lines.append(f"status: error, {type(exc).__name__}: {exc}")
        _append_log(log_file, log_lines)
        raise


def _collect_defaults(config: dict[str, Any]) -> dict[str, Any]:
    params = {}
    for section_name in ("io", "rotation"):
        section = config.get(section_name)
        if section:
            params.update(section)
    for key in (
        "input_dir",
        "output_dir",
        "input_template",
        "output_template",
        "azimuth",
        "back_azimuth",
        "obj_components",
        "components",
        "degrees",
        "overwrite",
        "name",
    ):
        if key in config:
            params[key] = config[key]
    return params


def _make_job(params: dict[str, Any], base_dir: Path | None) -> RotateCCFJob:
    missing = [key for key in ("input_dir", "output_dir") if key not in params]
    if missing:
        raise ValueError(f"Missing required rotate-ccf configuration keys: {missing}")

    obj_components = _parse_components(
        params.get("obj_components", params.get("components", ZRT_COMPONENTS))
    )
    input_template = params.get("input_template", "{component}.SAC")
    output_template = _normalize_output_template(params.get("output_template"))

    return RotateCCFJob(
        input_dir=_resolve_path(params["input_dir"], base_dir),
        output_dir=_resolve_path(params["output_dir"], base_dir),
        azimuth=_optional_float(params.get("azimuth")),
        back_azimuth=_optional_float(params.get("back_azimuth")),
        input_template=str(input_template),
        output_template=output_template,
        obj_components=obj_components,
        degrees=bool(params.get("degrees", True)),
        overwrite=bool(params.get("overwrite", False)),
        name=params.get("name"),
    )


def _resolve_azimuths(job: RotateCCFJob, trace) -> tuple[float, float, str, list[str]]:
    log_lines = []
    if job.azimuth is not None and job.back_azimuth is not None:
        log_lines.append(
            "azimuth/back_azimuth: specified in config or CLI; "
            "skip SAC HEADER and coordinate fallback"
        )
        return job.azimuth, job.back_azimuth, "configured", log_lines

    missing = []
    if job.azimuth is None:
        missing.append("azimuth")
    if job.back_azimuth is None:
        missing.append("back_azimuth")
    log_lines.append(
        "azimuth/back_azimuth: not fully specified; "
        f"missing {', '.join(missing)}"
    )

    inferred_azimuth, inferred_back_azimuth, inferred_source, inferred_log = (
        _infer_azimuths_from_sac(trace)
    )
    log_lines.extend(inferred_log)
    azimuth = job.azimuth if job.azimuth is not None else inferred_azimuth
    back_azimuth = (
        job.back_azimuth
        if job.back_azimuth is not None
        else inferred_back_azimuth
    )
    source_parts = []
    if job.azimuth is not None or job.back_azimuth is not None:
        source_parts.append("configured")
    source_parts.append(inferred_source)
    source = " + ".join(source_parts)
    if job.degrees:
        return azimuth, back_azimuth, source, log_lines
    if job.azimuth is None:
        azimuth = radians(azimuth)
    if job.back_azimuth is None:
        back_azimuth = radians(back_azimuth)
    return azimuth, back_azimuth, source, log_lines


def _infer_azimuths_from_sac(trace) -> tuple[float, float, str, list[str]]:
    from obspy.geodetics.base import gps2dist_azimuth

    log_lines = []
    sac = getattr(trace.stats, "sac", None)
    if sac is None:
        log_lines.append(
            "SAC HEADER az/baz: failed to read from SAC HEADER; "
            "trace has no SAC header"
        )
        raise ValueError(
            "azimuth/back_azimuth were not configured and SAC headers are missing."
        )

    azimuth = _sac_float(sac, "az")
    back_azimuth = _sac_float(sac, "baz")
    if azimuth is not None and back_azimuth is not None:
        log_lines.append(
            "SAC HEADER az/baz: success; "
            f"azimuth={azimuth:.6g}, back_azimuth={back_azimuth:.6g}"
        )
        log_lines.append("coordinate fallback: skipped because SAC HEADER az/baz worked")
        return azimuth, back_azimuth, "SAC headers az/baz", log_lines

    log_lines.append(
        "SAC HEADER az/baz: failed to read from SAC HEADER; "
        "az and/or baz missing or undefined"
    )

    evla = _sac_float(sac, "evla")
    evlo = _sac_float(sac, "evlo")
    stla = _sac_float(sac, "stla")
    stlo = _sac_float(sac, "stlo")
    if None in (evla, evlo, stla, stlo):
        log_lines.append(
            "coordinate fallback: failed; SAC headers evla, evlo, stla, stlo "
            "are required"
        )
        raise ValueError(
            "azimuth/back_azimuth were not configured and SAC headers az/baz "
            "or evla, evlo, stla, stlo are required for inference."
        )

    _, azimuth, back_azimuth = gps2dist_azimuth(evla, evlo, stla, stlo)
    log_lines.append(
        "coordinate fallback: success; computed from SAC evla/evlo/stla/stlo "
        f"azimuth={azimuth:.6g}, back_azimuth={back_azimuth:.6g}"
    )
    return (
        azimuth,
        back_azimuth,
        "computed from SAC evla/evlo/stla/stlo",
        log_lines,
    )


def _sac_float(sac, key: str) -> float | None:
    try:
        value = sac[key]
    except (KeyError, TypeError):
        value = getattr(sac, key, None)
    if value is None or float(value) == -12345.0:
        return None
    return float(value)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_components(value) -> tuple[str, ...]:
    if isinstance(value, str):
        components = tuple(comp.strip().upper() for comp in value.split(","))
    else:
        components = tuple(str(comp).strip().upper() for comp in value)
    components = tuple(comp for comp in components if comp)
    required_zne_components(components)
    return components


def _format_path(root: Path, template: str, component: str, name: str | None) -> Path:
    fields = {"component": component, "name": name or ""}
    try:
        path = template.format(**fields)
    except KeyError as exc:
        raise KeyError(
            f"Unknown template field {exc!s}; supported fields are component and name."
        ) from exc
    return root / path


def _resolve_input_file(path: Path) -> Path:
    path_text = str(path)
    if not glob.has_magic(path_text):
        if path.exists():
            return path
        raise FileNotFoundError(2, "No such file or directory", path_text)

    matches = sorted(glob.glob(path_text))
    if not matches:
        raise FileNotFoundError(f"No file matching file pattern: {path_text}")
    if len(matches) > 1:
        raise ValueError(f"Expected one file matching {path_text}, got {len(matches)}")
    return Path(matches[0])


def _format_output_path(
    job: RotateCCFJob,
    component: str,
    source_path: Path,
    source_component: str,
) -> Path:
    source_name = _replace_component_token(source_path.name, source_component, component)
    source_stem = Path(source_name).stem
    if job.output_template is None:
        return job.output_dir / source_name

    fields = {
        "component": component,
        "name": job.name or "",
        "source_component": source_component,
        "source_name": source_name,
        "source_stem": source_stem,
    }
    try:
        path = job.output_template.format(**fields)
    except KeyError as exc:
        raise KeyError(
            f"Unknown template field {exc!s}; supported fields are component, "
            "name, source_component, source_name, and source_stem."
        ) from exc
    return job.output_dir / path


def _replace_component_token(name: str, source_component: str, component: str) -> str:
    token_pattern = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(source_component)}(?![A-Za-z0-9])"
    )
    replaced, count = token_pattern.subn(component, name, count=1)
    if count:
        return replaced

    replaced = name.replace(source_component, component, 1)
    if replaced != name:
        return replaced
    raise ValueError(
        f"Could not replace component token {source_component!r} in {name!r}"
    )


def _normalize_output_template(value) -> str | None:
    if value is None:
        return None
    template = str(value)
    if template == _LEGACY_COMPONENT_OUTPUT_TEMPLATE:
        return None
    return template


def _resolve_path(path, base_dir: Path | None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = base_dir / resolved
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config or {}


def _start_log(job: RotateCCFJob) -> list[str]:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        "",
        f"[{timestamp}] rotate-ccf start",
        f"name: {job.name or '-'}",
        f"input_dir: {job.input_dir}",
        f"output_dir: {job.output_dir}",
        f"input_template: {job.input_template}",
        f"output_template: {job.output_template or '<preserve input naming>'}",
        f"obj_components: {','.join(job.obj_components)}",
        f"overwrite: {job.overwrite}",
    ]
    return lines


def _append_log(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
        file.write("\n")


def _set_component_header(trace, component: str) -> None:
    trace.stats.channel = component
    if hasattr(trace.stats, "sac"):
        trace.stats.sac.kcmpnm = component
