"""Rotate the ZNE CCF files of one station pair to ZRT (`seisforge ant rotate-ccf`).

One job is one station pair. Input and output are file path templates with a
``{component}`` field (``ZN``, ``EE``, ... for input; ``ZR``, ``TT``, ... for
output); an input template may contain shell wildcards if exactly one file
matches each component. Relative paths are relative to the config file.
Batches of pairs are a loop in the user's shell script.

Azimuth and back azimuth (degrees) come from the config or command line when
both are given, else from the SAC header of the first input file
(`seisforge.ant.rotation.interstation_azimuths`: ``az``/``baz``, else
``evla/evlo`` = virtual source, ``stla/stlo`` = receiver). The rotation and
the R/T axes are those of `seisforge.ant.rotation` (Lin et al. 2008); every
output keeps the header of its first input, with ``kcmpnm`` set to the ZRT
component and ``kuser0`` to ``lin2008`` (a record; nothing reads it).
Existing outputs are never replaced unless ``io.overwrite`` / ``--overwrite``.
``rotate_ccf.log`` next to the outputs records files, angles, their source,
and the software versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import glob
from pathlib import Path
from typing import Any

import yaml

from seisforge.ant.rotation import (
    RADIAL_CONVENTION,
    RADIAL_CONVENTION_SAC_HEADER,
    ZRT_COMPONENTS,
    interstation_azimuths,
    required_zne_components,
    rotate_zne_ccf_to_zrt,
)
from seisforge.provenance import software_provenance

_IO_KEYS = {"input_template", "output_template", "overwrite"}
_ROTATION_KEYS = {"components", "azimuth", "back_azimuth"}


@dataclass(frozen=True)
class RotateCCFJob:
    """One station pair: where to read, where to write, which components, which angles."""

    input_template: Path
    output_template: Path
    components: tuple[str, ...] = ZRT_COMPONENTS
    azimuth: float | None = None
    back_azimuth: float | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class RotateCCFResult:
    job: RotateCCFJob
    written: dict[str, Path]
    log_file: Path


def build_rotate_ccf_job(config: dict[str, Any], *, base_dir=None, **overrides) -> RotateCCFJob:
    """Validated job from the YAML mapping (``io``, ``rotation``) and CLI overrides (None = unset)."""
    io, rotation = config.get("io") or {}, config.get("rotation") or {}
    unknown = sorted(set(config) - {"io", "rotation"}) + sorted(
        [f"io.{k}" for k in set(io) - _IO_KEYS] + [f"rotation.{k}" for k in set(rotation) - _ROTATION_KEYS]
    )
    if unknown:
        raise ValueError(f"rotate-ccf config has unknown keys: {', '.join(unknown)}.")
    params = {**io, **rotation, **{k: v for k, v in overrides.items() if v is not None}}
    base = Path(base_dir) if base_dir is not None else Path()
    for key in ("input_template", "output_template"):
        if "{component}" not in str(params.get(key, "")):
            raise ValueError(f"rotate-ccf io.{key} is required and must contain {{component}}.")
    if (params.get("azimuth") is None) != (params.get("back_azimuth") is None):
        raise ValueError("give both azimuth and back_azimuth, or neither (then SAC headers are used).")
    components = params.get("components", ZRT_COMPONENTS)
    if isinstance(components, str):
        components = components.split(",")
    components = tuple(str(c).strip().upper() for c in components if str(c).strip())
    required_zne_components(components)  # validates the names
    return RotateCCFJob(
        input_template=base / str(params["input_template"]),
        output_template=base / str(params["output_template"]),
        components=components,
        azimuth=None if params.get("azimuth") is None else float(params["azimuth"]),
        back_azimuth=None if params.get("back_azimuth") is None else float(params["back_azimuth"]),
        overwrite=bool(params.get("overwrite", False)),
    )


def run_rotate_ccf_config(config_file, **overrides) -> RotateCCFResult:
    """Read the YAML config, apply CLI overrides, rotate one pair."""
    config_path = Path(config_file)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return rotate_ccf_job(build_rotate_ccf_job(config, base_dir=config_path.parent, **overrides))


def rotate_ccf_job(job: RotateCCFJob) -> RotateCCFResult:
    """Read the needed ZNE SAC files, rotate, write the ZRT SAC files and the log."""
    from obspy import read

    outputs = {c: Path(str(job.output_template).format(component=c)) for c in job.components}
    log_file = next(iter(outputs.values())).parent / "rotate_ccf.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "",
        f"[{datetime.now().isoformat(timespec='seconds')}] rotate-ccf start",
        f"input_template: {job.input_template}",
        f"output_template: {job.output_template}",
        f"components: {','.join(job.components)}",
        f"overwrite: {job.overwrite}",
        f"radial_convention: {RADIAL_CONVENTION} (SAC {RADIAL_CONVENTION_SAC_HEADER})",
        *(f"{key}: {value}" for key, value in software_provenance().items()),
    ]
    try:
        existing = [str(p) for p in outputs.values() if p.exists()]
        if existing and not job.overwrite:
            raise FileExistsError(f"{', '.join(existing)} exist; use --overwrite to replace them.")
        traces = {}
        for component in required_zne_components(job.components):
            path = _input_file(job.input_template, component)
            stream = read(str(path))
            if len(stream) != 1:
                raise ValueError(f"expected one trace in {path}, got {len(stream)}")
            traces[component] = stream[0]
            lines.append(f"read {component}: {path}")

        if job.azimuth is not None:
            azimuth, back_azimuth, source = job.azimuth, job.back_azimuth, "config or command line"
        else:
            header = getattr(next(iter(traces.values())).stats, "sac", None) or {}
            keys = ("az", "baz", "evla", "evlo", "stla", "stlo")
            azimuth, back_azimuth, source = interstation_azimuths(**{k: _sac_float(header, k) for k in keys})
        lines.append(f"azimuths: azimuth={azimuth:.6g}, back_azimuth={back_azimuth:.6g}, source={source}")

        rotated = rotate_zne_ccf_to_zrt(
            {c: tr.data for c, tr in traces.items()}, azimuth, back_azimuth, components=job.components
        )
        for component, data in rotated.items():
            trace = traces[required_zne_components([component])[0]].copy()
            trace.data = data
            trace.stats.channel = component
            if hasattr(trace.stats, "sac"):
                trace.stats.sac.kcmpnm = component
                trace.stats.sac[RADIAL_CONVENTION_SAC_HEADER] = RADIAL_CONVENTION
            outputs[component].parent.mkdir(parents=True, exist_ok=True)
            trace.write(str(outputs[component]), format="SAC")
            lines.append(f"wrote {component}: {outputs[component]}")
        lines.append(f"status: ok, wrote {len(rotated)} files")
        return RotateCCFResult(job=job, written=outputs, log_file=log_file)
    except Exception as exc:
        lines.append(f"status: error, {type(exc).__name__}: {exc}")
        raise
    finally:
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


def _input_file(template: Path, component: str) -> Path:
    pattern = str(template).format(component=component)
    if glob.has_magic(pattern):
        matches = sorted(glob.glob(pattern))
    else:
        matches = [pattern] if Path(pattern).exists() else []
    if len(matches) != 1:
        raise FileNotFoundError(f"{component}: expected one file for {pattern}, found {len(matches)}.")
    return Path(matches[0])


def _sac_float(header, key: str) -> float | None:
    value = header.get(key) if hasattr(header, "get") else None
    return None if value is None or float(value) == -12345.0 else float(value)
