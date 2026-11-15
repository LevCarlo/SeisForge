"""AFTAN runs on synthetic ZZ/ZR/RZ/RR/TT CCFs with a known phase velocity.

Shared by `tests/test_ant_aftan_components.py` (accuracy against the true
curve, pair mode vs station mode, bit-for-bit regression) and
`tools/make_aftan_component_baseline.py` (writes
`tests/data/aftan_component_baseline.npz`). The baseline was generated before
the T19 component fix (commit cb03b15 plus this module), so it pins that
components the fix does not touch keep their numbers. Regenerate it only with
the script, and only for a change listed under "Changed results" in
`CHANGELOG.md`.

Inputs: ZZ, ZR, RZ, RR from `synthetic_rayleigh_ccf_zne` (diffuse plane-wave
Rayleigh field, rotated with the package rotation; independent of the AFTAN
``pi_over_4`` table) and TT from `synthetic_ccf` with ``pi_over_4 = -1`` (a
Love-like waveform with the self-component phase constant). The 2 pi cycle is
fixed by giving the true curve as ``prediction_file``, so cycle selection
without a prediction (task T18) does not enter. 150 km is 2.4-9 wavelengths
over 5-14 s, so the far-field phase constants hold to within about
``1 / (8 k r)`` < 0.06 rad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from obspy import Trace

from seisforge.ant.aftan import run_aftan_config
from seisforge.ant.rotation import rotate_zne_ccf_to_zrt
from seisforge.ant.synthetic import (
    symmetric_ccf_time_axis,
    synthetic_ccf,
    synthetic_rayleigh_ccf_zne,
)

SOURCE, RECEIVER = "XX.A", "XX.B"
DISTANCE_KM = 150.0
AZIMUTH_DEG = 37.0
DELTA_S = 0.25
PERIOD_S = np.linspace(1.0, 40.0, 400)
PHASE_VELOCITY_KM_S = 2.8 + 0.04 * PERIOD_S
COMPONENTS = ("ZZ", "ZR", "RZ", "RR", "TT")
SYMMETRIC_COMPONENTS = ("ZZ", "RR", "TT")
MODES = ("pair", "station")
RESULT_KEYS = ("target_period", "period", "group_velocity", "phase_velocity")


def sac_name(component: str) -> str:
    return f"{SOURCE}_{RECEIVER}_{component}.SAC"


def write_ccfs(directory: Path, *, kcmpnm: bool = False) -> None:
    """Write one two-sided SAC CCF per component into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    time = symmetric_ccf_time_axis(300.0, DELTA_S)
    band = {"min_period_s": 4.0, "max_period_s": 16.0}
    zne = synthetic_rayleigh_ccf_zne(
        time, DISTANCE_KM, AZIMUTH_DEG, PERIOD_S, PHASE_VELOCITY_KM_S, **band
    )
    data = rotate_zne_ccf_to_zrt(zne, AZIMUTH_DEG, AZIMUTH_DEG + 180.0)
    data["TT"] = synthetic_ccf(
        time, DISTANCE_KM, PERIOD_S, PHASE_VELOCITY_KM_S, pi_over_4=-1.0, **band
    )
    for component in COMPONENTS:
        trace = Trace(data=np.asarray(data[component], dtype=np.float32))
        trace.stats.delta = DELTA_S
        trace.stats.sac = {"b": float(time[0]), "dist": DISTANCE_KM}
        if kcmpnm:
            trace.stats.sac["kcmpnm"] = component
        trace.write(str(directory / sac_name(component)), format="SAC")
    np.savetxt(
        directory / "prediction.dat", np.column_stack([PERIOD_S, PHASE_VELOCITY_KM_S])
    )


def _aftan_section(directory: Path, branch: str, extra: dict | None) -> dict:
    section = {
        "branch": branch,
        "min_period": 5.0,
        "max_period": 14.0,
        "period_sampling": {"mode": "uniform", "uniform": {"step": 1.0}},
        "velocity_window": {"min": 2.0, "max": 4.5},
        "basic": {"alpha": {"mode": "constant", "constant": {"value": 20.0}}},
        "prediction_file": str(directory / "prediction.dat"),
    }
    section.update(extra or {})
    return section


def run(
    mode: str,
    component: str,
    branch: str,
    directory: Path,
    output: Path,
    *,
    aftan: dict | None = None,
    pattern: str | None = None,
):
    """Run AFTAN on one component file; return the list of `AFTANResult`."""
    if mode == "pair":
        config = {
            "source_station": SOURCE,
            "receiver_station": RECEIVER,
            "component": component,
            "io": {
                "input_root_datadir": str(directory),
                "output_root_datadir": str(output),
                "input_template": "{source_station}_{receiver_station}_{component}.SAC",
            },
        }
    elif mode == "station":
        config = {
            "station": SOURCE,
            "io": {
                "input_dir": str(directory),
                "output_dir": str(output),
                "sac_pattern": pattern or f"*_{component}.SAC",
            },
        }
    else:
        raise ValueError(mode)
    config["aftan"] = _aftan_section(directory, branch, aftan)
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / f"{mode}_{component}_{branch}.yml"
    config_file.write_text(yaml.safe_dump(config))
    return run_aftan_config(config_file)


def baseline_cases():
    """(mode, component, branch) triples stored in the baseline."""
    for mode in MODES:
        for component in COMPONENTS:
            yield mode, component, "both"
            if component in SYMMETRIC_COMPONENTS:
                yield mode, component, "stack"


def result_arrays(directory: Path, workdir: Path) -> dict[str, np.ndarray]:
    """All baseline cases as ``{"<mode>/<component>/<branch>/<key>": array}``."""
    arrays = {}
    for mode, component, branch in baseline_cases():
        output = workdir / f"{mode}_{component}_{branch}"
        for result in run(mode, component, branch, directory, output):
            for key in RESULT_KEYS:
                arrays[f"{mode}/{component}/{result.branch}/{key}"] = np.asarray(
                    getattr(result, key)
                )
    return arrays


def phase_velocity_error(result) -> float:
    """Largest |measured - true| phase velocity (km/s) over the picked periods."""
    true = np.interp(result.period, PERIOD_S, PHASE_VELOCITY_KM_S)
    return float(np.max(np.abs(np.asarray(result.phase_velocity) - true)))
