"""Compute the diffuse-field H/V of a layered model from config.yml (examples/05_hvsr_forward).

    python hvsr_forward.py [config.yml]

Reads the model file and frequencies named in the config, calls
`seisforge.hvsr.predict_diffuse_field_hvsr` with the native solver, and
writes `frequency_hz hvsr` with a header recording the inputs. Refuses to
overwrite an existing output file. No formulas here: glue only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

from seisforge.hvsr import (
    DiffuseFieldSettings,
    ElasticLayerModel,
    NativeDiffuseFieldSolver,
    predict_diffuse_field_hvsr,
)
from seisforge.provenance import software_provenance


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    config_path = Path(argv[0] if argv else "config.yml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    unknown = set(config) - {"model_file", "frequency_hz", "output_file", "settings"}
    if unknown:
        raise SystemExit(f"Unknown keys in {config_path}: {sorted(unknown)}")
    output = base / config["output_file"]
    if output.exists():
        raise SystemExit(f"{output} exists; delete it first.")

    table = np.atleast_2d(np.loadtxt(base / config["model_file"], comments="#"))
    model = ElasticLayerModel(
        thickness_km=table[:-1, 0],
        vp_km_s=table[:, 1],
        vs_km_s=table[:, 2],
        density_g_cm3=table[:, 3],
    )
    band = config["frequency_hz"]
    frequency_hz = np.geomspace(float(band["min"]), float(band["max"]), int(band["count"]))
    settings = DiffuseFieldSettings(**(config.get("settings") or {}))
    prediction = predict_diffuse_field_hvsr(
        model, frequency_hz, solver=NativeDiffuseFieldSolver(), settings=settings
    )
    bad = [d for d in prediction.diagnostics if d.status.value != "ok"]
    for diagnostic in bad:
        print(f"warning: {diagnostic.frequency_hz:.4g} Hz: {diagnostic.message}")

    output.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "Diffuse-field H/V (seisforge.hvsr, native solver), written by hvsr_forward.py",
        f"model_file: {config['model_file']}",
        f"settings: {settings}",
        *(f"{key}: {value}" for key, value in software_provenance().items()),
        "frequency_hz  hvsr",
    ]
    np.savetxt(output, np.column_stack([prediction.frequency_hz, prediction.hvsr]),
               fmt="%.6g", header="\n".join(header))
    peak = int(np.nanargmax(prediction.hvsr))
    print(f"wrote {output}; {len(bad)} frequencies failed")
    print(f"H/V peak: {prediction.hvsr[peak]:.2f} at {prediction.frequency_hz[peak]:.3f} Hz")


if __name__ == "__main__":
    main()
