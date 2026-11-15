"""Regenerate `tests/data/sampler_baseline.npz` from `tests/sampler_baseline_cases.py`.

Run from the repository root with the code version the baseline should pin:

    PYTHONPATH=src python tools/make_sampler_baseline.py

Regenerate only for an intended change of sampler output, recorded under
"Changed results" in `CHANGELOG.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from sampler_baseline_cases import CASES, result_arrays, run_case  # noqa: E402


def main() -> None:
    arrays = {}
    for name in CASES:
        for key, value in result_arrays(run_case(name)).items():
            arrays[f"{name}/{key}"] = value
    path = ROOT / "tests" / "data" / "sampler_baseline.npz"
    np.savez_compressed(path, **arrays)
    print(f"wrote {path} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
