"""Regenerate `tests/data/constraint_grid_baseline.npz` from `tests/inv_constraint_grid_cases.py`.

The baseline pins the grid evaluation of the depth constraints (``vs_range``,
``monotonic_vs``, ``max_vs_gradient``) as it was before continuous evaluation
became the default. It was written with commit 8b6babf:

    PYTHONPATH=src python tools/make_constraint_grid_baseline.py

Do not regenerate it with later code: `constraints.evaluation: grid` exists to
reproduce exactly these decisions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from inv_constraint_grid_cases import BASELINE, all_models, constraint_items, decisions  # noqa: E402


def main() -> None:
    valid = decisions(all_models(), constraint_items())
    np.savez_compressed(BASELINE, valid=valid)
    print(f"wrote {BASELINE}: {valid.shape}, {valid.mean():.3f} valid")


if __name__ == "__main__":
    main()
