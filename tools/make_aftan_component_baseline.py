"""Write tests/data/aftan_component_baseline.npz from tests/aftan_component_cases.py.

Run from the repository root: ``PYTHONPATH=src:tests python tools/make_aftan_component_baseline.py``.
Regenerate only for a change listed under "Changed results" in CHANGELOG.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from aftan_component_cases import result_arrays, write_ccfs

OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "aftan_component_baseline.npz"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_ccfs(tmp_path / "ccf")
        arrays = result_arrays(tmp_path / "ccf", tmp_path / "out")
    np.savez(OUTPUT, **arrays)
    print(f"wrote {len(arrays)} arrays to {OUTPUT}")


if __name__ == "__main__":
    main()
