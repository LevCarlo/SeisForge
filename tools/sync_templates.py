"""Copy the example config files into the package templates (`seisforge init`).

    python tools/sync_templates.py

`examples/` is the source of truth; run this after editing an example's
config, and `tests/test_templates.py` will pass again.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seisforge.templates import WORKFLOWS  # noqa: E402


def main() -> None:
    for workflow, (example, names) in WORKFLOWS.items():
        target = ROOT / "src" / "seisforge" / "templates" / workflow.replace("-", "_")
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(ROOT / "examples" / example / name, target / name)
            print(f"{example}/{name} -> {target.relative_to(ROOT)}/{name}")


if __name__ == "__main__":
    main()
