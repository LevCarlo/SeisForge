"""Build the optional f2py HVSR extension in the source package."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "seisforge" / "hvsr" / "_fortran_src" / "hvsr_core.f90"
TYPE_MAP = SOURCE.parent / ".f2py_f2cmap"
DESTINATION = ROOT / "src" / "seisforge" / "hvsr"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="seisforge-f2py-") as temporary:
        build_dir = Path(temporary)
        shutil.copy2(TYPE_MAP, build_dir / TYPE_MAP.name)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                str(SOURCE),
                "-m",
                "_hvsr_fortran",
                "--build-dir",
                str(build_dir / "build"),
                "--backend",
                "meson",
            ],
            cwd=build_dir,
            check=True,
        )
        candidates = list(build_dir.glob("_hvsr_fortran*.so")) + list(
            build_dir.glob("_hvsr_fortran*.pyd")
        )
        if len(candidates) != 1:
            raise RuntimeError(f"expected one f2py extension, found {candidates}")
        destination = DESTINATION / candidates[0].name
        shutil.copy2(candidates[0], destination)
        print(destination)


if __name__ == "__main__":
    main()
