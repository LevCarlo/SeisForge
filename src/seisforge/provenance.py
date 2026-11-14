"""Software provenance recorded in every SeisForge output.

``software_provenance()`` returns the versions of SeisForge, its git state, and
the main numerical dependencies as a flat ``{name: str}`` dictionary. Every
value is a string so the dictionary can go straight into NetCDF attributes,
YAML snapshots, ``.dat`` comment headers, and log files.

Keys
----
``seisforge_version``
    Installed distribution version (``importlib.metadata``), or
    ``seisforge.__version__``'s placeholder when the package is imported from a
    source tree that was never installed.
``seisforge_git_commit``
    Full commit hash of the SeisForge checkout the code was imported from, or
    ``"unknown"`` (installed from a wheel/sdist, no ``git`` executable, not a
    git checkout, or the query failed).
``seisforge_git_dirty``
    ``"true"`` when tracked files differ from the commit, ``"false"`` when
    clean, ``"unknown"`` when it cannot be determined. Untracked files are
    ignored.
``python_version`` and ``<package>_version``
    Interpreter version and the distribution versions of numpy, scipy,
    obspy, xarray, disba and numba (``"not installed"`` when absent).

The git query runs only for a source checkout whose ``src/seisforge`` is this
package's directory, so an installed copy inside some other git repository
never reports that repository's commit. Values are computed on the first call
and cached for the life of the process; importing this module does not run
``git``.
"""

from __future__ import annotations

import os
import platform
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

UNKNOWN = "unknown"
NOT_INSTALLED = "not installed"

# Distribution names whose versions are recorded, in output order.
DEPENDENCIES = ("numpy", "scipy", "obspy", "xarray", "disba", "numba")

# Seconds; a local ``git rev-parse``/``git status`` takes milliseconds, the
# limit only guards against a hung filesystem or credential prompt.
_GIT_TIMEOUT_S = 5.0

_PACKAGE_DIR = Path(__file__).resolve().parent


def software_provenance() -> dict[str, str]:
    """Return SeisForge, git, Python and dependency versions as strings.

    Never raises: anything that cannot be determined is ``"unknown"`` (or
    ``"not installed"`` for a missing dependency). The result is cached after
    the first call; a fresh copy is returned each time so callers may modify
    it.
    """
    return dict(_cached_provenance())


@lru_cache(maxsize=1)
def _cached_provenance() -> tuple[tuple[str, str], ...]:
    commit, dirty = _git_state(_PACKAGE_DIR)
    items = {
        "seisforge_version": _seisforge_version(),
        "seisforge_git_commit": commit,
        "seisforge_git_dirty": dirty,
        "python_version": platform.python_version(),
    }
    for name in DEPENDENCIES:
        items[f"{name}_version"] = _distribution_version(name)
    return tuple(items.items())


def _seisforge_version() -> str:
    try:
        from seisforge import __version__
    except Exception:  # pragma: no cover - defensive, package always importable here
        return UNKNOWN
    return str(__version__)


def _distribution_version(name: str) -> str:
    try:
        return str(metadata.version(name))
    except metadata.PackageNotFoundError:
        return NOT_INSTALLED
    except Exception:
        return UNKNOWN


def _git_state(package_dir: Path) -> tuple[str, str]:
    """Commit and dirty flag of the checkout holding ``package_dir``."""
    toplevel = _run_git(package_dir, "rev-parse", "--show-toplevel")
    if toplevel is None:
        return UNKNOWN, UNKNOWN
    try:
        own_checkout = (Path(toplevel) / "src" / "seisforge").resolve() == package_dir
    except OSError:
        own_checkout = False
    if not own_checkout:
        return UNKNOWN, UNKNOWN
    commit = _run_git(package_dir, "rev-parse", "HEAD")
    if commit is None:
        return UNKNOWN, UNKNOWN
    status = _run_git(package_dir, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        dirty = UNKNOWN
    else:
        dirty = "true" if status else "false"
    return commit, dirty


def _run_git(cwd: Path, *args: str) -> str | None:
    """Stripped stdout of ``git <args>`` in ``cwd``, or ``None`` on any failure."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


__all__ = ["DEPENDENCIES", "NOT_INSTALLED", "UNKNOWN", "software_provenance"]
