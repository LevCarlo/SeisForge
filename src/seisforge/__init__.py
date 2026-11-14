"""SeisForge: tools for seismic data processing and analysis."""

from importlib.metadata import PackageNotFoundError, version

# The version number lives only in pyproject.toml. A source tree that was
# never installed has no distribution metadata; report a placeholder that
# cannot be mistaken for a release.
try:
    __version__ = version("seisforge")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
