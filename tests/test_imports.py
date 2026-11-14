from importlib import metadata

import seisforge


def test_package_version_comes_from_distribution_metadata():
    # pyproject.toml is the single source; __version__ reads the installed
    # distribution metadata.
    assert seisforge.__version__ == metadata.version("seisforge")
