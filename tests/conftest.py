"""Shared test helpers."""

from __future__ import annotations

import re

import pytest

from seisforge.provenance import software_provenance


@pytest.fixture
def assert_provenance():
    """Check that an output records every software-provenance field.

    Accepts a mapping (NetCDF/YAML attrs) or text (``.dat`` header, log).
    Text must contain ``key: value`` or ``key=value`` with a non-empty value.
    """

    def check(record) -> None:
        keys = software_provenance().keys()
        if isinstance(record, str):
            for key in keys:
                assert re.search(rf"\b{key}(: |=)\S", record), key
            return
        for key in keys:
            assert key in record, key
            assert isinstance(record[key], str) and record[key].strip(), key

    return check
