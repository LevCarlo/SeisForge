"""AFTAN component handling: phase constants per component, branch and run mode.

Inputs and the known answer come from `aftan_component_cases` (diffuse-field
Rayleigh CCFs rotated with the package rotation; see that module and
`seisforge.ant.synthetic`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aftan_component_cases import RESULT_KEYS, baseline_cases, run, write_ccfs

BASELINE = Path(__file__).parent / "data" / "aftan_component_baseline.npz"

# (mode, component) whose numbers must stay bit for bit as in the baseline.
UNCHANGED = {
    (mode, component)
    for mode in ("pair", "station")
    for component in ("ZZ", "ZR", "RZ", "RR", "TT")
}


@pytest.fixture(scope="module")
def ccf_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ccf")
    write_ccfs(directory)
    return directory


@pytest.mark.parametrize(
    "mode, component, branch",
    [case for case in baseline_cases() if case[:2] in UNCHANGED],
)
def test_unchanged_components_match_baseline_bitwise(mode, component, branch, ccf_dir, tmp_path):
    baseline = np.load(BASELINE)
    results = run(mode, component, branch, ccf_dir, tmp_path)
    assert results
    for result in results:
        for key in RESULT_KEYS:
            np.testing.assert_array_equal(
                np.asarray(getattr(result, key)),
                baseline[f"{mode}/{component}/{result.branch}/{key}"],
                err_msg=f"{mode} {component} {result.branch} {key}",
            )
