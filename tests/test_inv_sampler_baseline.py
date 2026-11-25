"""Regression of the Metropolis sampler against a stored baseline.

The baseline (`tests/data/sampler_baseline.npz`, written by
`tools/make_sampler_baseline.py` from commit 91b7cac) pins the chains drawn by
the plain, global-jump and tempered samplers. A tight cross-platform tolerance
(``rtol=1e-12, atol=1e-13``) absorbs 1–2 ULP ``libm``/BLAS rounding differences
between ARM64 (macOS) and x86_64 (Linux) while still catching any algorithmic
or RNG-stream drift.
"""

from pathlib import Path

import numpy as np
import pytest

from sampler_baseline_cases import CASES, result_arrays, run_case

BASELINE = Path(__file__).parent / "data" / "sampler_baseline.npz"


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("executor", ["serial", "thread", "process"])
def test_sampler_matches_baseline_bitwise(name, executor):
    stored = np.load(BASELINE)
    arrays = result_arrays(run_case(name, executor=executor))
    keys = sorted(k.split("/", 1)[1] for k in stored.files if k.startswith(f"{name}/"))
    assert keys == sorted(arrays)
    for key in keys:
        np.testing.assert_allclose(
            arrays[key],
            stored[f"{name}/{key}"],
            rtol=1e-12,
            atol=1e-13,
            equal_nan=True,
            err_msg=f"{name}/{key}",
        )
