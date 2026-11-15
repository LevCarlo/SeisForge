"""Bit-for-bit regression of the Metropolis sampler against a stored baseline.

The baseline (`tests/data/sampler_baseline.npz`, written by
`tools/make_sampler_baseline.py` from commit 91b7cac) pins the chains drawn by
the plain, global-jump and tempered samplers. Exact equality is the point:
with adaptation off the sampler must consume the same random numbers and do
the same arithmetic as before. The comparison assumes the same numpy
generator stream (PCG64, stable across numpy releases so far) and IEEE double
arithmetic; a platform whose ``exp``/``log`` differ in the last bit could
break it, in which case the baseline is regenerated there, not loosened.
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
        np.testing.assert_array_equal(arrays[key], stored[f"{name}/{key}"], err_msg=f"{name}/{key}")
