"""Trace branch selection for one-sided and symmetric CCFs."""

from __future__ import annotations

import numpy as np
from obspy import Trace

from .io import _sac_float, _trace_times
from .models import BranchTrace


def _prepare_branch_traces(trace: Trace, branch: str) -> list[BranchTrace]:
    if branch == "positive":
        return [BranchTrace("positive", _positive_branch_trace(trace))]
    if branch == "negative":
        if not _has_negative_lag(trace):
            return [
                BranchTrace(
                    "positive",
                    _positive_branch_trace(trace),
                    ("requested negative branch but trace is not symmetric; using positive branch",),
                )
            ]
        return [BranchTrace("negative", _negative_branch_trace(trace))]
    if branch == "both":
        if not _has_two_sided_lags(trace):
            return [
                BranchTrace(
                    "positive",
                    _positive_branch_trace(trace),
                    ("requested both branches but trace is not symmetric; using positive branch",),
                )
            ]
        return [
            BranchTrace("positive", _positive_branch_trace(trace)),
            BranchTrace("negative", _negative_branch_trace(trace)),
        ]
    if branch == "stack":
        if not _has_two_sided_lags(trace):
            return [
                BranchTrace(
                    "positive",
                    _positive_branch_trace(trace),
                    ("requested stack branch but trace is not symmetric; using positive branch",),
                )
            ]
        return [BranchTrace("stack", _stack_branch_trace(trace))]
    raise ValueError(f"Unsupported branch: {branch}")


def _positive_branch_trace(trace: Trace) -> Trace:
    times = _trace_times(trace)
    mask = times >= 0
    if not np.any(mask):
        raise ValueError("Trace does not contain positive lags.")
    return _branch_trace(trace, trace.data[mask], max(float(times[mask][0]), 0.0))


def _negative_branch_trace(trace: Trace) -> Trace:
    times = _trace_times(trace)
    mask = times <= 0
    if not np.any(mask):
        raise ValueError("Trace does not contain negative lags.")
    data = trace.data[mask][::-1]
    branch_times = np.abs(times[mask][::-1])
    order = np.argsort(branch_times)
    return _branch_trace(trace, data[order], max(float(branch_times[order][0]), 0.0))


def _stack_branch_trace(trace: Trace) -> Trace:
    positive = _positive_branch_trace(trace)
    negative = _negative_branch_trace(trace)
    npts = min(positive.stats.npts, negative.stats.npts)
    data = (positive.data[:npts] + negative.data[:npts]) / 2.0
    b = max(
        _sac_float(getattr(positive.stats, "sac", None), "b") or 0.0,
        _sac_float(getattr(negative.stats, "sac", None), "b") or 0.0,
    )
    return _branch_trace(trace, data, b)


def _branch_trace(trace: Trace, data: np.ndarray, b: float) -> Trace:
    branch = Trace(data=np.asarray(data, dtype=trace.data.dtype))
    branch.stats = trace.stats.copy()
    branch.stats.npts = branch.data.size
    if hasattr(branch.stats, "sac"):
        branch.stats.sac.b = float(b)
        branch.stats.sac.e = float(b + (branch.data.size - 1) * branch.stats.delta)
        branch.stats.sac.npts = branch.data.size
    return branch


def _has_negative_lag(trace: Trace) -> bool:
    return bool(np.any(_trace_times(trace) < 0))


def _has_two_sided_lags(trace: Trace) -> bool:
    times = _trace_times(trace)
    return bool(np.any(times < 0) and np.any(times >= 0))


def _split_or_symmetrize(trace: Trace, nlag_out: int = 1) -> list[Trace]:
    if nlag_out != 1:
        raise ValueError("Only symmetric one-lag output is supported for now.")
    sym = Trace()
    sym.data = _symmetrize_data(trace.data)
    sym.stats = trace.stats.copy()
    sym.stats.npts = sym.data.size
    sym.stats.starttime += _sac_float(trace.stats.sac, "e") or 0.0
    if hasattr(sym.stats, "sac"):
        sym.stats.sac.b = 0.0
        sym.stats.sac.npts = sym.data.size
    return [sym]


def _symmetrize_data(data: np.ndarray) -> np.ndarray:
    npts = data.size
    if npts % 2 == 0:
        left = npts // 2 - 1
        right = left + 1
        return data[left::-1] + data[right:]
    half = npts // 2
    return data[half::-1] + data[half:]
