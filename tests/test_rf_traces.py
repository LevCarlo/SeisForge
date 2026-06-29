import numpy as np
from obspy import Stream, Trace

from seisforge.rf.traces import RFstream


def _trace(data, *, slow=0.06, snr=5.0, b=-5.0, e=5.0):
    tr = Trace(data=np.asarray(data, dtype=float))
    tr.stats.sac = {
        "b": b,
        "e": e,
        "user0": slow,
        "user2": snr,
        "baz": 120.0,
    }
    return tr


def test_snr_select_keeps_files_aligned():
    st = Stream([
        _trace(np.ones(11), snr=2.0),
        _trace(np.ones(11), snr=8.0),
    ])
    rf = RFstream(files=["low.sac", "high.sac"], st=st)

    selected = rf.SNR_select(threshold=5.0)

    assert len(selected) == 1
    assert selected.files == ["high.sac"]
    assert selected.qc_metrics == [8.0]


def test_slowness_select_reverse_returns_outside_range():
    st = Stream([
        _trace(np.ones(11), slow=0.03),
        _trace(np.ones(11), slow=0.07),
        _trace(np.ones(11), slow=0.12),
    ])
    rf = RFstream(st=st)

    selected = rf.slowness_select(slow_min=0.04, slow_max=0.10, reverse=True)

    assert len(selected) == 2
    assert selected.qc_metrics == [0.03, 0.12]


def test_p_amp_select_requires_positive_peak_in_window():
    positive = np.zeros(11)
    positive[5] = 1.0
    negative = np.zeros(11)
    negative[5] = -1.0
    late = np.zeros(11)
    late[9] = 1.0
    rf = RFstream(st=Stream([_trace(positive), _trace(negative), _trace(late)]))

    selected = rf.P_amp_select(tmin=-1.0, tmax=1.0, window=[-5, 5])

    assert len(selected) == 1
    assert selected.qc_metrics == [[0.0, 1.0]]


def test_f_test_select_keeps_single_trace_by_default():
    rf = RFstream(st=Stream([_trace(np.ones(11))]))

    selected = rf.f_test_select()

    assert len(selected) == 1
    assert selected.qc_metrics == [1.0]
