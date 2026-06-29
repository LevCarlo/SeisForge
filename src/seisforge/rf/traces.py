"""Receiver-function trace and stream containers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from obspy import Stream, Trace, read
from scipy.stats import f as f_dist


def _default_window(window, default):
    return tuple(default) if window is None else tuple(window)


def _safe_std(data: np.ndarray) -> float:
    std = float(np.std(data))
    return std if std > 0 else 1.0


@dataclass
class RFtrace:
    """A receiver-function trace with commonly used SAC header metadata."""

    trace: Trace
    file: str | None = None
    slowness: float | None = None
    baz: float | None = None

    def __init__(
        self,
        trace: Trace | None = None,
        file: str | None = None,
        slowness: float | None = None,
        baz: float | None = None,
        slowness_key: str = "user0",
        baz_key: str = "baz",
    ):
        if trace is not None:
            self.trace = trace
            self.file = file
        elif file is not None:
            self.trace = read(file)[0]
            self.file = file
        else:
            raise ValueError("Must provide either `trace` or `file`.")

        self.slowness = slowness if slowness is not None else self._get_header(slowness_key)
        self.baz = baz if baz is not None else self._get_header(baz_key)

    def _get_header(self, key):
        try:
            return getattr(self.trace.stats.sac, key)
        except AttributeError:
            return None


class RFstream:
    """A stream of receiver functions with QC selection methods."""

    def __init__(self, files=None, st: Stream | None = None, qc_metrics=None):
        self.files = list(files) if files is not None else []
        self.st = st
        self.qc_metrics = list(qc_metrics) if qc_metrics is not None else []

    def __len__(self):
        return 0 if self.st is None else len(self.st)

    def load_data(self):
        """Load SAC files into an ObsPy stream."""
        st = Stream()
        for file in self.files:
            try:
                st += read(file)
            except Exception as exc:
                logging.warning("Error reading %s: %s", file, exc)
        self.st = st
        return self

    def copy(self):
        return RFstream(
            files=self.files.copy(),
            st=self.st.copy() if self.st else None,
            qc_metrics=self.qc_metrics.copy(),
        )

    def _require_loaded(self):
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        if len(self.st) == 0:
            raise ValueError("Stream is empty.")

    def _time_axis(self):
        self._require_loaded()
        tr0 = self.st[0]
        b = tr0.stats.sac.b
        e = tr0.stats.sac.e
        npts = tr0.stats.npts
        return np.linspace(b, e, npts)

    def _window_mask(self, window):
        t = self._time_axis()
        mask = (t >= window[0]) & (t <= window[1])
        if not np.any(mask):
            raise ValueError(f"Window {window} does not overlap trace time axis.")
        return t, mask

    def _select(self, indices, metrics):
        selected = self.copy()
        selected.st = Stream([selected.st[i] for i in indices])
        if len(selected.files) == len(self.st):
            selected.files = [selected.files[i] for i in indices]
        else:
            selected.files = []
        selected.qc_metrics = [metrics[i] for i in indices]
        return selected

    def SNR_select(self, threshold=5, reverse=False, sac_head="user2"):
        self._require_loaded()
        metrics = [tr.stats.sac[sac_head] for tr in self.st]
        keep = [i for i, value in enumerate(metrics) if value >= threshold]
        if reverse:
            keep = [i for i in range(len(self.st)) if i not in keep]
        return self._select(keep, metrics)

    def slowness_select(
        self,
        slow_min=0.04,
        slow_max=0.1,
        reverse=False,
        sac_head="user0",
    ):
        self._require_loaded()
        metrics = [tr.stats.sac[sac_head] for tr in self.st]
        keep = [i for i, value in enumerate(metrics) if slow_min <= value <= slow_max]
        if reverse:
            keep = [i for i in range(len(self.st)) if i not in keep]
        return self._select(keep, metrics)

    def P_amp_select(self, tmin=-2, tmax=2, window=None, reverse=False):
        window = _default_window(window, (-5, 30))
        t, t_idx = self._window_mask(window)

        metrics = []
        keep = []
        for i, tr in enumerate(self.st):
            max_amp_idx = np.argmax(np.abs(tr.data[t_idx]))
            max_amp_time = t[t_idx][max_amp_idx]
            max_amp_value = tr.data[t_idx][max_amp_idx]
            metrics.append([max_amp_time, max_amp_value])
            if tmin <= max_amp_time <= tmax and max_amp_value > 0:
                keep.append(i)

        if reverse:
            keep = [i for i in range(len(self.st)) if i not in keep]
        return self._select(keep, metrics)

    def late_rms_select(
        self,
        p_window=None,
        late_window=None,
        qmax=0.30,
        reverse=False,
    ):
        p_window = _default_window(p_window, (-2.0, 2.0))
        late_window = _default_window(late_window, (2.0, 10.0))
        _, p_idx = self._window_mask(p_window)
        _, late_idx = self._window_mask(late_window)

        metrics = []
        keep = []
        for i, tr in enumerate(self.st):
            p_amp = np.max(tr.data[p_idx])
            late_rms = np.sqrt(np.mean(tr.data[late_idx] ** 2))
            q = late_rms / p_amp if p_amp > 0 else np.inf
            metrics.append([q, p_amp, late_rms])
            if q <= qmax:
                keep.append(i)

        if reverse:
            keep = [i for i in range(len(self.st)) if i not in keep]
        return self._select(keep, metrics)

    def MAD_select(self, threshold=2.5, window=None, reverse=False):
        window = _default_window(window, (-5, 30))
        _, t_idx = self._window_mask(window)

        var_rf = np.array([np.var(tr.data[t_idx]) for tr in self.st])
        med = np.median(var_rf)
        mad = 1.4826 * np.median(np.abs(var_rf - med))
        mad = mad if mad > 0 else 1e-12
        metrics = np.abs((var_rf - med) / mad).tolist()

        if reverse:
            keep = [i for i, value in enumerate(metrics) if value > threshold]
        else:
            keep = [i for i, value in enumerate(metrics) if value <= threshold]
        return self._select(keep, metrics)

    def f_test_select(self, threshold=0.05, window=None, reverse=False):
        window = _default_window(window, (-5, 30))
        _, t_idx = self._window_mask(window)

        if len(self.st) < 2:
            metrics = [1.0 for _ in self.st]
            keep = list(range(len(self.st))) if not reverse else []
            return self._select(keep, metrics)

        rf_data = np.array([tr.data[t_idx] for tr in self.st])
        ref_rf = np.mean(rf_data, axis=0)
        res_all = (rf_data - ref_rf).reshape(-1)

        metrics = []
        for i in range(rf_data.shape[0]):
            reduced_rf = np.delete(rf_data, i, axis=0)
            ref_reduced = np.mean(reduced_rf, axis=0)
            res_reduced = (reduced_rf - ref_reduced).reshape(-1)
            metrics.append(_f_test(res_all, 1, res_reduced, 1))

        if reverse:
            keep = [i for i, value in enumerate(metrics) if value < threshold]
        else:
            keep = [i for i, value in enumerate(metrics) if value >= threshold]
        return self._select(keep, metrics)

    def CC_select(self, threshold=0.4, window=None, reverse=False):
        window = _default_window(window, (-5, 30))
        _, t_idx = self._window_mask(window)

        rf_data = np.array([tr.data[t_idx] for tr in self.st])
        ave_rf = np.mean(rf_data, axis=0)
        ave_rf = (ave_rf - np.mean(ave_rf)) / _safe_std(ave_rf)

        metrics = []
        for rf_i in rf_data:
            rf_i = (rf_i - np.mean(rf_i)) / _safe_std(rf_i)
            metrics.append(float(np.corrcoef(ave_rf, rf_i)[0, 1]))

        if reverse:
            keep = [i for i, value in enumerate(metrics) if value < threshold]
        else:
            keep = [i for i, value in enumerate(metrics) if value >= threshold]
        return self._select(keep, metrics)

    def plot(self, by="slow", window=None, save=False, save_path=None, show=False, **kwargs):
        from seisforge.rf.plotting import plot_rf_stream

        window = _default_window(window, (-5, 30))
        return plot_rf_stream(
            self,
            by=by,
            window=window,
            save=save,
            save_path=save_path,
            show=show,
        )


def _f_test(res1, pars1, res2, pars2):
    dof1 = len(res1) - pars1
    dof2 = len(res2) - pars2
    if dof1 <= 0 or dof2 <= 0:
        return 1.0

    ea_1 = np.sum(res1**2)
    ea_2 = np.sum(res2**2)
    if ea_2 == 0:
        return 1.0

    f_obs = (ea_1 / dof1) / (ea_2 / dof2)
    return 1 - (f_dist.cdf(f_obs, dof1, dof2) - f_dist.cdf(1 / f_obs, dof1, dof2))
