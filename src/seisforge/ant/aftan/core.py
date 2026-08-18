"""Core FTAN and PMF signal-processing routines."""

from __future__ import annotations

from math import tau
from pathlib import Path

import numpy as np
from obspy import Trace
from scipy import interpolate, signal
from scipy.fft import ifft

from seisforge.ant.filters import (
    GaussianFTANFilter,
    PhaseMatchedFilter,
    gaussian_alpha_aftan_scaling,
    gaussian_alpha_from_distance,
    gaussian_alpha_pyftan_constant,
)

from .io import _trace_distance_km, _trace_times
from .models import (
    AFTANAlphaConfig,
    AFTANConfig,
    AFTANPhaseCycleConfig,
    AFTANPeriodSamplingConfig,
    AFTANSNRConfig,
)
from .snr import _aftan_diagram_snr, _window_snr


_AFTAN_KEYS = (
    "period",
    "amplitude",
    "phase",
    "group_velocity",
    "phase_derivative",
    "hilbert_phase",
    "hilbert_phase_derivative",
    "hilbert_period",
)


class AFTANMeasurement:
    """A compact, config-local Python AFTAN implementation."""

    def __init__(self, trace: Trace, path: Path, config: AFTANConfig):
        self.trace = trace.copy()
        self.path = Path(path)
        self.config = config
        self.trace.taper(0.05)
        self.distance_km = _trace_distance_km(self.trace)
        self.basic_alpha = _resolve_alpha(config.basic.alpha, self.distance_km)
        self.basic_alpha_mode = config.basic.alpha.mode
        self.pmf_alpha = _resolve_alpha(config.pmf.alpha, self.distance_km)
        self.pmf_alpha_mode = config.pmf.alpha.mode
        self.alpha = self.basic_alpha
        self.alpha_mode = self.basic_alpha_mode
        self.min_period = config.min_period
        self.max_period = _effective_max_period(config, self.distance_km)
        if self.max_period <= self.min_period:
            raise ValueError(
                f"{self.path.name}: max_period {self.max_period:.3f} <= "
                f"min_period {self.min_period:.3f}"
            )
        self.target_periods = _build_period_grid(
            config.period_sampling,
            self.min_period,
            self.max_period,
        )
        self.nfft = int(2 ** np.ceil(np.log2(self.trace.stats.npts)))
        self.time = _trace_times(self.trace)
        self.pred_period, self.pred_velocity = _load_prediction(config)

    @property
    def nfreq(self) -> int:
        return self.target_periods.size

    @property
    def omega0(self) -> np.ndarray:
        return tau / self.target_periods

    @property
    def periods0(self) -> np.ndarray:
        return self.target_periods

    def measure(self) -> dict[str, np.ndarray]:
        return self._measure_signal(
            self.trace.data,
            self.target_periods,
            self.basic_alpha,
            self.config.basic.trig_threshold,
            self.config.basic.jump_points,
            snr_signal_data=self.trace.data,
        )

    def measure_pmf(self, basic: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        period_min, period_max, raw_period_min, raw_period_max = _pmf_period_bounds(
            basic["period"],
            self.config.min_period,
            self.config.max_period,
        )
        target_periods = _build_period_grid(
            self.config.period_sampling,
            period_min,
            period_max,
        )
        cleaned = PhaseMatchedFilter(
            dt=self.trace.stats.delta,
            distance_km=self.distance_km,
            periods=basic["period"],
            group_velocity=basic["group_velocity"],
            alpha=self.pmf_alpha,
            nalpha=self.config.pmf.nalpha,
            min_half_length=self.config.pmf.min_half_length,
            amplitude_ratio=self.config.pmf.amplitude_ratio,
            window_factor=self.config.pmf.window_factor,
            nfft=self.nfft,
        ).apply(self.trace.data)
        measured = self._measure_signal(
            cleaned,
            target_periods,
            self.pmf_alpha,
            self.config.pmf.trig_threshold,
            self.config.pmf.jump_points,
            snr_signal_data=self.trace.data,
        )
        measured["signal"] = cleaned
        measured["ftan_alpha"] = np.asarray(self.pmf_alpha)
        measured["ftan_alpha_mode"] = np.asarray(self.pmf_alpha_mode)
        measured["pmf_period_min"] = np.asarray(period_min)
        measured["pmf_period_max"] = np.asarray(period_max)
        measured["pmf_raw_period_min"] = np.asarray(raw_period_min)
        measured["pmf_raw_period_max"] = np.asarray(raw_period_max)
        return measured

    def _measure_signal(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
        trig_threshold: float,
        jump_points: int,
        snr_signal_data: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        par_all, raw = self._raw_ftan(signal_data, target_periods, alpha)
        corrected = self._jump_corrected(
            par_all,
            raw,
            trig_threshold,
            jump_points,
        )
        period = corrected["period"]
        group_velocity = corrected["group_velocity"]
        phase_velocity, phase_cycle_diagnostics = self._phase_velocity(
            period,
            group_velocity,
            corrected["phase"],
        )
        snr = self._spectral_snr(
            signal_data,
            period,
            group_velocity,
            self.config.snr,
            alpha,
            target_periods,
            snr_signal_data=snr_signal_data,
        )
        measured = {
            "target_period": corrected["center_period"],
            "initial_target_period_count": np.asarray(target_periods.size),
            "ftan_alpha": np.asarray(alpha),
            "period": period,
            "group_velocity": group_velocity,
            "phase_velocity": phase_velocity,
            "phase": corrected["phase"],
            "amplitude": corrected["amplitude"],
            "phase_derivative": corrected["phase_derivative"],
            "hilbert_phase": corrected["hilbert_phase"],
            "hilbert_phase_derivative": corrected["hilbert_phase_derivative"],
            "hilbert_period": corrected["hilbert_period"],
            "snr": snr,
        }
        measured.update(phase_cycle_diagnostics)
        return measured

    def _raw_ftan(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
        ft = self._ftan_complex(signal_data, target_periods, alpha)
        amplitude = np.abs(ft)
        phase = np.angle(ft)
        hilbert_phase = self._hilbert_phase(ft)
        out = {key: np.zeros(target_periods.size) for key in _AFTAN_KEYS}
        par_all = []

        for idx, (omega, amp_col, phase_col, hilbert_phase_col) in enumerate(
            zip(tau / target_periods, amplitude.T, phase.T, hilbert_phase.T)
        ):
            peak_indices = signal.find_peaks(amp_col[: self.trace.stats.npts])[0]
            if peak_indices.size == 0:
                peak_indices = np.array([self._fallback_peak_index(amp_col)])

            par = {key: np.zeros(peak_indices.size) for key in _AFTAN_KEYS}
            dphase, par["amplitude"], par["phase"], dt_offset, peak_indices = self._interp(
                amp_col,
                phase_col,
                peak_indices,
                omega,
            )
            hilbert_dphase, par["hilbert_phase"] = self._interp_phase_only(
                hilbert_phase_col,
                peak_indices,
                omega,
                dt_offset,
            )
            inst_period = tau / (dphase / self.trace.stats.delta)
            hilbert_period = tau / (hilbert_dphase / self.trace.stats.delta)
            t_max = self.time[0] + (peak_indices + dt_offset) * self.trace.stats.delta
            group_velocity = self.distance_km / t_max
            valid = np.isfinite(group_velocity)
            if not np.any(valid):
                valid = np.ones_like(group_velocity, dtype=bool)
            par["period"] = inst_period
            par["group_velocity"] = group_velocity
            par["phase_derivative"] = dphase / self.trace.stats.delta
            par["hilbert_phase_derivative"] = hilbert_dphase / self.trace.stats.delta
            par["hilbert_period"] = hilbert_period
            par_all.append(par)

            # Match AFTAN/PyAFTAN behavior: group-velocity picking follows the
            # envelope ridge. Instantaneous-period stability is reported as QC,
            # but should not reject envelope peaks before the group pick.
            score = np.where(valid, par["amplitude"], -np.inf)
            imax = int(np.argmax(score))
            for key in _AFTAN_KEYS:
                out[key][idx] = par[key][imax]

        out["center_period"] = target_periods
        return par_all, out

    def energy_map_dataset(
        self,
        measured: dict[str, np.ndarray],
        *,
        stage: str,
        source_file: Path,
        branch: str,
        velocity_count: int,
        normalize: bool,
        signal_data: np.ndarray | None = None,
    ):
        """Build an xarray dataset for the FTAN group-velocity energy map."""
        import xarray as xr

        if signal_data is None:
            signal_data = self.trace.data
        target_period = measured.get("energy_target_period", measured["target_period"])
        alpha = float(np.asarray(measured.get("ftan_alpha", self.basic_alpha)))
        amplitude = np.abs(
            self._ftan_complex(
                signal_data,
                np.asarray(target_period, dtype=float),
                alpha,
            )
        )
        amplitude = amplitude[: self.trace.stats.npts, :].T
        velocity = np.linspace(
            self.config.velocity_min,
            self.config.velocity_max,
            int(velocity_count),
        )
        map_amplitude = self._amplitude_time_to_velocity(amplitude, velocity)
        picked_target_period = measured["target_period"]
        data_vars = {
            "amplitude": (("target_period", "velocity"), map_amplitude),
            "picked_group_velocity": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["group_velocity"],
                ),
            ),
            "picked_instant_period": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["period"],
                ),
            ),
            "picked_amplitude": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["amplitude"],
                ),
            ),
            "picked_snr": (
                ("target_period",),
                _align_to_period_grid(
                    target_period,
                    picked_target_period,
                    measured["snr"],
                ),
            ),
        }
        if self.config.debug:
            data_vars.update(
                picked_hilbert_instant_period=(
                    ("target_period",),
                    _align_to_period_grid(
                        target_period,
                        picked_target_period,
                        measured["hilbert_period"],
                    ),
                ),
                picked_instant_period_delta=(
                    ("target_period",),
                    _align_to_period_grid(
                        target_period,
                        picked_target_period,
                        measured["hilbert_period"] - measured["period"],
                    ),
                ),
            )
        if normalize:
            scale = np.nanmax(map_amplitude, axis=1, keepdims=True)
            scale[~np.isfinite(scale) | (scale == 0)] = 1.0
            data_vars["normalized_amplitude"] = (
                ("target_period", "velocity"),
                map_amplitude / scale,
            )
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "target_period": target_period,
                "velocity": velocity,
            },
            attrs={
                "stage": stage,
                "branch": branch,
                "input_file": str(source_file),
                "distance_km": float(self.distance_km),
                "alpha": float(alpha),
                "alpha_mode": (
                    self.basic_alpha_mode if stage == "basic" else self.pmf_alpha_mode
                ),
                "velocity_min": float(self.config.velocity_min),
                "velocity_max": float(self.config.velocity_max),
                "period_min": float(np.nanmin(target_period)),
                "period_max": float(np.nanmax(target_period)),
                "period_sampling_mode": self.config.period_sampling.mode,
                "signal_source": "raw_waveform" if stage == "basic" else "pmf_clean_waveform",
                "debug": bool(self.config.debug),
            },
        )
        return dataset

    def phase_velocity_map_dataset(
        self,
        measured: dict[str, np.ndarray],
        *,
        stage: str,
        source_file: Path,
        branch: str,
        velocity_count: int,
        normalize: bool,
        cycle_count: int,
    ):
        """Build an xarray dataset that shows phase-velocity cycle candidates."""
        import xarray as xr

        alpha = self.basic_alpha if stage == "basic" else self.pmf_alpha
        alpha_mode = self.basic_alpha_mode if stage == "basic" else self.pmf_alpha_mode
        phase_velocity = np.linspace(
            self.config.velocity_min,
            self.config.velocity_max,
            int(velocity_count),
        )
        candidate_amplitude = _phase_velocity_candidate_map(
            measured["period"],
            measured["group_velocity"],
            measured["phase"],
            measured["phase_velocity"],
            measured["amplitude"],
            self.distance_km,
            phase_velocity,
            cycle_count,
            normalize,
        )
        return xr.Dataset(
            data_vars={
                "phase_candidate_amplitude": (
                    ("target_period", "phase_velocity"),
                    candidate_amplitude,
                ),
                "picked_phase_velocity": (
                    ("target_period",),
                    measured["phase_velocity"],
                    {
                        "long_name": "cycle-corrected phase velocity",
                        "units": "km/s",
                    },
                ),
                "picked_phase_velocity_uncorrected": (
                    ("target_period",),
                    measured["phase_velocity_uncorrected"],
                    {
                        "long_name": (
                            "phase velocity before global cycle correction"
                        ),
                        "units": "km/s",
                    },
                ),
                "reference_phase_velocity": (
                    ("target_period",),
                    measured["phase_reference_velocity"],
                    {
                        "long_name": (
                            "reference phase velocity evaluated at picked "
                            "instant period"
                        ),
                        "units": "km/s",
                    },
                ),
                "phase_cycle_residual_cycles_before": (
                    ("target_period",),
                    measured["phase_cycle_residual_cycles_before"],
                    {
                        "long_name": (
                            "reference travel-time residual before correction"
                        ),
                        "units": "cycles",
                    },
                ),
                "phase_cycle_residual_cycles_after": (
                    ("target_period",),
                    measured["phase_cycle_residual_cycles_after"],
                    {
                        "long_name": (
                            "reference travel-time residual after correction"
                        ),
                        "units": "cycles",
                    },
                ),
                "picked_group_velocity": (
                    ("target_period",),
                    measured["group_velocity"],
                ),
                "picked_instant_period": (
                    ("target_period",),
                    measured["period"],
                ),
                "picked_amplitude": (
                    ("target_period",),
                    measured["amplitude"],
                ),
                "phase_cycle_shift": (
                    (),
                    measured["phase_cycle_shift"],
                    {
                        "long_name": (
                            "global integer cycle shift applied to phase slowness"
                        )
                    },
                ),
                "phase_cycle_reference_score_cycles": (
                    (),
                    measured["phase_cycle_reference_score_cycles"],
                    {
                        "long_name": (
                            "median absolute reference residual after correction"
                        ),
                        "units": "cycles",
                    },
                ),
                "phase_cycle_candidate_score_gap_cycles": (
                    (),
                    measured["phase_cycle_candidate_score_gap_cycles"],
                    {
                        "long_name": (
                            "score gap between best and second-best cycle shifts"
                        ),
                        "units": "cycles",
                    },
                ),
                "phase_cycle_reference_rmse_km_s_before": (
                    (),
                    measured["phase_cycle_reference_rmse_km_s_before"],
                    {
                        "long_name": (
                            "phase-velocity RMSE to reference before correction"
                        ),
                        "units": "km/s",
                    },
                ),
                "phase_cycle_reference_rmse_km_s_after": (
                    (),
                    measured["phase_cycle_reference_rmse_km_s_after"],
                    {
                        "long_name": (
                            "phase-velocity RMSE to reference after correction"
                        ),
                        "units": "km/s",
                    },
                ),
                "phase_cycle_valid_count": (
                    (),
                    measured["phase_cycle_valid_count"],
                    {"long_name": "period count used for global cycle selection"},
                ),
                "phase_cycle_qc_pass": (
                    (),
                    measured["phase_cycle_qc_pass"],
                    {"long_name": "one when global phase-cycle diagnostics pass"},
                ),
                "phase_cycle_search_limit_reached": (
                    (),
                    measured["phase_cycle_search_limit_reached"],
                    {"long_name": "one when the selected shift reaches max_shift"},
                ),
            },
            coords={
                "target_period": measured["target_period"],
                "phase_velocity": phase_velocity,
            },
            attrs={
                "stage": stage,
                "branch": branch,
                "input_file": str(source_file),
                "distance_km": float(self.distance_km),
                "alpha": float(alpha),
                "alpha_mode": alpha_mode,
                "velocity_min": float(self.config.velocity_min),
                "velocity_max": float(self.config.velocity_max),
                "period_min": float(np.nanmin(measured["target_period"])),
                "period_max": float(np.nanmax(measured["target_period"])),
                "period_sampling_mode": self.config.period_sampling.mode,
                "phase_cycle_count": int(cycle_count),
                "phase_cycle_selection_method": (
                    "global minimum median absolute reference travel-time residual"
                ),
                "phase_cycle_status": str(measured["phase_cycle_status"].item()),
                "phase_cycle_enabled": bool(self.config.phase_cycle.enabled),
                "phase_cycle_max_shift": int(self.config.phase_cycle.max_shift),
                "phase_cycle_max_reference_residual_cycles": float(
                    self.config.phase_cycle.max_reference_residual_cycles
                ),
                "phase_cycle_min_candidate_score_gap_cycles": float(
                    self.config.phase_cycle.min_candidate_score_gap_cycles
                ),
                "description": "phase-velocity cycle-candidate diagnostic map",
            },
        )

    def _amplitude_time_to_velocity(
        self,
        amplitude: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        times = self.time[: self.trace.stats.npts]
        out = np.full((amplitude.shape[0], velocity.size), np.nan)
        for idx, amp in enumerate(amplitude):
            query_time = self.distance_km / velocity
            out[idx] = np.interp(query_time, times, amp, left=np.nan, right=np.nan)
        return out

    def _ftan_complex(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        spc_mat = self._multiple_filter(signal_data, target_periods, alpha)
        return ifft(spc_mat, axis=0)

    def _hilbert_phase(self, ft: np.ndarray) -> np.ndarray:
        real_filtered = ft.real[: self.trace.stats.npts, :]
        analytic = signal.hilbert(real_filtered, axis=0)
        phase = np.zeros_like(ft.real)
        phase[: self.trace.stats.npts, :] = np.angle(analytic)
        return phase

    def _multiple_filter(
        self,
        signal_data: np.ndarray,
        target_periods: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        return GaussianFTANFilter(
            self.trace.stats.delta,
            target_periods,
            alpha,
            nfft=self.nfft,
        ).spectra(signal_data)

    def _velocity_window_indices(self, indices: np.ndarray) -> np.ndarray:
        time = self.time[0] + indices * self.trace.stats.delta
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = self.distance_km / time
        mask = np.isfinite(velocity)
        mask &= velocity >= self.config.velocity_min
        mask &= velocity <= self.config.velocity_max
        return indices[mask]

    def _fallback_peak_index(self, amplitude: np.ndarray) -> int:
        npts = self.trace.stats.npts
        time = self.time[:npts]
        tmin = self.distance_km / self.config.velocity_max
        tmax = self.distance_km / self.config.velocity_min
        mask = (time >= tmin) & (time <= tmax)
        if np.any(mask):
            candidates = np.where(mask)[0]
            return int(candidates[np.argmax(amplitude[candidates])])
        return int(np.argmax(amplitude[:npts]))

    def _interp(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        indices: np.ndarray,
        omega: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = indices[(indices > 0) & (indices < self.trace.stats.npts - 1)]
        if indices.size == 0:
            indices = np.array([min(max(1, self.trace.stats.npts // 2), self.trace.stats.npts - 2)])

        left = indices - 1
        right = indices + 1
        denom = amplitude[left] + amplitude[right] - 2 * amplitude[indices]
        denom[denom == 0] = np.nan
        dt_offset = (amplitude[left] - amplitude[right]) / denom / 2
        dt_offset[np.isnan(denom)] = 0

        amp_i = _parabola(
            dt_offset,
            amplitude[left],
            amplitude[indices],
            amplitude[right],
        )
        phase_left = phase[left]
        phase_mid = phase[indices].copy()
        phase_right = phase[right].copy()
        delta = self.trace.stats.delta
        phase_mid -= tau * np.around((phase_mid - phase_left - omega * delta) / tau)
        phase_right -= tau * np.around((phase_right - phase_mid - omega * delta) / tau)
        dphase = dt_offset * (phase_left + phase_right - 2 * phase_mid)
        dphase += (phase_right - phase_left) / 2
        phase_i = (
            _parabola(dt_offset, phase_left, phase_mid, phase_right)
            + self.config.pi_over_4 * np.pi / 4
        )
        return dphase, amp_i, phase_i, dt_offset, indices

    def _interp_phase_only(
        self,
        phase: np.ndarray,
        indices: np.ndarray,
        omega: float,
        dt_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        left = indices - 1
        right = indices + 1
        delta = self.trace.stats.delta
        phase_left = phase[left]
        phase_mid = phase[indices].copy()
        phase_right = phase[right].copy()
        phase_mid -= tau * np.around((phase_mid - phase_left - omega * delta) / tau)
        phase_right -= tau * np.around((phase_right - phase_mid - omega * delta) / tau)
        dphase = dt_offset * (phase_left + phase_right - 2 * phase_mid)
        dphase += (phase_right - phase_left) / 2
        phase_i = (
            _parabola(dt_offset, phase_left, phase_mid, phase_right)
            + self.config.pi_over_4 * np.pi / 4
        )
        return dphase, phase_i

    def _jump_corrected(
        self,
        par_all: list[dict[str, np.ndarray]],
        raw: dict[str, np.ndarray],
        trig_threshold: float,
        jump_points: int,
    ) -> dict[str, np.ndarray]:
        out = {key: np.asarray(value).copy() for key, value in raw.items()}
        _, trig, on = self._trigger(raw["center_period"], raw["group_velocity"], trig_threshold)
        if not on:
            return out

        jumps = np.where(np.abs(trig[1:] - trig[:-1]) == 2)[0] + 1
        if jumps.size >= 2:
            close = np.where((jumps[1:] - jumps[:-1]) <= jump_points)[0]
            for item in close:
                temp = {key: value.copy() for key, value in out.items()}
                for k in range(jumps[item], jumps[item + 1]):
                    par = par_all[k]
                    delta = np.abs(par["group_velocity"] - temp["group_velocity"][k - 1])
                    best = int(np.argmin(delta))
                    for key in _AFTAN_KEYS:
                        temp[key][k] = par[key][best]
                _, test_trig, test_on = self._trigger(
                    raw["center_period"],
                    temp["group_velocity"],
                    trig_threshold,
                )
                if not np.any(np.abs(test_trig[jumps[item] - 1 : jumps[item + 1] + 1]) == 1):
                    out = temp
                    trig = test_trig
                    on = test_on

        if on:
            start, stop = _longest_branch(trig)
            out = {key: value[start : stop + 1] for key, value in out.items()}
        return out

    @staticmethod
    def _trigger(
        period: np.ndarray,
        velocity: np.ndarray,
        trig_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        curvature = _curvature(period, velocity)
        trig = np.zeros(curvature.size)
        trig[curvature > trig_threshold] = 1
        trig[curvature < -trig_threshold] = -1
        return curvature, trig, bool(np.any(np.abs(trig) == 1))

    def _phase_velocity(
        self,
        period: np.ndarray,
        group_velocity: np.ndarray,
        phase: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        distance = self.distance_km
        omega = tau / period
        time = distance / group_velocity
        slowness = 1 / group_velocity
        phase_velocity = np.zeros(group_velocity.size)

        c_pred = self._predicted_phase_velocity(period[-1])
        phase_pred = omega[-1] * time[-1] - omega[-1] / c_pred * distance
        cycle = round((phase_pred - phase[-1]) / tau)
        phase_velocity[-1] = omega[-1] * distance / (
            omega[-1] * time[-1] - (phase[-1] + cycle * tau)
        )

        domega = omega[:-1] - omega[1:]
        for idx in range(omega.size - 2, -1, -1):
            wave_number = omega[idx + 1] / phase_velocity[idx + 1]
            wave_number += domega[idx] * (slowness[idx] + slowness[idx + 1]) / 2
            phase_pred = omega[idx] * time[idx] - wave_number * distance
            cycle = round((phase[idx] - phase_pred) / tau)
            phase_velocity[idx] = omega[idx] * distance / (
                omega[idx] * time[idx] - phase[idx] + cycle * tau
            )
        reference_phase_velocity = np.asarray(
            interpolate.interp1d(
                self.pred_period,
                self.pred_velocity,
                fill_value="extrapolate",
                assume_sorted=True,
            )(period),
            dtype=float,
        )
        return _select_phase_cycle(
            period,
            phase_velocity,
            reference_phase_velocity,
            distance,
            self.config.phase_cycle,
        )

    def _predicted_phase_velocity(self, period: float) -> float:
        return float(
            interpolate.interp1d(
                self.pred_period,
                self.pred_velocity,
                fill_value="extrapolate",
                assume_sorted=True,
            )(period)
        )

    def _spectral_snr(
        self,
        signal_data: np.ndarray,
        period: np.ndarray,
        group_velocity: np.ndarray,
        config: AFTANSNRConfig,
        alpha: float,
        target_periods: np.ndarray,
        snr_signal_data: np.ndarray | None = None,
    ) -> np.ndarray:
        if config.definition == "pyftan":
            return self._pyftan_spectral_snr(
                self.trace.data if snr_signal_data is None else snr_signal_data,
                period,
                group_velocity,
                config,
                alpha,
            )

        snr_data = self.trace.data if snr_signal_data is None else snr_signal_data
        valid = np.isfinite(period) & (period > 0)
        snr = np.zeros(period.size)
        if not np.any(valid):
            return snr

        ft = self._ftan_complex(snr_data, period[valid], alpha)
        envelope = np.abs(ft[: self.trace.stats.npts, :])
        work_trace = Trace(header=self.trace.stats)
        valid_indices = np.flatnonzero(valid)
        for column, idx in enumerate(valid_indices):
            if config.definition == "aftan":
                peak_time = self.distance_km / group_velocity[idx]
                peak_index = int(
                    np.argmin(np.abs(self.time[: self.trace.stats.npts] - peak_time))
                )
                snr[idx] = _aftan_diagram_snr(envelope[:, column], peak_index, config)
            else:
                work_trace.data = envelope[:, column]
                snr[idx] = _window_snr(
                    work_trace,
                    self.distance_km,
                    period[idx],
                    group_velocity[idx],
                    self.max_period,
                    config,
                )
            if snr[idx] <= 0:
                break
        return snr

    def _pyftan_spectral_snr(
        self,
        signal_data: np.ndarray,
        period: np.ndarray,
        group_velocity: np.ndarray,
        config: AFTANSNRConfig,
        alpha: float,
    ) -> np.ndarray:
        valid = np.isfinite(period) & (period > 0)
        snr = np.zeros(period.size)
        if not np.any(valid):
            return snr

        filtered = self._ftan_complex(signal_data, period[valid], alpha).real[
            : self.trace.stats.npts,
            :,
        ]
        work_trace = Trace(header=self.trace.stats)
        valid_indices = np.flatnonzero(valid)
        for column, idx in enumerate(valid_indices):
            work_trace.data = filtered[:, column]
            snr[idx] = _window_snr(
                work_trace,
                self.distance_km,
                period[idx],
                group_velocity[idx],
                self.config.max_period,
                config,
            )
            if snr[idx] <= 0:
                break
        return snr


def _select_phase_cycle(
    period: np.ndarray,
    phase_velocity: np.ndarray,
    reference_phase_velocity: np.ndarray,
    distance_km: float,
    config: AFTANPhaseCycleConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select one global integer-cycle branch against a reference curve."""
    period = np.asarray(period, dtype=float)
    uncorrected = np.asarray(phase_velocity, dtype=float)
    reference = np.asarray(reference_phase_velocity, dtype=float)
    corrected = uncorrected.copy()
    residual_before = np.full(period.shape, np.nan, dtype=float)
    residual_after = np.full(period.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(period)
        & np.isfinite(uncorrected)
        & np.isfinite(reference)
        & (period > 0)
        & (uncorrected > 0)
        & (reference > 0)
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count:
        residual_before[valid] = (
            distance_km / uncorrected[valid]
            - distance_km / reference[valid]
        ) / period[valid]

    offsets = np.array([0], dtype=int)
    if config.enabled:
        offsets = np.arange(-config.max_shift, config.max_shift + 1, dtype=int)

    candidates: list[tuple[float, int]] = []
    for offset in offsets:
        candidate_slowness = (
            1.0 / uncorrected[valid] + offset * period[valid] / distance_km
        )
        if valid_count and np.all(candidate_slowness > 0):
            score = float(np.median(np.abs(residual_before[valid] + offset)))
            candidates.append((score, int(offset)))

    candidates.sort(key=lambda item: (item[0], abs(item[1]), item[1]))
    if candidates:
        best_score, cycle_shift = candidates[0]
        score_gap = (
            float(candidates[1][0] - best_score)
            if len(candidates) > 1
            else np.nan
        )
    else:
        best_score = np.nan
        cycle_shift = 0
        score_gap = np.nan

    if valid_count and cycle_shift:
        corrected_slowness = 1.0 / uncorrected + cycle_shift * period / distance_km
        if np.all(corrected_slowness[valid] > 0):
            corrected[valid] = 1.0 / corrected_slowness[valid]
        else:
            cycle_shift = 0

    if valid_count:
        residual_after[valid] = (
            distance_km / corrected[valid] - distance_km / reference[valid]
        ) / period[valid]
        rmse_before = float(
            np.sqrt(np.mean((uncorrected[valid] - reference[valid]) ** 2))
        )
        rmse_after = float(
            np.sqrt(np.mean((corrected[valid] - reference[valid]) ** 2))
        )
    else:
        rmse_before = np.nan
        rmse_after = np.nan

    search_limit_reached = bool(
        config.enabled
        and config.max_shift > 0
        and abs(cycle_shift) == config.max_shift
    )
    score_ok = bool(
        np.isfinite(best_score)
        and best_score <= config.max_reference_residual_cycles
    )
    gap_ok = bool(
        not config.enabled
        or config.max_shift == 0
        or (
            np.isfinite(score_gap)
            and score_gap >= config.min_candidate_score_gap_cycles
        )
    )
    qc_pass = bool(
        config.enabled
        and valid_count >= 2
        and score_ok
        and gap_ok
        and not search_limit_reached
    )

    if not config.enabled:
        status = "disabled"
    elif valid_count < 2:
        status = "insufficient_reference"
    elif search_limit_reached:
        status = "search_limit_reached"
    elif not score_ok:
        status = "reference_misfit"
    elif not gap_ok:
        status = "ambiguous"
    elif cycle_shift:
        status = "corrected"
    else:
        status = "unchanged"

    diagnostics = {
        "phase_velocity_uncorrected": uncorrected,
        "phase_reference_velocity": reference,
        "phase_cycle_residual_cycles_before": residual_before,
        "phase_cycle_residual_cycles_after": residual_after,
        "phase_cycle_shift": np.asarray(cycle_shift, dtype=np.int32),
        "phase_cycle_reference_score_cycles": np.asarray(best_score, dtype=float),
        "phase_cycle_candidate_score_gap_cycles": np.asarray(score_gap, dtype=float),
        "phase_cycle_reference_rmse_km_s_before": np.asarray(
            rmse_before, dtype=float
        ),
        "phase_cycle_reference_rmse_km_s_after": np.asarray(
            rmse_after, dtype=float
        ),
        "phase_cycle_valid_count": np.asarray(valid_count, dtype=np.int32),
        "phase_cycle_qc_pass": np.asarray(qc_pass, dtype=np.int8),
        "phase_cycle_search_limit_reached": np.asarray(
            search_limit_reached, dtype=np.int8
        ),
        "phase_cycle_status": np.asarray(status),
    }
    return corrected, diagnostics


def _build_period_grid(
    config: AFTANPeriodSamplingConfig,
    min_period: float,
    max_period: float,
) -> np.ndarray:
    if min_period <= 0 or max_period <= min_period:
        raise ValueError("period range must satisfy 0 < min_period < max_period.")
    if config.mode == "list":
        periods = np.asarray(config.periods, dtype=float)
        mask = (periods >= min_period) & (periods <= max_period)
        periods = np.unique(periods[mask])
        if periods.size == 0:
            raise ValueError("list period_sampling has no periods inside the usable range.")
        return periods

    if config.mode == "geomspace":
        count = _period_count_from_sampling(config, min_period, max_period)
        if count < 2:
            raise ValueError("period_sampling must produce at least two periods.")
        return np.geomspace(min_period, max_period, count)
    if config.mode == "uniform":
        if config.step is not None:
            periods = np.arange(min_period, max_period + config.step * 0.5, config.step)
            periods = periods[periods <= max_period]
            if periods.size == 0 or periods[-1] < max_period:
                periods = np.append(periods, max_period)
            return periods
        count = _period_count_from_sampling(config, min_period, max_period)
        if count < 2:
            raise ValueError("period_sampling must produce at least two periods.")
        return np.linspace(min_period, max_period, count)
    raise ValueError(f"Unsupported period sampling mode: {config.mode}")


def _period_count_from_sampling(
    config: AFTANPeriodSamplingConfig,
    min_period: float,
    max_period: float,
) -> int:
    if config.count is not None:
        return int(config.count)
    if config.mode == "geomspace":
        count = int(np.log(max_period / min_period) / float(config.dfreq))
        return max(int(config.min_count or 3), count)
    raise ValueError(f"period count cannot be inferred for mode {config.mode!r}.")


def _align_to_period_grid(
    target_period: np.ndarray,
    picked_target_period: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Align picked values to the full target-period grid."""
    aligned = np.full(target_period.size, np.nan, dtype=float)
    if picked_target_period.size == 0:
        return aligned

    indices = np.searchsorted(target_period, picked_target_period)
    for period, item, value in zip(picked_target_period, indices, values):
        candidates = []
        if item < target_period.size:
            candidates.append(item)
        if item > 0:
            candidates.append(item - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda idx: abs(target_period[idx] - period))
        tolerance = max(1e-8, abs(period) * 1e-8)
        if abs(target_period[best] - period) <= tolerance:
            aligned[best] = value
    return aligned


def _pmf_period_bounds(
    apparent_period: np.ndarray,
    config_min_period: float,
    config_max_period: float,
) -> tuple[float, float, float, float]:
    valid = np.asarray(apparent_period, dtype=float)
    valid = valid[np.isfinite(valid) & (valid > 0)]
    if valid.size < 2:
        raise ValueError("PMF requires at least two valid apparent periods.")
    period_min = max(float(config_min_period), float(np.min(valid)))
    period_max = min(float(config_max_period), float(np.max(valid)))
    if period_max <= period_min:
        raise ValueError(
            f"PMF period range is empty after first-pass bounds: "
            f"{period_min:.6g}-{period_max:.6g} s."
        )
    return period_min, period_max, period_min, period_max


def _phase_velocity_candidate_map(
    period: np.ndarray,
    group_velocity: np.ndarray,
    phase: np.ndarray,
    picked_phase_velocity: np.ndarray,
    amplitude: np.ndarray,
    distance_km: float,
    velocity: np.ndarray,
    cycle_count: int,
    normalize: bool,
) -> np.ndarray:
    out = np.zeros((period.size, velocity.size), dtype=float)
    if velocity.size < 2:
        return out

    sigma = max(float(np.median(np.diff(velocity))) * 1.5, 1e-6)
    offsets = np.arange(-int(cycle_count), int(cycle_count) + 1)
    for idx, (per, gv, ph, pv, amp) in enumerate(
        zip(period, group_velocity, phase, picked_phase_velocity, amplitude)
    ):
        if not all(np.isfinite(value) for value in (per, gv, ph, pv, amp)):
            continue
        if per <= 0 or gv <= 0 or pv <= 0:
            continue
        omega = tau / per
        time = distance_km / gv
        base = omega * time - ph
        selected_denominator = omega * distance_km / pv
        selected_cycle = int(np.rint((selected_denominator - base) / tau))
        cycles = selected_cycle + offsets
        for cycle in cycles:
            denominator = base + cycle * tau
            if abs(denominator) < 1e-10:
                continue
            candidate_velocity = omega * distance_km / denominator
            if not np.isfinite(candidate_velocity):
                continue
            if candidate_velocity < velocity[0] or candidate_velocity > velocity[-1]:
                continue
            out[idx] += amp * np.exp(
                -0.5 * ((velocity - candidate_velocity) / sigma) ** 2
            )

    if normalize:
        scale = np.nanmax(out, axis=1, keepdims=True)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.0
        out = out / scale
    return out


def _resolve_alpha(config: AFTANAlphaConfig, distance_km: float) -> float:
    if config.mode == "constant":
        return float(config.value)
    if config.mode == "empirical_distance_table":
        return gaussian_alpha_from_distance(
            distance_km,
            np.asarray(config.distance_nodes) if config.distance_nodes is not None else None,
            np.asarray(config.alpha_nodes) if config.alpha_nodes is not None else None,
        )
    if config.mode == "aftan_distance_scaling":
        return gaussian_alpha_aftan_scaling(distance_km, factor=config.factor)
    if config.mode == "pyftan_constant":
        return gaussian_alpha_pyftan_constant(factor=config.factor)
    raise ValueError(f"Unsupported alpha mode: {config.mode}")


def _effective_max_period(config: AFTANConfig, distance_km: float) -> float:
    if not config.short_distance_guard.enabled:
        return config.max_period
    guard = config.short_distance_guard
    guard_max_period = distance_km / (guard.reference_velocity * guard.max_period_nwl)
    return min(config.max_period, guard_max_period)


def _load_prediction(config: AFTANConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.prediction_file is not None:
        period, velocity = np.loadtxt(config.prediction_file, unpack=True)
        return np.asarray(period), np.asarray(velocity)
    reference_velocity = config.short_distance_guard.reference_velocity
    return (
        np.array([config.min_period, config.max_period], dtype=float),
        np.array([reference_velocity, reference_velocity], dtype=float),
    )


def _parabola(t, left, center, right):
    return ((left + right - 2 * center) / 2) * t**2 + ((right - left) / 2) * t + center


def _curvature(period: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    if period.size < 3:
        return np.zeros(period.size)
    x = np.log(period)
    y = velocity
    d10 = x[1:-1] - x[:-2]
    d21 = x[2:] - x[1:-1]
    d20 = x[2:] - x[:-2]
    curvature = ((y[2:] - y[1:-1]) / d21 - (y[1:-1] - y[:-2]) / d10) / d20
    return np.concatenate(([0.0], curvature, [0.0]))


def _longest_branch(trigger: np.ndarray) -> tuple[int, int]:
    indices = np.where(np.abs(trigger) == 1)[0]
    indices = np.concatenate(([0], indices, [trigger.size - 1]))
    branch = int(np.argmax(indices[1:] - indices[:-1]))
    return int(indices[branch]), int(indices[branch + 1])
