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
from .products import (
    amplitude_time_to_velocity,
    energy_map_dataset,
    phase_velocity_candidate_map,
    phase_velocity_map_dataset,
)
from .snr import spectral_snr
from .taper import (
    cosine_window_weights,
    group_velocity_window_bounds_s,
    legacy_edge_fraction_weights,
)


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
        self._apply_time_taper()
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

    def _apply_time_taper(self) -> None:
        """Taper the FTAN input; keep the SNR input free of the signal window.

        ``self.trace.data`` becomes the tapered FTAN/PMF input. ``self.snr_data``
        stays untapered for the window taper, because SNR noise windows lie
        outside the group-velocity window. The legacy mode reproduces the old
        behavior, in which SNR also used the edge-tapered trace.
        """
        taper_config = self.config.time_taper
        raw = np.asarray(self.trace.data, dtype=float)
        self.time_taper_window_s = (np.nan, np.nan)
        self.time_taper_ramp_s = np.nan
        if taper_config.mode == "velocity_window":
            start_s, end_s = group_velocity_window_bounds_s(
                self.distance_km,
                self.config.velocity_min,
                self.config.velocity_max,
            )
            ramp_s = (
                self.max_period if taper_config.ramp_s is None else taper_config.ramp_s
            )
            weights = cosine_window_weights(self.time, start_s, end_s, ramp_s)
            self.time_taper_window_s = (start_s, end_s)
            self.time_taper_ramp_s = float(ramp_s)
            self.snr_data = raw
        elif taper_config.mode == "legacy_edge_fraction":
            weights = legacy_edge_fraction_weights(
                self.trace, taper_config.legacy_edge_fraction
            )
            # ObsPy tapers in place, keeping the input dtype (often float32).
            tapered = (raw * weights).astype(self.trace.data.dtype, copy=False)
            self.snr_data = tapered
            self.trace.data = tapered
            return
        else:
            weights = np.ones(raw.size)
            self.snr_data = raw
        self.trace.data = raw * weights

    def time_taper_attrs(self) -> dict[str, object]:
        """Provenance attributes describing the FTAN input taper."""
        taper_config = self.config.time_taper
        start_s, end_s = self.time_taper_window_s
        return {
            "time_taper_mode": taper_config.mode,
            "time_taper_window_start_s": float(start_s),
            "time_taper_window_end_s": float(end_s),
            "time_taper_ramp_s": float(self.time_taper_ramp_s),
            "time_taper_legacy_edge_fraction": (
                float(taper_config.legacy_edge_fraction)
                if taper_config.mode == "legacy_edge_fraction"
                else np.nan
            ),
            "snr_input": (
                "legacy_edge_tapered_waveform"
                if taper_config.mode == "legacy_edge_fraction"
                else "untapered_waveform"
            ),
        }

    def measure(self) -> dict[str, np.ndarray]:
        return self._measure_signal(
            self.trace.data,
            self.target_periods,
            self.basic_alpha,
            self.config.basic.trig_threshold,
            self.config.basic.jump_points,
            snr_signal_data=self.snr_data,
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
            snr_signal_data=self.snr_data,
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
            # Peaks inside the zero-weight part of the time taper can have a
            # zero phase increment; they yield non-finite periods and are never
            # the envelope maximum when the window contains signal.
            with np.errstate(divide="ignore", invalid="ignore"):
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

    def _map_velocity_axis(self, velocity_count: int) -> np.ndarray:
        return np.linspace(
            self.config.velocity_min,
            self.config.velocity_max,
            int(velocity_count),
        )

    def _product_attrs(
        self,
        *,
        stage: str,
        source_file: Path,
        branch: str,
        alpha: float,
        target_period: np.ndarray,
    ) -> dict[str, object]:
        return {
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
        }

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
        if signal_data is None:
            signal_data = self.trace.data
        npts = self.trace.stats.npts
        target_period = measured.get("energy_target_period", measured["target_period"])
        alpha = float(np.asarray(measured.get("ftan_alpha", self.basic_alpha)))
        amplitude = np.abs(
            self._ftan_complex(signal_data, np.asarray(target_period, dtype=float), alpha)
        )[:npts, :].T
        velocity = self._map_velocity_axis(velocity_count)
        return energy_map_dataset(
            measured,
            target_period=target_period,
            velocity_km_s=velocity,
            map_amplitude=amplitude_time_to_velocity(
                amplitude, self.time[:npts], self.distance_km, velocity
            ),
            normalize=normalize,
            debug=self.config.debug,
            attrs={
                **self._product_attrs(
                    stage=stage,
                    source_file=source_file,
                    branch=branch,
                    alpha=alpha,
                    target_period=target_period,
                ),
                "signal_source": (
                    "raw_waveform" if stage == "basic" else "pmf_clean_waveform"
                ),
                "debug": bool(self.config.debug),
                **self.time_taper_attrs(),
            },
        )

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
        alpha = self.basic_alpha if stage == "basic" else self.pmf_alpha
        phase_velocity = self._map_velocity_axis(velocity_count)
        candidate_amplitude = phase_velocity_candidate_map(
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
        return phase_velocity_map_dataset(
            measured,
            phase_velocity_km_s=phase_velocity,
            candidate_amplitude=candidate_amplitude,
            attrs={
                **self._product_attrs(
                    stage=stage,
                    source_file=source_file,
                    branch=branch,
                    alpha=alpha,
                    target_period=measured["target_period"],
                ),
                "phase_cycle_count": int(cycle_count),
                "phase_cycle_selection_method": (
                    "global minimum median absolute reference travel-time residual"
                ),
                "phase_cycle_selection_source": "automatic",
                "phase_cycle_implementation": "seisforge_extension",
                "phase_cycle_enabled": bool(self.config.phase_cycle.enabled),
                "phase_cycle_max_shift": int(self.config.phase_cycle.max_shift),
                "description": "phase-velocity cycle-candidate diagnostic map",
                **self.time_taper_attrs(),
            },
        )

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
        snr_data = self.snr_data if snr_signal_data is None else snr_signal_data
        valid = np.isfinite(period) & (period > 0)
        if not np.any(valid):
            return np.zeros(period.size)
        # PyAFTAN windows scale with the configured max period; local windows
        # use the effective (short-distance guarded) max period.
        max_period_s = (
            self.config.max_period if config.definition == "pyftan" else self.max_period
        )
        return spectral_snr(
            self._ftan_complex(snr_data, period[valid], alpha),
            period,
            group_velocity,
            valid=valid,
            header=self.trace.stats,
            time_s=self.time,
            distance_km=self.distance_km,
            max_period_s=max_period_s,
            config=config,
        )


def _select_phase_cycle(
    period: np.ndarray,
    phase_velocity: np.ndarray,
    reference_phase_velocity: np.ndarray,
    distance_km: float,
    config: AFTANPhaseCycleConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select one global integer-cycle shift against the reference curve."""
    period = np.asarray(period, dtype=float)
    uncorrected = np.asarray(phase_velocity, dtype=float)
    reference = np.asarray(reference_phase_velocity, dtype=float)
    if period.shape != uncorrected.shape or period.shape != reference.shape:
        raise ValueError(
            "period, phase_velocity, and reference_phase_velocity must match."
        )
    if distance_km <= 0:
        raise ValueError("distance_km must be positive.")

    valid = (
        np.isfinite(period)
        & np.isfinite(uncorrected)
        & np.isfinite(reference)
        & (period > 0)
        & (uncorrected > 0)
        & (reference > 0)
    )
    valid_count = int(np.count_nonzero(valid))
    candidate_shifts = (
        np.arange(-config.max_shift, config.max_shift + 1, dtype=np.int32)
        if config.enabled
        else np.asarray([0], dtype=np.int32)
    )
    candidate_scores = np.full(candidate_shifts.shape, np.nan, dtype=float)
    cycle_shift = 0
    reference_score = np.nan
    score_gap = np.nan

    # A single reference sample cannot distinguish a stable global branch.
    if valid_count >= 2:
        residual_zero = (
            distance_km / uncorrected[valid]
            - distance_km / reference[valid]
        ) / period[valid]
        ranked: list[tuple[float, int]] = []
        for index, shift in enumerate(candidate_shifts):
            candidate_slowness = (
                1.0 / uncorrected[valid]
                + int(shift) * period[valid] / distance_km
            )
            if np.all(candidate_slowness > 0):
                score = float(np.median(np.abs(residual_zero + int(shift))))
                candidate_scores[index] = score
                ranked.append((score, int(shift)))
        ranked.sort(key=lambda item: (item[0], abs(item[1]), item[1]))
        if ranked:
            reference_score, cycle_shift = ranked[0]
            if len(ranked) > 1:
                score_gap = float(ranked[1][0] - reference_score)

    corrected = apply_phase_cycle_shift(
        period,
        uncorrected,
        distance_km,
        cycle_shift,
    )

    search_limit_reached = bool(
        config.enabled
        and config.max_shift > 0
        and abs(cycle_shift) == config.max_shift
    )
    diagnostics = {
        "phase_velocity_uncorrected": uncorrected,
        "phase_reference_velocity": reference,
        "phase_cycle_shift": np.asarray(cycle_shift, dtype=np.int32),
        "phase_cycle_reference_score_cycles": np.asarray(
            reference_score, dtype=float
        ),
        "phase_cycle_reference_score_gap_cycles": np.asarray(
            score_gap, dtype=float
        ),
        "phase_cycle_candidate_shift": candidate_shifts.astype(np.int32),
        "phase_cycle_candidate_reference_score_cycles": candidate_scores,
        "phase_cycle_valid_count": np.asarray(valid_count, dtype=np.int32),
        "phase_cycle_search_limit_reached": np.asarray(
            search_limit_reached, dtype=np.int8
        ),
    }
    return corrected, diagnostics


def apply_phase_cycle_shift(
    period: np.ndarray,
    phase_velocity_uncorrected: np.ndarray,
    distance_km: float,
    cycle_shift: int,
) -> np.ndarray:
    """Apply one global integer shift to phase slowness."""
    period = np.asarray(period, dtype=float)
    uncorrected = np.asarray(phase_velocity_uncorrected, dtype=float)
    if period.shape != uncorrected.shape:
        raise ValueError("period and phase_velocity_uncorrected must match.")
    if distance_km <= 0:
        raise ValueError("distance_km must be positive.")
    corrected = uncorrected.copy()
    valid = np.isfinite(period) & np.isfinite(uncorrected) & (period > 0) & (
        uncorrected > 0
    )
    corrected_slowness = (
        1.0 / uncorrected[valid] + int(cycle_shift) * period[valid] / distance_km
    )
    if np.any(corrected_slowness <= 0):
        raise ValueError("cycle shift produces non-positive phase slowness.")
    corrected[valid] = 1.0 / corrected_slowness
    return corrected


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
