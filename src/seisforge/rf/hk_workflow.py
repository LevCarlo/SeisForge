"""H-k stacking of receiver functions: per-RF stacks and the `rf hkseq` driver.

Formulas live in :mod:`seisforge.rf.hk_core`; this module builds grids,
samples traces, orchestrates the sediment and crust stages, and writes
outputs. Method and conventions: `docs/rf/hk.md`.
"""

from __future__ import annotations

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from joblib import Parallel, delayed, dump
from tqdm import tqdm

from seisforge.provenance import software_provenance
from seisforge.rf.config import HkLayerConfig, HkSeqConfig
from seisforge.rf.hk_core import (
    direct_p_mask,
    hk_grid,
    hk_phase_delays_s,
    hk_pick,
    hk_stack_amplitude,
    interpolate_at_times,
    median_and_robust_std,
    seispy_amplitude_correction,
    velocities_from_vpvs_ratio,
    weighted_mahalanobis_distance,
)
from seisforge.rf.traces import RFtrace

logger = logging.getLogger(__name__)

PHASE_SAMPLING = (
    "linear interpolation; delays outside the RF trace or post-critical contribute zero"
)
SUMMARY_METHOD = (
    "median +- 1.4826 * MAD over the RFs not dominated by direct P and kept by the "
    "Mahalanobis screen"
)


def _default_weights(weight):
    return [0.5, 0.2, 0.3] if weight is None else list(weight)


def _safe_normalize_abs(values):
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        return values
    return values / max_abs


def _check_output_dir(savedir, rootdir, input_dirs):
    """Refuse output directories that overlap the station root or RF inputs."""
    save = Path(savedir).resolve()
    root = Path(rootdir).resolve()
    if save == root or root.is_relative_to(save):
        raise ValueError(
            f"IO.SAVE must be a subdirectory of IO.ROOT, not {save}."
        )
    for input_dir in input_dirs:
        source = Path(input_dir).resolve()
        if save.is_relative_to(source) or source.is_relative_to(save):
            raise ValueError(
                f"IO.SAVE {save} overlaps the RF input directory {source}."
            )


def _hk_output_files(savedir, figdir, net, sta, stage, figure_name, plot):
    """Files written by one H-k stage (``HFreq`` or ``LFreq``)."""
    files = [
        os.path.join(savedir, f"{net}.{sta}_{stage}_Hk_results.joblib"),
        os.path.join(savedir, f"{net}.{sta}_{stage}_Hk_results_QC.joblib"),
        os.path.join(savedir, f"{net}.{sta}_{stage}_Hk_QC.dat"),
    ]
    if plot:
        files.append(os.path.join(figdir, figure_name))
    return files


def _check_existing_outputs(paths, overwrite):
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(
            "H-k outputs already exist (use --overwrite or IO.OVERWRITE: true): "
            + ", ".join(existing)
        )


def _estimate_hk_summary(hk_results, mask):
    """``(H, k, H_spread, k_spread)``: median and scaled MAD of the kept picks."""
    mask = np.asarray(mask, dtype=bool)
    if len(hk_results) == 0 or not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan
    h_best, h_spread = median_and_robust_std(
        np.array([res.attrs["H_best"] for res in hk_results])[mask]
    )
    k_best, k_spread = median_and_robust_std(
        np.array([res.attrs["k_best"] for res in hk_results])[mask]
    )
    return h_best, k_best, h_spread, k_spread


def _with_provenance(hk_results, provenance):
    """Copies of the per-RF H-k datasets with software provenance attrs."""
    return [res.assign_attrs(provenance) for res in hk_results]


def _write_qc_table(path, hk_results, mask, *, stage_settings, summary):
    """Kept picks with a header of provenance, stage settings and the estimate."""
    h_best, k_best, h_spread, k_spread = summary
    with open(path, "w", encoding="utf-8") as file:
        for key, value in software_provenance().items():
            file.write(f"# {key}: {value}\n")
        for key, value in stage_settings.items():
            file.write(f"# {key}: {value}\n")
        file.write(f"# n_rf: {len(hk_results)}\n")
        file.write(
            "# n_direct_p_dominates: "
            f"{sum(int(res.attrs['direct_p_dominates']) for res in hk_results)}\n"
        )
        file.write(f"# n_kept: {int(np.sum(mask))}\n")
        file.write(f"# summary_method: {SUMMARY_METHOD}\n")
        file.write(f"# H_km: {h_best:.4f} +- {h_spread:.4f}\n")
        file.write(f"# k: {k_best:.4f} +- {k_spread:.4f}\n")
        file.write("# filename H_best k_best Hk_energy\n")
        for res, keep in zip(hk_results, mask):
            if not keep:
                continue
            file.write(
                f"{os.path.basename(res.attrs['rf_file'])} "
                f"{res.attrs['H_best']:.4f} "
                f"{res.attrs['k_best']:.4f} "
                f"{res.attrs['amp_max']:.4f}\n"
            )


def _phase_delays(rf, H, k, Vp, Vs, **upper):
    """Ps/PpPs/PsPs delay grids for an RF trace and H-k parameter grids."""
    if rf.slowness is None:
        raise ValueError("RFtrace slowness is required for H-k stacking.")
    vp, vs = velocities_from_vpvs_ratio(k.values, vp_km_s=Vp, vs_km_s=Vs)
    return hk_phase_delays_s(H.values, vp, vs, rf.slowness, **upper)


def _rf_time_axis(rf):
    sac = rf.trace.stats.sac
    return sac.b, sac.delta


def _sample_rf(rf, data, delays):
    begin_s, delta_s = _rf_time_axis(rf)
    return interpolate_at_times(data, delays, begin_s, delta_s)


def _amplitude_corrected_rf(rf, method="seispy", P_wins=None):
    """Return RF data scaled for H-k stacking.

    - ``seispy``: multiply by :func:`seispy_amplitude_correction` of the ray
      parameter.
    - ``max_amp``: divide by the maximum absolute amplitude inside ``P_wins``
      (default ``[-2, 2]`` s around direct P).
    """
    data = rf.trace.data
    if method == "seispy":
        return seispy_amplitude_correction(rf.slowness) * data
    if method == "max_amp":
        P_wins = [-2, 2] if P_wins is None else P_wins
        begin_s, delta_s = _rf_time_axis(rf)
        t_axis = begin_s + delta_s * np.arange(rf.trace.stats.sac.npts)
        P_idx = (t_axis >= P_wins[0]) & (t_axis <= P_wins[1])
        if not np.any(P_idx):
            raise ValueError(f"P_wins {P_wins} does not overlap the RF trace.")
        max_amp = np.max(np.abs(data[P_idx]))
        if max_amp == 0:
            return data.copy()
        return data / max_amp
    raise ValueError(
        f"Unknown method: {method}. Supported methods are 'seispy' and 'max_amp'."
    )


class HkParam:
    """One search axis: ``vmin + dv * i`` up to ``vmax`` (``hk_core.hk_grid``), or ``values``.

    The step is exactly ``dv``. ``vmax`` is included when it lies on the
    grid; otherwise the grid ends at the last point below it.
    """

    def __init__(self, name: str, vmin, vmax, dv=None, values=None):
        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.dv = dv
        if dv is not None:
            grid = hk_grid(vmin, vmax, dv)
        elif values is not None:
            grid = np.array(values, dtype=float)
        else:
            raise ValueError("Either dv or values must be provided.")
        if grid.size == 0:
            raise ValueError("HkParam values cannot be empty.")
        grid.setflags(write=False)
        self.values = grid

    def __repr__(self):
        return f"HkParam(name={self.name}, vmin={self.vmin}, vmax={self.vmax}, dv={self.dv})"


class HkStack:
    """H-k stack of one RF over the ``(k, H)`` grid.

    ``mode=1`` stacks one layer of thickness ``H``. ``mode=2`` stacks the
    layer below a known upper layer (``h0``, ``k0`` and ``vp0`` or ``vs0``),
    with ``H`` the depth of its base. Grid points whose Ps delay is shorter
    than ``min_ps_delay_s`` lie on the direct-P pulse and are excluded from
    the pick (``hk_core.direct_p_mask``); this argument has no default.
    """

    def __init__(self, RFtrace: RFtrace, H: HkParam, k: HkParam, Vp: float = 6.3,
                 Vs: float = None, weight: list | None = None, mode: int = 1,
                 h0: float = 0.0, vp0: float = 4.5, vs0: float = None, k0: float = 2.0,
                 *, min_ps_delay_s: float):
        self.RFtrace = RFtrace
        self.H = H
        self.k = k
        self.Vp = Vp
        self.Vs = Vs
        self.weight = _default_weights(weight)
        self.mode = mode
        self.h0 = h0
        self.vp0 = vp0
        self.vs0 = vs0
        self.k0 = k0
        self.min_ps_delay_s = float(min_ps_delay_s)

    def _calc_phase_times(self):
        if self.mode == 1:
            return _phase_delays(self.RFtrace, self.H, self.k, self.Vp, self.Vs)
        if self.mode == 2:
            k0 = np.full_like(self.k.values, self.k0, dtype=float)
            upper_vp, upper_vs = velocities_from_vpvs_ratio(
                k0, vp_km_s=self.vp0, vs_km_s=self.vs0
            )
            return _phase_delays(
                self.RFtrace,
                self.H,
                self.k,
                self.Vp,
                self.Vs,
                upper_thickness_km=self.h0,
                upper_vp_km_s=upper_vp,
                upper_vs_km_s=upper_vs,
            )
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    def stack(self):
        t_Ps, t_PpPs, t_PsPs = self._calc_phase_times()
        data_norm = self._amp_correct(method="seispy")
        (amp_Ps, valid_Ps), (amp_PpPs, valid_PpPs), (amp_PsPs, valid_PsPs) = (
            _sample_rf(self.RFtrace, data_norm, delays)
            for delays in (t_Ps, t_PpPs, t_PsPs)
        )
        amp_stack = hk_stack_amplitude(amp_Ps, amp_PpPs, amp_PsPs, self.weight)
        excluded = direct_p_mask(t_Ps, self.min_ps_delay_s)
        i_k, j_H, dominates = hk_pick(amp_stack, excluded)
        H_best = self.H.values[j_H]
        k_best = self.k.values[i_k]

        hk_ds = xr.Dataset(
            coords={
                'H': np.array(self.H.values),
                'k': np.array(self.k.values),
            },
            data_vars={
                "amp_stack": (['k', 'H'], amp_stack),
                "amp_Ps": (['k', 'H'], amp_Ps),
                "amp_PpPs": (['k', 'H'], amp_PpPs),
                "amp_PsPs": (['k', 'H'], amp_PsPs),
                "t_Ps": (['k', 'H'], t_Ps),
                "t_PpPs": (['k', 'H'], t_PpPs),
                "t_PsPs": (['k', 'H'], t_PsPs),
                "excluded_direct_p": (['k', 'H'], excluded),
            },
            attrs={
                'H_best': H_best,
                'k_best': k_best,
                "idx_best": [i_k, j_H],
                'amp_max': amp_stack[i_k, j_H],
                'Vp': self.Vp if self.Vp is not None else np.nan,
                'Vs': self.Vs if self.Vs is not None else np.nan,
                'weight': self.weight,
                'mode': self.mode,
                'h0': self.h0 if self.h0 is not None else np.nan,
                'vp0': self.vp0 if self.vp0 is not None else np.nan,
                'vs0': self.vs0 if self.vs0 is not None else np.nan,
                'k0': self.k0 if self.k0 is not None else np.nan,
                'rf_file': self.RFtrace.file,
                'ray_parameter_s_per_km': float(self.RFtrace.slowness),
                'amplitude_correction': 'seispy',
                'phase_sampling': PHASE_SAMPLING,
                'min_ps_delay_s': self.min_ps_delay_s,
                'n_excluded_direct_p': int(np.count_nonzero(excluded)),
                'direct_p_dominates': int(dominates),
                'n_invalid_Ps': int(np.size(valid_Ps) - np.count_nonzero(valid_Ps)),
                'n_invalid_PpPs': int(
                    np.size(valid_PpPs) - np.count_nonzero(valid_PpPs)
                ),
                'n_invalid_PsPs': int(
                    np.size(valid_PsPs) - np.count_nonzero(valid_PsPs)
                ),
            }
        )

        return hk_ds

    def _amp_correct(self, method="seispy", P_wins=None):
        return _amplitude_corrected_rf(self.RFtrace, method=method, P_wins=P_wins)

    def plot(self):
        hk_ds = self.stack()
        amp_stack = hk_ds.amp_stack.values
        amp = [hk_ds.amp_Ps.values, hk_ds.amp_PpPs.values, -hk_ds.amp_PsPs.values]
        H_best = hk_ds.attrs['H_best']
        k_best = hk_ds.attrs['k_best']
        logger.info("Best H: %.2f km, Best k: %.2f Vp/Vs", H_best, k_best)

        H_grid = self.H.values
        k_grid = self.k.values
        fig, ax = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
        kk, HH = np.meshgrid(k_grid, H_grid, indexing='xy')
        cmap = "viridis"
        ax[0, 0].pcolormesh(HH, kk, _safe_normalize_abs(amp[0].T), cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax[0, 1].pcolormesh(HH, kk, _safe_normalize_abs(amp[1].T), cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax[1, 0].pcolormesh(HH, kk, _safe_normalize_abs(amp[2].T), cmap=cmap, shading='auto', vmin=0, vmax=1)
        img = ax[1, 1].pcolormesh(HH, kk, _safe_normalize_abs(amp_stack.T), cmap=cmap, shading='auto', vmin=0, vmax=1)

        ax[1, 1].scatter(H_best, k_best, marker="*", facecolor='tab:red', edgecolor='black', s=80)
        ax[1, 1].axvline(H_best, color='white', linestyle='--', linewidth=1)
        ax[1, 1].axhline(k_best, color='white', linestyle='--', linewidth=1)
        ax[1, 1].text(
            H_best,
            k_best,
            f"({H_best:.2f}, {k_best:.2f})",
            color='white',
            fontdict={'family': 'Times New Roman', 'size': 12},
            ha='left', va='bottom'
        )

        fontdict_label = {'weight': 'bold', 'family': 'Times New Roman'}
        ax[0, 0].set_title('Ps', fontsize=12)
        ax[0, 1].set_title('PpPs', fontsize=12)
        ax[1, 0].set_title('-(PpSs+PsPs)', fontsize=12)
        ax[1, 1].set_title(
            f"[{self.weight[0]:.1f}, {self.weight[1]:.1f}, {self.weight[2]:.1f}]",
            fontsize=12
        )
        ax[0, 0].set_ylabel('Vp/Vs', fontdict=fontdict_label)
        ax[1, 0].set_ylabel('Vp/Vs', fontdict=fontdict_label)
        ax[1, 0].set_xlabel('H (km)', fontdict=fontdict_label)
        ax[1, 1].set_xlabel('H (km)', fontdict=fontdict_label)
        fig.colorbar(img, ax=ax.ravel().tolist(), shrink=0.55)
        return fig, ax


class HkQController:
    """Screens single-RF H-k picks before the station estimate."""

    def __init__(self, hk_results: list[xr.Dataset], QC_params: dict):
        """
        Parameters:
        - hk_results: per-RF H-k datasets (``HkStack.stack``).
        - QC_params: ``{"mahalanobis_dist_threshold": r}``.
        """
        self.hk_results = hk_results
        self.QC_params = QC_params

    def mahalanobis_dist_select(self, return_mask_only=True):
        """Keep picks with Mahalanobis distance below the threshold.

        Mean and covariance of the ``(H, k)`` picks are weighted by the stack
        maximum of each RF (``hk_core.weighted_mahalanobis_distance``).
        Fewer than two picks are all kept.
        """
        if len(self.hk_results) == 0:
            return np.array([], dtype=bool), None if return_mask_only else []
        if len(self.hk_results) < 2:
            mask = np.ones(len(self.hk_results), dtype=bool)
            QC_results = None if return_mask_only else self.hk_results.copy()
            return mask, QC_results

        points = np.column_stack((
            [res.attrs['H_best'] for res in self.hk_results],
            [res.attrs['k_best'] for res in self.hk_results],
        ))
        energy = np.array([res.attrs['amp_max'] for res in self.hk_results])
        distance = weighted_mahalanobis_distance(points, energy)
        mask = distance < self.QC_params['mahalanobis_dist_threshold']
        if return_mask_only:
            QC_results = None
        else:
            QC_results = [res for res, m in zip(self.hk_results, mask) if m]
        return mask, QC_results


def Hk_analysis(rf, H, k, Vp, Vs, weight, mode, h0=None, vp0=None, vs0=None, k0=None,
                *, min_ps_delay_s):
    """
    Perform H-k analysis on a single RF trace.

    Parameters:
    - rf: RFtrace object.
    - H: HkParam object for H values.
    - k: HkParam object for k values.
    - Vp: float, P-wave velocity.
    - Vs: float, S-wave velocity (optional).
    - weight: weights (w1, w2, w3) of A(t_Ps), A(t_PpPs) and -A(t_PsPs), the
      last being the negative PpSs + PsPs pulse.
    - mode: int, stacking mode (1 or 2).
    - h0: float, upper-layer thickness for mode 2.
    - vp0: float, upper-layer P-wave velocity for mode 2.
    - vs0: float, upper-layer S-wave velocity for mode 2 (optional).
    - k0: float, upper-layer Vp/Vs ratio for mode 2.
    - min_ps_delay_s: grid points with a shorter Ps delay are not picked.

    Returns:
    - xarray Dataset with stacking results and best H-k estimates.
    """
    hk_stack = HkStack(
        RFtrace=rf,
        H=H,
        k=k,
        Vp=Vp,
        Vs=Vs,
        weight=weight,
        mode=mode,
        h0=h0,
        vp0=vp0,
        vs0=vs0,
        k0=k0,
        min_ps_delay_s=min_ps_delay_s,
    )
    return hk_stack.stack()


def init_RFtraces(RFs_datadir, suffix=".sac"):
    RF_files = sorted(
        f for f in os.listdir(RFs_datadir) if f.endswith(suffix) and not f.startswith('.')
    )
    return [RFtrace(file=os.path.join(RFs_datadir, rf_file)) for rf_file in RF_files]


def plot_hk_QC(Hk_results, kept_mask, title=None):
    """All single-RF picks and the ones kept by the Mahalanobis screen.

    Each panel shows the mean of the max-normalised stacks of its RFs
    (grey), the picks coloured by stack maximum, and the station estimate
    (median +- 1.4826 MAD) of the kept picks.
    """
    if len(Hk_results) == 0:
        raise ValueError("No H-k results available for plotting.")
    masks = [np.ones(len(Hk_results), dtype=bool), np.asarray(kept_mask, dtype=bool)]
    keys = ["All", "Mahalanobis screen"]
    title = "" if title is None else title

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    Hvals = Hk_results[0].coords['H'].values
    kvals = Hk_results[0].coords['k'].values
    kk, HH = np.meshgrid(kvals, Hvals, indexing='xy')
    energy_all = np.array([res.attrs['amp_max'] for res in Hk_results])
    for axis, mask, key in zip(ax, masks, keys):
        selected = [res for res, m in zip(Hk_results, mask) if m]
        if selected:
            amp_stack = np.stack([res.amp_stack.values for res in selected])
            amp_max = np.max(amp_stack, axis=(1, 2), keepdims=True)
            amp_max[amp_max == 0] = 1
            amp_mean = np.mean(amp_stack / amp_max, axis=0)
        else:
            amp_mean = np.zeros((len(kvals), len(Hvals)))
        axis.pcolormesh(HH, kk, _safe_normalize_abs(amp_mean.T), cmap='binary',
                        shading='auto', vmin=0, vmax=1)
        points = axis.scatter(
            [res.attrs['H_best'] for res in selected],
            [res.attrs['k_best'] for res in selected],
            c=[res.attrs['amp_max'] for res in selected], cmap='viridis', s=50,
            edgecolor='black', vmin=energy_all.min(), vmax=energy_all.max(),
        )
        h_best, k_best, h_spread, k_spread = _estimate_hk_summary(selected, np.ones(len(selected), bool))
        axis.scatter(h_best, k_best, marker="*", facecolor='tab:red', edgecolor='black', s=200)
        axis.text(0.01, 0.97, f"{h_best:.2f} ± {h_spread:.2f} km\n{k_best:.3f} ± {k_spread:.3f}",
                  transform=axis.transAxes, va='top', ha='left', color='tab:red',
                  fontweight='bold')
        axis.set_title(f"{key} ({len(selected)})")
        axis.set_xlabel('H (km)')
    ax[0].set_ylabel('Vp/Vs')
    fig.suptitle(title)
    fig.colorbar(points, ax=list(ax), label='stack maximum of each RF', shrink=0.8)
    return fig


def _stage_settings(layer: HkLayerConfig, threshold: float, **extra) -> dict:
    return {
        "H_grid_km": f"{layer.H.min} to {layer.H.max} step {layer.H.delta}",
        "k_grid": f"{layer.k.min} to {layer.k.max} step {layer.k.delta}",
        "Vp_km_s": layer.vp_km_s,
        "Vs_km_s": layer.vs_km_s,
        "weight": list(layer.weights),
        "min_ps_delay_s": layer.min_ps_delay_s,
        "phase_sampling": PHASE_SAMPLING,
        "mahalanobis_dist_threshold": threshold,
        **extra,
    }


def hkSeq(Params, mode="both", plot=False, overwrite=None):
    """Sequential sediment (high-frequency) then crust (low-frequency) H-k.

    ``Params`` is the YAML mapping (validated by ``HkSeqConfig``). Only the
    files of the requested stage(s) are written; existing outputs raise
    ``FileExistsError`` unless ``overwrite`` (or ``IO.OVERWRITE``) is set.
    """
    if mode not in ["both", "low", "high"]:
        raise ValueError("Invalid mode. Supported modes are 'both', 'low', and 'high'.")
    config = HkSeqConfig.from_mapping(Params)
    if overwrite is None:
        overwrite = config.overwrite
    net, sta = config.network, config.station

    _check_output_dir(config.save_dir, config.root, [config.high_rf_dir, config.low_rf_dir])
    planned = []
    if mode in ["both", "high"]:
        planned += _hk_output_files(
            config.save_dir, config.figure_dir, net, sta, "HFreq", f"{net}.{sta}_HFreq_QC.png", plot
        )
    if mode in ["both", "low"]:
        low_figure = (
            f"{net}.{sta}_HFreq_LFreq_QC.png" if mode == "both" else f"{net}.{sta}_LFreq_QC.png"
        )
        planned += _hk_output_files(
            config.save_dir, config.figure_dir, net, sta, "LFreq", low_figure, plot
        )
    _check_existing_outputs(planned, overwrite)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.figure_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    handler = logging.FileHandler(
        os.path.join(config.log_dir, f"{net}.{sta}_hkSeq.log"), mode="w", encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        _run_hkseq(config, mode, plot)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def _layer_grids(layer: HkLayerConfig):
    return (
        HkParam("H", layer.H.min, layer.H.max, dv=layer.H.delta),
        HkParam("k", layer.k.min, layer.k.max, dv=layer.k.delta),
    )


MIN_RESOLVED_FRACTION = 0.5
"""Assumed: a stage has an estimate only if direct P does not dominate the
stacks of at least this fraction of its RFs. 0.5 reproduces the median rule
of earlier versions, where the station H was the median of all picks and the
direct-P ridge supplied H near 0 when most RFs picked it."""


def _screen_and_estimate(results, threshold, label):
    """Mask of RFs used for the estimate, and the estimate (NaN if unresolved).

    RFs whose stack is dominated by direct P (``direct_p_dominates``) are
    dropped first; the Mahalanobis screen runs on the rest.
    """
    n_rf = len(results)
    resolved = np.array([not res.attrs["direct_p_dominates"] for res in results], dtype=bool)
    logger.info(f"{label} RFs: {n_rf} total")
    logger.info(f"{label} RFs not dominated by direct P: {np.sum(resolved)}")
    mask = np.zeros(n_rf, dtype=bool)
    if n_rf == 0 or np.sum(resolved) < MIN_RESOLVED_FRACTION * n_rf:
        logger.warning(
            f"{label} RFs: direct P dominates the stack of more than "
            f"{1 - MIN_RESOLVED_FRACTION:.0%} of the RFs; no layer is resolved"
        )
        return mask, (np.nan, np.nan, np.nan, np.nan)
    candidates = [res for res, keep in zip(results, resolved) if keep]
    kept, _ = HkQController(
        candidates, {"mahalanobis_dist_threshold": threshold}
    ).mahalanobis_dist_select(return_mask_only=True)
    mask[np.flatnonzero(resolved)[kept]] = True
    logger.info(f"{label} RFs after Mahalanobis distance select: {np.sum(mask)}")
    return mask, _estimate_hk_summary(results, mask)


def _run_stage(config, stage, band, label, rf_dir, hk_func, settings, plot, figure_name,
               provenance):
    """Stack every RF of one band, screen the picks, write outputs; return the estimate."""
    net, sta = config.network, config.station
    results = Parallel(n_jobs=-1)(delayed(hk_func)(rf) for rf in tqdm(init_RFtraces(rf_dir)))
    results = _with_provenance(results, provenance)
    mask, summary = _screen_and_estimate(results, config.mahalanobis_threshold[band], label)
    h_best, k_best, h_spread, k_spread = summary
    logger.info(f"{label} RFs: H = {h_best:.2f} ± {h_spread:.2f} km")
    logger.info(f"{label} RFs: k = {k_best:.2f} ± {k_spread:.2f} Vp/Vs")
    logger.info(f"Summary: {SUMMARY_METHOD}")

    dump(results, os.path.join(config.save_dir, f"{net}.{sta}_{stage}_Hk_results.joblib"))
    kept = [res for res, m in zip(results, mask) if m]
    dump(kept, os.path.join(config.save_dir, f"{net}.{sta}_{stage}_Hk_results_QC.joblib"))
    _write_qc_table(
        os.path.join(config.save_dir, f"{net}.{sta}_{stage}_Hk_QC.dat"),
        results, mask, stage_settings=settings, summary=summary,
    )
    logger.info(f"Hk analysis for {label} RFs completed and saved.")
    if plot and results:
        fig = plot_hk_QC(results, mask, title=f"{net}.{sta} ({label} RFs)")
        fig.savefig(os.path.join(config.figure_dir, figure_name), dpi=300, bbox_inches='tight')
        plt.close(fig)
    return summary


def _run_hkseq(config: HkSeqConfig, mode, plot):
    net, sta = config.network, config.station
    logger.info(f"Starting H-k analysis for {net}.{sta}")
    provenance = software_provenance()
    for key, value in provenance.items():
        logger.info(f"{key}: {value}")

    sediment = config.sediment
    H0_best = k0_best = np.nan
    if mode in ["both", "high"]:
        H_grid, k_grid = _layer_grids(sediment)
        hk_func = partial(
            Hk_analysis, H=H_grid, k=k_grid, Vp=sediment.vp_km_s, Vs=sediment.vs_km_s,
            weight=list(sediment.weights), mode=1, min_ps_delay_s=sediment.min_ps_delay_s,
        )
        H0_best, k0_best, _, _ = _run_stage(
            config, "HFreq", "HighFreq", "High Frequency", config.high_rf_dir, hk_func,
            _stage_settings(sediment, config.mahalanobis_threshold["HighFreq"], layers=1),
            plot, f"{net}.{sta}_HFreq_QC.png", provenance,
        )

    if mode in ["both", "low"]:
        crust = config.crust
        H_grid, k_grid = _layer_grids(crust)
        common = dict(
            H=H_grid, k=k_grid, Vp=crust.vp_km_s, Vs=crust.vs_km_s,
            weight=list(crust.weights), min_ps_delay_s=crust.min_ps_delay_s,
        )
        limit = config.two_layer_min_thickness_km
        if np.isnan(H0_best) or np.isnan(k0_best) or H0_best <= limit:
            logger.warning(
                "No sediment estimate, or sediment H <= %.2f km "
                "(Model.Sediment.two_layer_min_thickness_km)", limit,
            )
            logger.info("Only use one-layer Hk analysis for crust")
            hk_func = partial(Hk_analysis, mode=1, **common)
            settings = _stage_settings(crust, config.mahalanobis_threshold["LowFreq"], layers=1)
        else:
            logger.info("Use two-layer Hk analysis for crust")
            hk_func = partial(
                Hk_analysis, mode=2, h0=H0_best, vp0=sediment.vp_km_s,
                vs0=sediment.vs_km_s, k0=k0_best, **common,
            )
            settings = _stage_settings(
                crust, config.mahalanobis_threshold["LowFreq"], layers=2,
                sediment_h0_km=H0_best, sediment_k0=k0_best,
                sediment_Vp_km_s=sediment.vp_km_s, sediment_Vs_km_s=sediment.vs_km_s,
            )
        figure = f"{net}.{sta}_HFreq_LFreq_QC.png" if mode == "both" else f"{net}.{sta}_LFreq_QC.png"
        _run_stage(
            config, "LFreq", "LowFreq", "Low Frequency", config.low_rf_dir, hk_func,
            settings, plot, figure, provenance,
        )

    logger.info(f"Completed H-k analysis for {net}.{sta}")


def load_parse_args():
    parser = argparse.ArgumentParser(description="Sequential Hk Analysis")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Hk analysis configuration file in YAML format")
    parser.add_argument("-m", "--mode", type=str, default="both", choices=["both", "high", "low"], help="Mode of Hk analysis: 'both', 'high' or 'low'")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the results if set")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing H-k outputs of the requested stage(s)")
    return parser.parse_args()


def main():
    args = load_parse_args()
    with open(args.config_file, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    hkSeq(params, mode=args.mode, plot=args.plot, overwrite=args.overwrite or None)
