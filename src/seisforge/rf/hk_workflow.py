import os
from pathlib import Path
from seisforge.rf.hk_core import (
    hk_phase_delays_s,
    hk_stack_amplitude,
    sample_at_times,
    seispy_amplitude_correction,
    velocities_from_vpvs_ratio,
)
from seisforge.rf.traces import RFtrace
from seisforge.provenance import software_provenance
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import argparse
import yaml
from functools import partial
from joblib import Parallel, delayed, dump
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)


def _default_weights(weight):
    return [0.5, 0.2, 0.3] if weight is None else weight


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
    if len(hk_results) == 0 or np.sum(mask) == 0:
        return np.nan, np.nan, np.nan, np.nan

    h_estimates = np.array([res.attrs["H_best"] for res in hk_results])[mask]
    k_estimates = np.array([res.attrs["k_best"] for res in hk_results])[mask]
    if len(h_estimates) < 15:
        h_best = np.mean(h_estimates)
        k_best = np.mean(k_estimates)
        h_uncert = np.std(h_estimates)
        k_uncert = np.std(k_estimates)
    else:
        h_best = np.median(h_estimates)
        k_best = np.median(k_estimates)
        h_uncert = np.median(np.abs(h_estimates - h_best))
        k_uncert = np.median(np.abs(k_estimates - k_best))

    return h_best, k_best, h_uncert, k_uncert


def _with_provenance(hk_results, provenance):
    """Copies of the per-RF H-k datasets with software provenance attrs."""
    return [res.assign_attrs(provenance) for res in hk_results]


def _write_qc_table(path, hk_results, mask):
    with open(path, "w", encoding="utf-8") as file:
        for key, value in software_provenance().items():
            file.write(f"# {key}: {value}\n")
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


def _sample_rf(rf, data, delays, return_valid=False):
    begin_s, delta_s = _rf_time_axis(rf)
    amplitude, valid = sample_at_times(data, delays, begin_s, delta_s)
    return (amplitude, valid) if return_valid else amplitude


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
    def __init__(self, name: str, vmin, vmax, dv=None, values=None):

        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.dv = dv
        self.values = values

        if dv is not None:
            if dv <= 0:
                raise ValueError("dv must be positive.")
            N = int((vmax - vmin) / dv) + 1
            self.values = np.linspace(vmin, vmax, N)
        elif values is not None:
            self.values = np.asarray(values)
        else:
            raise ValueError("Either dv or values must be provided.")

        if len(self.values) == 0:
            raise ValueError("HkParam values cannot be empty.")
    
    def __repr__(self):
        return f"HkParam(name={self.name}, vmin={self.vmin}, vmax={self.vmax}, dv={self.dv})"

class HkStack_classic:
    def __init__(self, RFtrace: RFtrace, H: HkParam, k: HkParam, Vp=6.3, Vs=None, weight=None):
        self.RFtrace = RFtrace
        self.H = H
        self.k = k
        self.Vp = Vp
        self.Vs = Vs
        self.weight = _default_weights(weight)

    def stack(self):
        t_Ps, t_PpPs, t_PsPs = _phase_delays(self.RFtrace, self.H, self.k, self.Vp, self.Vs)
        data_norm = self._amp_correct(method="seispy")
        amp_Ps, amp_PpPs, amp_PsPs = (
            _sample_rf(self.RFtrace, data_norm, delays)
            for delays in (t_Ps, t_PpPs, t_PsPs)
        )
        amp_stack = hk_stack_amplitude(amp_Ps, amp_PpPs, amp_PsPs, self.weight)
        return amp_stack, (t_Ps, t_PpPs, t_PsPs), (amp_Ps, amp_PpPs, amp_PsPs)

    def estimate_best_Hk(self, smooth=False):
        amp_stack, _, _ = self.stack()
        if smooth:
            amp = gaussian_filter(amp_stack, sigma=1.5)
        else:
            amp = amp_stack
        H_grid = self.H.values
        k_grid = self.k.values

        i_k, j_H = np.unravel_index(np.argmax(amp), amp.shape)
        H_best = H_grid[j_H]
        k_best = k_grid[i_k]

        return (H_best, k_best, amp_stack[i_k, j_H])

    def _amp_correct(self, method="seispy", P_wins=None):
        return _amplitude_corrected_rf(self.RFtrace, method=method, P_wins=P_wins)

    def plot(self):
        amp_stack, _, amp = self.stack()
        H_best, k_best, _ = self.estimate_best_Hk()
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
        ax[1, 0].set_title('PsPs+PpPs', fontsize=12)
        ax[1, 1].set_title(
            f"[{self.weight[0]:.1f}, {self.weight[1]:.1f}, {self.weight[2]:.1f}]",
            fontsize=12
        )
        ax[0, 0].set_ylabel('Vp/Vs', fontdict=fontdict_label)
        ax[1, 0].set_ylabel('Vp/Vs', fontdict=fontdict_label)
        ax[1, 0].set_xlabel('H (km)', fontdict=fontdict_label)
        ax[1, 1].set_xlabel('H (km)', fontdict=fontdict_label)
        fig.colorbar(img, ax=ax.ravel().tolist(), shrink=0.55)
        fig.tight_layout()
        return fig, ax

class HkStack:
    def __init__(self, RFtrace: RFtrace, H: HkParam, k: HkParam, Vp: float = 6.3, Vs: float = None, weight: list | None = None,
                 mode: int = 1, h0: float = 0.0, vp0: float = 4.5, vs0: float = None, k0: float = 2.0):
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

    def stack(self, smooth=False):
        t_Ps, t_PpPs, t_PsPs = self._calc_phase_times()
        data_norm = self._amp_correct(method="seispy")
        (amp_Ps, valid_Ps), (amp_PpPs, valid_PpPs), (amp_PsPs, valid_PsPs) = (
            _sample_rf(self.RFtrace, data_norm, delays, return_valid=True)
            for delays in (t_Ps, t_PpPs, t_PsPs)
        )
        amp_stack = hk_stack_amplitude(amp_Ps, amp_PpPs, amp_PsPs, self.weight)

        if smooth:
            amp_new = gaussian_filter(amp_stack, sigma=1.5)
        else:
            amp_new = amp_stack
        
        i_k, j_H = np.unravel_index(np.argmax(amp_new), amp_new.shape)
        H_best = self.H.values[j_H]
        k_best = self.k.values[i_k]

        hk_ds = xr.Dataset(
            coords={
                'H': self.H.values,
                'k': self.k.values
            },
            data_vars={
                "amp_stack": (['k', 'H'], amp_stack),
                "amp_Ps": (['k', 'H'], amp_Ps),
                "amp_PpPs": (['k', 'H'], amp_PpPs),
                "amp_PsPs": (['k', 'H'], amp_PsPs),
                "t_Ps": (['k', 'H'], t_Ps),
                "t_PpPs": (['k', 'H'], t_PpPs),
                "t_PsPs": (['k', 'H'], t_PsPs),
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
                'phase_sampling': (
                    'nearest sample; delays outside the RF trace or post-critical '
                    'contribute zero'
                ),
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
        amp = [hk_ds.amp_Ps.values, hk_ds.amp_PpPs.values, hk_ds.amp_PsPs.values]
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
        ax[1, 0].set_title('PsPs+PpPs', fontsize=12)
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
    def __init__(self, hk_results: list[xr.Dataset], QC_params: dict):
        """
        Class to handle H-k stacking results and apply quality control (QC).

        Parameters:
        - hk_results: xarray Dataset containing H-k stacking results.
        - QC_params: QC parameters as a dict
        """
        self.hk_results = hk_results
        self.QC_params = QC_params

    def hk_energy_select(self, return_mask_only=True):
        if len(self.hk_results) == 0:
            return np.array([], dtype=bool), None if return_mask_only else []

        hk_energys = np.array([res.attrs['amp_max'] for res in self.hk_results])
        maximum_energy = np.max(hk_energys)
        
        energy_threshold = self.QC_params['energy_threshold']
        mask = hk_energys >= energy_threshold * maximum_energy
        if return_mask_only:
            QC_results = None
        else:
            QC_results = [res for res, m in zip(self.hk_results, mask) if m]
        return mask, QC_results
    
    def elliptical_mad_select(self, return_mask_only=True):
        if len(self.hk_results) == 0:
            return np.array([], dtype=bool), None if return_mask_only else []

        H_estimates = np.array([res.attrs['H_best'] for res in self.hk_results])
        k_estimates = np.array([res.attrs['k_best'] for res in self.hk_results])
        sigma = self.QC_params['ell_mad_sigma']
        
        mask = elliptical_mad_filter(H_estimates, k_estimates, sigma=sigma)
        if return_mask_only:
            QC_results = None
        else:
            QC_results = [res for res, m in zip(self.hk_results, mask) if m]
        return mask, QC_results

    def mahalanobis_dist_select(self, return_mask_only=True):
        if len(self.hk_results) == 0:
            return np.array([], dtype=bool), None if return_mask_only else []
        if len(self.hk_results) < 2:
            mask = np.ones(len(self.hk_results), dtype=bool)
            QC_results = None if return_mask_only else self.hk_results.copy()
            return mask, QC_results

        H_estimates = np.array([res.attrs['H_best'] for res in self.hk_results])
        k_estimates = np.array([res.attrs['k_best'] for res in self.hk_results])
        Hk_energy = np.array([res.attrs['amp_max'] for res in self.hk_results])

        mahalanobis_dist_threshold = self.QC_params['mahalanobis_dist_threshold']

        X = np.column_stack((H_estimates, k_estimates))
        E = Hk_energy.reshape(-1, 1)
        E_sum = np.sum(E)
        if E_sum <= 0:
            E = np.ones_like(E)
            E_sum = np.sum(E)

        # weighted mean
        mu = np.sum(X * E, axis=0) / E_sum

        # weighted covariance
        X_centered = X - mu
        cov = (E * X_centered).T @ X_centered / E_sum
        cov_inv = np.linalg.pinv(cov)

        # Mahalanobis distance
        m_dist = np.sqrt(np.sum((X_centered @ cov_inv) * X_centered, axis=1))
        mask = m_dist < mahalanobis_dist_threshold
        if return_mask_only:
            QC_results = None
        else:
            QC_results = [res for res, m in zip(self.hk_results, mask) if m]
        return mask, QC_results

def elliptical_mad_filter(x0, x1, sigma=2.0, scaled_MAD=True):
    """
    2D elliptical outlier filtering using the median and MAD (Median Absolute Deviation)
    """
    x0_median = np.median(x0)
    x1_median = np.median(x1)
    x0_mad = np.median(np.abs(x0 - x0_median))
    x1_mad = np.median(np.abs(x1 - x1_median))

    if scaled_MAD:
        x0_mad = x0_mad * 1.4826  # Scale MAD to match standard deviation for normal distribution
        x1_mad = x1_mad * 1.4826  # Scale MAD to match standard deviation for normal distribution

    x0_mad = x0_mad if x0_mad > 1e-6 else 1e-6
    x1_mad = x1_mad if x1_mad > 1e-6 else 1e-6

    x0_norm = (x0 - x0_median) / x0_mad
    x1_norm = (x1 - x1_median) / x1_mad
    ell_dist = x0_norm**2 + x1_norm**2
    mask = ell_dist < sigma**2
    return mask

def Hk_analysis(rf, H, k, Vp, Vs, weight, mode, h0=None, vp0=None, vs0=None, k0=None):
    """
    Perform H-k analysis on a single RF trace.

    Parameters:
    - rf: RFtrace object.
    - H: HkParam object for H values.
    - k: HkParam object for k values.
    - Vp: float, P-wave velocity.
    - Vs: float, S-wave velocity (optional).
    - weight: list of weights for Ps, PpPs, and PsPs+PpPs.
    - mode: int, stacking mode (1 or 2).
    - h0: float, reference depth for mode 2.
    - vp0: float, reference P-wave velocity for mode 2.
    - vs0: float, reference S-wave velocity for mode 2 (optional).
    - k0: float, reference Vp/Vs ratio for mode 2.

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
        k0=k0
    )
    hk_ds = hk_stack.stack()
    return hk_ds

def init_RFtraces(RFs_datadir, suffix=".sac"):
    RFtrace_list = []
    RF_files = [f for f in os.listdir(RFs_datadir) if f.endswith(suffix) and not f.startswith('.')]
    for rf_file in RF_files:
        rf = RFtrace(file=os.path.join(RFs_datadir, rf_file))
        RFtrace_list.append(rf)
    return RFtrace_list

def plot_hk_QC(Hk_results, QC_masks=None, QC_key=None, title=None):
    if len(Hk_results) == 0:
        raise ValueError("No H-k results available for plotting.")

    all_masks = np.ones(len(Hk_results), dtype=bool)
    QC_masks = [] if QC_masks is None else QC_masks
    QC_key = [] if QC_key is None else QC_key
    title = "" if title is None else title
    QC_masks = [all_masks] + QC_masks
    QC_key = ["Raw"] + QC_key

    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(16, 3)
    
    ax_all = fig.add_subplot(gs[3:10, 0])
    ax_qc = [
    fig.add_subplot(gs[0:7, 1], sharex=ax_all, sharey=ax_all),
    fig.add_subplot(gs[0:7, 2], sharex=ax_all, sharey=ax_all),
    fig.add_subplot(gs[-7::, 1], sharex=ax_all, sharey=ax_all),
    fig.add_subplot(gs[-7::, 2], sharex=ax_all, sharey=ax_all)
    ]
    cb1_ax = fig.add_subplot(gs[-4, 0])
    cb2_ax = fig.add_subplot(gs[-2, 0])
    pos1 = cb1_ax.get_position()
    cb1_ax.set_position([pos1.x0, pos1.y0 + 0.25 * pos1.height, pos1.width, 0.5 * pos1.height])

    
    pos2 = cb2_ax.get_position()
    cb2_ax.set_position([pos2.x0, pos2.y0 + 0.25 * pos2.height, pos2.width, 0.5 * pos2.height])

    ax = [ax_all] + ax_qc

    Hvals = Hk_results[0].coords['H'].values
    kvals = Hk_results[0].coords['k'].values
    kk, HH = np.meshgrid(kvals, Hvals, indexing='xy')
    Hk_energy_all = np.array([res.attrs['amp_max'] for res in Hk_results])
    vmin = np.min(Hk_energy_all)
    vmax = np.max(Hk_energy_all)
    for i, mask in enumerate(QC_masks):
        QC_results = [res for res, m in zip(Hk_results, mask) if m]
        H_estimates = np.array([res.attrs['H_best'] for res in QC_results])
        k_estimates = np.array([res.attrs['k_best'] for res in QC_results])
        Hk_energy = np.array([res.attrs['amp_max'] for res in QC_results])

        if len(QC_results) == 0:
            amp_stack_mean = np.zeros((len(kvals), len(Hvals)))
            H_best = np.nan
            k_best = np.nan
            H_uncert = np.nan
            k_uncert = np.nan
        else:
            if len(H_estimates) < 15:
                H_best = np.mean(H_estimates)
                k_best = np.mean(k_estimates)
                H_uncert = np.std(H_estimates)
                k_uncert = np.std(k_estimates)
            else:
                H_best = np.median(H_estimates)
                H_uncert = np.median(np.abs(H_estimates - H_best))
                k_best = np.median(k_estimates)
                k_uncert = np.median(np.abs(k_estimates - k_best))

            amp_stack = np.stack([res.amp_stack.values for res in QC_results])
            amp_stack_max = np.max(amp_stack, axis=(1, 2), keepdims=True)
            amp_stack_max[amp_stack_max == 0] = 1
            amp_stack_norm = amp_stack / amp_stack_max
            amp_stack_mean = np.mean(amp_stack_norm, axis=0)
        
        # tmp = amp_stack_mean.T / np.max(np.abs(amp_stack_mean.T))
        # if i == 0:
        #     print(np.min(tmp), np.max(tmp))
        ax[i].pcolormesh(
            HH, kk, _safe_normalize_abs(amp_stack_mean.T),
            cmap='binary', shading='auto', vmin=0, vmax=1
        )
        ax[i].scatter(
            H_estimates, k_estimates, c=Hk_energy, cmap='viridis', s=60, edgecolor='black',
            vmin=vmin, vmax=vmax
        )
        ax[i].scatter(
            H_best,
            k_best,
            marker="*",
            facecolor='tab:red',
            edgecolor='black',
            s=200
        )

        if i==len(QC_masks) - 1 or i==len(QC_masks) - 2:
            ax[i].text(
                0.01, 0.95,
                f"{H_best:.2f} ± {H_uncert:.2f} km \n{k_best:.2f} ± {k_uncert:.2f}",
                transform=ax[i].transAxes, fontsize=14, fontweight='bold',
                fontfamily='Times New Roman', va='top', ha='left',
                color='tab:red'
            )

        if i == 0:
            ax[i].set_title(title + "\n" + f"{QC_key[i]}({len(QC_results)})", fontweight='bold', fontfamily='Times New Roman')
        else:
            ax[i].set_title(f"{QC_key[i]}({len(QC_results)})", fontweight='bold', fontfamily='Times New Roman')
            
    fig.colorbar(
        ax[0].collections[0], cax=cb1_ax, label='Hk Energy (stack)', orientation='horizontal'
    )
    fig.colorbar(
        ax[0].collections[1], cax=cb2_ax, label='Hk Energy (individual)', orientation='horizontal'
    )

    ax_all.set_xlabel('H (km)', fontweight='bold', fontfamily='Times New Roman')
    ax_all.set_ylabel('Vp/Vs', fontweight='bold', fontfamily='Times New Roman')
    # ax[1].set_ylabel('Vp/Vs', fontweight='bold', fontfamily='Times New Roman')
    # ax[3].set_ylabel('Vp/Vs', fontweight='bold', fontfamily='Times New Roman')
    ax[3].set_xlabel('H (km)', fontweight='bold', fontfamily='Times New Roman') 
    ax[4].set_xlabel('H (km)', fontweight='bold', fontfamily='Times New Roman')

    pannel_idx = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    for i, ax_i in enumerate(ax):
        ax_i.text(
            0.01, 0.01, pannel_idx[i],
            transform=ax_i.transAxes, fontsize=16, fontweight='bold',
            fontfamily='Times New Roman', va='bottom', ha='left'
        )

    return fig

def hkSeq(Params, mode="both", plot=False, overwrite=None):
    """Sequential sediment (high-frequency) then crust (low-frequency) H-k.

    Only the files of the requested stage(s) are written; existing outputs
    raise ``FileExistsError`` unless ``overwrite`` (or ``IO.OVERWRITE``) is set.
    """
    if mode not in ["both", "low", "high"]:
        raise ValueError("Invalid mode. Supported modes are 'both', 'low', and 'high'.")
    IO_params = Params['IO']
    rootdir = IO_params['ROOT']
    HighFreq_RF_dir = os.path.join(rootdir, IO_params['HighFreq_RF'])
    LowFreq_RF_dir = os.path.join(rootdir, IO_params['LowFreq_RF'])
    savedir = os.path.join(rootdir, IO_params['SAVE'])
    figdir = os.path.join(rootdir, IO_params['FIGURE'])
    logdir = os.path.join(rootdir, IO_params['LOG'])
    if overwrite is None:
        overwrite = bool(IO_params.get('OVERWRITE', False))

    meta_params = Params['Meta']
    sta = meta_params['station']
    net = meta_params['network']

    _check_output_dir(savedir, rootdir, [HighFreq_RF_dir, LowFreq_RF_dir])
    planned = []
    if mode in ["both", "high"]:
        planned += _hk_output_files(
            savedir, figdir, net, sta, "HFreq", f"{net}.{sta}_HFreq_QC.png", plot
        )
    if mode in ["both", "low"]:
        low_figure = (
            f"{net}.{sta}_HFreq_LFreq_QC.png" if mode == "both" else f"{net}.{sta}_LFreq_QC.png"
        )
        planned += _hk_output_files(savedir, figdir, net, sta, "LFreq", low_figure, plot)
    _check_existing_outputs(planned, overwrite)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    handler = logging.FileHandler(
        os.path.join(logdir, f"{net}.{sta}_hkSeq.log"), mode="w", encoding="utf-8"
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
        _run_hkseq(
            Params, mode, plot, HighFreq_RF_dir, LowFreq_RF_dir, savedir, figdir, net, sta
        )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def _run_hkseq(Params, mode, plot, HighFreq_RF_dir, LowFreq_RF_dir, savedir, figdir, net, sta):
    logger.info(f"Starting H-k analysis for {net}.{sta}")
    provenance = software_provenance()
    for key, value in provenance.items():
        logger.info(f"{key}: {value}")

    if mode in ["both", "high"]:
        # High Frequency RF analysis

        sediment_params = Params['Model']['Sediment']
        Hparam_sediment = HkParam(
            name="H",
            vmin=sediment_params['H']['min'],
            vmax=sediment_params['H']['max'],
            dv=sediment_params['H']['delta']
        )
        kparam_sediment = HkParam(
            name="k",
            vmin=sediment_params['k']['min'],
            vmax=sediment_params['k']['max'],
            dv=sediment_params['k']['delta']
        )
        vp_sediment = sediment_params['Vp']
        vs_sediment = sediment_params['Vs']
        weight_sediment = sediment_params['weight']

        HighFreq_RF_list = init_RFtraces(HighFreq_RF_dir)
        hk_func = partial(
            Hk_analysis,
            H=Hparam_sediment,
            k=kparam_sediment,
            Vp=vp_sediment,
            Vs=vs_sediment,
            weight=weight_sediment,
            mode=1
        )
    

        HFreq_results = Parallel(n_jobs=-1)(
            delayed(hk_func)(rf) for rf in tqdm(HighFreq_RF_list)
        )
        HFreq_results = _with_provenance(HFreq_results, provenance)
        HFreq_QC_params = Params['QC']['HighFreq']
        HFreq_QC = HkQController(HFreq_results, HFreq_QC_params)

        mask0, _ = HFreq_QC.hk_energy_select(return_mask_only=True) # Energy select
        mask1, _ = HFreq_QC.elliptical_mad_select(return_mask_only=True) # Elliptical MAD select
        mask2    = mask0 & mask1 # Combined mask
        mask3, _ = HFreq_QC.mahalanobis_dist_select(return_mask_only=True) # Mahalanobis distance select

        logger.info(f"High Frequency RFs: {len(HFreq_results)} total")
        logger.info(f"High Frequency RFs after energy select: {np.sum(mask0)}")
        logger.info(f"High Frequency RFs after elliptical MAD select: {np.sum(mask1)}")
        logger.info(f"High Frequency RFs after [Energy & Elliptical] select: {np.sum(mask2)}")
        logger.info(f"High Frequency RFs after Mahalanobis distance select: {np.sum(mask3)}")

        logger.info("Mahalanobis distance-based QC is taken")
        H0_best, k0_best, H0_uncert, k0_uncert = _estimate_hk_summary(
            HFreq_results,
            mask3,
        )
        logger.info(f"High Frequency RFs: H = {H0_best:.2f} ± {H0_uncert:.2f} km")
        logger.info(f"High Frequency RFs: k = {k0_best:.2f} ± {k0_uncert:.2f} Vp/Vs")
        # Save results
        # datasets = []
        # for res in HFreq_results:
        #     res = res.expand_dims(rf=[0])
        #     datasets.append(res)
        # HFreq_results_all = xr.concat(datasets, dim='rf')
        # HFreq_results_all.attrs['station'] = sta
        # HFreq_results_all.attrs['network'] = net
        # HFreq_results_all.to_netcdf(os.path.join(savedir, f"{net}.{sta}_HFreq_Hk_results.nc"), mode='w')
        dump(HFreq_results, os.path.join(savedir, f"{net}.{sta}_HFreq_Hk_results.joblib"))

        # Save QC results
        # qc_datasets = []
        # for res, m in zip(HFreq_results, mask3):
        #     if m:
        #         res = res.expand_dims(rf=[0])
        #         qc_datasets.append(res)
        
        # Hk_results_QC = xr.concat(qc_datasets, dim='rf')
        # Hk_results_QC.attrs['station'] = sta
        # Hk_results_QC.attrs['network'] = net
        # Hk_results_QC.attrs['H_best'] = H0_best
        # Hk_results_QC.attrs['k_best'] = k0_best
        # Hk_results_QC.attrs['H_uncert'] = H0_uncert
        # Hk_results_QC.attrs['k_uncert'] = k0_uncert
        # Hk_results_QC.to_netcdf(os.path.join(savedir, f"{net}.{sta}_HFreq_Hk_results_QC.nc"), mode='w')
        HFreq_results_QC = [res for res, m in zip(HFreq_results, mask3) if m]
        dump(HFreq_results_QC, os.path.join(savedir, f"{net}.{sta}_HFreq_Hk_results_QC.joblib"))

        _write_qc_table(
            os.path.join(savedir, f"{net}.{sta}_HFreq_Hk_QC.dat"),
            HFreq_results,
            mask3,
        )
        logger.info("Hk analysis for High Frequency RFs completed and saved.")
        if plot and HFreq_results:
            QC_key = ["Energy Select", "Elliptical MAD Select", "Energy & Elliptical Combined", "Mahalanobis Select"]
            fig_title = f"{net}.{sta} (High Frequency RFs)"
        
            fig = plot_hk_QC(HFreq_results, QC_masks=[mask0, mask1, mask2, mask3], QC_key=QC_key, title=fig_title)
            fig.savefig(os.path.join(figdir, f"{net}.{sta}_HFreq_QC.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

    if mode in ["both", "low"]:
        # Low Frequency RF analysis
        crust_params = Params['Model']['Crust']
        Hparam_crust = HkParam(
            name="H",
            vmin=crust_params['H']['min'],
            vmax=crust_params['H']['max'],
            dv=crust_params['H']['delta']
        )
        kparam_crust = HkParam(
            name="k",
            vmin=crust_params['k']['min'],
            vmax=crust_params['k']['max'],
            dv=crust_params['k']['delta']
        )
        vp_crust = crust_params['Vp']
        vs_crust = crust_params['Vs']
        weight_crust = crust_params['weight']

        if mode == "low":
            H0_best = np.nan
            k0_best = np.nan

        if np.isnan(H0_best) or np.isnan(k0_best) or H0_best <= 0.4:
            logger.warning("No valid H and k estimates from sediment layer or Only consider crust layer")
            logger.info("Only use one-layer Hk analysis for crust")
            hk_func = partial(
                Hk_analysis,
                H=Hparam_crust,
                k=kparam_crust,
                Vp=vp_crust,
                Vs=vs_crust,
                weight=weight_crust,
                mode=1
            )
        else:
            logger.info("Use two-layer Hk analysis for crust")
            hk_func = partial(
            Hk_analysis,
            H=Hparam_crust,
            k=kparam_crust,
            Vp=vp_crust,
            Vs=vs_crust,
            weight=weight_crust,
            mode=2,
            h0=H0_best,
            vp0=sediment_params['Vp'],
            vs0= sediment_params['Vs'],
            k0=k0_best
            )
        
        LowFreq_RF_list = init_RFtraces(LowFreq_RF_dir)
        LFreq_results = Parallel(n_jobs=-1)(
            delayed(hk_func)(rf) for rf in tqdm(LowFreq_RF_list)
        )
        LFreq_results = _with_provenance(LFreq_results, provenance)
        LFreq_QC_params = Params['QC']['LowFreq']
        LFreq_QC = HkQController(LFreq_results, LFreq_QC_params)
        mask0, _ = LFreq_QC.hk_energy_select(return_mask_only=True)
        mask1, _ = LFreq_QC.elliptical_mad_select(return_mask_only=True)
        mask2    = mask0 & mask1
        mask3, _ = LFreq_QC.mahalanobis_dist_select(return_mask_only=True)
        logger.info(f"Low Frequency RFs: {len(LFreq_results)} total")
        logger.info(f"Low Frequency RFs after energy select: {np.sum(mask0)}")
        logger.info(f"Low Frequency RFs after elliptical MAD select: {np.sum(mask1)}")
        logger.info(f"Low Frequency RFs after [Energy & Elliptical] select: {np.sum(mask2)}")
        logger.info(f"Low Frequency RFs after Mahalanobis distance select: {np.sum(mask3)}")

        logger.info("Mahalanobis distance-based QC is taken")
        H1_best, k1_best, H1_uncert, k1_uncert = _estimate_hk_summary(
            LFreq_results,
            mask3,
        )
        logger.info(f"Low Frequency RFs: H = {H1_best:.2f} ± {H1_uncert:.2f} km")
        logger.info(f"Low Frequency RFs: k = {k1_best:.2f} ± {k1_uncert:.2f} Vp/Vs")
        # Save results
        dump(LFreq_results, os.path.join(savedir, f"{net}.{sta}_LFreq_Hk_results.joblib"))
        LFreq_results_QC = [res for res, m in zip(LFreq_results, mask3) if m]
        dump(LFreq_results_QC, os.path.join(savedir, f"{net}.{sta}_LFreq_Hk_results_QC.joblib"))
        _write_qc_table(
            os.path.join(savedir, f"{net}.{sta}_LFreq_Hk_QC.dat"),
            LFreq_results,
            mask3,
        )


        logger.info("Hk analysis for Low Frequency RFs completed and saved.")
        if plot and LFreq_results:
            QC_key = ["Energy Select", "Elliptical MAD Select", "Energy & Elliptical Combined", "Mahalanobis Select"]
            fig_title = f"{net}.{sta} (Low Frequency RFs)"
            fig = plot_hk_QC(LFreq_results, QC_masks=[mask0, mask1, mask2, mask3], QC_key=QC_key, title=fig_title)
            if mode == "both":
                figfile = f"{net}.{sta}_HFreq_LFreq_QC.png"
            else:
                figfile = f"{net}.{sta}_LFreq_QC.png"
            
            fig.savefig(os.path.join(figdir, figfile), dpi=300, bbox_inches='tight')
            plt.close(fig)

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
    
    plot_flag = args.plot

    hkSeq(params, mode=args.mode, plot=plot_flag, overwrite=args.overwrite or None)
