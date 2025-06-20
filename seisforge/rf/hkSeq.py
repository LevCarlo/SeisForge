import os
from seisforge.rf.rf import RFstream, RFtrace
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import argparse
import yaml
from functools import partial
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging


radius=6371.0 # Earth radius in km

class HkParam:
    def __init__(self, name: str, vmin, vmax, dv=None, values=None):

        self.name = name
        self.vmin = vmin
        self.vmax = vmax
        self.dv = dv
        self.values = values

        if dv is not None:
            N = int((vmax - vmin) / dv) + 1
            self.values = np.linspace(vmin, vmax, N)
    
    def __repr__(self):
        return f"HkParam(name={self.name}, vmin={self.vmin}, vmax={self.vmax}, dv={self.dv})"

class HkStack_classic:
    def __init__(self, RFtrace: RFtrace, H: HkParam, k: HkParam, Vp=6.3, Vs=None, weight=[0.5, 0.2, 0.3]):
        self.RFtrace = RFtrace
        self.H = H
        self.k = k
        self.Vp = Vp
        self.Vs = Vs
        self.weight = weight

    def stack(self):
        p = self.RFtrace.slowness / radius  # s/rad --> s/km

        H_grid = self.H.values
        k_grid = self.k.values

        if self.Vs is None:
            vs = self.Vp / k_grid
            vp = np.full_like(k_grid, self.Vp)
        else:
            vs = np.full_like(k_grid, self.Vs)
            vp = self.Vs * k_grid
        
        t_Ps = np.outer(np.sqrt(1 / vs**2 - p**2) - np.sqrt(1 / vp**2 - p**2), H_grid)
        t_PpPs = np.outer(np.sqrt(1 / vs**2 - p**2) + np.sqrt(1 / vp**2 - p**2), H_grid)
        t_PsPs = np.outer(2 * np.sqrt(1 / vs**2 - p**2), H_grid)

        tr = self.RFtrace.trace
        b = tr.stats.sac.b
        dt = tr.stats.sac.delta
        npts = tr.stats.sac.npts
        t_axis = b + dt * np.arange(npts)

        t_Ps_idx = self._times2idx(t_axis, t_Ps)
        t_PpPs_idx = self._times2idx(t_axis, t_PpPs)
        t_PsPs_idx = self._times2idx(t_axis, t_PsPs)

        data_norm = self._amp_correct(method="seispy")
        amp_Ps = data_norm[t_Ps_idx]
        amp_PpPs = data_norm[t_PpPs_idx]
        amp_PsPs = data_norm[t_PsPs_idx]

        amp_stack = self.weight[0] * amp_Ps + self.weight[1] * amp_PpPs - self.weight[2] * amp_PsPs
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

    def _times2idx(self, t_axis, t_grid):
        t_grid_flat = t_grid.ravel()
        idx_flat = np.abs(t_axis[:, None] - t_grid_flat[None, :]).argmin(axis=0)
        return idx_flat.reshape(t_grid.shape)

    def _amp_correct(self, method="seispy", P_wins=[-2, 2]):
        """
        Apply amplitude correction to the RF trace before H-k stacking.

        Parameters:
        - method: str, default "seispy".
        "seispy": empirical correction based on slowness to unify energy scaling.
        "max_amp": normalize the trace by its maximum amplitude within the P-wave window.
        - P_wins: list, default [-2, 2]. Time window for P

        Returns:
        - Corrected RF data (np.ndarray).
     
        """
        if method == "seispy":
            p = self.RFtrace.slowness / radius
            amp_corr = 151.5478 * p**2 + 3.2896 * p + 0.2618
            data = self.RFtrace.trace.data
            return amp_corr * data
        elif method == "max_amp":
            tr = self.RFtrace.trace
            b = tr.stats.sac.b
            dt = tr.stats.sac.delta
            npts = tr.stats.sac.npts
            t_axis = b + dt * np.arange(npts)
            P_idx = (t_axis >= P_wins[0]) & (t_axis <= P_wins[1])
            data = self.RFtrace.trace.data
            max_amp = np.max(np.abs(data[P_idx]))
            return data / max_amp
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods are 'seispy' and 'max_amp'.")


    def plot(self):
        amp_stack, _, amp = self.stack()
        H_best, k_best, _ = self.estimate_best_Hk()
        print(f"Best H: {H_best:.2f} km, Best k: {k_best:.2f} Vp/Vs")
        H_grid = self.H.values
        k_grid = self.k.values
        fig, ax = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
        kk, HH = np.meshgrid(k_grid, H_grid, indexing='xy')
        cmap = "viridis"
        ax[0, 0].pcolormesh(HH, kk, amp[0].T / np.max(np.abs(amp[0])), cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax[0, 1].pcolormesh(HH, kk, amp[1].T / np.max(np.abs(amp[1])), cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax[1, 0].pcolormesh(HH, kk, amp[2].T / np.max(np.abs(amp[2])), cmap=cmap, shading='auto', vmin=0, vmax=1)
        img = ax[1, 1].pcolormesh(HH, kk, amp_stack.T / np.max(np.abs(amp_stack.T)), cmap=cmap, shading='auto', vmin=0, vmax=1)

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
        cbar = fig.colorbar(img, ax=ax.ravel().tolist(), shrink=0.55)
        fig.tight_layout()

class HkStack:
    def __init__(self, RFtrace: RFtrace, H: HkParam, k: HkParam, Vp: float = 6.3, Vs: float = None, weight: list = [0.5, 0.2, 0.3],
                 mode: int = 1, h0: float = 0.0, vp0: float = 4.5, vs0: float = None, k0: float = 2.0):
        self.RFtrace = RFtrace
        self.H = H
        self.k = k
        self.Vp = Vp
        self.Vs = Vs
        self.weight = weight
        self.mode = mode
        self.h0 = h0
        self.vp0 = vp0
        self.vs0 = vs0
        self.k0 = k0
    
    def _calc_phase_times(self):
        p = self.RFtrace.slowness / radius  # s/rad --> s/km

        H_grid = self.H.values
        k_grid = self.k.values
        
        if self.Vs is None:
            vs = self.Vp / k_grid
            vp = np.full_like(k_grid, self.Vp)
        else:
            vs = np.full_like(k_grid, self.Vs)
            vp = self.Vs * k_grid
        
        if self.mode == 1:
            t_Ps = np.outer(np.sqrt(1 / vs**2 - p**2) - np.sqrt(1 / vp**2 - p**2), H_grid)
            t_PpPs = np.outer(np.sqrt(1 / vs**2 - p**2) + np.sqrt(1 / vp**2 - p**2), H_grid)
            t_PsPs = np.outer(2 * np.sqrt(1 / vs**2 - p**2), H_grid)
        elif self.mode == 2:
            h0 = self.h0
            if self.vs0 is None:
                vs0 = np.full_like(k_grid, self.vp0 / self.k0)
                vp0 = np.full_like(k_grid, self.vp0)
            else:
                vs0 = np.full_like(k_grid, self.vs0)
                vp0 = vs0 * self.k0
            
            t_Ps = np.outer(np.sqrt(1 / vs**2 - p**2) - np.sqrt(1 / vp**2 - p**2), H_grid - h0) + \
                           np.outer(np.sqrt(1 / vs0**2 - p**2) - np.sqrt(1 / vp0**2 - p**2), h0)
            t_PpPs = np.outer(np.sqrt(1 / vs**2 - p**2) + np.sqrt(1 / vp**2 - p**2), H_grid - h0) + \
                             np.outer(np.sqrt(1 / vs0**2 - p**2) + np.sqrt(1 / vp0**2 - p**2), h0)
            t_PsPs = np.outer(2 * np.sqrt(1 / vs**2 - p**2), H_grid - h0) + \
                             np.outer(2 * np.sqrt(1 / vs0**2 - p**2), h0)
        else:
            raise ValueError("Invalid mode. Supported modes are 1 and 2.")
        return t_Ps, t_PpPs, t_PsPs

    def stack(self, smooth=False):
        t_Ps, t_PpPs, t_PsPs = self._calc_phase_times()

        tr = self.RFtrace.trace
        b = tr.stats.sac.b
        dt = tr.stats.sac.delta
        npts = tr.stats.sac.npts
        t_axis = b + dt * np.arange(npts)

        t_Ps_idx = self._times2idx(t_axis, t_Ps)
        t_PpPs_idx = self._times2idx(t_axis, t_PpPs)
        t_PsPs_idx = self._times2idx(t_axis, t_PsPs)

        data_norm = self._amp_correct(method="seispy")
        amp_Ps = data_norm[t_Ps_idx]
        amp_PpPs = data_norm[t_PpPs_idx]
        amp_PsPs = data_norm[t_PsPs_idx]

        amp_stack = self.weight[0] * amp_Ps + self.weight[1] * amp_PpPs - self.weight[2] * amp_PsPs

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
                'rf_file': self.RFtrace.file
            }
        )

        return hk_ds

    def _times2idx(self, t_axis, t_grid):
        t_grid_flat = t_grid.ravel()
        idx_flat = np.abs(t_axis[:, None] - t_grid_flat[None, :]).argmin(axis=0)
        return idx_flat.reshape(t_grid.shape)

    def _amp_correct(self, method="seispy", P_wins=[-2, 2]):
        if method == "seispy":
            p = self.RFtrace.slowness / radius
            amp_corr = 151.5478 * p**2 + 3.2896 * p + 0.2618
            data = self.RFtrace.trace.data
            return amp_corr * data
        elif method == "max_amp":
            tr = self.RFtrace.trace
            b = tr.stats.sac.b
            dt = tr.stats.sac.delta
            npts = tr.stats.sac.npts
            t_axis = b + dt * np.arange(npts)
            P_idx = (t_axis >= P_wins[0]) & (t_axis <= P_wins[1])
            data = self.RFtrace.trace.data
            max_amp = np.max(np.abs(data[P_idx]))
            return data / max_amp
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods are 'seispy' and 'max_amp'.")
    
    def plot(self):
        hk_ds = self.stack()
        amp_stack = hk_ds.amp_stack.values
        amp = [hk_ds.amp_Ps.values, hk_ds.amp_PpPs.values, hk_ds.amp_PsPs.values]
        H_best = hk_ds.attrs['H_best']
        k_best = hk_ds.attrs['k_best']
        print(f"Best H: {H_best:.2f} km, Best k: {k_best:.2f} Vp/Vs")

        H_grid = self.H.values
        k_grid = self.k.values
        fig, ax = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
        kk, HH = np.meshgrid(k_grid, H_grid, indexing='xy')
        cmap = "viridis"
        ax[0, 0].pcolormesh(HH, kk, amp[0].T / np.max(np.abs(amp[0])), cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax[0, 1].pcolormesh(HH, kk, amp[1].T / np.max(np.abs(amp[1])), cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax[1, 0].pcolormesh(HH, kk, amp[2].T / np.max(np.abs(amp[2])), cmap=cmap, shading='auto', vmin=0, vmax=1)
        img = ax[1, 1].pcolormesh(HH, kk, amp_stack.T / np.max(np.abs(amp_stack.T)), cmap=cmap, shading='auto', vmin=0, vmax=1)

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
        cbar = fig.colorbar(img, ax=ax.ravel().tolist(), shrink=0.55)

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
        H_estimates = np.array([res.attrs['H_best'] for res in self.hk_results])
        k_estimates = np.array([res.attrs['k_best'] for res in self.hk_results])
        Hk_energy = np.array([res.attrs['amp_max'] for res in self.hk_results])

        mahalanobis_dist_threshold = self.QC_params['mahalanobis_dist_threshold']

        X = np.column_stack((H_estimates, k_estimates))
        E = Hk_energy.reshape(-1, 1)
        E_sum = np.sum(E)

        # weighted mean
        mu = np.sum(X * E, axis=0) / E_sum

        # weighted covariance
        X_centered = X - mu
        cov = (E * X_centered).T @ X_centered / E_sum
        cov_inv = np.linalg.inv(cov)

        # Mahalanobis distance
        m_dist = np.sqrt(np.sum((X_centered @ cov_inv) * X_centered, axis=1))
        mask = m_dist < mahalanobis_dist_threshold
        if return_mask_only:
            QC_results = None
        else:
            QC_results = [res for res, m in zip(self.hk_results, mask) if m]
        return mask, QC_results

def elliptical_mad_filter(x0, x1, sigma=2.0):
    """
    2D elliptical outlier filtering using the median and MAD (Median Absolute Deviation)
    """
    x0_median = np.median(x0)
    x1_median = np.median(x1)
    x0_mad = np.median(np.abs(x0 - x0_median))
    x1_mad = np.median(np.abs(x1 - x1_median))

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
    RF_files = [f for f in os.listdir(RFs_datadir) if f.endswith(suffix)]
    for rf_file in RF_files:
        rf = RFtrace(file=os.path.join(RFs_datadir, rf_file))
        RFtrace_list.append(rf)
    return RFtrace_list

def plot_hk_QC(Hk_results, QC_masks=None, QC_key=None, title=None):
    all_masks = np.ones(len(Hk_results), dtype=bool)
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
    cb2_ax = fig.add_subplot(gs[-1, 0])

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

        H_best = np.median(H_estimates)
        H_mad = np.median(np.abs(H_estimates - H_best))
        k_best = np.median(k_estimates)
        k_mad = np.median(np.abs(k_estimates - k_best))

        if len(QC_results) == 0:
            amp_stack_mean = np.zeros((len(kvals), len(Hvals)))
        else:
            amp_stack = np.stack([res.amp_stack.values for res in QC_results])
            amp_stack_max = np.max(amp_stack, axis=(1, 2), keepdims=True)
            amp_stack_norm = amp_stack / amp_stack_max
            amp_stack_mean = np.mean(amp_stack_norm, axis=0)

        ax[i].pcolormesh(
            HH, kk, amp_stack_mean.T / np.max(np.abs(amp_stack_mean.T)),
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
                f"{H_best:.2f} ± {H_mad:.2f} km \n{k_best:.2f} ± {k_mad:.2f}",
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


def hkSeq(Params, plot=False):
    IO_params = Params['IO']
    rootdir = IO_params['ROOT']
    HighFreq_RF_dir = os.path.join(rootdir, IO_params['HighFreq_RF'])
    LowFreq_RF_dir = os.path.join(rootdir, IO_params['LowFreq_RF'])
    savedir = os.path.join(rootdir, IO_params['SAVE'])
    figdir = os.path.join(rootdir, IO_params['FIGURE'])
    logdir = os.path.join(rootdir, IO_params['LOG'])
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(logdir, 'HkSeq.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='w'
    )

    meta_params = Params['Meta']
    sta = meta_params['station']
    net = meta_params['network']
    logging.info(f"Starting H-k analysis for {net}.{sta}")

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
    HFreq_QC_params = Params['QC']['HighFreq']
    HFreq_QC = HkQController(HFreq_results, HFreq_QC_params)

    mask0, _ = HFreq_QC.hk_energy_select(return_mask_only=True) # Energy select
    mask1, _ = HFreq_QC.elliptical_mad_select(return_mask_only=True) # Elliptical MAD select
    mask2    = mask0 & mask1 # Combined mask
    mask3, _ = HFreq_QC.mahalanobis_dist_select(return_mask_only=True) # Mahalanobis distance select

    logging.info(f"High Frequency RFs: {len(HFreq_results)} total")
    logging.info(f"High Frequency RFs after energy select: {np.sum(mask0)}")
    logging.info(f"High Frequency RFs after elliptical MAD select: {np.sum(mask1)}")
    logging.info(f"High Frequency RFs after [Energy & Elliptical] select: {np.sum(mask2)}")
    logging.info(f"High Frequency RFs after Mahalanobis distance select: {np.sum(mask3)}")

    H_estimates = np.array([res.attrs['H_best'] for res in HFreq_results])
    k_estimates = np.array([res.attrs['k_best'] for res in HFreq_results])
    logging.info(f"Mahalanobis distance-based QC is taken")
    H0_best = np.median(H_estimates[mask3])
    k0_best = np.median(k_estimates[mask3])
    H0_mad = np.median(np.abs(H_estimates[mask3] - H0_best))
    k0_mad = np.median(np.abs(k_estimates[mask3] - k0_best))
    logging.info(f"High Frequency RFs: H = {H0_best:.2f} ± {H0_mad:.2f} km")
    logging.info(f"High Frequency RFs: k = {k0_best:.2f} ± {k0_mad:.2f} Vp/Vs")
    # Save results
    Hk_results_all = xr.concat(HFreq_results, dim='rf')
    Hk_results_all.attrs['station'] = sta
    Hk_results_all.attrs['network'] = net
    Hk_results_all.to_netcdf(os.path.join(savedir, f"{sta}.{net}_HFreq_Hk_results.nc"), mode='w')
    with open(os.path.join(savedir, f"{sta}.{net}_HFreq_Hk_QC.dat"), 'w') as f:
        f.write("# filename H_best k_best Hk_energy\n")
        QC_results = [res for res, m in zip(HFreq_results, mask3) if m]
        for res in QC_results:
            f.write(
                f"{os.path.basename(res.attrs['rf_file'])} "
                f"{res.attrs['H_best']:.4f} {res.attrs['k_best']:.4f} {res.attrs['amp_max']:.4f}\n"
            )
                    
    if plot:
        QC_key = ["Energy Select", "Elliptical MAD Select", "Energy & Elliptical Combined", "Mahalanobis Select"]
        fig_title = f"{sta}.{net} (High Frequency RFs)"
        
        fig = plot_hk_QC(HFreq_results, QC_masks=[mask0, mask1, mask2, mask3], QC_key=QC_key, title=fig_title)
        fig.savefig(os.path.join(figdir, f"{sta}.{net}_HFreq_QC.png"), dpi=300, bbox_inches='tight')

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

    if np.isnan(H0_best) or np.isnan(k0_best) or H0_best <= 0.25:
        logging.warning("No valid H and k estimates from sediment layer")
        logging.info("Only use one-layer Hk analysis for crust")
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
        logging.info(f"Use two-layer Hk analysis for crust")
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
    LFreq_QC_params = Params['QC']['LowFreq']
    LFreq_QC = HkQController(LFreq_results, LFreq_QC_params)
    mask0, _ = LFreq_QC.hk_energy_select(return_mask_only=True)
    mask1, _ = LFreq_QC.elliptical_mad_select(return_mask_only=True)
    mask2    = mask0 & mask1
    mask3, _ = LFreq_QC.mahalanobis_dist_select(return_mask_only=True)
    logging.info(f"Low Frequency RFs: {len(LFreq_results)} total")
    logging.info(f"Low Frequency RFs after energy select: {np.sum(mask0)}")
    logging.info(f"Low Frequency RFs after elliptical MAD select: {np.sum(mask1)}")
    logging.info(f"Low Frequency RFs after [Energy & Elliptical] select: {np.sum(mask2)}")
    logging.info(f"Low Frequency RFs after Mahalanobis distance select: {np.sum(mask3)}")

    H_estimates = np.array([res.attrs['H_best'] for res in LFreq_results])
    k_estimates = np.array([res.attrs['k_best'] for res in LFreq_results])
    logging.info(f"Mahalanobis distance-based QC is taken")
    H1_best = np.median(H_estimates[mask3])
    k1_best = np.median(k_estimates[mask3])
    H1_mad = np.median(np.abs(H_estimates[mask3] - H1_best))
    k1_mad = np.median(np.abs(k_estimates[mask3] - k1_best))
    logging.info(f"Low Frequency RFs: H = {H1_best:.2f} ± {H1_mad:.2f} km")
    logging.info(f"Low Frequency RFs: k = {k1_best:.2f} ± {k1_mad:.2f} Vp/Vs")
    # Save results
    Hk_results_all = xr.concat(LFreq_results, dim='rf')
    Hk_results_all.attrs['station'] = sta
    Hk_results_all.attrs['network'] = net   
    Hk_results_all.to_netcdf(os.path.join(savedir, f"{sta}.{net}_LFreq_Hk_results.nc"), mode='w')
    with open(os.path.join(savedir, f"{sta}.{net}_LFreq_Hk_QC.dat"), 'w') as f:
        f.write("# filename H_best k_best Hk_energy\n")
        QC_results = [res for res, m in zip(LFreq_results, mask3) if m]
        for res in QC_results:
            f.write(
                f"{os.path.basename(res.attrs['rf_file'])} "
                f"{res.attrs['H_best']:.4f} {res.attrs['k_best']:.4f} {res.attrs['amp_max']:.4f}\n"
            )


    if plot:
        QC_key = ["Energy Select", "Elliptical MAD Select", "Energy & Elliptical Combined", "Mahalanobis Select"]
        fig_title = f"{sta}.{net} (Low Frequency RFs)"
        fig = plot_hk_QC(LFreq_results, QC_masks=[mask0, mask1, mask2, mask3], QC_key=QC_key, title=fig_title)
        fig.savefig(os.path.join(figdir, f"{sta}.{net}_LFreq_QC.png"), dpi=300, bbox_inches='tight')
    logging.info(f"Completed H-k analysis for {net}.{sta}")

    
    

def load_parse_args():
    parser = argparse.ArgumentParser(description="Sequential Hk Analysis")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Hk analysis configuration file in YAML format")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the results if set")
    return parser.parse_args()

def main():
    args = load_parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    
    plot_flag = args.plot

    hkSeq(params, plot=plot_flag)
