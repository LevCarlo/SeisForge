import numpy as np
from scipy.stats import f as f_dist
from obspy import Stream, Trace, read
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

class RFtrace():
    def __init__(self, trace: Trace=None, file=None, slowness=None, baz=None, slowness_key='user0', baz_key='user1'):
        if trace is not None:
            self.trace = trace
            self.file = file
        elif file is not None:
            self.trace = read(file)[0]
            self.file = file
        else:
            raise ValueError("Must provide either `trace` or `file` to initialize RFtrace.")

        self.slowness = slowness if slowness is not None else self._get_header(slowness_key)
        self.baz = baz if baz is not None else self._get_header(baz_key)


    def _get_header(self, key):
        try:
            return getattr(self.trace.stats.sac, key)
        except:
            return None
    def __repr__(self):
        return f"RFtrace(file={self.file}, \nslowness={self.slowness}, \nbaz={self.baz})"

class RFstream():
    """
    A class for quality control and visualization of Receiver Functions (RFs).

    Attributes:
        files (list): List of SAC file paths.
        st (Stream): ObsPy Stream object containing RF traces.
    """
    def __init__(self, files=None, st=None, qc_metrics=None):
        """
        Initialize RFstream with file list and optional ObsPy Stream.
        """
        self.files = files
        self.st = st
        self.qc_metrics = qc_metrics if qc_metrics is not None else []

    def load_data(self):
        """
        Load SAC files into an ObsPy Stream.
        """
        st = Stream()
        for file in self.files:
            try:
                st += read(file)
            except Exception as e:
                print(f"Error reading {file}: {e}")
        self.st = st
    
    def copy(self):
        """
        Create a deep copy of the RFstream instance, duplicating file references and Stream object.
        """
        return RFstream(files=self.files.copy(), st=self.st.copy() if self.st else None, qc_metrics=self.qc_metrics.copy())
    
    def SNR_select(self, threshold=5, reverse=False):
        """
        Select traces based on a signal-to-noise ratio threshold.

        Args:
            threshold (float): Minimum SNR required to retain a trace.
            reverse (bool): If True, keep traces below the threshold.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        
        new = self.copy()


        to_keep = []
        for i, tr in enumerate(self.st):
            isnr = tr.stats.sac.user2
            new.qc_metrics.append(isnr)
            if isnr >= threshold:
                to_keep.append(i)
        
        if reverse:
            to_keep = [i for i in range(len(self.st)) if i not in to_keep]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new
    
    def slowness_select(self, slow_min=0.04, slow_max=0.1, reverse=False):
        """
        Select traces based on slowness range.

        Args:
            slow_min (float): Minimum slowness (s/deg).
            slow_max (float): Maximum slowness (s/deg).
            reverse (bool): If True, keep traces outside the range.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        new = self.copy()

        to_keep = []
        for i, tr in enumerate(self.st):
            slowness = tr.stats.sac.user0
            new.qc_metrics.append(slowness)
            if slow_min <= slowness <= slow_max:
                to_keep.append(i)
        
        if reverse:
            to_keep = [i for i in range(len(self.st)) if i not in to_keep]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new

    def P_amp_select(self, tmin=-2, tmax=2, window=[-5, 30], reverse=False):
        """
        Select traces where the maximum amplitude is positive and falls within a P-wave window.

        Args:
            tmin (float): Minimum expected time of P-wave peak.
            tmax (float): Maximum expected time of P-wave peak.
            window (list): Time window [start, end] in seconds.
            reverse (bool): If True, keep traces that do not meet criteria.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        new = self.copy()

        b = new.st[0].stats.sac.b
        e = new.st[0].stats.sac.e
        npts = new.st[0].stats.npts
        t = np.linspace(b, e, npts)
        t_idx = (t >= window[0]) & (t <= window[1])

        to_keep = []
        for i, tr in enumerate(new.st):
            max_amp_idx = np.argmax(np.abs(tr.data[t_idx]))
            max_amp_time = t[t_idx][max_amp_idx]
            max_amp_value = tr.data[t_idx][max_amp_idx]
            new.qc_metrics.append([max_amp_time, max_amp_value])
            if tmin <= max_amp_time <= tmax and max_amp_value > 0:
                to_keep.append(i)

        if reverse:
            to_keep = [i for i in range(len(new.st)) if i not in to_keep]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new


    def MAD_select(self, threshold=2.5, window=[-5, 30], reverse=False):
        """
        Select traces using robust outlier detection based on Median Absolute Deviation (MAD) of variance.

        Args:
            threshold (float): MAD threshold for filtering.
            window (list): Time window [start, end] for computation.
            reverse (bool): If True, retain outliers instead of filtering them.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        
        new = self.copy()

        b = new.st[0].stats.sac.b
        e = new.st[0].stats.sac.e
        npts = new.st[0].stats.npts
        t = np.linspace(b, e, npts)
        t_idx = (t >= window[0]) & (t <= window[1])

        var_rf = np.array([np.var(tr.data[t_idx]) for tr in new.st])
        med = np.median(var_rf)
        mad = 1.4826 * np.median(np.abs(var_rf - med))
        robust_score = np.abs((var_rf - med) / mad)
        new.qc_metrics = robust_score.tolist()

        if reverse:
            to_keep = [i for i in range(len(robust_score)) if robust_score[i] > threshold]
        else:
            to_keep = [i for i in range(len(robust_score)) if robust_score[i] <= threshold]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new
    
    def f_test_select(self, threshold=0.05, window=[-5, 30], reverse=False):
        """
        Select traces by comparing residual variance using an F-test.

        Args:
            threshold (float): P-value threshold. Retain traces with p >= threshold.
            window (list): Time window [start, end] for analysis.
            reverse (bool): If True, keep traces with p < threshold.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        
        new = self.copy()

        b = new.st[0].stats.sac.b
        e = new.st[0].stats.sac.e
        npts = new.st[0].stats.npts
        t = np.linspace(b, e, npts)
        t_idx = (t >= window[0]) & (t <= window[1])

        rf_data = np.array([tr.data[t_idx] for tr in new.st])
        ref_rf = np.mean(rf_data, axis=0)
        res_all = (rf_data - ref_rf).reshape(-1)

        def _ftest(res1, pars1, res2, pars2):
            N1 = len(res1)
            N2 = len(res2)
            
            dof1 = N1 - pars1
            dof2 = N2 - pars2

            Ea_1 = np.sum(res1**2)
            Ea_2 = np.sum(res2**2)

            Fobs = (Ea_1 / dof1) / (Ea_2 / dof2)

            P = 1 - (f_dist.cdf(Fobs, dof1, dof2) - f_dist.cdf(1 / Fobs, dof1, dof2))
            return P

        Pvals = []
        for i in range(rf_data.shape[0]):
            reduced_rf = np.delete(rf_data, i, axis=0)
            ref_reduced = np.mean(reduced_rf, axis=0)
            res_reduced = (reduced_rf - ref_reduced).reshape(-1)
            pval = _ftest(res_all, 1, res_reduced, 1)
            Pvals.append(pval)
        
        new.qc_metrics = Pvals

        if reverse:
            to_keep = [i for i in range(len(Pvals)) if Pvals[i] < threshold]
        else:
            to_keep = [i for i in range(len(Pvals)) if Pvals[i] >= threshold]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new
    
    def CC_select(self, threshold=0.4, window=[-5, 30], reverse=False):
        """
        Select traces based on cross-correlation with the average waveform.

        Args:
            threshold (float): Minimum correlation coefficient to retain a trace.
            window (list): Time window [start, end] for analysis.
            reverse (bool): If True, keep traces below the threshold.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        
        new = self.copy()

        b = new.st[0].stats.sac.b
        e = new.st[0].stats.sac.e
        npts = new.st[0].stats.npts
        t = np.linspace(b, e, npts)
        t_idx = (t >= window[0]) & (t <= window[1])

        rf_data = np.array([tr.data[t_idx] for tr in new.st])
        ave_rf = np.mean(rf_data, axis=0)
        ave_rf -= np.mean(ave_rf)
        ave_rf /= np.std(ave_rf)

        cc_values = []
        for i in range(rf_data.shape[0]):
            rf_i = (rf_data[i] - np.mean(rf_data[i])) / np.std(rf_data[i])
            cc = np.corrcoef(ave_rf, rf_i)[0, 1]
            cc_values.append(cc)
        
        new.qc_metrics = cc_values

        if reverse:
            to_keep = [i for i in range(len(cc_values)) if cc_values[i] < threshold]
        else:
            to_keep = [i for i in range(len(cc_values)) if cc_values[i] >= threshold]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new
    
    def plot(self, window=[-5, 30], save=False, save_path=None, **kwargs):
        """
        Plot normalized RF traces by slowness and the average waveform.

        Args:
            window (list): Time window [start, end] to display.
            **kwargs: Additional arguments for matplotlib customization.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
            
        b = self.st[0].stats.sac.b
        e = self.st[0].stats.sac.e
        npts = self.st[0].stats.npts
        t = np.linspace(b, e, npts)
        t_idx = (t >= window[0]) & (t <= window[1])

        fig, ax = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[0.15, 1.0], sharex=True)
        rf_data = []
        for tr in self.st:
            slow = tr.stats.sac.user0
            data_norm = tr.data[t_idx] / np.max(np.abs(tr.data[t_idx])) * 1.2e-3
            shifted = data_norm + slow
            ax[1].plot(t[t_idx], shifted, c='k', lw=0.5)
            ax[1].fill_between(t[t_idx], 
                               slow,
                               shifted, 
                               where=(shifted > slow),
                               facecolor='tab:red'
                                 )
            ax[1].fill_between(t[t_idx],
                               slow,
                               shifted, 
                               where=(shifted < slow),
                               facecolor='blue'
                                 )
            rf_data.append(tr.data[t_idx])
        
        ave_rf = np.mean(rf_data, axis=0)
        ax[0].plot(t[t_idx], ave_rf, c='k', lw=1.0)
        ax[0].fill_between(t[t_idx],
                           ave_rf,
                            where=(ave_rf > 0),
                            facecolor='tab:red'
        )
        ax[0].fill_between(t[t_idx],
                           ave_rf,
                            where=(ave_rf < 0),
                            facecolor='blue'
        )


        ax[1].set_ylabel('Slowness (s/deg)', fontdict={'family': 'Times New Roman', 'weight': 'bold'})
        ax[1].set_xlabel('Time (s)', fontdict={'family': 'Times New Roman', 'weight': 'bold'})
        ax[1].set_xlim(window[0], window[1])
        ax[1].axvline(0, color='k', lw=1.0, ls='--')

        ax[0].set_ylabel('Amplitude', fontdict={'family': 'Times New Roman', 'weight': 'bold'})
        ax[0].tick_params(top=True, labeltop=True)
        ax[0].axvline(0, color='k', lw=1.0, ls='--')
        fig.tight_layout()

        if save:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()

        


if __name__ == "__main__":
    import os
    import glob
    datadir = "/home/mengjie/data/CREST.Data/RFs/RFs_SEISPY"
    sta = "IW.SMCO"
    gauss_factor = [2.5, 5.0]
    sta_datadir = os.path.join(datadir, sta)
    filelst0 = glob.glob(os.path.join(sta_datadir, f"Gauss_{gauss_factor[0]}", f"*R*"))
    rf0 = RFstream(files=filelst0)
    rf0.load_data()
    rf0_1 = rf0.SNR_select()