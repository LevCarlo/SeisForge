import os
import glob
import numpy as np
import shutil
import logging
from scipy.stats import f as f_dist
from obspy import Stream, Trace, read
import matplotlib.pyplot as plt
import argparse
import yaml
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
    
    def SNR_select(self, threshold=5, reverse=False, sac_head='user2'):
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
            isnr = tr.stats.sac[sac_head]
            new.qc_metrics.append(isnr)
            if isnr >= threshold:
                to_keep.append(i)
        
        if reverse:
            to_keep = [i for i in range(len(self.st)) if i not in to_keep]

        new.st = Stream([new.st[i] for i in to_keep])
        new.files = [new.files[i] for i in to_keep]
        new.qc_metrics = [new.qc_metrics[i] for i in to_keep]
        return new
    
    def slowness_select(self, slow_min=0.04, slow_max=0.1, reverse=False, sac_head='user0'):
        """
        Select traces based on slowness range.

        Args:
            slow_min (float): Minimum slowness (s/km).
            slow_max (float): Maximum slowness (s/km).
            reverse (bool): If True, keep traces outside the range.
        
        Returns:
            RFstream: A new RFstream instance with filtered traces.
        """
        if self.st is None:
            raise ValueError("Stream is not loaded. Please load data first.")
        new = self.copy()

        to_keep = []
        for i, tr in enumerate(self.st):
            slowness = tr.stats.sac[sac_head]
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


        ax[1].set_ylabel('Slowness (s/km)', fontdict={'family': 'Times New Roman', 'weight': 'bold'})
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

def rf_QC(params, type='LowFreq', plot=True):
    # IO setup
    IO_params = params['IO']
    rootdir = IO_params['ROOT']
    raw_RF_dir = os.path.join(rootdir, IO_params[f"RAW_{type}_RF"])
    clean_RF_dir = os.path.join(rootdir, IO_params[f"CLEAN_{type}_RF"])
    figdir = os.path.join(rootdir, IO_params['FIGURE'])
    logdir = os.path.join(rootdir, IO_params['LOG'])
    os.makedirs(clean_RF_dir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)


    # meta 
    meta_params = params['META']
    sta = meta_params['station']
    net = meta_params['network']
    comp = meta_params.get('component', 'R')

    # QC
    qc_params = params['QC'][type]

    # High Frequency RF QC
    filelst = glob.glob(os.path.join(raw_RF_dir, f"*{comp}*"))
    if not filelst:
        logging.warning(f"No RF files found in {raw_RF_dir}. Skipping QC for {net}.{sta}.")
        return
    
    rf = RFstream(files=filelst)
    rf.load_data()
    logging.info(f"Number of RF traces loaded: {len(rf.st)}")
    # SNR selection
    SNR_params = qc_params['SNR']
    if SNR_params["ENABLE"]:
        rf0 = rf.SNR_select(threshold=SNR_params['THRESHOLD'], sac_head=SNR_params['SAC_HEADER'])
        logging.info(f"Number of RF traces after SNR selection: {len(rf0.st)}")
    else:
        rf0 = rf.copy()
        logging.info("No SNR selection applied.")
    
    # Slowness selection
    slowness_params = qc_params['SLOWNESS']
    if slowness_params["ENABLE"]:
        slow_min = slowness_params['MIN']
        slow_max = slowness_params['MAX']
        rf1 = rf0.slowness_select(slow_min=slow_min, slow_max=slow_max, sac_head=slowness_params['SAC_HEADER'])
        logging.info(f"Number of RF traces after slowness selection: {len(rf1.st)}")
    else:
        rf1 = rf0.copy()
        logging.info("No slowness selection applied.")
    
    # P amplitude selection
    Pamp_params = qc_params['PAMP']
    if Pamp_params["ENABLE"]:
        rf2 = rf1.P_amp_select(
            tmin=Pamp_params['TMIN'], 
            tmax=Pamp_params['TMAX'], 
            window=Pamp_params['WINDOW'])
        logging.info(f"Number of RF traces after P amplitude selection: {len(rf2.st)}")
    else:
        rf2 = rf1.copy()
        logging.info("No P amplitude selection applied.")
    # MAD selection
    MAD_params = qc_params['MAD']
    if MAD_params["ENABLE"]:
        rf3 = rf2.MAD_select(threshold=MAD_params['THRESHOLD'], window=MAD_params['WINDOW'])
        logging.info(f"Number of RF traces after MAD selection: {len(rf3.st)}")
    else:
        rf3 = rf2.copy()
        logging.info("No MAD selection applied.")
    # F-test selection
    F_test_params = qc_params['F_TEST']
    if F_test_params["ENABLE"]:
        rf4 = rf3.f_test_select(threshold=F_test_params['THRESHOLD'], window=F_test_params['WINDOW'])
        logging.info(f"Number of RF traces after F-test selection: {len(rf4.st)}")
    else:
        rf4 = rf3.copy()
        logging.info("No F-test selection applied.")
    # Cross-correlation selection
    CC_params = qc_params['CC']
    if CC_params["ENABLE"]:
        rf5 = rf4.CC_select(threshold=CC_params['THRESHOLD'], window=CC_params['WINDOW'])
        logging.info(f"Number of RF traces after CC selection: {len(rf5.st)}")
    else:
        rf5 = rf4.copy()
        logging.info("No CC selection applied.")
    
    # Save cleaned RFs
    for file in rf5.files:
        shutil.copy(file, clean_RF_dir)
    logging.info(f"Cleaned RF traces saved")

    if plot:
        # Plot the cleaned RFs
        rf5.plot(save=True, save_path=os.path.join(figdir, f"{net}.{sta}_RF_QC_{type}.png"))
        logging.info(f"RF QC plot completed and saved")
    

def load_parse_args():
    parser = argparse.ArgumentParser(description="Receiver Function Quality Control")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the QC config YAML file.")
    parser.add_argument("-p", "--plot", action='store_true', help="Enable plotting of RF QC results.")
    return parser.parse_args()


def main():
    args = load_parse_args()
    with open(args.config_file, 'r') as f:
        params = yaml.safe_load(f)
    
    IO_params = params['IO']
    rootdir = IO_params['ROOT']
    logdir = os.path.join(rootdir, IO_params['LOG'])
    os.makedirs(logdir, exist_ok=True)

    sta = params['META']['station']
    net = params['META']['network']
    logging.basicConfig(
        filename=os.path.join(logdir, f"{net}.{sta}_RF_QC.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )

    print(f"Running RF QC for {params['META']['network']}.{params['META']['station']}...")
    logging.info(f"=" * 20 + "High-Frequency RF QC" + "=" * 20)
    rf_QC(params, type='HighFreq', plot=args.plot)
    logging.info(f"=" * 20 + "Low-Frequency RF QC" + "=" * 20)
    rf_QC(params, type='LowFreq', plot=args.plot)
