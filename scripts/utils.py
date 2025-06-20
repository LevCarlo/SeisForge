'''
Author: Mengjie Zheng
Email: mengjie.zheng@utoronto.ca;zhengmengjie18@mails.ucas.ac.cn
Date: 2025-02-26 11:00:37
LastEditTime: 2025-02-26 11:00:44
LastEditors: Mengjie Zheng
Description: 
FilePath: /Project/RISTRA.Proj/DATA/ANCC/ROVER/utils.py
'''


import numpy as np
from scipy.fft import fft
from obspy import Stream, UTCDateTime


def trace_shift(tr, t_new):

    dt = t_new - tr.stats.starttime
    freq = np.fft.fftfreq(tr.stats.npts, d=tr.stats.delta)
    F = np.fft.fft(tr.data)

    # Apply phase shift
    F *= np.exp(-2j * np.pi * freq * dt)

    tr_copy = tr.copy()
    tr_copy.data = np.fft.ifft(F).real
    tr_copy.stats.starttime = t_new

    return tr_copy

def QC_Streams(st, tstart, tend):

    # Check Start Time
    if not all([tr.stats.starttime == tstart for tr in st]):
        # print("* Start times are not all close to true start: ")
        # for tr in st:
        #     print("*   "+ tr.stats.channel + " " +str(tr.stats.starttime)+" " + str(tr.stats.endtime))
        # print("*   True start: "+str(tstart))
        # print("* -> Shifting traces to true start")
        st_shifted = Stream(traces=[trace_shift(tr, tstart) for tr in st])
        st = st_shifted.copy()
    
    # Try trimming
    dt = st[0].stats.delta
    try:
        # st.trim(tstart, tend-dt, pad=True, fill_value=0)
        st.trim(tstart, tend, pad=True, fill_value=0)
    except:
        print("* Unable to trim")
        print("* -> Skipping")
        print("**************************************************")
        return False, None
    
    # Check if all traces are the same length
    fs = st[0].stats.sampling_rate
    if not np.allclose([tr.stats.npts for tr in st[1:]], st[0].stats.npts):
        print("* Lengths are incompatible: ")
        [print("*     "+str(tr.stats.npts)) for tr in st]
        print("* -> Skipping")
        print("**************************************************")

        return False, None
    
    elif not np.allclose([st[0].stats.npts], int((tend - tstart)*fs), atol=1):
        print("* Length is too short: ")
        print("*    "+str(st[0].stats.npts) +
              " ~= " +int((tend - tstart)*fs))
        print("* -> Skipping")
        print("**************************************************")

        return False, None
    
    else:
        return True, st
    

def rssq(x):
    """ Root sum square
    :param x: Input array
    :type x: np.ndarray
    :return: Root sum square
    :rtype: float
    """
    return np.sqrt(np.sum(x**2)/len(x))

def snr(x, y):
    """ Signal to noise ratio
    :param x: Signal
    :type x: np.ndarray
    :param y: Noise
    :type y: np.ndarray
    :return: Signal to noise ratio
    :rtype: float
    """
    spow = rssq(x) ** 2
    npow = rssq(y) ** 2
    if npow == 0:
        npow = 1e-3
    return 10 * np.log10(spow / npow)