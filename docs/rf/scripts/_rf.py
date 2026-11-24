"""Shared helpers for docs/rf/scripts/*.py (not a figure itself).

Imports the figure style of the ambient-noise pages (`docs/ant/scripts/_style.py`)
and redirects its output to `docs/rf/img/`. Every figure script calls
``save(draw, __file__)``. Also builds in-memory receiver functions with SAC
headers, so the package QC and H-k code run on synthetic data with a known
answer.

Synthetic truth used across the pages (assumed values, not a real station):
a 35 km crust (Vp 6.3 km/s, Vp/Vs 1.75, density 2.8 g/cm^3) over a mantle
half-space (Vp 8.0, Vs 4.5 km/s, density 3.3 g/cm^3), the model of
`examples/04_rf_qc_hk`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from obspy import Stream, Trace
from obspy.core.util import AttribDict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ant" / "scripts"))

import _style  # noqa: E402
from _style import plt  # noqa: E402,F401

from seisforge.rf.synthetic import receiver_function  # noqa: E402
from seisforge.rf.traces import RFstream, RFtrace  # noqa: E402

IMG_DIR = HERE.parent / "img"
GENERATED_DIR = HERE.parent / "generated"

CRUST = {
    "thickness_km": np.array([35.0, 0.0]),
    "vp_km_s": np.array([6.3, 8.0]),
    "vs_km_s": np.array([6.3 / 1.75, 4.5]),
    "rho_g_cm3": np.array([2.8, 3.3]),
}
H_TRUE_KM, K_TRUE, VP_CRUST_KM_S = 35.0, 1.75, 6.3


def save(draw, script_file: str):
    """Light and dark SVG of ``draw()`` in docs/rf/img/."""
    return _style.save_themed(draw, script_file, out_dir=IMG_DIR)


def synthetic_rf(model=CRUST, p=0.06, *, gauss_a=2.5, dt_s=0.05, begin_s=-10.0,
                 end_s=60.0, amplitude="gaussian_peak"):
    """``(time_s, rf)`` of ``seisforge.rf.synthetic.receiver_function`` on [begin, end]."""
    n_samples = int(round((end_s - begin_s) / dt_s)) + 1
    return receiver_function(
        model["thickness_km"], model["vp_km_s"], model["vs_km_s"], model["rho_g_cm3"], p,
        dt_s=dt_s, n_samples=n_samples, gauss_a=gauss_a, time_shift_s=-begin_s,
        amplitude=amplitude,
    )


def sac_trace(time_s, data, p, snr=20.0, name="SYN"):
    """ObsPy trace with the SAC headers the RF code reads (b, e, delta, user0, user2)."""
    trace = Trace(np.asarray(data, dtype=float))
    dt = float(time_s[1] - time_s[0])
    trace.stats.delta = dt
    trace.stats.sac = AttribDict(
        b=float(time_s[0]), e=float(time_s[-1]), delta=dt, npts=len(data),
        user0=float(p), user2=float(snr), kevnm=name,
    )
    return trace


def rf_trace(time_s, data, p, **kwargs) -> RFtrace:
    return RFtrace(trace=sac_trace(time_s, data, p, **kwargs), file=kwargs.get("name"))


def rf_stream(traces) -> RFstream:
    return RFstream(st=Stream(list(traces)))
