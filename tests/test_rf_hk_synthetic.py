"""H-k stacking recovers known answers from synthetic receiver functions.

The RFs come from `seisforge.rf.synthetic.receiver_function` (Thomson-Haskell
propagator), which is independent of the H-k delay formulas, so these tests
check the stack, the sampling of the trace, the direct-P exclusion and the
sequential sediment-then-crust driver end to end. Models are assumed values
(not a real station): a 35 km crust (Vp 6.3 km/s, Vp/Vs 1.75, 2.8 g/cm^3)
over a mantle half-space (Vp 8.0, Vs 4.5, 3.3 g/cm^3), optionally under a
top layer.
"""

from __future__ import annotations

import numpy as np
import pytest
from obspy import Trace
from obspy.core.util import AttribDict

from seisforge.rf.config import HkSeqConfig
from seisforge.rf.hk import HkParam, HkStack, hkSeq
from seisforge.rf.synthetic import receiver_function
from seisforge.rf.traces import RFtrace

CRUST_H_KM, CRUST_K, CRUST_VP = 35.0, 1.75, 6.3


def _model(sediment=None):
    """Crust over mantle, or ``sediment = (h_km, vs, k, rho)`` on top of it."""
    if sediment is None:
        return (np.array([CRUST_H_KM, 0.0]), np.array([CRUST_VP, 8.0]),
                np.array([CRUST_VP / CRUST_K, 4.5]), np.array([2.8, 3.3]))
    h, vs, k, rho = sediment
    return (np.array([h, CRUST_H_KM - h, 0.0]), np.array([vs * k, CRUST_VP, 8.0]),
            np.array([vs, CRUST_VP / CRUST_K, 4.5]), np.array([rho, 2.8, 3.3]))


def _trace(model, p, gauss_a, dt_s, begin_s=-10.0, end_s=60.0):
    n = int(round((end_s - begin_s) / dt_s)) + 1
    _, data = receiver_function(*model, p, dt_s=dt_s, n_samples=n, gauss_a=gauss_a,
                                time_shift_s=-begin_s, amplitude="gaussian_peak")
    trace = Trace(np.asarray(data, dtype=np.float32))
    trace.stats.delta = dt_s
    trace.stats.sac = AttribDict(b=begin_s, e=begin_s + (n - 1) * dt_s, delta=dt_s, npts=n,
                                 user0=p, user2=20.0)
    return trace


def _pick(trace, p, h_grid, k_grid, *, vp=None, vs=None, min_ps_delay_s):
    result = HkStack(RFtrace(trace=trace, slowness=p), h_grid, k_grid, Vp=vp, Vs=vs,
                     min_ps_delay_s=min_ps_delay_s).stack()
    return result.attrs["H_best"], result.attrs["k_best"], result


SEDIMENT_H = HkParam("H", 0.0, 5.0, 0.05)
SEDIMENT_K = HkParam("k", 1.5, 3.0, 0.01)
CRUST_GRID = (HkParam("H", 25.0, 45.0, 0.1), HkParam("k", 1.6, 1.9, 0.005))


@pytest.mark.parametrize("p", [0.04, 0.06, 0.08])
def test_one_layer_crust_recovered_on_the_grid(p):
    h, k, _ = _pick(_trace(_model(), p, 2.5, 0.05), p, *CRUST_GRID, vp=CRUST_VP,
                    min_ps_delay_s=0.6)
    assert (h, k) == pytest.approx((CRUST_H_KM, CRUST_K), abs=1e-9)


def test_one_layer_crust_at_coarse_sampling():
    # dt = 0.1 s: interpolating the a = 2.5 pulse between samples moves the
    # pick by up to two grid steps along the H-k trade-off.
    for p in (0.04, 0.06, 0.08):
        h, k, _ = _pick(_trace(_model(), p, 2.5, 0.1), p, *CRUST_GRID, vp=CRUST_VP,
                        min_ps_delay_s=0.6)
        assert abs(h - CRUST_H_KM) <= 0.25 and abs(k - CRUST_K) <= 0.011


@pytest.mark.parametrize("h_min", [0.0, 0.5])
def test_weak_contrast_sediment_from_h_zero(h_min):
    # The case of docs/rf/hk.md: 2 km, Vs 2.5 km/s, Vp/Vs 2.0, a = 5, dt 0.1 s.
    # Before 0.2.0 the grid from 0 picked (0.2 km, 1.5) on direct P and the
    # grid from 0.5 km picked kappa 1.76: nearest-sample lookup made the stack
    # flat over kappa 1.76-2.1 at H = 2 km and argmax took the first point.
    trace = _trace(_model((2.0, 2.5, 2.0, 2.2)), 0.06, 5.0, 0.1, end_s=30.0)
    h, k, result = _pick(trace, 0.06, HkParam("H", h_min, 5.0, 0.05), SEDIMENT_K, vs=2.5,
                         min_ps_delay_s=0.3)
    # 0.06 in kappa: the a = 5 pulse sampled at 0.1 s (at dt 0.01 s the pick is 1.96).
    assert abs(h - 2.0) <= 0.05 and abs(k - 2.0) <= 0.06
    assert not result.attrs["direct_p_dominates"]


def test_direct_p_ridge_is_excluded_and_flagged():
    # Very weak contrast (Vs 2.8 over 3.6 km/s, 3 km): with no exclusion the
    # maximum is the direct-P ridge at H < 0.5 km. Excluding Ps delays below
    # 0.3 s gives H; the ridge is still higher than that stack, so the pick
    # is flagged as not resolved. Kappa of so weak a layer is poorly resolved
    # (1.87 against 2.0) and not asserted.
    trace = _trace(_model((3.0, 2.8, 2.0, 2.4)), 0.06, 5.0, 0.1, end_s=30.0)
    h_ridge, _, _ = _pick(trace, 0.06, SEDIMENT_H, SEDIMENT_K, vs=2.8, min_ps_delay_s=0.0)
    h, _, result = _pick(trace, 0.06, SEDIMENT_H, SEDIMENT_K, vs=2.8, min_ps_delay_s=0.3)
    assert h_ridge < 0.5
    assert abs(h - 3.0) <= 0.05
    assert result.attrs["n_excluded_direct_p"] > 0
    assert result.attrs["direct_p_dominates"]


def test_no_top_layer_is_dominated_by_direct_p():
    trace = _trace(_model(), 0.06, 5.0, 0.1, end_s=30.0)
    _, _, result = _pick(trace, 0.06, SEDIMENT_H, SEDIMENT_K, vs=1.8, min_ps_delay_s=0.3)
    assert result.attrs["direct_p_dominates"]


def _write_station(root, sediment):
    rays = np.linspace(0.045, 0.075, 7)
    for folder, gauss_a in (("high", 5.0), ("low", 2.5)):
        (root / folder).mkdir(parents=True)
        for i, p in enumerate(rays):
            end_s = 30.0 if gauss_a == 5.0 else 60.0
            _trace(_model(sediment), p, gauss_a, 0.05, end_s=end_s).write(
                str(root / folder / f"EV{i:02d}.sac"), format="SAC")


def _station_config(root):
    def layer(h, k, vp, vs, min_ps):
        return {"H": h, "k": k, "Vp": vp, "Vs": vs, "weight": [0.5, 0.2, 0.3],
                "min_ps_delay_s": min_ps}

    return {
        "IO": {"ROOT": str(root), "HighFreq_RF": "high", "LowFreq_RF": "low",
               "SAVE": "hk", "FIGURE": "fig", "LOG": "log"},
        "Meta": {"station": "SYN", "network": "XX"},
        "Model": {
            "Sediment": {**layer({"min": 0.0, "max": 5.0, "delta": 0.05},
                                 {"min": 1.5, "max": 3.0, "delta": 0.01}, None, 1.8, 0.3),
                         "two_layer_min_thickness_km": 0.4},
            "Crust": layer({"min": 25.0, "max": 45.0, "delta": 0.1},
                           {"min": 1.6, "max": 1.9, "delta": 0.005}, CRUST_VP, None, 0.6),
        },
        "QC": {"HighFreq": {"mahalanobis_dist_threshold": 2.0},
               "LowFreq": {"mahalanobis_dist_threshold": 2.0}},
    }


def _estimate(path):
    header = dict(line[2:].split(": ", 1) for line in path.read_text().splitlines()
                  if line.startswith("# ") and ": " in line)
    return (float(header["H_km"].split()[0]), float(header["k"].split()[0]), header)


def test_hkseq_recovers_sediment_and_crust(tmp_path):
    # Slow sediment 2 km, Vs 1.8 km/s (fixed at the truth), Vp/Vs 2.2, over
    # the crust. 0.05 in kappa: interference of the sediment phases with the
    # direct-P tail at a = 5 (single-RF picks 2.17-2.19).
    root = tmp_path / "XX.SYN"
    _write_station(root, (2.0, 1.8, 2.2, 2.2))
    hkSeq(_station_config(root), mode="both")
    h0, k0, _ = _estimate(root / "hk" / "XX.SYN_HFreq_Hk_QC.dat")
    h, k, header = _estimate(root / "hk" / "XX.SYN_LFreq_Hk_QC.dat")
    assert abs(h0 - 2.0) <= 0.05 and abs(k0 - 2.2) <= 0.05
    assert header["layers"] == "2"
    assert abs(h - CRUST_H_KM) <= 0.2 and abs(k - CRUST_K) <= 0.01


def test_hkseq_one_layer_without_top_layer(tmp_path):
    # Crust only: direct P dominates every sediment stack, so the sediment
    # stage has no estimate and the crust is stacked as one layer.
    root = tmp_path / "XX.SYN"
    _write_station(root, None)
    hkSeq(_station_config(root), mode="both")
    h0, _, sediment = _estimate(root / "hk" / "XX.SYN_HFreq_Hk_QC.dat")
    h, k, header = _estimate(root / "hk" / "XX.SYN_LFreq_Hk_QC.dat")
    assert np.isnan(h0) and sediment["n_direct_p_dominates"] == sediment["n_rf"]
    assert header["layers"] == "1"
    assert abs(h - CRUST_H_KM) <= 0.1 + 1e-9 and abs(k - CRUST_K) <= 0.005 + 1e-9  # one grid step


@pytest.mark.parametrize("key", ["energy_threshold", "ell_mad_sigma"])
def test_removed_hk_qc_keys_raise(tmp_path, key):
    config = _station_config(tmp_path)
    config["QC"]["LowFreq"][key] = 0.6
    with pytest.raises(ValueError, match=f"QC.LowFreq.{key} was removed"):
        HkSeqConfig.from_mapping(config)


@pytest.mark.parametrize(
    ("block", "key"),
    [("Crust", "min_ps_delay_s"), ("Sediment", "two_layer_min_thickness_km")],
)
def test_new_hk_keys_are_required(tmp_path, block, key):
    config = _station_config(tmp_path)
    del config["Model"][block][key]
    with pytest.raises(ValueError, match=f"Model.{block}.{key}"):
        HkSeqConfig.from_mapping(config)
