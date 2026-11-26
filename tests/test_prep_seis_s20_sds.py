"""Synthetic unit tests for scripts/prep_seis_s20_sds.py."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read
import pytest
from scipy import fft as sp_fft

from seisforge.ant.ccf.io import load_station_table, read_sds_day_zz, sds_file_path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prep_seis_s20_sds.py"

SEIS_S20_PZ_TEXT = """\
ZEROS   5
        +0.000000e+00 +0.000000e+00
        +0.000000e+00 +0.000000e+00
        +0.000000e+00 +0.000000e+00
        -1.466276e+02 +0.000000e+00
        -3.786600e+00 +0.000000e+00
POLES   5
        -1.379583e+02 +0.000000e+00
        -3.505500e+00 +0.000000e+00
        -1.056000e-01 +0.000000e+00
        -1.056000e-01 +0.000000e+00
        -3.030000e-02 +0.000000e+00
CONSTANT        +1.602386e+02
"""

SAMPLE_LOG_TEXT = """\
@Sampling Started 26-03-01 03:03 GPS_Lock:T,SPS=200, Pos:228612440,1079387204,67991
@Sampling Started 26-03-01 03:42 GPS_Lock:T,SPS=200, Pos:228660693,1079297933,156188
#Status:26-03-02 00 BAT=8.2V CHG=8.2V Pitch=-0.3 Roll=-1.3
#Status:26-03-02 12 BAT=8.1V CHG=8.1V Pitch=-0.4 Roll=-1.2
#Status:26-03-03 00 BAT=8.0V CHG=8.0V Pitch=-0.2 Roll=-1.4
"""


def _load_prep_module():
    spec = importlib.util.spec_from_file_location("prep_seis_s20_sds", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = _load_prep_module()


def _write_pz(tmp_path: Path) -> Path:
    pz_path = tmp_path / "SEIS-S20.PZ"
    pz_path.write_text(SEIS_S20_PZ_TEXT, encoding="utf-8")
    return pz_path


def _apply_forward_pz(x: np.ndarray, fs: float, pz) -> np.ndarray:
    """Convolve a velocity signal forward with H(s) (counts per velocity unit)."""
    nfft = int(sp_fft.next_fast_len(x.size + int(round(30.0 * fs)), real=True))
    freqs = sp_fft.rfftfreq(nfft, d=1.0 / fs)
    spec = sp_fft.rfft(x, n=nfft) * prep.eval_pz_transfer_function(pz, freqs)
    return sp_fft.irfft(spec, n=nfft)[: x.size]


def test_parse_sac_pz_and_to_none_deconvolution_gain_and_phase(tmp_path: Path) -> None:
    """SEIS-S20.PZ 'to none' deconvolution must recover exact velocity gain & phase."""
    pz_path = _write_pz(tmp_path)
    pz = prep.parse_sac_pz(pz_path)
    assert len(pz.zeros) == 5
    assert len(pz.poles) == 5
    assert np.isclose(pz.constant, 160.2386, rtol=1e-7)

    # Check passband transfer function amplitude is ~170 (not 1/omega scaled).
    h_1hz = prep.eval_pz_transfer_function(pz, np.array([1.0]))[0]
    assert 165.0 < abs(h_1hz) < 180.0
    assert abs(np.degrees(np.angle(h_1hz))) < 1.0

    # Multi-tone velocity signal in passband [0.25, 1.0, 4.0] Hz at 200 Hz.
    fs_in = 200.0
    fs_out = 20.0
    dur_s = 1200.0
    n_in = int(round(dur_s * fs_in))
    t0 = UTCDateTime("2026-03-02T00:00:00.000000Z")
    t_in = np.arange(n_in, dtype=np.float64) / fs_in

    vel_true_in = (
        1.5 * np.sin(2.0 * np.pi * 0.25 * t_in + 0.3)
        + 0.8 * np.cos(2.0 * np.pi * 1.0 * t_in - 0.7)
        + 0.4 * np.sin(2.0 * np.pi * 4.0 * t_in + 1.1)
    )
    raw_counts = _apply_forward_pz(vel_true_in, fs_in, pz)

    y_out, k_out = prep.process_contiguous_segment(
        raw_counts,
        t0,
        fs_in=fs_in,
        fs_out=fs_out,
        pz=pz,
        pre_filt=(0.05, 0.10, 8.0, 9.0),
        water_level_db=60.0,
        edge_taper_s=30.0,
    )

    t_out = (k_out + np.arange(y_out.size, dtype=np.float64)) / fs_out - t0.timestamp
    vel_true_out = (
        1.5 * np.sin(2.0 * np.pi * 0.25 * t_out + 0.3)
        + 0.8 * np.cos(2.0 * np.pi * 1.0 * t_out - 0.7)
        + 0.4 * np.sin(2.0 * np.pi * 4.0 * t_out + 1.1)
    )

    # Compare interior (away from 30 s edge tapers and filter startup).
    interior = (t_out >= 150.0) & (t_out <= dur_s - 150.0)
    rel_err = np.linalg.norm(y_out[interior] - vel_true_out[interior]) / np.linalg.norm(
        vel_true_out[interior]
    )
    corr = float(np.corrcoef(y_out[interior], vel_true_out[interior])[0, 1])
    assert rel_err < 2e-3
    assert corr > 0.99999


def test_pz_deconvolution_matches_obspy_simulate_seismometer(tmp_path: Path) -> None:
    """Cross-check custom PZ 'to none' deconvolution point-by-point against ObsPy."""
    from obspy.signal.invsim import simulate_seismometer
    from scipy import signal as sp_signal

    pz_path = _write_pz(tmp_path)
    pz = prep.parse_sac_pz(pz_path)

    fs = 20.0
    dur_s = 600.0
    n = int(round(dur_s * fs))
    t0 = UTCDateTime("2026-03-02T00:00:00.000000Z")
    rng = np.random.default_rng(2026)

    # Broadband synthetic signal spanning 0.15–6 Hz (inside flat pre_filt passband).
    t = np.arange(n, dtype=np.float64) / fs
    raw = (
        1.2 * np.sin(2.0 * np.pi * 0.2 * t + 0.4)
        + 0.7 * np.cos(2.0 * np.pi * 1.3 * t - 0.5)
        + 0.3 * rng.standard_normal(n)
    )
    pre_filt = (0.05, 0.10, 8.0, 9.0)

    # Custom implementation (fs_in == fs_out, on-grid so shift_s == 0).
    y_custom, _ = prep.process_contiguous_segment(
        raw,
        t0,
        fs_in=fs,
        fs_out=fs,
        pz=pz,
        pre_filt=pre_filt,
        water_level_db=60.0,
        edge_taper_s=30.0,
    )

    # Independent ObsPy reference: apply identical detrend + edge taper, then
    # simulate_seismometer with SAC 'to none' convention (gain = CONSTANT, sensitivity = 1).
    x_prep = sp_signal.detrend(raw.astype(np.float64), type="linear")
    n_taper = min(int(round(30.0 * fs)), max(0, int(0.05 * n)))
    x_prep = prep.apply_edge_cosine_taper(x_prep, n_taper)

    y_obspy = simulate_seismometer(
        x_prep,
        samp_rate=fs,
        paz_remove={
            "zeros": list(pz.zeros),
            "poles": list(pz.poles),
            "gain": float(pz.constant),
            "sensitivity": 1.0,
        },
        water_level=60.0,
        pre_filt=pre_filt,
        zero_mean=False,
        taper=False,
        remove_sensitivity=False,
        sacsim=False,
        pitsasim=False,
    )

    # Interior away from the 30 s edge tapers (slight FFT pad length difference only affects edges).
    interior = (t >= 90.0) & (t <= dur_s - 90.0)
    rel_err = np.linalg.norm(y_custom[interior] - y_obspy[interior]) / np.linalg.norm(
        y_obspy[interior]
    )
    corr = float(np.corrcoef(y_custom[interior], y_obspy[interior])[0, 1])
    assert rel_err < 1e-4
    assert corr > 0.9999999


@pytest.mark.parametrize("offset_s", [0.0018, -0.0021, 0.0168])
def test_fractional_sample_alignment_and_decimation_phase(offset_s: float) -> None:
    """Off-grid 200 Hz traces (including k0 % 10 != 0) must align exactly to k / 20."""
    fs_in = 200.0
    fs_out = 20.0
    dur_s = 400.0
    n_in = int(round(dur_s * fs_in))
    base_epoch = UTCDateTime("2026-03-02T00:00:00Z").timestamp
    t0 = UTCDateTime(base_epoch + offset_s)

    t_in_abs = (base_epoch + offset_s) + np.arange(n_in, dtype=np.float64) / fs_in
    t_rel = t_in_abs - base_epoch
    # Analytic band-limited signal well below 20 Hz Nyquist (10 Hz).
    sig_in = np.sin(2.0 * np.pi * 1.3 * t_rel + 0.4) + 0.5 * np.cos(
        2.0 * np.pi * 3.7 * t_rel - 0.9
    )

    y_out, k_out = prep.process_contiguous_segment(
        sig_in,
        t0,
        fs_in=fs_in,
        fs_out=fs_out,
        pz=None,
        pre_filt=None,
        water_level_db=None,
        edge_taper_s=20.0,
    )

    # Check output start lies on exact k / fs_out grid (float64 epoch ULP ~1e-6 samples).
    t_out_start = k_out / fs_out
    grid_res = abs((t_out_start - base_epoch) * fs_out - round((t_out_start - base_epoch) * fs_out))
    assert grid_res < 1e-5

    # Check waveform matches exact analytical values at k / fs_out in interior.
    t_out_rel = (k_out + np.arange(y_out.size, dtype=np.float64)) / fs_out - base_epoch
    truth = np.sin(2.0 * np.pi * 1.3 * t_out_rel + 0.4) + 0.5 * np.cos(
        2.0 * np.pi * 3.7 * t_out_rel - 0.9
    )
    interior = (t_out_rel >= 60.0) & (t_out_rel <= dur_s - 60.0)
    rel_err = np.linalg.norm(y_out[interior] - truth[interior]) / np.linalg.norm(truth[interior])
    corr = float(np.corrcoef(y_out[interior], truth[interior])[0, 1])
    assert rel_err < 1.5e-3
    assert corr > 0.999999


def test_end_to_end_rolling_files_cross_midnight_and_read_sds_day_zz(tmp_path: Path) -> None:
    """End-to-end test: rolling non-aligned files -> SDS -> read_sds_day_zz & cross-midnight."""
    pz_path = _write_pz(tmp_path)
    pz = prep.parse_sac_pz(pz_path)

    # Build a synthetic SEIS-S20 station directory: 2164-0519
    sta_dir = tmp_path / "2164-0519"
    mseed_dir = sta_dir / "Mseed0519"
    log_dir = sta_dir / "Log0519"
    mseed_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    (log_dir / "L2603.0519").write_text(SAMPLE_LOG_TEXT, encoding="utf-8")

    # Use 200 Hz -> 20 Hz over a 2-day span (2026-03-02 and 2026-03-03) with
    # rolling non-UTC-aligned files starting at +1.8 ms off-grid.
    # For speed in CI, we test two full days? Actually, let's check:
    # 2026-03-02 00:00 - 120s .. 2026-03-03 00:00 + 300s is ~86820 s (~17.36M samples at 200 Hz).
    # Instead, to keep unit tests snappy (<2s) while still testing 200 Hz -> 20 Hz
    # across midnight with read_sds_day_zz(extra_next_day_s=60.0), we populate:
    # - File 1: 2026-03-01T23:55:00.001800Z .. 2026-03-02T12:00:00.001800Z (12h05m)
    # - File 2: 2026-03-02T12:00:00.006800Z .. 2026-03-03T00:05:00.001800Z (12h05m)
    # Wait: even better, read_sds_day_zz reads 2026-03-02 + first 60 s of 2026-03-03!
    # At 200 Hz, 24h10m = 17.4M samples (~1.5 s total). Let's use 200 Hz directly!
    fs_in = 200.0
    fs_out = 20.0
    dt_in = 1.0 / fs_in
    offset_s = 0.0018
    day0 = UTCDateTime("2026-03-02T00:00:00Z")

    # Span from 2026-03-01T23:55:00.0018Z to 2026-03-03T00:05:00.0018Z (87000 s)
    t_start = day0 - 300.0 + offset_s
    n_total = int(round(87000.0 * fs_in))
    # Piecewise split at a non-aligned time (2026-03-02T04:59:51.651800Z)
    split_idx = int(round((5.0 * 3600.0 - 8.35) * fs_in))

    # Forward apply exact SEIS-S20 response at 0.5 Hz + 2.0 Hz (exact steady-state)
    t_rel = (t_start.timestamp - day0.timestamp) + np.arange(n_total, dtype=np.float64) * dt_in
    h_vals = prep.eval_pz_transfer_function(pz, np.array([0.5, 2.0]))
    raw_f64 = (
        2.0 * abs(h_vals[0]) * np.sin(2.0 * np.pi * 0.5 * t_rel + 0.2 + np.angle(h_vals[0]))
        + 1.0 * abs(h_vals[1]) * np.cos(2.0 * np.pi * 2.0 * t_rel - 0.5 + np.angle(h_vals[1]))
    )
    # Scale to realistic integer counts (STEIM2 encoding)
    scale = 100.0
    raw_int = np.rint(raw_f64 * scale).astype(np.int32)

    tr1 = Trace(
        data=raw_int[:split_idx],
        header={
            "network": "WT",
            "station": "00519",
            "location": "00",
            "channel": "BHZ",
            "sampling_rate": fs_in,
            "starttime": t_start,
        },
    )
    tr2 = Trace(
        data=raw_int[split_idx:],
        header={
            "network": "WT",
            "station": "00519",
            "location": "00",
            "channel": "BHZ",
            "sampling_rate": fs_in,
            "starttime": t_start + split_idx * dt_in,
        },
    )
    Stream([tr1]).write(str(mseed_dir / "MD26030123.0519"), format="MSEED", encoding="STEIM2")
    Stream([tr2]).write(str(mseed_dir / "MD26030204.0519"), format="MSEED", encoding="STEIM2")

    out_sds = tmp_path / "sds_out"
    qc_jsonl = tmp_path / "qc.jsonl"
    rc = prep.main(
        [
            str(sta_dir),
            "--out-sds",
            str(out_sds),
            "--pz",
            str(pz_path),
            "--channels",
            "BHZ",
            "--start-date",
            "2026-03-02",
            "--end-date",
            "2026-03-03",
            "--fs-out",
            "20.0",
            "--pad-s",
            "180.0",
            "--edge-taper-s",
            "30.0",
            "--qc-jsonl",
            str(qc_jsonl),
        ]
    )
    assert rc == 0

    # Verify station table snippet loads via seisforge.ant.ccf.io.load_station_table
    sta_csv = out_sds / "_qc" / "2164_station.csv"
    assert sta_csv.is_file()
    stations = load_station_table(sta_csv, expected_response_state="velocity")
    assert len(stations) == 1
    sta = stations[0]
    assert sta.network == "WT"
    assert sta.station == "2164"
    assert sta.location == "00"
    assert sta.channel == "BHZ"
    assert np.isclose(sta.latitude_deg, 22.8660693, atol=1e-7)
    assert np.isclose(sta.longitude_deg, 107.9297933, atol=1e-7)
    assert np.isclose(sta.elevation_m, 156.188, atol=1e-3)
    assert sta.instrument_model == "SEIS-S20"

    # Verify SDS path & FLOAT32 MiniSEED encoding
    day0_path = sds_file_path(out_sds, sta, "2026-03-02")
    assert day0_path.is_file()
    st_check = read(str(day0_path))
    assert len(st_check) == 1
    assert st_check[0].stats.station == "2164"
    assert st_check[0].stats.mseed.encoding == "FLOAT32"
    assert st_check[0].stats.npts == int(round(86400.0 * fs_out))

    # Verify read_sds_day_zz reads both 2026-03-02 and 60 s of 2026-03-03 cleanly!
    day_data = read_sds_day_zz(
        out_sds,
        stations,
        "2026-03-02",
        sampling_rate_hz=fs_out,
        extra_next_day_s=60.0,
        grid_tolerance_samples=1e-3,
        allow_missing_next_day=False,
    )
    assert day_data.missing_stations == ()
    assert day_data.next_day_missing_stations == ()
    assert np.all(day_data.valid_mask)

    # Check cross-midnight continuity across 2026-03-02T23:59:30 .. 2026-03-03T00:00:30
    day_samples = int(round(86400.0 * fs_out))
    seam_slice = slice(day_samples - 600, day_samples + 600)
    t_seam = np.arange(day_samples - 600, day_samples + 600, dtype=np.float64) / fs_out
    expected_seam = (
        2.0 * np.sin(2.0 * np.pi * 0.5 * t_seam + 0.2)
        + 1.0 * np.cos(2.0 * np.pi * 2.0 * t_seam - 0.5)
    ) * scale
    got_seam = day_data.waveforms[0, seam_slice]
    rel_err = np.linalg.norm(got_seam - expected_seam) / np.linalg.norm(expected_seam)
    assert rel_err < 2e-3

    # Verify QC CSV & JSONL contents for 2026-03-02 (full day, zero gaps)
    qc_rows = [json.loads(line) for line in qc_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(qc_rows) == 2
    row_d0 = next(r for r in qc_rows if r["utc_date"] == "2026-03-02")
    assert row_d0["status"] == "ok"
    assert row_d0["raw_station"] == "00519"
    assert row_d0["station"] == "2164"
    assert row_d0["n_source_files"] == 2
    assert row_d0["gap_count"] == 0
    assert row_d0["overlap_count"] == 0
    assert row_d0["day_coverage_pct"] == pytest.approx(100.0, abs=1e-3)
    assert row_d0["out_grid_residual_samples"] < 1e-6
    assert row_d0["gps_lock"] == "T"
    assert row_d0["bat_min_v"] == pytest.approx(8.1)
    assert row_d0["bat_max_v"] == pytest.approx(8.2)
    assert row_d0["anomalies"] == ""

    # Verify default no-overwrite skips existing files cleanly and preserves existing qc.csv
    rc_skip = prep.main(
        [
            str(sta_dir),
            "--out-sds",
            str(out_sds),
            "--pz",
            str(pz_path),
            "--channels",
            "BHZ",
            "--date",
            "2026-03-02",
        ]
    )
    assert rc_skip == 0
    with (out_sds / "_qc" / "2164_qc.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"

    # Writing to a fresh --qc-csv path still records skipped_exists
    fresh_qc = tmp_path / "skip_only_qc.csv"
    rc_skip2 = prep.main(
        [
            str(sta_dir),
            "--out-sds",
            str(out_sds),
            "--pz",
            str(pz_path),
            "--channels",
            "BHZ",
            "--date",
            "2026-03-02",
            "--qc-csv",
            str(fresh_qc),
        ]
    )
    assert rc_skip2 == 0
    with fresh_qc.open("r", encoding="utf-8") as handle:
        skip_rows = list(csv.DictReader(handle))
    assert len(skip_rows) == 1
    assert skip_rows[0]["status"] == "skipped_exists"


def test_anomaly_detection_gap_overlap_clip_jump_fs_and_no_remove_response(
    tmp_path: Path,
) -> None:
    """Verify QC flags gaps, overlaps, clipping, sub-sample jumps, GPS unlock, and gain_only."""
    sta_dir = tmp_path / "2199-0777"
    mseed_dir = sta_dir / "Mseed0777"
    log_dir = sta_dir / "Log0777"
    mseed_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)

    # Log with GPS_Lock:F
    (log_dir / "L2603.0777").write_text(
        "@Sampling Started 26-03-05 00:00 GPS_Lock:F,SPS=200, Pos:228660693,1079297933,156188\n"
        "#Status:26-03-05 01 BAT=7.1V CHG=7.1V Pitch=-2.5 Roll=3.1\n",
        encoding="utf-8",
    )

    fs_in = 100.0  # intentionally != expected 200.0 Hz to trigger FS_ANOMALY
    day0 = UTCDateTime("2026-03-05T00:00:00Z")

    # Segment 1: 00:00:00.0020 .. 00:05:00 (300 s), with clipped samples
    n1 = int(round(300.0 * fs_in))
    d1 = np.sin(2.0 * np.pi * 1.0 * np.arange(n1) / fs_in).astype(np.float32) * 1000.0
    d1[100:110] = 8388607.0  # clipped
    tr1 = Trace(
        data=d1,
        header={
            "network": "WT",
            "station": "00777",
            "location": "00",
            "channel": "BHZ",
            "sampling_rate": fs_in,
            "starttime": day0 + 0.0020,
        },
    )

    # Segment 2: 00:05:10.0040 .. 00:10:10 (10 s gap + 2 ms sub-sample jump!)
    n2 = int(round(300.0 * fs_in))
    d2 = np.cos(2.0 * np.pi * 1.0 * np.arange(n2) / fs_in).astype(np.float32) * 1000.0
    tr2 = Trace(
        data=d2,
        header={
            "network": "WT",
            "station": "00777",
            "location": "00",
            "channel": "BHZ",
            "sampling_rate": fs_in,
            "starttime": day0 + 310.0 + 0.0040,
        },
    )

    # Segment 3: 00:10:05.0040 .. 00:15:05 (5 s overlap with segment 2!)
    n3 = int(round(300.0 * fs_in))
    d3 = np.cos(2.0 * np.pi * 1.0 * np.arange(n3) / fs_in).astype(np.float32) * 1000.0
    tr3 = Trace(
        data=d3,
        header={
            "network": "WT",
            "station": "00777",
            "location": "00",
            "channel": "BHZ",
            "sampling_rate": fs_in,
            "starttime": day0 + 605.0 + 0.0040,
        },
    )

    Stream([tr1]).write(str(mseed_dir / "MD26030500.0777"), format="MSEED", encoding="FLOAT32")
    Stream([tr2]).write(str(mseed_dir / "MD26030505.0777"), format="MSEED", encoding="FLOAT32")
    Stream([tr3]).write(str(mseed_dir / "MD26030510.0777"), format="MSEED", encoding="FLOAT32")

    out_sds = tmp_path / "sds_anom"
    qc_jsonl = tmp_path / "anom_qc.jsonl"
    rc = prep.main(
        [
            str(sta_dir),
            "--out-sds",
            str(out_sds),
            "--no-remove-response",
            "--date",
            "2026-03-05",
            "--fs-out",
            "20.0",
            "--expected-fs-in",
            "200.0",
            "--qc-jsonl",
            str(qc_jsonl),
        ]
    )
    assert rc == 0

    rows = [json.loads(line) for line in qc_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["status"] == "ok"
    assert rec["response_state"] == "gain_only"
    assert rec["fs_anomaly"] is True
    assert rec["gap_count"] >= 2  # internal 10 s gap + trailing gap to 24:00:00
    assert rec["overlap_count"] == 1
    assert rec["overlap_total_s"] == pytest.approx(5.0, abs=0.05)
    assert rec["clip_count"] == 10
    assert rec["frac_offset_jump"] is True
    assert rec["gps_lock"] == "F"
    assert rec["pitch_max_abs_deg"] == pytest.approx(2.5)
    assert rec["roll_max_abs_deg"] == pytest.approx(3.1)

    flags = set(rec["anomalies"].split(";"))
    assert {"FS_ANOMALY", "GAP", "OVERLAP", "FRAC_OFFSET_JUMP", "CLIPPING", "GPS_UNLOCK", "LOW_COVERAGE"} <= flags

    # Verify read_sds_day_zz reads the gappy output cleanly (and marks gaps invalid).
    sta_csv = out_sds / "_qc" / "2199_station.csv"
    stations = load_station_table(sta_csv, expected_response_state="gain_only")
    day_data = read_sds_day_zz(
        out_sds,
        stations,
        "2026-03-05",
        sampling_rate_hz=20.0,
        grid_tolerance_samples=1e-3,
    )
    assert day_data.missing_stations == ()
    # Valid samples should be roughly (300 + 300 + 295) * 20 = 17900 samples.
    valid_count = int(np.count_nonzero(day_data.valid_mask))
    assert 17000 < valid_count < 18500

    # Re-running without --overwrite skips existing outputs and preserves qc.csv / qc.jsonl
    qc_csv_path = out_sds / "_qc" / "2199_qc.csv"
    before_csv = qc_csv_path.read_text(encoding="utf-8")
    before_jsonl = qc_jsonl.read_text(encoding="utf-8")
    rc2 = prep.main(
        [
            str(sta_dir),
            "--out-sds",
            str(out_sds),
            "--no-remove-response",
            "--date",
            "2026-03-05",
            "--fs-out",
            "20.0",
            "--expected-fs-in",
            "200.0",
            "--qc-jsonl",
            str(qc_jsonl),
        ]
    )
    assert rc2 == 0
    assert qc_csv_path.read_text(encoding="utf-8") == before_csv
    assert qc_jsonl.read_text(encoding="utf-8") == before_jsonl


def test_min_segment_s_drops_heavily_fragmented_bad_channel(tmp_path: Path) -> None:
    """Heavily fragmented channels (e.g. 60 x 15 s traces) are skipped cleanly without writing junk SDS."""
    sta_dir = tmp_path / "2343-0779"
    mseed_dir = sta_dir / "Mseed0779"
    mseed_dir.mkdir(parents=True)

    fs_in = 50.0
    t0 = UTCDateTime("2026-04-04T00:00:00")
    traces = []
    for i in range(60):
        seg_start = t0 + i * 25.0  # 15 s data + 10 s gap repeated 60 times
        traces.append(
            Trace(
                data=np.ones(int(15.0 * fs_in), dtype=np.int32) * 1000,
                header={
                    "network": "WT",
                    "station": "00779",
                    "location": "00",
                    "channel": "BHZ",
                    "sampling_rate": fs_in,
                    "starttime": seg_start,
                },
            )
        )
    Stream(traces).write(str(mseed_dir / "MD26040400.0779"), format="MSEED", encoding="STEIM2")

    out_sds = tmp_path / "sds_out"
    qc_jsonl = tmp_path / "qc_frag.jsonl"
    rc = prep.main(
        [
            str(sta_dir),
            "--out-sds",
            str(out_sds),
            "--no-remove-response",
            "--date",
            "2026-04-04",
            "--fs-out",
            "50.0",
            "--expected-fs-in",
            "50.0",
            "--min-segment-s",
            "60.0",
            "--qc-jsonl",
            str(qc_jsonl),
        ]
    )
    assert rc == 0
    rows = [json.loads(line) for line in qc_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    rec = rows[0]
    assert rec["status"] == "no_data"
    assert rec["dropped_short_segments"] == 60
    assert rec["out_n_traces"] == 0
    flags = set(rec["anomalies"].split(";"))
    assert {"SHORT_SEGMENTS_ONLY", "EXCESSIVE_FRAGMENTATION"} <= flags
    assert not Path(rec["output_path"]).exists()
