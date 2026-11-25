"""End-to-end driver, triple-hash resume, stacking, and CLI tests for ant.ccf."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read as obspy_read
import pytest
import yaml

from seisforge.ant.ccf import (
    StationRecord,
    read_netcdf,
    run_ccf_config,
    run_stack_config,
    sds_file_path,
)
from seisforge.cli.main import main as cli_main


def _write_station_csv(path: Path, stations: list[StationRecord]) -> Path:
    lines = ["network,station,location,channel,latitude_deg,longitude_deg,response_state"]
    for s in stations:
        lines.append(
            f"{s.network},{s.station},{s.location},{s.channel},"
            f"{s.latitude_deg},{s.longitude_deg},{s.response_state}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_sds_day(
    sds_root: Path,
    station: StationRecord,
    utc_date: str,
    data: np.ndarray,
    fs: float,
) -> Path:
    d = date.fromisoformat(utc_date)
    t0 = UTCDateTime(d.year, d.month, d.day, 0, 0, 0)
    tr = Trace(
        data=np.asarray(data, dtype=np.float32),
        header={
            "network": station.network,
            "station": station.station,
            "location": station.location,
            "channel": station.channel,
            "sampling_rate": fs,
            "starttime": t0,
        },
    )
    path = sds_file_path(sds_root, station, utc_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    Stream([tr]).write(str(path), format="MSEED", encoding="FLOAT32")
    return path


def test_end_to_end_sds_resume_stack_and_cli_export(tmp_path: Path):
    fs = 1.0  # 1 Hz keeps 2 full UTC days (2 * 86,400 samples) fast in unit tests
    stations = [
        StationRecord("XX", "S01", "00", "LHZ", 35.0, 100.0, response_state="velocity"),
        StationRecord("XX", "S02", "00", "LHZ", 35.0, 101.0, response_state="velocity"),
    ]
    sds_root = tmp_path / "sds"
    rng = np.random.default_rng(1234)
    # Construct correlated wavefield with +5 s delay from S01 to S02 across 2 days.
    for utc_date in ("2026-01-01", "2026-01-02"):
        base = rng.standard_normal(86400)
        delayed = np.roll(base, 5) + 0.2 * rng.standard_normal(86400)
        _write_sds_day(sds_root, stations[0], utc_date, base, fs)
        _write_sds_day(sds_root, stations[1], utc_date, delayed, fs)

    stations_csv = _write_station_csv(tmp_path / "stations.csv", stations)
    ccf_dir = tmp_path / "daily_ccf"
    ccf_cfg_path = tmp_path / "ccf.yml"
    ccf_cfg = {
        "input": {
            "sds_root": str(sds_root),
            "station_table": str(stations_csv),
            "sampling_rate_hz": fs,
            "response_state": "velocity",
            "valid_passband_hz": [0.02, 0.45],
        },
        "processing": {
            "cc_type": "cc",
            "window_length_s": 500.0,  # 86400 % 500 = 400 -> 172 windows/day + cross-midnight test
            "max_lag_s": 20.0,
            "lag_guard_s": 5.0,
            "include_autocorr": False,
        },
        "output": {"dir": str(ccf_dir), "overwrite": False},
        "run": {"start_date": "2026-01-01", "end_date": "2026-01-02"},
    }
    ccf_cfg_path.write_text(yaml.safe_dump(ccf_cfg), encoding="utf-8")

    # 1. First run computes both days.
    summary1 = run_ccf_config(ccf_cfg_path)
    assert [o.status for o in summary1.outcomes] == ["computed", "computed"]
    assert summary1.log_file.exists()

    # 2. Second run with unchanged inputs/params/stations resumes both days.
    summary2 = run_ccf_config(ccf_cfg_path)
    assert [o.status for o in summary2.outcomes] == ["resumed", "resumed"]

    # 3. Changing a scientific parameter triggers recomputation.
    ccf_cfg["processing"]["max_lag_s"] = 25.0
    ccf_cfg_path.write_text(yaml.safe_dump(ccf_cfg), encoding="utf-8")
    summary3 = run_ccf_config(ccf_cfg_path)
    assert [o.status for o in summary3.outcomes] == ["computed", "computed"]

    # 4. Changing station coordinates updates station_list_hash -> recompute.
    stations_mod = [
        StationRecord("XX", "S01", "00", "LHZ", 35.01, 100.0, response_state="velocity"),
        stations[1],
    ]
    _write_station_csv(stations_csv, stations_mod)
    summary4 = run_ccf_config(ccf_cfg_path)
    assert [o.status for o in summary4.outcomes] == ["computed", "computed"]

    # 5. Stack via CLI (`seisforge ant stack-ccf`)
    stack_nc = tmp_path / "stacked.nc"
    stack_cfg_path = tmp_path / "stack.yml"
    stack_cfg_path.write_text(
        yaml.safe_dump(
            {
                "input": {"daily_dir": str(ccf_dir)},
                "stack": {"method": "linear", "trim_guard": True},
                "output": {"file": str(stack_nc)},
            }
        ),
        encoding="utf-8",
    )
    assert cli_main(["ant", "stack-ccf", "-c", str(stack_cfg_path)]) == 0
    stacked_ds = read_netcdf(stack_nc)
    lags = stacked_ds["lag_s"].values
    ccf_val = stacked_ds["ccf"].values[0]
    # Peak must fall cleanly on +5.0 s lag (S01 -> S02).
    assert lags[int(np.argmax(ccf_val))] == pytest.approx(5.0, abs=1e-12)

    # 6. Independent post-processing NetCDF -> SAC via CLI (`seisforge ant export-sac`)
    sac_dir = tmp_path / "sac"
    assert (
        cli_main(
            [
                "ant",
                "export-sac",
                "-i",
                str(stack_nc),
                "-o",
                str(sac_dir),
                "--min-distance-km",
                "10",
            ]
        )
        == 0
    )
    sac_files = list(sac_dir.glob("*.sac"))
    assert len(sac_files) == 1
    tr = obspy_read(str(sac_files[0]))[0]
    assert tr.stats.sac.b == pytest.approx(-25.0, abs=1e-6)
    assert tr.stats.sac.kcmpnm.strip() == "ZZ"

    # 7. Refuses overwrite by default in stack-ccf
    with pytest.raises(FileExistsError):
        run_stack_config(stack_cfg_path, overwrite=False)

    # 8. tf-PWS stacking via YAML config (also exercising pair chunking)
    tf_nc = tmp_path / "stacked_tf_pws.nc"
    tf_cfg_path = tmp_path / "stack_tf_pws.yml"
    tf_cfg_path.write_text(
        yaml.safe_dump(
            {
                "input": {"daily_dir": str(ccf_dir)},
                "stack": {
                    "method": "tf_pws",
                    "pws_power": 1.0,
                    "day_weighting": "equal",
                    "tf_pws_freq_min_hz": 0.02,
                    "tf_pws_freq_max_hz": 0.45,
                    "tf_pws_pair_chunk_size": 1,
                },
                "output": {"file": str(tf_nc)},
            }
        ),
        encoding="utf-8",
    )
    tf_summary = run_stack_config(tf_cfg_path)
    assert tf_summary.n_pairs == 1
    tf_ds = read_netcdf(tf_nc)
    assert tf_ds.attrs["stack_method"] == "tf_pws"
    assert tf_ds.attrs["pws_power"] == pytest.approx(1.0)
    assert tf_ds.attrs["tf_pws_freq_min_hz"] == pytest.approx(0.02)
    assert tf_ds.attrs["tf_pws_freq_max_hz"] == pytest.approx(0.45)
    tf_lags = tf_ds["lag_s"].values
    assert tf_lags[int(np.argmax(tf_ds["ccf"].values[0]))] == pytest.approx(5.0, abs=1e-12)


def test_run_stack_config_tf_pws_chunked_matches_unchunked(tmp_path):
    """Pair-chunked tf-PWS streaming matches single-chunk tf-PWS to machine precision."""
    from seisforge.ant.ccf import DailyCCFResult, daily_ccf_to_dataset, lag_axis_s, write_netcdf

    fs = 10.0
    lags = lag_axis_s(fs, max_lag_s=2.0, lag_guard_s=1.0)
    stations = [
        StationRecord("XX", f"S0{i}", "00", "LHZ", 35.0, 100.0 + 0.1 * i, response_state="velocity")
        for i in range(1, 5)
    ]
    pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
    rng = np.random.default_rng(42)

    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    for d, date_str in enumerate(("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04")):
        ccf_sum = (rng.standard_normal((pairs.shape[0], lags.size)) * 20.0).astype(np.float64)
        n_win = np.full(pairs.shape[0], 20, dtype=np.int64)
        if d == 1:
            ccf_sum[2] = 0.0
            n_win[2] = 0
        res = DailyCCFResult(lags, pairs, ccf_sum, n_win, "cc", fs, 2.0, 1.0)
        ds = daily_ccf_to_dataset(
            res,
            stations,
            utc_date=date_str,
            param_hash="abcd",
            station_list_hash="ef01",
        )
        write_netcdf(ds, daily_dir / f"ccf_ZZ_{date_str}.nc")

    out_full = tmp_path / "full.nc"
    out_chunked = tmp_path / "chunked.nc"
    for out_file, chunk_size in ((out_full, 64), (out_chunked, 2)):
        cfg_path = tmp_path / f"cfg_{chunk_size}.yml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "input": {"daily_dir": str(daily_dir)},
                    "stack": {
                        "method": "tf_pws",
                        "pws_power": 1.5,
                        "tf_pws_freq_min_hz": 0.2,
                        "tf_pws_freq_max_hz": 3.0,
                        "tf_pws_pair_chunk_size": chunk_size,
                    },
                    "output": {"file": str(out_file)},
                }
            ),
            encoding="utf-8",
        )
        run_stack_config(cfg_path)

    ds_full = read_netcdf(out_full)
    ds_chunked = read_netcdf(out_chunked)
    np.testing.assert_allclose(ds_chunked["ccf"].values, ds_full["ccf"].values, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(ds_chunked["n_windows"].values, ds_full["n_windows"].values)
    np.testing.assert_array_equal(ds_chunked["n_days"].values, ds_full["n_days"].values)
