"""Tests for seisforge.ant.ccf config, SDS IO, NetCDF products, and SAC export."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime, read as obspy_read
import pytest
import yaml

from seisforge.ant.ccf import (
    CCFProcessingConfig,
    CCFRunConfig,
    SDSInputConfig,
    StationRecord,
    WhitenSpec,
    correlate_zz_day,
    daily_ccf_to_dataset,
    dataset_to_daily_ccf,
    dataset_to_stacked_ccf,
    export_ccf_to_sac,
    linear_stack_daily,
    load_ccf_config,
    load_stack_config,
    load_station_table,
    read_netcdf,
    read_sds_day_zz,
    sds_file_path,
    stacked_ccf_to_dataset,
    station_table_hash,
    write_netcdf,
)


def _write_station_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(row[h]) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_sds_mseed(
    sds_root: Path,
    station: StationRecord,
    utc_date: str,
    segments: list[tuple[float, np.ndarray]],
    sampling_rate_hz: float,
) -> Path:
    d = date.fromisoformat(utc_date)
    day_start = UTCDateTime(d.year, d.month, d.day, 0, 0, 0)
    traces = []
    for offset_s, data in segments:
        tr = Trace(
            data=np.asarray(data, dtype=np.float32),
            header={
                "network": station.network,
                "station": station.station,
                "location": station.location,
                "channel": station.channel,
                "sampling_rate": sampling_rate_hz,
                "starttime": day_start + offset_s,
            },
        )
        traces.append(tr)
    path = sds_file_path(sds_root, station, utc_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    Stream(traces).write(str(path), format="MSEED", encoding="FLOAT32")
    return path


def test_config_validation_unknown_keys_passband_and_hash(tmp_path: Path):
    stations_csv = _write_station_csv(
        tmp_path / "stations.csv",
        [
            {
                "network": "XX",
                "station": "S01",
                "location": "00",
                "channel": "HHZ",
                "latitude_deg": 35.0,
                "longitude_deg": 100.0,
            }
        ],
    )
    cfg_data = {
        "input": {
            "sds_root": "sds",
            "station_table": str(stations_csv),
            "sampling_rate_hz": 10.0,
            "response_state": "velocity",
            "valid_passband_hz": [0.1, 4.0],
        },
        "processing": {
            "cc_type": "cc",
            "window_length_s": 100.0,
            "max_lag_s": 20.0,
            "lag_guard_s": 5.0,
            "whiten": {
                "taper_min_hz": 0.2,
                "freq_min_hz": 0.3,
                "freq_max_hz": 3.0,
                "taper_max_hz": 3.5,
            },
        },
        "output": {"dir": "out"},
        "run": {"start_date": "2026-01-01", "end_date": "2026-01-02"},
    }
    cfg_path = tmp_path / "ccf.yml"
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")
    cfg = load_ccf_config(cfg_path)
    assert cfg.dates == ("2026-01-01", "2026-01-02")

    # Runtime-only knobs (block_size, max_workers, overwrite) must not change parameter_hash.
    cfg_fast = CCFRunConfig(
        input=cfg.input,
        processing=CCFProcessingConfig(
            **{**cfg.processing.__dict__, "block_size": 16}
        ),
        output_dir=cfg.output_dir,
        dates=cfg.dates,
        overwrite=True,
        max_workers=4,
    )
    assert cfg_fast.parameter_hash() == cfg.parameter_hash()

    # Changing a scientific parameter must change parameter_hash.
    cfg_diff = CCFRunConfig(
        input=cfg.input,
        processing=CCFProcessingConfig(
            **{**cfg.processing.__dict__, "max_lag_s": 25.0}
        ),
        output_dir=cfg.output_dir,
        dates=cfg.dates,
    )
    assert cfg_diff.parameter_hash() != cfg.parameter_hash()

    # Unknown key must raise with close-match hint.
    bad = dict(cfg_data)
    bad["processing"] = {**cfg_data["processing"], "max_lag": 20.0}
    bad_path = tmp_path / "bad.yml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="max_lag_s"):
        load_ccf_config(bad_path)

    # Whitening outside declared valid_passband_hz must be rejected.
    with pytest.raises(ValueError, match="valid_passband_hz"):
        CCFRunConfig(
            input=SDSInputConfig(
                sds_root=tmp_path / "sds",
                station_table=stations_csv,
                sampling_rate_hz=10.0,
                response_state="velocity",
                valid_passband_hz=(0.5, 3.0),
            ),
            processing=CCFProcessingConfig(
                window_length_s=100.0,
                max_lag_s=20.0,
                lag_guard_s=5.0,
                whiten=WhitenSpec(0.6, 2.5, 0.2, 2.8),  # taper_min_hz < 0.5
            ),
            output_dir=tmp_path / "out",
            dates=("2026-01-01",),
        )

    # Stack config loader smoke test.
    stack_path = tmp_path / "stack.yml"
    stack_path.write_text(
        yaml.safe_dump(
            {
                "input": {"daily_dir": "out", "dates": ["2026-01-01", "2026-01-02"]},
                "stack": {"method": "linear", "trim_guard": True},
                "output": {"file": "stacked.nc"},
            }
        ),
        encoding="utf-8",
    )
    stk_cfg = load_stack_config(stack_path)
    assert stk_cfg.method == "linear"
    assert stk_cfg.trim_guard is True


def test_station_table_rejects_mixed_response_and_warns_gain_only_models(tmp_path: Path):
    mixed_csv = _write_station_csv(
        tmp_path / "mixed.csv",
        [
            {
                "network": "XX",
                "station": "S01",
                "location": "00",
                "channel": "HHZ",
                "latitude_deg": 35.0,
                "longitude_deg": 100.0,
                "response_state": "velocity",
            },
            {
                "network": "XX",
                "station": "S02",
                "location": "00",
                "channel": "DPZ",
                "latitude_deg": 35.1,
                "longitude_deg": 100.1,
                "response_state": "gain_only",
            },
        ],
    )
    with pytest.raises(ValueError, match="mixing response states"):
        load_station_table(mixed_csv, expected_response_state="velocity")

    gain_csv = _write_station_csv(
        tmp_path / "gain.csv",
        [
            {
                "network": "XX",
                "station": "S01",
                "location": "00",
                "channel": "DPZ",
                "latitude_deg": 35.0,
                "longitude_deg": 100.0,
                "instrument_model": "SmartSolo-5Hz",
            },
            {
                "network": "XX",
                "station": "S02",
                "location": "00",
                "channel": "DPZ",
                "latitude_deg": 35.1,
                "longitude_deg": 100.1,
                "instrument_model": "Fairfield-ZLand",
            },
        ],
    )
    with pytest.warns(UserWarning, match="multiple instrument models"):
        stations = load_station_table(gain_csv, expected_response_state="gain_only")
    assert len(stations) == 2
    assert len(station_table_hash(stations)) == 16


def test_sds_reader_grid_gaps_cross_midnight_and_rejections(tmp_path: Path):
    fs = 1.0  # 1 Hz keeps synthetic 1-day arrays small (86,400 samples) and fast
    sta1 = StationRecord("XX", "S01", "00", "LHZ", 35.0, 100.0, response_state="velocity")
    sta2 = StationRecord("XX", "S02", "00", "LHZ", 35.0, 101.0, response_state="velocity")
    sds_root = tmp_path / "sds"

    # Day 1: S01 has a 100-sample gap at [1000:1100]; S02 is missing on day 1.
    seg1 = np.ones(1000, dtype=np.float32) * 2.0
    seg2 = np.ones(86400 - 1100, dtype=np.float32) * 3.0
    _write_sds_mseed(sds_root, sta1, "2026-01-01", [(0.0, seg1), (1100.0, seg2)], fs)
    # Day 2 head for S01 (cross-midnight context)
    _write_sds_mseed(
        sds_root, sta1, "2026-01-02", [(0.0, np.ones(600, dtype=np.float32) * 4.0)], fs
    )

    day = read_sds_day_zz(
        sds_root,
        [sta1, sta2],
        "2026-01-01",
        sampling_rate_hz=fs,
        extra_next_day_s=300.0,
        gap_guard_s=5.0,
    )
    assert day.waveforms.shape == (2, 86400 + 300)
    assert day.missing_stations == ("XX.S02.00.LHZ",)
    assert day.next_day_missing_stations == ()
    # Gap [1000:1100] dilated by 5 s -> [995:1105] invalid and NaN.
    assert not np.any(day.valid_mask[0, 995:1105])
    assert np.all(np.isnan(day.waveforms[0, 995:1105]))
    assert day.valid_mask[0, 994] and day.valid_mask[0, 1105]
    # Next-day head mapped cleanly right after 86400.
    np.testing.assert_allclose(day.waveforms[0, 86400:86700], 4.0)

    # Off-grid starttime (> grid_tolerance_samples) must be rejected.
    bad_root = tmp_path / "sds_offgrid"
    _write_sds_mseed(
        bad_root, sta1, "2026-01-01", [(0.25, np.ones(100, dtype=np.float32))], fs
    )
    with pytest.raises(ValueError, match="off the absolute k/fs grid"):
        read_sds_day_zz(bad_root, [sta1], "2026-01-01", sampling_rate_hz=fs)

    # Mismatched sampling_rate must be rejected.
    with pytest.raises(ValueError, match="sampling_rate"):
        read_sds_day_zz(sds_root, [sta1], "2026-01-01", sampling_rate_hz=2.0)

    # Missing next-day file rejected when allow_missing_next_day=False.
    with pytest.raises(FileNotFoundError, match="next-day SDS"):
        read_sds_day_zz(
            sds_root,
            [sta1],
            "2026-01-02",
            sampling_rate_hz=fs,
            extra_next_day_s=100.0,
            allow_missing_next_day=False,
        )


def test_netcdf_roundtrip_and_selective_sac_export(tmp_path: Path):
    fs = 10.0
    stations = (
        StationRecord("XX", "S01", "00", "HHZ", 35.0, 100.0, response_state="velocity"),
        StationRecord("XX", "S02", "00", "HHZ", 35.0, 101.0, response_state="velocity"),
        StationRecord("XX", "S03", "00", "HHZ", 36.0, 100.0, response_state="velocity"),
    )
    rng = np.random.default_rng(7)
    wave = rng.standard_normal((3, 500))
    mask = np.ones((3, 500), dtype=bool)
    daily = correlate_zz_day(
        wave, mask, fs, window_length_s=10.0, max_lag_s=2.0, lag_guard_s=0.5
    )

    # Daily NetCDF round-trip
    ds_daily = daily_ccf_to_dataset(
        daily,
        stations,
        utc_date="2026-01-01",
        param_hash="abcd1234",
        station_list_hash=station_table_hash(stations),
        input_fingerprint="ef567890",
    )
    daily_nc = write_netcdf(ds_daily, tmp_path / "daily_2026-01-01.nc")
    with pytest.raises(FileExistsError):
        write_netcdf(ds_daily, daily_nc, overwrite=False)

    loaded_ds = read_netcdf(daily_nc)
    assert loaded_ds.attrs["param_hash"] == "abcd1234"
    assert "seisforge_version" in loaded_ds.attrs
    recovered_daily = dataset_to_daily_ccf(loaded_ds)
    np.testing.assert_allclose(recovered_daily.ccf_sum, daily.ccf_sum)
    np.testing.assert_array_equal(recovered_daily.n_windows, daily.n_windows)

    # Stacked NetCDF round-trip
    stacked = linear_stack_daily([daily], trim_guard=True)
    ds_stack = stacked_ccf_to_dataset(stacked, stations, dates=["2026-01-01"])
    stack_nc = write_netcdf(ds_stack, tmp_path / "stacked.nc")
    recovered_stack = dataset_to_stacked_ccf(read_netcdf(stack_nc))
    np.testing.assert_allclose(recovered_stack.ccf, stacked.ccf)

    # Independent NetCDF -> SAC export
    sac_dir = tmp_path / "sac_out"
    written = export_ccf_to_sac(
        stack_nc,
        sac_dir,
        pairs=[("XX.S01", "XX.S02")],
        include_autocorr=False,
    )
    assert len(written) == 1
    assert written[0].name == "XX.S01_XX.S02_ZZ.sac"

    tr = obspy_read(str(written[0]))[0]
    assert tr.stats.sac.b == pytest.approx(-2.0, abs=1e-6)
    assert tr.stats.sac.delta == pytest.approx(0.1, abs=1e-6)
    # 1 deg longitude at 35 deg N on WGS84 is ~91.29 km, azimuth ~90 deg.
    assert 90.5 < tr.stats.sac.dist < 92.0
    assert 89.0 < tr.stats.sac.az < 91.0
    assert tr.stats.sac.kcmpnm.strip() == "ZZ"
    # Pair (0, 1) is index 1 in [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)].
    np.testing.assert_allclose(tr.data, stacked.ccf[1].astype(np.float32), rtol=1e-6, atol=1e-6)

    with pytest.raises(FileExistsError):
        export_ccf_to_sac(stack_nc, sac_dir, pairs=[("XX.S01", "XX.S02")], overwrite=False)
