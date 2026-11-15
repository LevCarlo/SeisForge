"""Glue commands: AFTAN -> Eikonal dataset, Eikonal stack -> inversion file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from obspy import Trace
import pytest

from seisforge.ant.aftan import run_aftan_config
from seisforge.ant.eikonal import load_measurement_dataset
from seisforge.ant.glue import (
    CollectDispersionConfig,
    collect_dispersion,
    read_aftan_dat,
    resample_to_periods,
    run_collect_dispersion,
    wrap_longitude_deg,
)
from seisforge.cli.main import main

# Station coordinates: assumed synthetic layout near the equator.
STATIONS = {
    "XX.A": (100.0, 0.0),
    "XX.B": (101.0, 0.0),
    "XX.C": (100.0, 1.0),
}


# --- resampling: independent hand-computed values ---------------------------


def test_resample_matches_hand_computed_linear_values():
    period = np.array([1.0, 2.0, np.nan, 4.0])
    value = np.array([3.0, 3.2, 9.9, 3.6])
    # 3 s lies halfway between (2, 3.2) and (4, 3.6) -> 3.4; 0.5 s and 5 s are
    # outside [1, 4] and must not be extrapolated; the NaN row is ignored.
    out = resample_to_periods(period, value, np.array([0.5, 1, 2, 3, 5]))
    np.testing.assert_allclose(out[1:4], [3.0, 3.2, 3.4], rtol=0, atol=1e-12)
    assert np.isnan(out[0]) and np.isnan(out[4])


def test_resample_rejects_non_increasing_period_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_to_periods(np.array([1.0, 3.0, 2.0]), np.ones(3), np.array([2.0]))


def test_wrap_longitude_conventions():
    np.testing.assert_allclose(
        wrap_longitude_deg(np.array([-90.0, 180.0, 270.0]), "-180_180"),
        [-90.0, 180.0, -90.0],
    )
    np.testing.assert_allclose(
        wrap_longitude_deg(np.array([-90.0, 0.0]), "0_360"), [270.0, 0.0]
    )


# --- AFTAN -> collect -> Eikonal reader --------------------------------------


def _write_ccf(path: Path, source: str, receiver: str, velocity_km_s=3.0):
    """Symmetric synthetic CCF with SAC pair coordinates (evlo/evla = source)."""
    from obspy.geodetics.base import gps2dist_azimuth

    (slon, slat), (rlon, rlat) = STATIONS[source], STATIONS[receiver]
    distance_km = gps2dist_azimuth(slat, slon, rlat, rlon)[0] / 1000.0
    delta = 0.5
    time = np.arange(-400.0, 400.0 + delta, delta)
    center = distance_km / velocity_km_s
    envelope = np.exp(-0.5 * ((time - center) / 10.0) ** 2) + np.exp(
        -0.5 * ((time + center) / 10.0) ** 2
    )
    trace = Trace(data=(envelope * np.cos(2 * np.pi * time / 5.0)).astype(np.float32))
    trace.stats.delta = delta
    trace.stats.sac = {
        "b": time[0],
        "dist": distance_km,
        "evlo": slon,
        "evla": slat,
        "stlo": rlon,
        "stla": rlat,
    }
    trace.write(str(path), format="SAC")


def _run_aftan(tmp_path: Path, branch: str = "both") -> Path:
    ccf = tmp_path / "ccf"
    ccf.mkdir(exist_ok=True)
    _write_ccf(ccf / "XX.A_XX.B_ZZ_pws.SAC", "XX.A", "XX.B")
    _write_ccf(ccf / "XX.A_XX.C_ZZ_pws.SAC", "XX.A", "XX.C")
    config = tmp_path / "aftan.yml"
    config.write_text(
        "\n".join(
            [
                "station: XX.A",
                "io:",
                "  input_dir: ccf",
                "  output_dir: ftan",
                "  sac_pattern: '*_ZZ_pws.SAC'",
                "aftan:",
                f"  branch: {branch}",
                "  min_period: 2.0",
                "  max_period: 8.0",
                "  period_sampling:",
                "    mode: uniform",
                "    step: 0.25",
                "  reference_velocity: 3.5",
                "  velocity_window:",
                "    min: 2.0",
                "    max: 4.5",
                "  trig_threshold: 9999.0",
                "  basic:",
                "    alpha:",
                "      mode: constant",
                "      value: 20.0",
                "  snr:",
                "    dsn: 20.0",
                "    nlen: 40.0",
            ]
        )
    )
    run_aftan_config(config)
    return tmp_path / "ftan"


def _station_table(tmp_path: Path) -> Path:
    path = tmp_path / "stations.txt"
    path.write_text(
        "# station longitude_deg latitude_deg\n"
        + "".join(f"{name} {lon} {lat}\n" for name, (lon, lat) in STATIONS.items())
    )
    return path


PERIODS = (3, 4, 5, 6, 7)


def test_collect_round_trip_is_accepted_by_eikonal_reader(tmp_path, assert_provenance):
    ftan = _run_aftan(tmp_path)
    files = sorted(ftan.glob("*.dat"))
    assert len(files) == 4  # two pairs x (positive, negative)
    config = CollectDispersionConfig(
        periods_s=PERIODS, station_table=_station_table(tmp_path)
    )
    output = tmp_path / "measurements.nc"

    run_collect_dispersion(files, config, output)

    dataset = load_measurement_dataset(
        output,
        requested_periods_s=PERIODS,
        min_snr=1.0,
        longitude_convention="-180_180",
    )
    assert dataset.attrs["component"] == "ZZ"
    assert dataset.attrs["snr_scale"] == "linear"
    assert dataset.attrs["period_axis"] == "instant_period_s"
    assert_provenance(dataset.attrs)
    pairs = set(
        zip(
            dataset["source_station"].values.astype(str),
            dataset["receiver_station"].values.astype(str),
        )
    )
    # Negative lags of the stored pair A_B are the physical path B -> A.
    assert pairs == {
        ("XX.A", "XX.B"),
        ("XX.B", "XX.A"),
        ("XX.A", "XX.C"),
        ("XX.C", "XX.A"),
    }
    for index in range(dataset.sizes["path"]):
        source = str(dataset["source_station"].values[index])
        assert dataset["source_longitude_deg"].values[index] == STATIONS[source][0]
        # Independent check of the resampling against the raw .dat columns.
        raw = np.loadtxt(str(dataset["aftan_file"].values[index]), comments="#")
        expected = np.interp(PERIODS, raw[:, 1], raw[:, 3])
        np.testing.assert_allclose(
            dataset["phase_velocity_km_s"].values[index], expected, rtol=0, atol=1e-12
        )
        assert np.all(np.isfinite(expected)) and np.all(expected > 0)


def test_collect_reads_coordinates_from_sac_headers(tmp_path):
    ftan = _run_aftan(tmp_path, branch="negative")
    config = CollectDispersionConfig(periods_s=PERIODS, sac_dir=tmp_path / "ccf")

    result = collect_dispersion(sorted(ftan.glob("*.dat")), config)

    dataset = result.dataset
    assert set(dataset["source_station"].values.astype(str)) == {"XX.B", "XX.C"}
    for index in range(dataset.sizes["path"]):
        for role in ("source", "receiver"):
            station = str(dataset[f"{role}_station"].values[index])
            assert (
                dataset[f"{role}_longitude_deg"].values[index],
                dataset[f"{role}_latitude_deg"].values[index],
            ) == pytest.approx(STATIONS[station])


def test_collect_rejects_duplicate_directed_paths(tmp_path):
    ftan = _run_aftan(tmp_path, branch="positive")
    duplicate = ftan / "copy.dat"
    duplicate.write_text((ftan / "XX.A_XX.B_ZZ_pws.positive.dat").read_text())
    config = CollectDispersionConfig(
        periods_s=PERIODS, station_table=_station_table(tmp_path)
    )
    with pytest.raises(ValueError, match="appears twice"):
        collect_dispersion(sorted(ftan.glob("*.dat")), config)


def test_collect_mirrors_stack_branch_only_when_asked(tmp_path):
    ftan = _run_aftan(tmp_path, branch="stack")
    files = sorted(ftan.glob("*.dat"))
    table = _station_table(tmp_path)
    plain = collect_dispersion(
        files, CollectDispersionConfig(periods_s=PERIODS, station_table=table)
    ).dataset
    mirrored = collect_dispersion(
        files,
        CollectDispersionConfig(periods_s=PERIODS, station_table=table, mirror_stack=True),
    ).dataset
    assert plain.sizes["path"] == 2
    assert mirrored.sizes["path"] == 4
    assert mirrored.attrs["mirror_stack"] == 1


def test_read_aftan_dat_reports_header_and_snr_scale(tmp_path):
    ftan = _run_aftan(tmp_path, branch="positive")
    record = read_aftan_dat(ftan / "XX.A_XX.B_ZZ_pws.positive.dat")
    assert record["metadata"]["input_file"] == "XX.A_XX.B_ZZ_pws.SAC"
    assert record["metadata"]["selected_branch"] == "positive"
    assert record["snr_scale"] == "linear"


def test_collect_config_validates_choices(tmp_path):
    with pytest.raises(ValueError, match="integers"):
        CollectDispersionConfig(periods_s=(5.5,), station_table=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        CollectDispersionConfig(periods_s=(5,))
    with pytest.raises(ValueError, match="component"):
        CollectDispersionConfig(
            periods_s=(5,),
            station_table=tmp_path,
            name_pattern="{source_station}_{receiver_station}*",
        )


def test_collect_dispersion_cli_writes_and_refuses_overwrite(tmp_path, capsys):
    ftan = _run_aftan(tmp_path, branch="positive")
    output = tmp_path / "measurements.nc"
    argv = [
        "ant",
        "collect-dispersion",
        str(ftan / "*.dat"),
        "-o",
        str(output),
        "--periods",
        "3:7:1",
        "--stations",
        str(_station_table(tmp_path)),
    ]
    assert main(argv) == 0
    assert "2 paths from 2 files, 5 periods" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main(argv)
    assert main([*argv, "--overwrite"]) == 0
