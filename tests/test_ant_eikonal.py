from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import xarray as xr

from seisforge.ant.eikonal import (
    build_eikonal_config,
    run_eikonal_config,
    stack_event_fields,
    validate_measurement_dataset,
)
from seisforge.ant.eikonal.core import (
    apparent_slowness_s_per_km,
    travel_time_gradient_s_per_km,
    travel_time_laplacian_s_per_km2,
)
from seisforge.ant.eikonal.cycle_skip import correct_cycle_skips
from seisforge.ant.eikonal.grid import build_geographic_grid
from seisforge.ant.eikonal.interpolation import interpolate_travel_time_field
from seisforge.ant.eikonal.helmholtz import (
    amplitude_laplacian_correction_s2_per_km2,
    corrected_slowness_squared_s2_per_km2,
    helmholtz_slowness_and_velocity,
)
from seisforge.ant.eikonal.models import (
    AzimuthWeightingConfig,
    GridConfig,
    InterpolationConfig,
    InterpolationStabilityConfig,
    OutlierRejectionConfig,
    StackingConfig,
)


def test_linear_travel_time_field_recovers_constant_gradient():
    grid = build_geographic_grid(
        GridConfig(
            longitude_convention="-180_180",
            min_longitude_deg=0.0,
            max_longitude_deg=0.4,
            min_latitude_deg=40.0,
            max_latitude_deg=40.4,
            longitude_step_deg=0.1,
            latitude_step_deg=0.1,
        )
    )
    field = 0.12 * grid.x_km_by_latitude + 0.20 * grid.y_km[:, None]

    north, east = travel_time_gradient_s_per_km(field, grid)
    slowness = apparent_slowness_s_per_km(north, east)
    laplacian = travel_time_laplacian_s_per_km2(field, grid)

    # Eastward metric distance changes slightly with latitude on WGS84, so a
    # field containing x also has a small north derivative at fixed longitude.
    np.testing.assert_allclose(north, 0.20, atol=7e-4)
    np.testing.assert_allclose(east, 0.12, atol=2e-4)
    np.testing.assert_allclose(slowness, np.hypot(0.20, 0.12), atol=5e-4)
    np.testing.assert_allclose(laplacian, 0.0, atol=3e-5)


def test_cycle_skip_correction_removes_one_integer_period():
    longitude = np.asarray([0.1, 0.2, 0.3, 0.4])
    latitude = np.zeros(4)
    distance = np.asarray([100.0, 110.0, 120.0, 130.0])
    expected = distance / 4.0
    observed = expected.copy()
    observed[-1] += 10.0

    result = correct_cycle_skips(
        longitude,
        latitude,
        distance,
        observed,
        period_s=10.0,
        max_residual_s=2.0,
        trial_count=1,
        seed_station_stride=1,
    )

    assert result.valid.all()
    np.testing.assert_allclose(result.travel_time_s, expected)
    assert result.cycle_count[-1] == -1


def test_helmholtz_amplitude_correction_formula():
    amplitude = np.asarray([2.0, 4.0])
    laplacian = np.asarray([0.02, -0.04])
    omega = 2.0 * np.pi / 10.0

    correction = amplitude_laplacian_correction_s2_per_km2(
        amplitude, laplacian, period_s=10.0
    )
    scaled_correction = amplitude_laplacian_correction_s2_per_km2(
        10.0 * amplitude, 10.0 * laplacian, period_s=10.0
    )
    corrected_squared = corrected_slowness_squared_s2_per_km2(
        np.full(2, 0.25), correction
    )
    slowness, velocity = helmholtz_slowness_and_velocity(corrected_squared)

    np.testing.assert_allclose(
        correction,
        laplacian / (amplitude * omega**2),
    )
    np.testing.assert_allclose(scaled_correction, correction)
    np.testing.assert_allclose(slowness**2, 0.25**2 - correction)
    np.testing.assert_allclose(velocity, 1.0 / slowness)


def test_measurement_contract_rejects_non_integer_periods():
    dataset = _measurement_dataset(periods=np.asarray([5.0, 8.5]))

    try:
        validate_measurement_dataset(
            dataset, requested_periods_s=(5,), min_snr=None
        )
    except ValueError as exc:
        assert "integer periods" in str(exc)
    else:
        raise AssertionError("non-integer period_s should be rejected")


def test_measurement_contract_requires_amplitude_for_helmholtz():
    dataset = _measurement_dataset(periods=np.asarray([5], dtype=np.int32))

    try:
        validate_measurement_dataset(
            dataset,
            requested_periods_s=(5,),
            min_snr=None,
            require_amplitude=True,
        )
    except ValueError as exc:
        assert "requires amplitude" in str(exc)
    else:
        raise AssertionError("Helmholtz input without amplitude should fail")


def test_azimuth_peer_weighting_discards_isolated_direction():
    shape = (3, 2, 2)
    slowness = np.empty(shape)
    slowness[0] = 0.25
    slowness[1] = 0.25
    slowness[2] = 0.50
    azimuth = np.empty(shape)
    azimuth[0] = 0.0
    azimuth[1] = 5.0
    azimuth[2] = 180.0
    event_fields = xr.Dataset(
        data_vars={
            "slowness_s_per_km": (
                ("source", "latitude", "longitude"),
                slowness,
            ),
            "great_circle_azimuth_deg": (
                ("source", "latitude", "longitude"),
                azimuth,
            ),
            "rejection_reason": (
                ("source", "latitude", "longitude"),
                np.zeros(shape, dtype=np.uint8),
            ),
            "source_accepted": ("source", np.ones(3, dtype=bool)),
        },
        coords={
            "source": ["A", "B", "C"],
            "latitude": [0.0, 0.1],
            "longitude": [0.0, 0.1],
        },
    )
    config = StackingConfig(
        min_raw_measurements_per_cell=1,
        azimuth_weighting=AzimuthWeightingConfig(
            enabled=True,
            neighbor_width_deg=20.0,
            require_peer=True,
        ),
        outlier_rejection=OutlierRejectionConfig(enabled=False),
    )

    stacked = stack_event_fields(event_fields, config)

    np.testing.assert_allclose(stacked["phase_velocity_km_s"], 4.0)
    np.testing.assert_array_equal(stacked["qc_measurement_count"], 2)


def test_eikonal_workflow_writes_event_fields_and_stack(tmp_path):
    measurement_path = tmp_path / "measurements.nc"
    output_directory = tmp_path / "output"
    dataset = _workflow_measurement_dataset()
    dataset.to_netcdf(measurement_path, engine="h5netcdf")
    config_path = tmp_path / "eikonal.yml"
    config_path.write_text(
        _workflow_config_text(measurement_path, output_directory),
        encoding="utf-8",
    )

    outputs = run_eikonal_config(config_path)

    assert outputs == [output_directory / "event_fields" / "10s.nc", output_directory / "stack.nc"]
    with xr.open_dataset(outputs[0], engine="h5netcdf") as fields:
        assert fields.attrs["stage"] == "eikonal_event_fields"
        assert fields.sizes["source"] == 2
        assert "rejection_reason" in fields
        assert "config_yaml" in fields.attrs
        assert fields["helmholtz_applied"].values.all()
        assert np.any(np.isfinite(fields["helmholtz_phase_velocity_km_s"]))
    with xr.open_dataset(outputs[1], engine="h5netcdf") as stacked:
        assert stacked.attrs["stage"] == "eikonal_isotropic_stack"
        assert stacked["period_s"].values.tolist() == [10]
        assert np.any(np.isfinite(stacked["phase_velocity_km_s"]))
        assert np.isclose(
            np.nanmedian(stacked["phase_velocity_km_s"]), 3.5, atol=0.1
        )
        assert np.isclose(
            np.nanmedian(stacked["helmholtz_phase_velocity_km_s"]),
            3.5,
            atol=0.1,
        )


def test_build_eikonal_config_resolves_paths_and_defaults(tmp_path):
    raw = {
        "input": {"dataset": "measurements.nc", "periods_s": [5, 10]},
        "grid": {
            "min_longitude_deg": 0.0,
            "max_longitude_deg": 1.0,
            "min_latitude_deg": 0.0,
            "max_latitude_deg": 1.0,
            "longitude_step_deg": 0.1,
            "latitude_step_deg": 0.1,
        },
        "output": {"directory": "results"},
    }

    config = build_eikonal_config(raw, base_dir=tmp_path)

    assert config.input.dataset == tmp_path / "measurements.nc"
    assert config.output.directory == tmp_path / "results"
    assert config.interpolation.method == "scipy_rbf"
    assert config.input.periods_s == (5, 10)


def test_build_eikonal_config_rejects_fractional_requested_period(tmp_path):
    raw = {
        "input": {"dataset": "measurements.nc", "periods_s": [5, 8.5]},
        "grid": {
            "min_longitude_deg": 0.0,
            "max_longitude_deg": 1.0,
            "min_latitude_deg": 0.0,
            "max_latitude_deg": 1.0,
            "longitude_step_deg": 0.1,
            "latitude_step_deg": 0.1,
        },
        "output": {"directory": "results"},
    }

    try:
        build_eikonal_config(raw, base_dir=tmp_path)
    except ValueError as exc:
        assert "integer periods" in str(exc)
    else:
        raise AssertionError("fractional input.periods_s should be rejected")


def test_build_eikonal_config_accepts_gmt_surface(tmp_path):
    raw = {
        "input": {"dataset": "measurements.nc", "periods_s": [5]},
        "grid": {
            "min_longitude_deg": 0.0,
            "max_longitude_deg": 1.0,
            "min_latitude_deg": 0.0,
            "max_latitude_deg": 1.0,
            "longitude_step_deg": 0.1,
            "latitude_step_deg": 0.1,
        },
        "interpolation": {
            "method": "gmt_surface",
            "block_reduce": "median",
            "tension": 0.1,
            "stability_check": {"comparison_tension": 0.3},
        },
        "output": {"directory": "results"},
    }

    config = build_eikonal_config(raw, base_dir=tmp_path)

    assert config.interpolation.method == "gmt_surface"
    assert config.interpolation.block_reduce == "median"
    assert config.interpolation.tension == 0.1
    assert config.interpolation.stability_check.comparison_tension == 0.3


def test_gmt_surface_backend_uses_primary_and_comparison_tensions(monkeypatch):
    grid = build_geographic_grid(
        GridConfig(
            longitude_convention="-180_180",
            min_longitude_deg=0.0,
            max_longitude_deg=0.2,
            min_latitude_deg=0.0,
            max_latitude_deg=0.2,
            longitude_step_deg=0.1,
            latitude_step_deg=0.1,
        )
    )
    calls = []

    def blockmedian(*, data, region, spacing):
        calls.append(("blockmedian", region, spacing))
        return data

    def surface(*, data, region, spacing, tension):
        del data, region, spacing
        calls.append(("surface", tension))
        return xr.DataArray(
            np.full(grid.shape, tension),
            dims=("lat", "lon"),
            coords={"lat": grid.latitude_deg, "lon": grid.longitude_deg},
        )

    monkeypatch.setitem(
        sys.modules,
        "pygmt",
        SimpleNamespace(blockmedian=blockmedian, surface=surface),
    )
    config = InterpolationConfig(
        method="gmt_surface",
        block_reduce="median",
        tension=0.1,
        stability_check=InterpolationStabilityConfig(
            comparison_tension=0.3
        ),
    )
    longitude = np.asarray([0.0, 0.2, 0.0, 0.2])
    latitude = np.asarray([0.0, 0.0, 0.2, 0.2])
    travel_time = np.asarray([0.0, 1.0, 1.0, 2.0])

    primary = interpolate_travel_time_field(
        longitude, latitude, travel_time, grid, config
    )
    comparison = interpolate_travel_time_field(
        longitude,
        latitude,
        travel_time,
        grid,
        config,
        comparison=True,
    )

    np.testing.assert_allclose(primary, 0.1)
    np.testing.assert_allclose(comparison, 0.3)
    assert [call for call in calls if call[0] == "surface"] == [
        ("surface", 0.1),
        ("surface", 0.3),
    ]


def _measurement_dataset(periods: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "phase_velocity_km_s": (
                ("path", "period"),
                np.full((2, periods.size), 3.5),
            )
        },
        coords={
            "path": np.arange(2),
            "period": np.arange(periods.size),
            "period_s": ("period", periods),
            "source_station": ("path", ["S1", "S2"]),
            "receiver_station": ("path", ["R1", "R2"]),
            "source_longitude_deg": ("path", [-1.0, 1.0]),
            "source_latitude_deg": ("path", [0.0, 0.0]),
            "receiver_longitude_deg": ("path", [0.0, 0.1]),
            "receiver_latitude_deg": ("path", [0.0, 0.1]),
        },
        attrs={
            "schema_version": 1,
            "component": "ZZ",
            "longitude_convention": "-180_180",
        },
    )


def _workflow_measurement_dataset() -> xr.Dataset:
    receiver_longitude = np.asarray([0.0, 0.2, 0.4, 0.0, 0.4, 0.0, 0.2, 0.4])
    receiver_latitude = np.asarray([0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.4])
    source_station = np.repeat(["S.WEST", "S.EAST"], receiver_longitude.size)
    source_longitude = np.repeat([-1.0, 1.4], receiver_longitude.size)
    source_latitude = np.repeat([0.2, 0.2], receiver_longitude.size)
    target_longitude = np.tile(receiver_longitude, 2)
    target_latitude = np.tile(receiver_latitude, 2)
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    _, _, distance_m = geod.inv(
        source_longitude,
        source_latitude,
        target_longitude,
        target_latitude,
    )
    phase_velocity = np.full((source_station.size, 1), 3.5)
    return xr.Dataset(
        data_vars={
            "phase_velocity_km_s": (
                ("path", "period"),
                phase_velocity,
            ),
            "snr": (("path", "period"), np.full_like(phase_velocity, 20.0)),
            "distance_km": ("path", distance_m / 1000.0),
            "amplitude": (
                ("path", "period"),
                np.ones_like(phase_velocity),
            ),
        },
        coords={
            "path": np.arange(source_station.size),
            "period": [0],
            "period_s": ("period", np.asarray([10], dtype=np.int32)),
            "source_station": ("path", source_station),
            "receiver_station": (
                "path",
                np.tile([f"R{index}" for index in range(8)], 2),
            ),
            "source_longitude_deg": ("path", source_longitude),
            "source_latitude_deg": ("path", source_latitude),
            "receiver_longitude_deg": ("path", target_longitude),
            "receiver_latitude_deg": ("path", target_latitude),
        },
        attrs={
            "schema_version": 1,
            "component": "ZZ",
            "snr_scale": "linear",
            "longitude_convention": "-180_180",
        },
    )


def _workflow_config_text(dataset: Path, output: Path) -> str:
    return f"""
schema_version: 1
project: synthetic
input:
  dataset: {dataset}
  periods_s: [10]
  min_snr: 15.0
grid:
  longitude_convention: "-180_180"
  min_longitude_deg: 0.0
  max_longitude_deg: 0.4
  min_latitude_deg: 0.0
  max_latitude_deg: 0.4
  longitude_step_deg: 0.1
  latitude_step_deg: 0.1
cycle_skip:
  enabled: false
interpolation:
  method: scipy_rbf
  smoothing: 0.0
  stability_check:
    enabled: false
eikonal:
  min_apparent_velocity_km_s: 1.0
  max_apparent_velocity_km_s: 8.0
  boundary_cells_to_mask: 0
helmholtz:
  enabled: true
  min_period_s: 0.0
  min_amplitude: 0.0
  max_relative_interpolation_difference: 0.01
  curvature_reference_velocity_km_s: 4.0
qc:
  max_abs_travel_time_laplacian_s_per_km2: 1.0
  station_coverage:
    mode: nearby
    nearby_radius_km: 1000.0
  far_field:
    min_wavelengths: 0.0
    reference_phase_velocity_km_s: 3.5
  source_field:
    min_valid_fraction: 0.0
stacking:
  min_raw_measurements_per_cell: 1
  azimuth_weighting:
    enabled: false
  outlier_rejection:
    enabled: false
output:
  directory: {output}
  stack_file: stack.nc
  overwrite: false
compute:
  workers: 1
"""
