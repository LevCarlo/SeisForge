"""Self-describing xarray/NetCDF4 product builders for ZZ CCFs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pyproj import Geod
import xarray as xr

from seisforge.provenance import software_provenance

from .config import StationRecord
from .models import DailyCCFResult, StackedCCFResult

__all__ = [
    "SCHEMA_VERSION",
    "daily_ccf_to_dataset",
    "dataset_to_daily_ccf",
    "dataset_to_stacked_ccf",
    "pair_geometry",
    "stacked_ccf_to_dataset",
]

SCHEMA_VERSION = 1
_WGS84 = Geod(ellps="WGS84")
_LAG_CONVENTION = "C_AB(t) = sum_tau u_A(tau) * u_B(tau + t); positive lag is A -> B"


def pair_geometry(
    stations: Sequence[StationRecord],
    pair_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute WGS84 pair geometry arrays for canonical station pairs ``(i, j)``."""
    pairs = np.asarray(pair_indices, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pair_indices must have shape (n_pairs, 2).")

    n_pairs = pairs.shape[0]
    src_idx = pairs[:, 0]
    rcv_idx = pairs[:, 1]

    src_sta = np.asarray([stations[i].station_id for i in src_idx], dtype=str)
    rcv_sta = np.asarray([stations[j].station_id for j in rcv_idx], dtype=str)
    src_lat = np.asarray([stations[i].latitude_deg for i in src_idx], dtype=np.float64)
    src_lon = np.asarray([stations[i].longitude_deg for i in src_idx], dtype=np.float64)
    rcv_lat = np.asarray([stations[j].latitude_deg for j in rcv_idx], dtype=np.float64)
    rcv_lon = np.asarray([stations[j].longitude_deg for j in rcv_idx], dtype=np.float64)

    distance_km = np.zeros(n_pairs, dtype=np.float64)
    azimuth_deg = np.full(n_pairs, np.nan, dtype=np.float64)
    back_azimuth_deg = np.full(n_pairs, np.nan, dtype=np.float64)

    different = (src_idx != rcv_idx) & (
        (np.abs(src_lat - rcv_lat) > 0.0) | (np.abs(src_lon - rcv_lon) > 0.0)
    )
    if np.any(different):
        az, baz, dist_m = _WGS84.inv(
            src_lon[different],
            src_lat[different],
            rcv_lon[different],
            rcv_lat[different],
        )
        distance_km[different] = np.asarray(dist_m, dtype=np.float64) / 1000.0
        azimuth_deg[different] = np.mod(np.asarray(az, dtype=np.float64), 360.0)
        back_azimuth_deg[different] = np.mod(np.asarray(baz, dtype=np.float64), 360.0)

    return {
        "source_station": src_sta,
        "receiver_station": rcv_sta,
        "source_latitude_deg": src_lat,
        "source_longitude_deg": src_lon,
        "receiver_latitude_deg": rcv_lat,
        "receiver_longitude_deg": rcv_lon,
        "distance_km": distance_km,
        "azimuth_deg": azimuth_deg,
        "back_azimuth_deg": back_azimuth_deg,
    }


def daily_ccf_to_dataset(
    result: DailyCCFResult,
    stations: Sequence[StationRecord],
    *,
    utc_date: str,
    param_hash: str = "",
    station_list_hash: str = "",
    input_fingerprint: str = "",
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Encode a :class:`DailyCCFResult` into a self-describing ``xarray.Dataset``."""
    geom = pair_geometry(stations, result.pair_indices)
    attrs: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "ccf_daily",
        "component": "ZZ",
        "utc_date": str(utc_date),
        "cc_type": result.cc_type,
        "sampling_rate_hz": float(result.sampling_rate_hz),
        "max_lag_s": float(result.max_lag_s),
        "lag_guard_s": float(result.lag_guard_s),
        "lag_convention": _LAG_CONVENTION,
        "zero_count_semantics": "ccf_sum stores 0.0 where n_windows == 0; divide by n_windows and set NaN where 0",
        "param_hash": str(param_hash),
        "station_list_hash": str(station_list_hash),
        "input_fingerprint": str(input_fingerprint),
        **software_provenance(),
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    ds = xr.Dataset(
        data_vars={
            "ccf_sum": (("pair", "lag_s"), result.ccf_sum.copy()),
            "n_windows": (("pair",), result.n_windows.copy()),
            "pair_i": (("pair",), result.pair_indices[:, 0].copy()),
            "pair_j": (("pair",), result.pair_indices[:, 1].copy()),
            "source_station": (("pair",), geom["source_station"]),
            "receiver_station": (("pair",), geom["receiver_station"]),
            "source_latitude_deg": (("pair",), geom["source_latitude_deg"]),
            "source_longitude_deg": (("pair",), geom["source_longitude_deg"]),
            "receiver_latitude_deg": (("pair",), geom["receiver_latitude_deg"]),
            "receiver_longitude_deg": (("pair",), geom["receiver_longitude_deg"]),
            "distance_km": (("pair",), geom["distance_km"]),
            "azimuth_deg": (("pair",), geom["azimuth_deg"]),
            "back_azimuth_deg": (("pair",), geom["back_azimuth_deg"]),
        },
        coords={"lag_s": result.lag_s.copy()},
        attrs=attrs,
    )
    ds["lag_s"].attrs["units"] = "s"
    ds["distance_km"].attrs["units"] = "km"
    ds["azimuth_deg"].attrs["units"] = "degree"
    ds["back_azimuth_deg"].attrs["units"] = "degree"
    return ds


def dataset_to_daily_ccf(dataset: xr.Dataset) -> DailyCCFResult:
    """Reconstruct a :class:`DailyCCFResult` from a daily CCF ``xarray.Dataset``."""
    if dataset.attrs.get("stage") != "ccf_daily":
        raise ValueError(
            f"Expected dataset stage='ccf_daily', got {dataset.attrs.get('stage')!r}."
        )
    pair_indices = np.column_stack(
        [
            np.asarray(dataset["pair_i"].values, dtype=np.int64),
            np.asarray(dataset["pair_j"].values, dtype=np.int64),
        ]
    )
    return DailyCCFResult(
        lag_s=np.asarray(dataset["lag_s"].values, dtype=np.float64),
        pair_indices=pair_indices,
        ccf_sum=np.asarray(dataset["ccf_sum"].values, dtype=np.float64),
        n_windows=np.asarray(dataset["n_windows"].values, dtype=np.int64),
        cc_type=str(dataset.attrs["cc_type"]),
        sampling_rate_hz=float(dataset.attrs["sampling_rate_hz"]),
        max_lag_s=float(dataset.attrs["max_lag_s"]),
        lag_guard_s=float(dataset.attrs.get("lag_guard_s", 0.0)),
    )


def stacked_ccf_to_dataset(
    result: StackedCCFResult,
    stations: Sequence[StationRecord],
    *,
    dates: Sequence[str],
    param_hash: str = "",
    station_list_hash: str = "",
    extra_attrs: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Encode a :class:`StackedCCFResult` into a self-describing ``xarray.Dataset``."""
    geom = pair_geometry(stations, result.pair_indices)
    attrs: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "ccf_stacked",
        "component": "ZZ",
        "stack_method": result.method,
        "cc_type": result.cc_type,
        "sampling_rate_hz": float(result.sampling_rate_hz),
        "max_lag_s": float(result.max_lag_s),
        "lag_guard_s": float(result.lag_guard_s),
        "lag_convention": _LAG_CONVENTION,
        "dates": ",".join(str(d) for d in dates),
        "n_input_days": int(len(dates)),
        "param_hash": str(param_hash),
        "station_list_hash": str(station_list_hash),
        **software_provenance(),
    }
    if result.pws_power is not None:
        attrs["pws_power"] = float(result.pws_power)
    if extra_attrs:
        attrs.update(extra_attrs)

    ds = xr.Dataset(
        data_vars={
            "ccf": (("pair", "lag_s"), result.ccf.copy()),
            "n_windows": (("pair",), result.n_windows.copy()),
            "n_days": (("pair",), result.n_days.copy()),
            "pair_i": (("pair",), result.pair_indices[:, 0].copy()),
            "pair_j": (("pair",), result.pair_indices[:, 1].copy()),
            "source_station": (("pair",), geom["source_station"]),
            "receiver_station": (("pair",), geom["receiver_station"]),
            "source_latitude_deg": (("pair",), geom["source_latitude_deg"]),
            "source_longitude_deg": (("pair",), geom["source_longitude_deg"]),
            "receiver_latitude_deg": (("pair",), geom["receiver_latitude_deg"]),
            "receiver_longitude_deg": (("pair",), geom["receiver_longitude_deg"]),
            "distance_km": (("pair",), geom["distance_km"]),
            "azimuth_deg": (("pair",), geom["azimuth_deg"]),
            "back_azimuth_deg": (("pair",), geom["back_azimuth_deg"]),
        },
        coords={"lag_s": result.lag_s.copy()},
        attrs=attrs,
    )
    ds["lag_s"].attrs["units"] = "s"
    ds["distance_km"].attrs["units"] = "km"
    ds["azimuth_deg"].attrs["units"] = "degree"
    ds["back_azimuth_deg"].attrs["units"] = "degree"
    return ds


def dataset_to_stacked_ccf(dataset: xr.Dataset) -> StackedCCFResult:
    """Reconstruct a :class:`StackedCCFResult` from a stacked CCF ``xarray.Dataset``."""
    if dataset.attrs.get("stage") != "ccf_stacked":
        raise ValueError(
            f"Expected dataset stage='ccf_stacked', got {dataset.attrs.get('stage')!r}."
        )
    pair_indices = np.column_stack(
        [
            np.asarray(dataset["pair_i"].values, dtype=np.int64),
            np.asarray(dataset["pair_j"].values, dtype=np.int64),
        ]
    )
    pws_power = dataset.attrs.get("pws_power")
    return StackedCCFResult(
        lag_s=np.asarray(dataset["lag_s"].values, dtype=np.float64),
        pair_indices=pair_indices,
        ccf=np.asarray(dataset["ccf"].values, dtype=np.float64),
        n_windows=np.asarray(dataset["n_windows"].values, dtype=np.int64),
        n_days=np.asarray(dataset["n_days"].values, dtype=np.int64),
        method=str(dataset.attrs["stack_method"]),
        cc_type=str(dataset.attrs["cc_type"]),
        sampling_rate_hz=float(dataset.attrs["sampling_rate_hz"]),
        max_lag_s=float(dataset.attrs["max_lag_s"]),
        lag_guard_s=float(dataset.attrs.get("lag_guard_s", 0.0)),
        pws_power=None if pws_power is None else float(pws_power),
    )
