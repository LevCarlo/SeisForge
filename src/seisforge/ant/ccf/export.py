"""Independent post-processing exporter: CCF NetCDF4/HDF5 -> SAC.

This module is intentionally separate from daily CCF computation (Phase B).
It reads a stacked or daily CCF NetCDF4/HDF5 dataset and writes selected
station pairs as standard SAC files with geometry headers (``dist``, ``az``,
``baz``, ``evla``, ``evlo``, ``stla``, ``stlo``, ``b``, ``delta``) ready for
``seisforge ant aftan`` or external tools.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from .io import read_netcdf

__all__ = ["EXPORT_LAYOUTS", "export_ccf_to_sac"]

EXPORT_LAYOUTS: tuple[str, ...] = ("by-source", "flat")
_LAYOUT_TEMPLATES: dict[str, str] = {
    "by-source": "{source}/{source}_{receiver}_{component}.sac",
    "flat": "{source}_{receiver}_{component}.sac",
}


def export_ccf_to_sac(
    source: xr.Dataset | str | Path,
    output_dir: str | Path,
    *,
    pairs: Iterable[tuple[str, str]] | None = None,
    min_distance_km: float | None = None,
    max_distance_km: float | None = None,
    min_n_windows: int = 1,
    include_autocorr: bool = False,
    layout: str = "by-source",
    filename_template: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Selectively export station pairs from a CCF NetCDF4/HDF5 dataset to SAC.

    Parameters
    ----------
    source : xarray.Dataset or path-like
        Daily (``stage='ccf_daily'``) or stacked (``stage='ccf_stacked'``)
        CCF dataset. For daily datasets, ``ccf_sum / n_windows`` is exported.
    output_dir : path-like
        Destination directory for SAC files.
    pairs : iterable of (source_station, receiver_station), optional
        Explicit subset of station IDs (e.g. ``[("XX.STA1", "XX.STA2")]``) to
        export. If None, all pairs passing the filters below are exported.
    min_distance_km, max_distance_km : float, optional
        Inclusive interstation distance bounds in km.
    min_n_windows : int, optional
        Minimum valid window count required to export a pair (default: 1).
    include_autocorr : bool, optional
        Whether to include same-station autocorrelations (default: False).
    layout : {"by-source", "flat"}, optional
        Directory organization when ``filename_template`` is None.
        ``"by-source"`` (default) writes ``{source}/{source}_{receiver}_{component}.sac``
        to avoid huge flat directories on large arrays; ``"flat"`` writes directly
        under ``output_dir``.
    filename_template : str, optional
        Explicit relative output path template with ``{source}``, ``{receiver}``,
        ``{component}``. Overrides ``layout`` when provided.
    overwrite : bool, optional
        If False (default), raise ``FileExistsError`` if any target SAC exists.

    Returns
    -------
    written : list of Path
        Paths of all SAC files written.
    """
    from obspy import Trace, UTCDateTime

    if filename_template is None:
        if layout not in _LAYOUT_TEMPLATES:
            raise ValueError(
                f"layout must be one of {EXPORT_LAYOUTS}, got {layout!r}."
            )
        resolved_template = _LAYOUT_TEMPLATES[layout]
    else:
        resolved_template = filename_template

    ds = source if isinstance(source, xr.Dataset) else read_netcdf(source)
    stage = ds.attrs.get("stage")
    if stage not in {"ccf_daily", "ccf_stacked"}:
        raise ValueError(
            f"Unsupported dataset stage {stage!r}; expected 'ccf_daily' or 'ccf_stacked'."
        )

    component = str(ds.attrs.get("component", "ZZ"))
    lag_s = np.asarray(ds["lag_s"].values, dtype=np.float64)
    fs = float(ds.attrs["sampling_rate_hz"])
    dt_s = 1.0 / fs
    n_windows = np.asarray(ds["n_windows"].values, dtype=np.int64)
    src_sta = np.asarray(ds["source_station"].values).astype(str)
    rcv_sta = np.asarray(ds["receiver_station"].values).astype(str)
    dist_km = np.asarray(ds["distance_km"].values, dtype=np.float64)
    az_deg = np.asarray(ds["azimuth_deg"].values, dtype=np.float64)
    baz_deg = np.asarray(ds["back_azimuth_deg"].values, dtype=np.float64)
    evla = np.asarray(ds["source_latitude_deg"].values, dtype=np.float64)
    evlo = np.asarray(ds["source_longitude_deg"].values, dtype=np.float64)
    stla = np.asarray(ds["receiver_latitude_deg"].values, dtype=np.float64)
    stlo = np.asarray(ds["receiver_longitude_deg"].values, dtype=np.float64)

    if stage == "ccf_stacked":
        ccf_all = np.asarray(ds["ccf"].values, dtype=np.float64)
    else:
        ccf_sum = np.asarray(ds["ccf_sum"].values, dtype=np.float64)
        ccf_all = np.full_like(ccf_sum, np.nan)
        valid_cnt = n_windows > 0
        ccf_all[valid_cnt] = ccf_sum[valid_cnt] / n_windows[valid_cnt, None]

    pair_filter = None if pairs is None else {(str(a), str(b)) for a, b in pairs}
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[int, Path]] = []
    for idx in range(n_windows.size):
        s_name = str(src_sta[idx])
        r_name = str(rcv_sta[idx])
        if not include_autocorr and s_name == r_name:
            continue
        if pair_filter is not None and (s_name, r_name) not in pair_filter:
            continue
        if int(n_windows[idx]) < int(min_n_windows):
            continue
        d_km = float(dist_km[idx])
        if min_distance_km is not None and d_km < float(min_distance_km):
            continue
        if max_distance_km is not None and d_km > float(max_distance_km):
            continue
        if not np.all(np.isfinite(ccf_all[idx])):
            continue

        out_path = out_root / resolved_template.format(
            source=s_name,
            receiver=r_name,
            component=component,
        )
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"SAC output exists and overwrite is False: {out_path}"
            )
        selected.append((idx, out_path))

    written: list[Path] = []
    for idx, out_path in selected:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rcv_str = str(rcv_sta[idx])
        if "." in rcv_str:
            rcv_net, rcv_station = rcv_str.split(".", 1)
        else:
            rcv_net, rcv_station = "", rcv_str

        sac_header: dict[str, object] = {
            "b": float(lag_s[0]),
            "e": float(lag_s[-1]),
            "delta": float(dt_s),
            "dist": float(dist_km[idx]),
            "evla": float(evla[idx]),
            "evlo": float(evlo[idx]),
            "stla": float(stla[idx]),
            "stlo": float(stlo[idx]),
            "kevnm": str(src_sta[idx])[:16],
            "kstnm": rcv_station[:8],
            "kcmpnm": component[:8],
            "lcalda": True,
        }
        if rcv_net:
            sac_header["knetwk"] = rcv_net[:8]
        if np.isfinite(az_deg[idx]):
            sac_header["az"] = float(az_deg[idx])
        if np.isfinite(baz_deg[idx]):
            sac_header["baz"] = float(baz_deg[idx])

        tr = Trace(
            data=np.asarray(ccf_all[idx], dtype=np.float32),
            header={
                "network": rcv_net,
                "station": rcv_station,
                "channel": component,
                "sampling_rate": fs,
                "starttime": UTCDateTime(0) + float(lag_s[0]),
                "sac": sac_header,
            },
        )
        tr.write(str(out_path), format="SAC")
        written.append(out_path)

    return written
