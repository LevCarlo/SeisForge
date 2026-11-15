"""File formats for the AFTAN -> Eikonal -> inversion glue commands.

Formats read here (as written by the current code, not by the docs):

- AFTAN ``*.dat`` (``seisforge.ant.aftan.io``): ``# key: value`` metadata
  lines (``stage``, ``input_file`` = SAC basename, ``selected_branch``, ...),
  then a ``#`` column-name line ``target_period_s instant_period_s
  group_velocity_km_s phase_velocity_km_s amplitude snr|snr_db`` (debug files
  have four more columns before the SNR). Station names, component and
  coordinates are not recorded; they come from the SAC file name, a station
  table or the SAC headers.
- Station table: whitespace text, ``#`` comments, columns ``station
  longitude_deg latitude_deg`` (more columns are ignored).
- SAC CCF headers: ``evlo/evla`` are the stored source station, ``stlo/stla``
  the stored receiver station (the convention ``ant rotate-ccf`` uses).
- Eikonal stack (``eikonal_isotropic_stack``): variables on
  ``(period, latitude, longitude)`` with coordinate ``period_s(period)``.
"""

from __future__ import annotations

import glob
from pathlib import Path
import re

import numpy as np
import xarray as xr

from seisforge.ant.eikonal.io import write_netcdf
from seisforge.provenance import software_provenance

AFTAN_DAT_REQUIRED_COLUMNS = (
    "target_period_s",
    "instant_period_s",
    "phase_velocity_km_s",
    "amplitude",
)


def expand_inputs(patterns: list[str] | tuple[str, ...]) -> list[Path]:
    """Expand files and (recursive ``**``) glob patterns, sorted and unique."""
    found: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [str(pattern)]
        found.update(Path(item) for item in matches if Path(item).is_file())
    return sorted(found)


def read_aftan_dat(path: str | Path) -> dict:
    """Read one AFTAN dispersion ``.dat`` into metadata and named columns."""
    path = Path(path)
    metadata: dict[str, str] = {}
    columns: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            continue
        text = line[1:].strip()
        if text.startswith("target_period_s"):
            columns = text.split()
        elif ":" in text:
            key, value = text.split(":", 1)
            metadata[key.strip()] = value.strip()
    if columns is None:
        raise ValueError(f"{path}: no AFTAN column-name header line.")
    missing = [name for name in AFTAN_DAT_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{path}: missing AFTAN columns {missing}.")
    snr_names = [name for name in columns if name in {"snr", "snr_db"}]
    if len(snr_names) != 1:
        raise ValueError(f"{path}: expected one snr or snr_db column.")
    for key in ("stage", "input_file", "selected_branch"):
        if key not in metadata:
            raise ValueError(f"{path}: header is missing '{key}'.")
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.size and data.shape[1] != len(columns):
        raise ValueError(f"{path}: {data.shape[1]} columns, header names {len(columns)}.")
    values = {
        name: (data[:, index] if data.size else np.empty(0))
        for index, name in enumerate(columns)
    }
    return {
        "path": path,
        "metadata": metadata,
        "columns": values,
        "snr_scale": "db" if snr_names[0] == "snr_db" else "linear",
        "snr": values[snr_names[0]],
    }


def parse_pair_name(name: str, pattern: str) -> dict[str, str]:
    """Match a SAC stem against ``pattern`` and return its named fields."""
    regex = ""
    for token in re.split(r"(\{[a-z_]+\}|\*)", pattern):
        if token == "*":
            regex += ".*"
        elif token.startswith("{") and token.endswith("}"):
            field = token[1:-1]
            if field not in {"source_station", "receiver_station", "component"}:
                raise ValueError(f"unknown name_pattern field {token}.")
            regex += f"(?P<{field}>[^_/]+)"
        else:
            regex += re.escape(token)
    match = re.fullmatch(regex, name)
    if match is None:
        raise ValueError(f"'{name}' does not match name_pattern '{pattern}'.")
    return match.groupdict()


def read_station_table(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read ``station longitude_deg latitude_deg`` rows."""
    stations: dict[str, tuple[float, float]] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        fields = text.split()
        if len(fields) < 3:
            raise ValueError(f"{path}:{number}: need station longitude latitude.")
        name, longitude, latitude = fields[0], float(fields[1]), float(fields[2])
        if name in stations and stations[name] != (longitude, latitude):
            raise ValueError(f"{path}:{number}: station {name} listed twice.")
        stations[name] = (longitude, latitude)
    return stations


def find_sac_file(sac_dir: str | Path, basename: str) -> Path:
    """Find the unique file called ``basename`` below ``sac_dir``."""
    matches = sorted(Path(sac_dir).rglob(basename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one '{basename}' below {sac_dir}, found {len(matches)}."
        )
    return matches[0]


def read_sac_pair_coordinates(path: str | Path) -> tuple[float, float, float, float]:
    """Return stored ``(source_lon, source_lat, receiver_lon, receiver_lat)``."""
    from obspy import read

    sac = read(str(path), headonly=True)[0].stats.sac
    values = []
    for key in ("evlo", "evla", "stlo", "stla"):
        value = sac.get(key)
        if value is None or float(value) == -12345.0:
            raise ValueError(f"{path}: SAC header {key} is not set.")
        values.append(float(value))
    return tuple(values)


def measurement_dataset(
    *,
    periods_s: tuple[int, ...],
    paths: list[dict],
    attrs: dict[str, object],
    include_snr: bool,
    include_amplitude: bool,
) -> xr.Dataset:
    """Build the Eikonal input dataset (``docs/eikonal_workflow.md``)."""

    def column(key: str) -> np.ndarray:
        return np.asarray([item[key] for item in paths])

    data_vars: dict[str, tuple] = {
        "phase_velocity_km_s": (("path", "period"), column("phase_velocity_km_s")),
    }
    if include_snr:
        data_vars["snr"] = (("path", "period"), column("snr"))
    if include_amplitude:
        data_vars["amplitude"] = (("path", "period"), column("amplitude"))
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "path": np.arange(len(paths)),
            "period": np.arange(len(periods_s)),
            "period_s": ("period", np.asarray(periods_s, dtype=np.int32)),
            "source_station": ("path", column("source_station").astype(str)),
            "receiver_station": ("path", column("receiver_station").astype(str)),
            "source_longitude_deg": ("path", column("source_longitude_deg").astype(float)),
            "source_latitude_deg": ("path", column("source_latitude_deg").astype(float)),
            "receiver_longitude_deg": (
                "path",
                column("receiver_longitude_deg").astype(float),
            ),
            "receiver_latitude_deg": (
                "path",
                column("receiver_latitude_deg").astype(float),
            ),
            "aftan_file": ("path", column("aftan_file").astype(str)),
            "aftan_branch": ("path", column("aftan_branch").astype(str)),
        },
        attrs={**attrs, **software_provenance()},
    )
    dataset["phase_velocity_km_s"].attrs["units"] = "km s-1"
    dataset["period_s"].attrs["units"] = "s"
    for name in ("source_longitude_deg", "receiver_longitude_deg"):
        dataset[name].attrs["units"] = "degree_east"
    for name in ("source_latitude_deg", "receiver_latitude_deg"):
        dataset[name].attrs["units"] = "degree_north"
    dataset["aftan_branch"].attrs["long_name"] = (
        "CCF lag branch measured; negative means the stored pair is reversed"
    )
    return dataset


def write_measurement_dataset(
    dataset: xr.Dataset, path: str | Path, *, overwrite: bool
) -> Path:
    """Write the collected dataset as HDF5 NetCDF (the Eikonal reader's engine)."""
    return write_netcdf(dataset, path, overwrite=overwrite, compression_level=4)


def read_stack(path: str | Path) -> xr.Dataset:
    """Load an Eikonal isotropic-stack NetCDF file into memory."""
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        dataset = opened.load()
    if dataset.attrs.get("stage") != "eikonal_isotropic_stack":
        raise ValueError(f"{path} is not an eikonal_isotropic_stack file.")
    return dataset


def write_three_column_dat(
    path: str | Path,
    period_s: np.ndarray,
    value: np.ndarray,
    sigma: np.ndarray,
    *,
    header: dict[str, object],
    column_names: str,
    overwrite: bool,
) -> Path:
    """Write ``period value sigma`` rows with ``# key: value`` header lines."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {item}" for key, item in {**header, **software_provenance()}.items()]
    lines.append(column_names)
    np.savetxt(
        path,
        np.column_stack([period_s, value, sigma]),
        fmt=("%g", "%.5f", "%.5f"),
        header="\n".join(lines),
    )
    return path
