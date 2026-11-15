"""Glue between AFTAN, Eikonal tomography and the Vs(z) inversion.

``collect_dispersion`` turns AFTAN ``.dat`` files into the Eikonal input
dataset; ``extract_station_curve`` turns an Eikonal stack into the
inversion's three-column ``period value sigma`` file.
"""

from .core import (
    resample_to_periods,
    sample_grid,
    station_sigma_km_s,
    velocity_std_from_slowness_std,
    wrap_longitude_deg,
)
from .driver import (
    CollectResult,
    StationCurve,
    collect_dispersion,
    extract_station_curve,
    run_collect_dispersion,
    run_extract_station_curve,
)
from .io import read_aftan_dat, read_station_table
from .models import (
    DEFAULT_SIGMA_FLOOR_KM_S,
    CollectDispersionConfig,
    StationCurveConfig,
)

__all__ = [
    "DEFAULT_SIGMA_FLOOR_KM_S",
    "CollectDispersionConfig",
    "CollectResult",
    "StationCurve",
    "StationCurveConfig",
    "collect_dispersion",
    "extract_station_curve",
    "read_aftan_dat",
    "read_station_table",
    "resample_to_periods",
    "run_collect_dispersion",
    "run_extract_station_curve",
    "sample_grid",
    "station_sigma_km_s",
    "velocity_std_from_slowness_std",
    "wrap_longitude_deg",
]
