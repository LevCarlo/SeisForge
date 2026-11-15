"""Validated settings for the AFTAN -> Eikonal -> inversion glue commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LONGITUDE_CONVENTIONS = ("-180_180", "0_360")

# Assumed value (not taken from a source): 0.01 km/s is about 0.3 % of a
# typical crustal Rayleigh phase velocity, near the smallest standard errors
# reported for dense-array Eikonal stacks. Change it per project.
DEFAULT_SIGMA_FLOOR_KM_S = 0.01


@dataclass(frozen=True)
class CollectDispersionConfig:
    """Choices for gathering AFTAN ``.dat`` files into the Eikonal dataset.

    ``name_pattern`` is matched against the stem of the SAC file recorded in
    each ``.dat`` header (``input_file``); ``{source_station}``,
    ``{receiver_station}`` and ``{component}`` match one ``_``-free field and
    ``*`` matches anything. The SAC file's stored pair is (source, receiver)
    for its positive lags; the negative branch is the reversed pair.
    """

    periods_s: tuple[int, ...]
    station_table: Path | None = None
    sac_dir: Path | None = None
    name_pattern: str = "{source_station}_{receiver_station}_{component}*"
    component: str | None = None
    aftan_stage: str = "basic"
    period_axis: str = "instant"
    longitude_convention: str = "-180_180"
    mirror_stack: bool = False
    include_amplitude: bool = False

    def __post_init__(self) -> None:
        periods = tuple(int(value) for value in self.periods_s)
        if not periods or any(value <= 0 for value in periods):
            raise ValueError("periods_s must contain positive integer periods.")
        if any(int(a) != a for a in self.periods_s):
            raise ValueError("periods_s must be integers (Eikonal input contract).")
        if any(b <= a for a, b in zip(periods, periods[1:])):
            raise ValueError("periods_s must be strictly increasing.")
        object.__setattr__(self, "periods_s", periods)
        if (self.station_table is None) == (self.sac_dir is None):
            raise ValueError("give exactly one of station_table or sac_dir.")
        if "{source_station}" not in self.name_pattern or (
            "{receiver_station}" not in self.name_pattern
        ):
            raise ValueError(
                "name_pattern must contain {source_station} and {receiver_station}."
            )
        if self.component is None and "{component}" not in self.name_pattern:
            raise ValueError("give component or put {component} in name_pattern.")
        if self.aftan_stage not in {"basic", "pmf"}:
            raise ValueError("aftan_stage must be 'basic' or 'pmf'.")
        if self.period_axis not in {"instant", "target"}:
            raise ValueError("period_axis must be 'instant' or 'target'.")
        if self.longitude_convention not in LONGITUDE_CONVENTIONS:
            raise ValueError("longitude_convention must be '-180_180' or '0_360'.")


@dataclass(frozen=True)
class StationCurveConfig:
    """Choices for extracting one station's curve from an Eikonal stack."""

    field: str = "eikonal"
    sigma_source: str = "sem"
    sigma_floor_km_s: float = DEFAULT_SIGMA_FLOOR_KM_S
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        if self.field not in {"eikonal", "helmholtz"}:
            raise ValueError("field must be 'eikonal' or 'helmholtz'.")
        if self.sigma_source not in {"sem", "std"}:
            raise ValueError("sigma_source must be 'sem' or 'std'.")
        if not (self.sigma_floor_km_s >= 0) or self.sigma_floor_km_s == float("inf"):
            raise ValueError("sigma_floor_km_s must be finite and non-negative.")
        if self.interpolation not in {"bilinear", "nearest"}:
            raise ValueError("interpolation must be 'bilinear' or 'nearest'.")
