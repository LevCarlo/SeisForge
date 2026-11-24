"""Validated configuration of the RF workflows (`rf qc`, `rf hkseq`).

The YAML layout is documented in `docs/rf/qc.md` and `docs/rf/hk.md`. Every
key is required (there are no defaults) except ``IO.OVERWRITE``, which is
false when absent. Keys removed in 0.2.0 raise a ``ValueError`` that says so,
and unknown keys in the blocks parsed here raise too, so a misspelt key
cannot be silently ignored.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

QC_BANDS = ("HighFreq", "LowFreq")

_REMOVED_HK_QC_KEYS = {
    "energy_threshold": "the energy screen",
    "ell_mad_sigma": "the elliptical MAD screen",
}

_QC_STEP_KEYS = {
    "SNR": ("THRESHOLD", "SAC_HEADER"),
    "SLOWNESS": ("MIN", "MAX", "SAC_HEADER"),
    "PAMP": ("TMIN", "TMAX", "WINDOW"),
    "MAD": ("THRESHOLD", "WINDOW"),
    "F_TEST": ("THRESHOLD", "WINDOW"),
    "CC": ("THRESHOLD", "WINDOW"),
}


def _require(block: Mapping, key: str, where: str):
    if not isinstance(block, Mapping):
        raise ValueError(f"{where} must be a mapping.")
    if key not in block:
        raise ValueError(f"Missing config key {where}.{key}.")
    return block[key]


def _check_keys(block: Mapping, allowed, where: str) -> None:
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown config keys in {where}: {', '.join(map(str, unknown))}.")


def _number(value, where: str, *, minimum=None, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}.")
    value = float(value)
    if positive and not value > 0:
        raise ValueError(f"{where} must be positive, got {value}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum}, got {value}.")
    return value


@dataclass(frozen=True)
class HkGridConfig:
    """Search range ``min..max`` with step ``delta`` (see ``hk_core.hk_grid``)."""

    min: float
    max: float
    delta: float

    @classmethod
    def from_mapping(cls, block: Mapping, where: str, *, minimum: float) -> HkGridConfig:
        _check_keys(block, ("min", "max", "delta"), where)
        grid = cls(
            min=_number(_require(block, "min", where), f"{where}.min", minimum=minimum),
            max=_number(_require(block, "max", where), f"{where}.max"),
            delta=_number(_require(block, "delta", where), f"{where}.delta", positive=True),
        )
        if grid.max < grid.min:
            raise ValueError(f"{where}.max ({grid.max}) is below {where}.min ({grid.min}).")
        return grid


@dataclass(frozen=True)
class HkLayerConfig:
    """One H-k stage: grids, the fixed velocity, weights, direct-P exclusion."""

    H: HkGridConfig
    k: HkGridConfig
    vp_km_s: float | None
    vs_km_s: float | None
    weights: tuple[float, float, float]
    min_ps_delay_s: float

    @classmethod
    def from_mapping(cls, block: Mapping, where: str, extra=()) -> HkLayerConfig:
        _check_keys(
            block, ("H", "k", "Vp", "Vs", "weight", "min_ps_delay_s", *extra), where
        )
        vp = _require(block, "Vp", where)
        vs = _require(block, "Vs", where)
        if (vp is None) == (vs is None):
            raise ValueError(f"Set exactly one of {where}.Vp and {where}.Vs; the other is null.")
        weights = _require(block, "weight", where)
        if not isinstance(weights, (list, tuple)) or len(weights) != 3:
            raise ValueError(f"{where}.weight must be three numbers (Ps, PpPs, PsPs).")
        return cls(
            H=HkGridConfig.from_mapping(_require(block, "H", where), f"{where}.H", minimum=0.0),
            k=HkGridConfig.from_mapping(_require(block, "k", where), f"{where}.k", minimum=1.0),
            vp_km_s=None if vp is None else _number(vp, f"{where}.Vp", positive=True),
            vs_km_s=None if vs is None else _number(vs, f"{where}.Vs", positive=True),
            weights=tuple(_number(w, f"{where}.weight") for w in weights),
            min_ps_delay_s=_number(
                _require(block, "min_ps_delay_s", where), f"{where}.min_ps_delay_s", minimum=0.0
            ),
        )


@dataclass(frozen=True)
class HkSeqConfig:
    """Configuration of `seisforge rf hkseq` (paths resolved against ``IO.ROOT``)."""

    root: str
    high_rf_dir: str
    low_rf_dir: str
    save_dir: str
    figure_dir: str
    log_dir: str
    overwrite: bool
    station: str
    network: str
    sediment: HkLayerConfig
    crust: HkLayerConfig
    two_layer_min_thickness_km: float
    mahalanobis_threshold: Mapping[str, float]

    @classmethod
    def from_mapping(cls, params: Mapping) -> HkSeqConfig:
        io = _require(params, "IO", "config")
        meta = _require(params, "Meta", "config")
        model = _require(params, "Model", "config")
        qc = _require(params, "QC", "config")
        root = str(_require(io, "ROOT", "IO"))
        sediment_block = _require(model, "Sediment", "Model")
        thresholds = {}
        for band in QC_BANDS:
            block = _require(qc, band, "QC")
            for key, screen in _REMOVED_HK_QC_KEYS.items():
                if key in block:
                    raise ValueError(
                        f"QC.{band}.{key} was removed in SeisForge 0.2.0: {screen} never "
                        "changed the H-k estimate (only the Mahalanobis screen did). "
                        "Delete the key."
                    )
            _check_keys(block, ("mahalanobis_dist_threshold",), f"QC.{band}")
            thresholds[band] = _number(
                _require(block, "mahalanobis_dist_threshold", f"QC.{band}"),
                f"QC.{band}.mahalanobis_dist_threshold",
                positive=True,
            )
        return cls(
            root=root,
            high_rf_dir=os.path.join(root, _require(io, "HighFreq_RF", "IO")),
            low_rf_dir=os.path.join(root, _require(io, "LowFreq_RF", "IO")),
            save_dir=os.path.join(root, _require(io, "SAVE", "IO")),
            figure_dir=os.path.join(root, _require(io, "FIGURE", "IO")),
            log_dir=os.path.join(root, _require(io, "LOG", "IO")),
            overwrite=bool(io.get("OVERWRITE", False)),
            station=str(_require(meta, "station", "Meta")),
            network=str(_require(meta, "network", "Meta")),
            sediment=HkLayerConfig.from_mapping(
                sediment_block, "Model.Sediment", extra=("two_layer_min_thickness_km",)
            ),
            crust=HkLayerConfig.from_mapping(_require(model, "Crust", "Model"), "Model.Crust"),
            two_layer_min_thickness_km=_number(
                _require(sediment_block, "two_layer_min_thickness_km", "Model.Sediment"),
                "Model.Sediment.two_layer_min_thickness_km",
                minimum=0.0,
            ),
            mahalanobis_threshold=MappingProxyType(thresholds),
        )


@dataclass(frozen=True)
class QCStepConfig:
    """One QC test: whether it runs and its settings (keys as in the YAML)."""

    name: str
    enable: bool
    settings: Mapping

    @classmethod
    def from_mapping(cls, name: str, block: Mapping, where: str) -> QCStepConfig:
        required = _QC_STEP_KEYS[name]
        _check_keys(block, ("ENABLE", *required), where)
        enable = _require(block, "ENABLE", where)
        if not isinstance(enable, bool):
            raise ValueError(f"{where}.ENABLE must be true or false.")
        settings = {}
        if enable:
            settings = {key: _require(block, key, where) for key in required}
        return cls(name=name, enable=enable, settings=MappingProxyType(settings))


@dataclass(frozen=True)
class RFQCConfig:
    """Configuration of `seisforge rf qc` (paths resolved against ``IO.ROOT``)."""

    root: str
    raw_dirs: Mapping[str, str]
    clean_dirs: Mapping[str, str]
    figure_dir: str
    log_dir: str
    overwrite: bool
    station: str
    network: str
    component: str
    steps: Mapping[str, tuple[QCStepConfig, ...]]

    @classmethod
    def from_mapping(cls, params: Mapping) -> RFQCConfig:
        io = _require(params, "IO", "config")
        meta = _require(params, "META", "config")
        qc = _require(params, "QC", "config")
        root = str(_require(io, "ROOT", "IO"))
        steps = {}
        for band in QC_BANDS:
            block = _require(qc, band, "QC")
            _check_keys(block, tuple(_QC_STEP_KEYS), f"QC.{band}")
            steps[band] = tuple(
                QCStepConfig.from_mapping(name, _require(block, name, f"QC.{band}"),
                                          f"QC.{band}.{name}")
                for name in _QC_STEP_KEYS
            )
        return cls(
            root=root,
            raw_dirs=MappingProxyType({
                band: os.path.join(root, _require(io, f"RAW_{band}_RF", "IO"))
                for band in QC_BANDS
            }),
            clean_dirs=MappingProxyType({
                band: os.path.join(root, _require(io, f"CLEAN_{band}_RF", "IO"))
                for band in QC_BANDS
            }),
            figure_dir=os.path.join(root, _require(io, "FIGURE", "IO")),
            log_dir=os.path.join(root, _require(io, "LOG", "IO")),
            overwrite=bool(io.get("OVERWRITE", False)),
            station=str(_require(meta, "station", "META")),
            network=str(_require(meta, "network", "META")),
            component=str(meta.get("component", "R")),
            steps=MappingProxyType(steps),
        )
