"""Configuration-driven receiver-function quality-control workflow (`rf qc`).

Each band's tests run in a fixed order on the survivors of the previous test
(`docs/rf/qc.md`). Outputs per band: copies of the passing SAC files in
``CLEAN_<band>_RF``, a table ``LOG/NET.STA_RF_QC_<band>.dat`` with every
input file's metrics and fate plus the settings and software provenance,
and with ``--plot`` a figure. The log ``LOG/NET.STA_RF_QC.log`` records the
counts. Existing outputs stop the run unless ``IO.OVERWRITE`` or
``--overwrite`` is set.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import yaml

from seisforge.provenance import software_provenance
from seisforge.rf.config import QC_BANDS, QCStepConfig, RFQCConfig
from seisforge.rf.traces import RFstream, RFtrace

logger = logging.getLogger(__name__)

# Loggers of the RF package whose messages go into the QC log file.
_RF_LOGGER = logging.getLogger("seisforge.rf")

_STEP_LABELS = {
    "SNR": "SNR",
    "SLOWNESS": "slowness",
    "PAMP": "P amplitude",
    "MAD": "MAD",
    "F_TEST": "F-test",
    "CC": "cross-correlation",
}

_SELECTORS: dict[str, Callable[[RFstream, dict], RFstream]] = {
    "SNR": lambda s, c: s.SNR_select(threshold=c["THRESHOLD"], sac_head=c["SAC_HEADER"]),
    "SLOWNESS": lambda s, c: s.slowness_select(
        slow_min=c["MIN"], slow_max=c["MAX"], sac_head=c["SAC_HEADER"]
    ),
    "PAMP": lambda s, c: s.P_amp_select(tmin=c["TMIN"], tmax=c["TMAX"], window=c["WINDOW"]),
    "MAD": lambda s, c: s.MAD_select(threshold=c["THRESHOLD"], window=c["WINDOW"]),
    "F_TEST": lambda s, c: s.f_test_select(threshold=c["THRESHOLD"], window=c["WINDOW"]),
    "CC": lambda s, c: s.CC_select(threshold=c["THRESHOLD"], window=c["WINDOW"]),
}

_TABLE_COLUMNS = ("SNR", "slowness_s_per_km", "P_time_s", "P_amp", "MAD_z", "F_p", "CC")


def _metric_columns(step: str, metric) -> dict:
    if step == "SNR":
        return {"SNR": metric}
    if step == "SLOWNESS":
        return {"slowness_s_per_km": metric}
    if step == "PAMP":
        return {"P_time_s": metric[0], "P_amp": metric[1]}
    if step == "MAD":
        return {"MAD_z": metric}
    if step == "F_TEST":
        return {"F_p": metric}
    return {"CC": metric}


def _band_outputs(config: RFQCConfig, band: str, plot: bool) -> list[str]:
    """Existing files that this band's run would replace."""
    net, sta = config.network, config.station
    existing = sorted(glob.glob(os.path.join(config.clean_dirs[band], f"*{config.component}*")))
    for path in (
        _table_path(config, band),
        os.path.join(config.figure_dir, f"{net}.{sta}_RF_QC_{band}.png") if plot else None,
    ):
        if path is not None and os.path.exists(path):
            existing.append(path)
    return existing


def _table_path(config: RFQCConfig, band: str) -> str:
    return os.path.join(config.log_dir, f"{config.network}.{config.station}_RF_QC_{band}.dat")


def _check_clean_dir(config: RFQCConfig, band: str) -> None:
    clean = Path(config.clean_dirs[band]).resolve()
    for raw in config.raw_dirs.values():
        raw = Path(raw).resolve()
        if clean.is_relative_to(raw) or raw.is_relative_to(clean):
            raise ValueError(
                f"IO.CLEAN_{band}_RF {clean} overlaps the RF input directory {raw}."
            )


def _write_table(config, band, steps, rows, n_in, n_out):
    with open(_table_path(config, band), "w", encoding="utf-8") as file:
        for key, value in software_provenance().items():
            file.write(f"# {key}: {value}\n")
        file.write(f"# band: {band}\n")
        file.write(f"# raw_dir: {config.raw_dirs[band]}\n")
        file.write(f"# clean_dir: {config.clean_dirs[band]}\n")
        for step in steps:
            settings = dict(step.settings) if step.enable else {}
            file.write(f"# QC.{band}.{step.name}: enable={step.enable} {settings}\n")
        file.write(
            "# F_p: one-sided leave-one-out F-test p-value with effective degrees of "
            "freedom (seisforge.rf.qc_core)\n"
        )
        file.write(f"# n_in: {n_in}\n# n_kept: {n_out}\n")
        file.write("# filename status " + " ".join(_TABLE_COLUMNS) + "\n")
        for name, (status, metrics) in rows.items():
            values = " ".join(
                f"{float(metrics[c]):.6g}" if c in metrics else "nan" for c in _TABLE_COLUMNS
            )
            file.write(f"{name} {status} {values}\n")


def rf_QC(params, type: str = "LowFreq", plot: bool = True,
          overwrite: bool | None = None) -> RFstream | None:
    """Run RF quality control for one band (``HighFreq`` or ``LowFreq``).

    ``params`` is the YAML mapping or an ``RFQCConfig``. Existing outputs of
    this band raise ``FileExistsError`` unless ``overwrite`` (or
    ``IO.OVERWRITE``) is set; with it, previous copies in the CLEAN folder
    are removed first, so the folder holds exactly this run's survivors.
    """
    config = params if isinstance(params, RFQCConfig) else RFQCConfig.from_mapping(params)
    band = type
    if band not in QC_BANDS:
        raise ValueError(f"Unknown band {band!r}; use one of {QC_BANDS}.")
    overwrite = config.overwrite if overwrite is None else overwrite
    _check_clean_dir(config, band)
    existing = _band_outputs(config, band, plot)
    if existing and not overwrite:
        raise FileExistsError(
            "RF QC outputs already exist (use --overwrite or IO.OVERWRITE: true): "
            + ", ".join(existing)
        )
    net, sta = config.network, config.station
    clean_rf_dir = config.clean_dirs[band]
    for directory in (clean_rf_dir, config.figure_dir, config.log_dir):
        os.makedirs(directory, exist_ok=True)

    filelst = sorted(glob.glob(os.path.join(config.raw_dirs[band], f"*{config.component}*")))
    if not filelst:
        logger.warning(
            "No RF files found in %s. Skipping QC for %s.%s.", config.raw_dirs[band], net, sta
        )
        return None

    for path in sorted(glob.glob(os.path.join(clean_rf_dir, f"*{config.component}*"))):
        os.remove(path)
        logger.info("Removed previous QC output %s (overwrite)", path)

    rf = RFstream(files=filelst).load_data()
    logger.info("Number of RF traces loaded: %s", len(rf))
    rows = {os.path.basename(f): ["kept", {}] for f in rf.files}
    steps: tuple[QCStepConfig, ...] = config.steps[band]
    for step in steps:
        label = _STEP_LABELS[step.name]
        if not step.enable:
            logger.info("No %s selection applied.", label)
            continue
        entering = [os.path.basename(f) for f in rf.files]
        selected = _SELECTORS[step.name](rf, dict(step.settings))
        for name, metric in zip(entering, selected.step_metrics):
            rows[name][1].update(_metric_columns(step.name, metric))
        kept = {os.path.basename(f) for f in selected.files}
        for name in entering:
            if name not in kept:
                rows[name][0] = f"removed:{step.name}"
        rf = selected
        logger.info("Number of RF traces after %s selection: %s", label, len(rf))

    for file in rf.files:
        shutil.copy2(file, clean_rf_dir)
    logger.info("Cleaned RF traces saved to %s", clean_rf_dir)
    _write_table(config, band, steps, rows, len(filelst), len(rf))

    if plot and len(rf) > 0:
        rf.plot(save=True, save_path=os.path.join(config.figure_dir, f"{net}.{sta}_RF_QC_{band}.png"))
        logger.info("RF QC plot completed and saved")

    return rf


def run_rf_qc_config(config_file: str, plot: bool = False, overwrite: bool | None = None) -> None:
    """Run high- and low-frequency RF QC from a YAML config file."""
    with open(config_file, "r", encoding="utf-8") as file:
        config = RFQCConfig.from_mapping(yaml.safe_load(file))
    overwrite = config.overwrite if overwrite is None else overwrite
    net, sta = config.network, config.station
    log_path = os.path.join(config.log_dir, f"{net}.{sta}_RF_QC.log")

    for band in QC_BANDS:
        _check_clean_dir(config, band)
    existing = [path for band in QC_BANDS for path in _band_outputs(config, band, plot)]
    if os.path.exists(log_path):
        existing.append(log_path)
    if existing and not overwrite:
        raise FileExistsError(
            "RF QC outputs already exist (use --overwrite or IO.OVERWRITE: true): "
            + ", ".join(existing)
        )
    os.makedirs(config.log_dir, exist_ok=True)

    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    previous_level = _RF_LOGGER.level
    _RF_LOGGER.addHandler(handler)
    _RF_LOGGER.setLevel(logging.INFO)
    try:
        print(f"Running RF QC for {net}.{sta}...")
        for key, value in software_provenance().items():
            logger.info("%s: %s", key, value)
        logger.info("%sHigh-Frequency RF QC%s", "=" * 20, "=" * 20)
        rf_QC(config, type="HighFreq", plot=plot, overwrite=overwrite)
        logger.info("%sLow-Frequency RF QC%s", "=" * 20, "=" * 20)
        rf_QC(config, type="LowFreq", plot=plot, overwrite=overwrite)
    finally:
        _RF_LOGGER.removeHandler(handler)
        _RF_LOGGER.setLevel(previous_level)
        handler.close()


def load_parse_args():
    parser = argparse.ArgumentParser(description="Receiver Function Quality Control")
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        required=True,
        help="Path to the QC config YAML file.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Enable plotting of RF QC results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing QC outputs.",
    )
    return parser.parse_args()


def main():
    args = load_parse_args()
    run_rf_qc_config(args.config_file, plot=args.plot, overwrite=args.overwrite or None)


__all__ = [
    "RFstream",
    "RFtrace",
    "load_parse_args",
    "main",
    "rf_QC",
    "run_rf_qc_config",
]
