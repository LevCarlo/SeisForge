"""Configuration-driven receiver-function quality-control workflow."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
from collections.abc import Callable

import yaml

from seisforge.rf.traces import RFstream, RFtrace


def _is_enabled(config: dict) -> bool:
    return bool(config.get("ENABLE", False))


def _copy_step(stream: RFstream, name: str) -> RFstream:
    logging.info("No %s selection applied.", name)
    return stream.copy()


def _apply_step(
    stream: RFstream,
    name: str,
    config: dict,
    selector: Callable[[RFstream, dict], RFstream],
) -> RFstream:
    if not _is_enabled(config):
        return _copy_step(stream, name)

    selected = selector(stream, config)
    logging.info("Number of RF traces after %s selection: %s", name, len(selected))
    return selected


def rf_QC(params: dict, type: str = "LowFreq", plot: bool = True) -> RFstream | None:
    """Run RF quality control for one frequency band from a config dictionary."""
    io_params = params["IO"]
    rootdir = io_params["ROOT"]
    raw_rf_dir = os.path.join(rootdir, io_params[f"RAW_{type}_RF"])
    clean_rf_dir = os.path.join(rootdir, io_params[f"CLEAN_{type}_RF"])
    figdir = os.path.join(rootdir, io_params["FIGURE"])
    logdir = os.path.join(rootdir, io_params["LOG"])
    os.makedirs(clean_rf_dir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    meta_params = params["META"]
    sta = meta_params["station"]
    net = meta_params["network"]
    comp = meta_params.get("component", "R")
    qc_params = params["QC"][type]

    filelst = sorted(glob.glob(os.path.join(raw_rf_dir, f"*{comp}*")))
    if not filelst:
        logging.warning("No RF files found in %s. Skipping QC for %s.%s.", raw_rf_dir, net, sta)
        return None

    rf = RFstream(files=filelst).load_data()
    logging.info("Number of RF traces loaded: %s", len(rf))

    rf = _apply_step(
        rf,
        "SNR",
        qc_params["SNR"],
        lambda stream, config: stream.SNR_select(
            threshold=config["THRESHOLD"],
            sac_head=config["SAC_HEADER"],
        ),
    )
    rf = _apply_step(
        rf,
        "slowness",
        qc_params["SLOWNESS"],
        lambda stream, config: stream.slowness_select(
            slow_min=config["MIN"],
            slow_max=config["MAX"],
            sac_head=config["SAC_HEADER"],
        ),
    )
    rf = _apply_step(
        rf,
        "P amplitude",
        qc_params["PAMP"],
        lambda stream, config: stream.P_amp_select(
            tmin=config["TMIN"],
            tmax=config["TMAX"],
            window=config["WINDOW"],
        ),
    )
    rf = _apply_step(
        rf,
        "MAD",
        qc_params["MAD"],
        lambda stream, config: stream.MAD_select(
            threshold=config["THRESHOLD"],
            window=config["WINDOW"],
        ),
    )
    rf = _apply_step(
        rf,
        "F-test",
        qc_params["F_TEST"],
        lambda stream, config: stream.f_test_select(
            threshold=config["THRESHOLD"],
            window=config["WINDOW"],
        ),
    )
    rf = _apply_step(
        rf,
        "cross-correlation",
        qc_params["CC"],
        lambda stream, config: stream.CC_select(
            threshold=config["THRESHOLD"],
            window=config["WINDOW"],
        ),
    )

    for file in rf.files:
        shutil.copy2(file, clean_rf_dir)
    logging.info("Cleaned RF traces saved to %s", clean_rf_dir)

    if plot and len(rf) > 0:
        rf.plot(save=True, save_path=os.path.join(figdir, f"{net}.{sta}_RF_QC_{type}.png"))
        logging.info("RF QC plot completed and saved")

    return rf


def run_rf_qc_config(config_file: str, plot: bool = False) -> None:
    """Run high- and low-frequency RF QC from a YAML config file."""
    with open(config_file, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    io_params = params["IO"]
    rootdir = io_params["ROOT"]
    logdir = os.path.join(rootdir, io_params["LOG"])
    os.makedirs(logdir, exist_ok=True)

    sta = params["META"]["station"]
    net = params["META"]["network"]
    logging.basicConfig(
        filename=os.path.join(logdir, f"{net}.{sta}_RF_QC.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
        force=True,
    )

    print(f"Running RF QC for {net}.{sta}...")
    logging.info("%sHigh-Frequency RF QC%s", "=" * 20, "=" * 20)
    rf_QC(params, type="HighFreq", plot=plot)
    logging.info("%sLow-Frequency RF QC%s", "=" * 20, "=" * 20)
    rf_QC(params, type="LowFreq", plot=plot)


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
    return parser.parse_args()


def main():
    args = load_parse_args()
    run_rf_qc_config(args.config_file, plot=args.plot)


__all__ = [
    "RFstream",
    "RFtrace",
    "load_parse_args",
    "main",
    "rf_QC",
    "run_rf_qc_config",
]
