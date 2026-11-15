#!/usr/bin/env bash
# CCF rotation example: ZNE -> ZRT, then the figure. Usage: bash run.sh
# Outputs go to output/; delete that folder before running again.
set -euo pipefail
cd "$(dirname "$0")"
SEISFORGE="${SEISFORGE:-seisforge}"   # e.g. SEISFORGE="python -m seisforge.cli.main"
PYTHON="${PYTHON:-python}"

$SEISFORGE ant rotate-ccf -c config.yml
$PYTHON plot_result.py -o result.png
