#!/usr/bin/env bash
# Inversion example: prior-only run, posterior run, figure. Usage: bash run.sh
# Outputs go to output/; delete that folder before running again.
set -euo pipefail
cd "$(dirname "$0")"
SEISFORGE="${SEISFORGE:-seisforge}"   # e.g. SEISFORGE="python -m seisforge.cli.main"
PYTHON="${PYTHON:-python}"

$SEISFORGE inv disp-hv --inv inv.yaml -o output/prior --prior-only
$SEISFORGE inv disp-hv --inv inv.yaml --obs obs.yaml -o output/posterior
$PYTHON plot_result.py --prior output/prior --posterior output/posterior -o result.png
