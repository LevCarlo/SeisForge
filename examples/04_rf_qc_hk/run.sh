#!/usr/bin/env bash
# RF example: quality control, H-k stacking of the crust, figure. Usage: bash run.sh
# Outputs go to output/; delete that folder before running again.
set -euo pipefail
cd "$(dirname "$0")"
SEISFORGE="${SEISFORGE:-seisforge}"   # e.g. SEISFORGE="python -m seisforge.cli.main"
PYTHON="${PYTHON:-python}"

if [ -e output ]; then echo "output/ exists; delete it first" >&2; exit 1; fi
$SEISFORGE rf qc -c rf_qc.yml --plot
$SEISFORGE rf hkseq -c hkseq.yml --mode low --plot
$PYTHON plot_result.py -o result.png
