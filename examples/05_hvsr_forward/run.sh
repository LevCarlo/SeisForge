#!/usr/bin/env bash
# H/V forward example: compute the H/V curve of data/model.txt, then the figure. Usage: bash run.sh
# Outputs go to output/; delete that folder before running again.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

$PYTHON hvsr_forward.py config.yml
$PYTHON plot_result.py -o result.png
