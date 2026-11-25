#!/usr/bin/env bash
# ZZ cross-correlation, multi-day stacking, and selective SAC export example.
# Usage: bash run.sh
# Outputs go to output/; delete output/ before running from scratch.
set -euo pipefail
cd "$(dirname "$0")"
SEISFORGE="${SEISFORGE:-seisforge}"   # e.g. SEISFORGE="python -m seisforge.cli.main"

$SEISFORGE ant ccf -c ccf.yml
$SEISFORGE ant stack-ccf -c ccf_stack.yml
$SEISFORGE ant export-sac -i output/stacked_ZZ.nc -o output/sac_ZZ --min-distance-km 1
