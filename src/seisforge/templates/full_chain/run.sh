#!/usr/bin/env bash
# End-to-end example: CCFs -> AFTAN -> collect -> Eikonal -> station curve -> inversion.
# Usage: bash run.sh   (outputs go to output/; delete it before running again)
set -euo pipefail
cd "$(dirname "$0")"
SEISFORGE="${SEISFORGE:-seisforge}"   # e.g. SEISFORGE="python -m seisforge.cli.main"
PYTHON="${PYTHON:-python}"
STATION=XX.A22                        # the station whose Vs(z) is inverted

if [ -e output ]; then echo "output/ exists; delete it first" >&2; exit 1; fi
# 1. group and phase velocity of every CCF
$SEISFORGE ant aftan -c aftan.yml
# 2. gather them into one file per (path, period); integer periods 4-8 s
#    (shorter periods are not resolved by this 40 km station spacing, see README)
$SEISFORGE ant collect-dispersion 'output/aftan/*.dat' --periods 4:8:1 \
    --stations data/stations.txt -o output/measurements.nc
# 3. phase-velocity maps
$SEISFORGE ant eikonal -c eikonal.yml
# 4. the curve under one station; sigma at least 0.05 km/s (assumed: the map
#    error of this sparse array, see README; the default floor is 0.01)
$SEISFORGE ant extract-station-curve output/eikonal/eikonal_stack.nc \
    --station "$STATION" --stations data/stations.txt -o output/vph_disp.dat \
    --sigma-floor-km-s 0.05
# 5. Vs(z) from that curve and the H/V curve
$SEISFORGE inv disp-hv --inv inv.yaml --obs obs.yaml -o output/inversion
$PYTHON plot_result.py --station "$STATION" -o result.png
