#!/usr/bin/env bash
# One possible route: CCFs -> AFTAN -> Eikonal maps -> station curve -> inversion.
# glue.py (this example's own script, not a SeisForge command) links the steps.
# Usage: bash run.sh   (outputs go to output/; delete it before running again)
set -euo pipefail
cd "$(dirname "$0")"
SEISFORGE="${SEISFORGE:-seisforge}"   # e.g. SEISFORGE="python -m seisforge.cli.main"
PYTHON="${PYTHON:-python}"
STATION=XX.A22                        # the station whose Vs(z) is inverted

if [ -e output ]; then echo "output/ exists; delete it first" >&2; exit 1; fi
# 1. group and phase velocity of every CCF
$SEISFORGE ant aftan -c aftan.yml
# 2. glue.py: one directed path per AFTAN branch file, phase velocity at the
#    integer periods 4-8 s (shorter periods are not resolved by this 40 km
#    station spacing, see README)
$PYTHON glue.py measurements
# 3. phase-velocity maps
$SEISFORGE ant eikonal -c eikonal.yml
# 4. glue.py: the curve under one station; sigma at least 0.05 km/s (assumed:
#    the map error of this sparse array, see README and glue.py)
$PYTHON glue.py station-curve "$STATION"
# 5. Vs(z) from that curve and the H/V curve
$SEISFORGE inv disp-hv --inv inv.yaml --obs obs.yaml -o output/inversion
$PYTHON plot_result.py --station "$STATION" -o result.png
