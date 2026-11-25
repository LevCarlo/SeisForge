"""Generate the synthetic 3-station, 2-day 1 Hz SDS dataset for examples/07_ccf."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from seisforge.ant.ccf import load_station_table, sds_file_path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "07_ccf"
SDS_ROOT = EXAMPLE_DIR / "data" / "SDS"
STATIONS_CSV = EXAMPLE_DIR / "stations.csv"

FS_HZ = 1.0
DATES = ("2026-01-01", "2026-01-02")
# Known propagation delays relative to STA01 (s): STA01->STA02 = +8 s, STA01->STA03 = +14 s.
DELAYS_S = {"STA01": 0, "STA02": 8, "STA03": 14}


def main() -> None:
    stations = load_station_table(STATIONS_CSV, expected_response_state="velocity")
    rng = np.random.default_rng(20260101)
    n_day = int(round(86400.0 * FS_HZ))
    # Continuous 2-day source wavefield so cross-midnight windows stay coherent.
    source_wave = rng.standard_normal(len(DATES) * n_day)

    for day_idx, utc_date in enumerate(DATES):
        d = date.fromisoformat(utc_date)
        day_start = UTCDateTime(d.year, d.month, d.day, 0, 0, 0)
        sl = slice(day_idx * n_day, (day_idx + 1) * n_day)

        for sta in stations:
            delay = DELAYS_S[sta.station]
            shifted = np.roll(source_wave, delay)[sl]
            data = (shifted + 0.3 * rng.standard_normal(n_day)).astype(np.float32)

            tr = Trace(
                data=data,
                header={
                    "network": sta.network,
                    "station": sta.station,
                    "location": sta.location,
                    "channel": sta.channel,
                    "sampling_rate": FS_HZ,
                    "starttime": day_start,
                },
            )
            out_path = sds_file_path(SDS_ROOT, sta, utc_date)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Stream([tr]).write(str(out_path), format="MSEED", encoding="FLOAT32")
            print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
