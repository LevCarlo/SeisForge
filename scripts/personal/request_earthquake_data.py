import os
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from obspy import UTCDateTime
from time import sleep
import logging
import calendar
import pandas as pd
import numpy as np

# Parameters
# ==============================================================================
domain = RectangularDomain(
    minlongitude=4.5,
    maxlongitude=19.0,
    minlatitude=36.4,
    maxlatitude=48.5
)

max_retries = 5

window_lenth = 2 * 3600 # 2 hours in seconds

# IO
# ==============================================================================
workdir = os.path.dirname(os.path.abspath(__file__))
datadir = "/mnt/data/Italy.Data"
ESW_datadir = os.path.join(datadir, "ESW")
outdir = os.path.join(ESW_datadir, "RAW_DATA")

catalog_file = os.path.join(workdir, "2016_M5.8_catalog.csv")


logdir = os.path.join(workdir, "LOG")
os.makedirs(logdir, exist_ok=True)
log_file = os.path.join(logdir, os.path.basename(catalog_file).replace(".csv", ".log"))

logging.getLogger('obspy.clients.fdsn.mass_downloader').setLevel(logging.CRITICAL)
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode="w"
                    )

# ===============================================================================
cat = pd.read_csv(catalog_file, index_col=0)
cat.sort_values("Magnitude", ascending=False, inplace=True)
cat["Time"] = cat["Time"].apply(lambda x: UTCDateTime(x))

for i, event in enumerate(cat.index):
    t0 = cat.loc[event, "Time"]
    t1 = t0 + window_lenth

    mag = cat.loc[event, "Magnitude"]
    logging.info(f"{i + 1}/{len(cat)}: {event} (M{mag}): Start downloading...")
    restrictions = Restrictions(
        starttime=t0,
        endtime=t1,
        reject_channels_with_gaps=False,
        minimum_length=0.90,
        minimum_interstation_distance_in_m=1E3,
        channel_priorities=["HH[ZNE12]", "BH[ZNE12]"]
    )

    # savedir = os.path.join(outdir, f"M{mag:.1f}_{t0.strftime('%Y%m%d%H%M%S')}")
    savedir = os.path.join(outdir, t0.strftime('%Y'), f"{t0.strftime('%Y%m%d%H%M%S')}_M{mag:.1f}")
    os.makedirs(savedir, exist_ok=True)

    mdl = MassDownloader(configure_logging=False)

    retrieval_flag = False
    for attempt in range(1, max_retries + 1):
        try:
            mdl.download(domain, 
                         restrictions,
                         threads_per_client=8,
                         mseed_storage=os.path.join(savedir, "waveforms"),
                         stationxml_storage=os.path.join(savedir, "stations"),
                        )
            logging.info(f"{i + 1}/{len(cat)}: {event} (M{mag}): Successfully retrieved data.")
            retrieval_flag = True
            break
        except Exception as e:
            if attempt == max_retries:
                logging.error(f"{i + 1}/{len(cat)}: {event} (M{mag}): Failed to retrieve data after {max_retries} attempts.")
                logging.error(e)
        
        wait_time = 2 * (2 ** (attempt - 1))
        sleep(wait_time)
    if not retrieval_flag:
        continue

    station_dir = os.path.join(savedir, "stations")
    if os.path.exists(station_dir):
        inv_files = os.listdir(station_dir)
        logging.info(f"{i + 1}/{len(cat)}: {event} (M{mag}): Found {len(inv_files)} station files.")
    else:
        logging.warning(f"{i + 1}/{len(cat)}: {event} (M{mag}): No station files found.")
    
    sleep(np.random.randint(1, 10))  # Random sleep to avoid server overload