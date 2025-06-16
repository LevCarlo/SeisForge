'''
Author: Mengjie Zheng
Email: mengjie.zheng@utoronto.ca;zhengmengjie18@mails.ucas.ac.cn
Date: 2025-02-13 20:31:15
LastEditTime: 2025-02-28 19:47:39
LastEditors: Mengjie Zheng
Description: 
FilePath: /Project/Antarctica.Proj/retrieve_noise_data.py
'''

import os
import sys
from obspy.clients.fdsn.mass_downloader import CircularDomain, Restrictions, MassDownloader, RectangularDomain
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from time import sleep
import logging
import calendar
import pandas as pd
from dateutil.relativedelta import relativedelta

# Parameters
# ==============================================================================
domain = RectangularDomain(
    minlongitude=4.5,
    maxlongitude=19.0,
    minlatitude=36.4,
    maxlatitude=48.5
) # # Italy
max_retries = 5

Tstart = UTCDateTime(2014, 1, 1, 0, 0, 0)
Tend = UTCDateTime(2014, 12, 31, 23, 59, 59)
# Tend = UTCDateTime(2014, 1, 1, 23, 59, 59)

# IO
# ==============================================================================
workdir = os.path.dirname(os.path.abspath(__file__))
datadir = "/mnt/data/Italy.Data"
ANCC_datadir = os.path.join(datadir, "ANCC")
outdir = os.path.join(ANCC_datadir, "RAW_DATA")

logdir = os.path.join(workdir, "LOG")
os.makedirs(logdir, exist_ok=True)
log_file = os.path.join(logdir, f"retrieve_{Tstart.strftime('%Y%m%d')}_{Tend.strftime('%Y%m%d')}.log")



logging.getLogger('obspy.clients.fdsn.mass_downloader').setLevel(logging.CRITICAL)
logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode="w"
                    )

# ==============================================================================

t0 = UTCDateTime()

tbegin = Tstart
while tbegin < Tend:
    days_in_month = calendar.monthrange(tbegin.year, tbegin.month)[1]
    tend = UTCDateTime(tbegin.year, tbegin.month, days_in_month, 23, 59, 59)
    if tend > Tend:
        tend = Tend

    logging.info(f"{tbegin} - {tend}: Start downloading...")
    restrictions = Restrictions(
        starttime=tbegin,
        endtime=tend,
        # Chunk it to have one file per day.
        chunklength_in_sec=86400,
        # The typical use case for such a data set are noise correlations where
        # gaps are dealt with at a later stage.
        reject_channels_with_gaps=False,
        # Same is true with the minimum length. All data might be useful.
        minimum_length=0.0,
        # Guard against the same station having different names.
        minimum_interstation_distance_in_m=100.0,
        channel_priorities=["BH[ZNE12]", "HH[ZNE12]"]
    )

    month_datadir = os.path.join(outdir, tbegin.strftime("%Y"), f"{tbegin.month:02d}")
    os.makedirs(month_datadir, exist_ok=True)

    mdl = MassDownloader(configure_logging=False)
    retrieval_flag = False
    for attempt in range(1, max_retries + 1):
        try:
            mdl.download(domain, 
                         restrictions,
                         threads_per_client=5,
                         mseed_storage=os.path.join(month_datadir, "waveforms"),
                         stationxml_storage=os.path.join(month_datadir, "stations"),
                        )
            logging.info(f"{tbegin} - {tend}: Downloaded completed.")
            retrieval_flag = True
            break
        except Exception as e:
            if attempt == max_retries:
                logging.error(f"{tbegin} - {tend}: Failed retrieving data after {max_retries} attempts.")
                logging.error(e)
        
        wait_time = 2 * (2 ** (attempt - 1))
        sleep(wait_time)
    
    if not retrieval_flag:
        tbegin = tend
        continue

    station_dir = os.path.join(month_datadir, "stations")
    if os.path.exists(station_dir):
        inv_files = os.listdir(station_dir)
        logging.info(f"{tbegin} - {tend}: {len(inv_files)} stationXML files downloaded.")
    else:
        logging.warning(f"{tbegin} - {tend}: Station directory does not exist.")

    tbegin = tend + 1
    sleep(10)

t1 = UTCDateTime()
# print(f"------------------------{t1}------------------------")
# print(f"Download completed. Time elapsed: {(t1 - t0) / 3600:.2f} hours.")
logging.info(f"Download completed. Time elapsed: {(t1 - t0) / 3600:.2f} hours.")