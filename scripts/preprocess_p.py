import os
import glob
import logging
import pandas as pd
from obspy import read, read_inventory, UTCDateTime, Stream
from utils import QC_Streams
from obspy.io.sac import SACTrace
from multiprocessing import Pool



# Parameters
# =========================================================
pre_filt = (1/500, 1/400, 40, 50)
sampling_rate = 1 # Hz
unit = "DISP"
win_len = 7200

channel_map = {
    "Z": "BHZ",
    "N": "BHN",
    "E": "BHE",
}
orientation_map = {
    "Z": {"cmpaz": 0, "cmpinc": 0},
    "N": {"cmpaz": 0, "cmpinc": 90},
    "E": {"cmpaz": 90, "cmpinc": 90},
}
# IO
# ==========================================================
workdir = os.path.dirname(os.path.abspath(__file__))
datadir = "/mnt/data/Italy.Data/ESW"
raw_datadir = os.path.join(datadir, "RAW_DATA")
outdir = os.path.join(datadir, "DATA")
logdir = os.path.join(datadir, "LOGS")
os.makedirs(outdir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

catalog_file = os.path.join(workdir, "2014_M5.8_catalog.csv")

# ===========================================================

def setup_event_logger(logfile):
    logger = logging.getLogger(logfile)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(logfile, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def process_event(catalog_df, event_idx):
    catalog = catalog_df
    idx = event_idx
    t = catalog.loc[idx, "Time"]
    evla, evlo = catalog.loc[idx, "Latitude"], catalog.loc[idx, "Longitude"]
    evdp = catalog.loc[idx, "Depth(km)"]
    mag = catalog.loc[idx, "Magnitude"]

    logfile = os.path.join(logdir, f"{t.strftime('%Y%m%d%H%M%S')}_M{mag:.1f}.log")
    logger = setup_event_logger(logfile)
    logger.info("=" * 50 + f"{t.strftime('%Y-%m-%d %H:%M:%S')} (M{mag:.1f})" + "=" * 50)

    event_datadir = os.path.join(raw_datadir, t.strftime("%Y"), f"{t.strftime('%Y%m%d%H%M%S')}_M{mag:.1f}")
    if not os.path.exists(event_datadir):
        return
    
    station_datadir = os.path.join(event_datadir, "stations")
    waveform_datadir = os.path.join(event_datadir, "waveforms")

    filelst = os.listdir(station_datadir)
    count = 0
    for i, file in enumerate(filelst):
        inv = read_inventory(os.path.join(station_datadir, file))
        net = inv[0].code
        sta = inv[0][0].code
        stla = inv[0][0].latitude
        stlo = inv[0][0].longitude
        stel = inv[0][0].elevation

        try:
            st = read(os.path.join(waveform_datadir, f"{net}.{sta}*.mseed"))
        except:
            logger.error(f"Skipping {net}.{sta} due to missing waveform data.")
            continue
            
        try:
            st.merge(method=1, fill_value=0)
        except:
            logger.error(f"Skipping {net}.{sta} due to merge error.")
            continue

        if len(st) == 0:
            logger.error(f"Skipping {net}.{sta} due to empty stream.")
            continue

        st.detrend("demean")
        st.detrend("linear")
        if st[0].stats.sampling_rate > sampling_rate:
            st.filter("lowpass", freq=0.5 * sampling_rate, corners=2, zerophase=True)
        st.resample(sampling_rate)

        is_ok, st = QC_Streams(st, t, t + win_len)
        if not is_ok:
            logger.error(f"Skipping {net}.{sta} due to QC failure.")
            continue

        try:
            trZ = st.select(component="Z")[0]
            trN = st.select(component="N")[0]
            trE = st.select(component="E")[0]
            st = Stream(traces=[trE, trN, trZ])
        except:
            try:
                trZ = st.select(component="Z")[0]
                tr1 = st.select(component="1")[0]
                tr2 = st.select(component="2")[0]
                st = Stream(traces=[tr1, tr2, trZ])
            except:
                logger.error(f"Skipping {net}.{sta} due to missing components.")
                continue
        
        try:
            st.rotate(method="->ZNE", inventory=inv)
        except:
            logger.error(f"Skipping {net}.{sta} due to ZNE rotation error.")
            continue

        try:
            st.remove_response(
                inventory=inv, 
                pre_filt=pre_filt, 
                output=unit, 
                water_level=None
            )
        except:
            logger.error(f"Skipping {net}.{sta} due to response removal error.")
            continue

        savedir = os.path.join(outdir, f"{net}.{sta}", f"{t.strftime('%Y%m%d%H%M%S')}_M{mag:.1f}")
        os.makedirs(savedir, exist_ok=True)

        for tr in st:
            sac = SACTrace.from_obspy_trace(tr)
            component = tr.stats.component
            channel = channel_map[component]
            cmpaz = orientation_map[component]["cmpaz"]
            cmpinc = orientation_map[component]["cmpinc"]

            sac.knetwk = net
            sac.kstnm = sta
            sac.stla = stla
            sac.stlo = stlo
            sac.stel = stel
            sac.kcmpnm = channel
            sac.cmpaz = cmpaz
            sac.cmpinc = cmpinc
            
            sac.evla = evla
            sac.evlo = evlo
            sac.evdp = evdp
            sac.mag = mag

            sac.lcalda = True

            filename = f"{t.strftime('%Y%m%d%H%M%S')}_M{mag:.1f}_{net}.{sta}.{channel}.SAC"
            sac.write(os.path.join(savedir, filename))
        
        count += 1
    
    logger.info("=" * 50 + "Summary" + "=" * 50)
    logger.info(f"{len(filelst)} stations inventory files found.")
    logger.info(f"{count} stations processed successfully.")
    
    
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

        
    

if __name__ == "__main__":
    catalog = pd.read_csv(catalog_file, index_col=0)
    catalog.sort_values("Magnitude", ascending=False, inplace=True)
    catalog["Time"] = catalog["Time"].apply(lambda x: UTCDateTime(x))

    with Pool(processes=4) as pool:
        pool.starmap(process_event, [(catalog, idx) for idx in catalog.index])