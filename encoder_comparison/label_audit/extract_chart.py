"""Filter chartevents for CSFM downstream labels to planned ICU stays. Read-only input."""
import os
from pathlib import Path
import sqlite3

import pandas as pd

HERE = Path(__file__).resolve().parent
PRIVATE = HERE / "private"
PLAN = HERE.parents[1] / "waveform/icu_cut/output/plan/plan.sqlite"
CHART = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/icu/chartevents.csv.gz"
ITEMS = {220045: "hr", 220048: "rhythm", 224650: "ectopy", 220210: "rr", 224689: "rr_spont", 224690: "rr_total",
         220050: "abp_s", 220051: "abp_d", 220052: "abp_m", 225309: "art_s", 225310: "art_d", 225312: "art_m",
         220179: "nbp_s", 220180: "nbp_d", 220181: "nbp_m", 220277: "spo2", 223761: "temp_f", 223762: "temp_c",
         226730: "height_cm", 226512: "weight_kg", 224639: "weight_daily_kg",
         226732: "o2_device", 223849: "vent_mode", 229314: "vent_mode_h", 223835: "fio2", 220339: "peep",
         229322: "breath_pattern", 224685: "tidal"}


def main():
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    stays = {stay for stay, in db.execute("SELECT stay_id FROM stays")}
    db.close()
    parts, seen = [], 0
    columns = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    # Charted text such as "None" is a real value, so default NA parsing stays off.
    for chunk in pd.read_csv(CHART, usecols=columns, chunksize=4_000_000, dtype={"value": str}, low_memory=False,
                             keep_default_na=False, na_values={"valuenum": [""]}):
        seen += len(chunk)
        chunk = chunk[chunk.itemid.isin(ITEMS) & chunk.stay_id.isin(stays)]
        if len(chunk):
            parts.append(chunk)
        print("chart", seen, flush=True)
    table = pd.concat(parts, ignore_index=True)
    table["item"] = table.itemid.map(ITEMS)
    table.to_pickle(PRIVATE / "chart.pkl")
    print("chart DONE", len(table), flush=True)


if __name__ == "__main__":
    main()
