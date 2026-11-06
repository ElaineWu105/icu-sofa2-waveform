"""E2 inputs of README section 7.4 for every anchor, written to private/e2.pkl.

Per anchor, with t = intime + 24k hours and the baseline day k - 1:
- previous day's SOFA-2 organ scores, from day k - 2, real scores only; missing on the first ICU day. Day
  scores use records stored by the end of their day (score.py), so these are known at t;
- ICU day k;
- hourly medians of charted heart rate, respiratory rate, SpO2 and mean arterial pressure in each hour of
  [t - 24h, t), from records charted in that hour and stored by t. Items and plausible ranges follow mimic-code
  concepts/measurement/vitalsign.sql; MAP pools invasive and non-invasive means, as that file does.

Missing values stay NaN; the level 3 head turns them into zeros with missing flags.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PRIVATE = HERE / "private"
E = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/"
ORGANS = ("brain", "resp", "cv", "liver", "kidney", "hemo")
HOUR, DAY, HOURS = pd.Timedelta(hours=1), pd.Timedelta(hours=24), 24
# name: item ids, lower bound (exclusive), upper bound, upper bound inclusive. mimic-code vitalsign.sql.
VITALS = {"heart_rate": ((220045,), 0, 300, False), "resp_rate": ((220210, 224690), 0, 70, False),
          "spo2": ((220277,), 0, 100, True), "mbp": ((220052, 220181, 225312), 0, 300, False)}


def columns():
    return [f"{name}_{hour:02d}" for name in VITALS for hour in range(HOURS)]


def previous_scores(anchors, daily):
    previous = daily[["stay_id", "day"]].copy()
    for organ in ORGANS:
        previous[f"prev_{organ}"] = daily[f"{organ}_s2"].where(daily[f"{organ}_fresh"].eq(True))
    previous["day"] += 1
    return anchors.merge(previous, on=["stay_id", "day"], how="left")


def hourly_vitals(anchors, vitals, intime):
    """Median per anchor, vital and hour of the baseline day, from records stored by the anchor time."""
    vitals = vitals[vitals.stay_id.notna()].copy()
    vitals["stay_id"] = vitals.stay_id.astype(np.int64)
    vitals = vitals[vitals.stay_id.isin(set(anchors.stay_id))].copy()
    vitals["valuenum"] = pd.to_numeric(vitals.valuenum, errors="coerce")
    vitals["charttime"] = pd.to_datetime(vitals.charttime, errors="coerce")
    vitals["storetime"] = pd.to_datetime(vitals.storetime, errors="coerce")
    vitals["name"] = None
    for name, (items, low, high, inclusive) in VITALS.items():
        upper = vitals.valuenum <= high if inclusive else vitals.valuenum < high
        vitals.loc[vitals.itemid.isin(items) & (vitals.valuenum > low) & upper, "name"] = name
    vitals = vitals[vitals.name.notna() & vitals.charttime.notna() & vitals.storetime.notna()]
    start = vitals.stay_id.map(intime)
    offset = (vitals.charttime - start) / HOUR
    vitals = vitals[offset >= 0]
    offset = offset[offset >= 0]
    vitals["day"] = (offset // HOURS).astype(int)
    vitals["hour"] = (offset % HOURS).astype(int)
    anchor_time = vitals.stay_id.map(intime) + (vitals.day + 1) * DAY
    vitals = vitals[vitals.storetime <= anchor_time]
    keys = pd.MultiIndex.from_frame(anchors[["stay_id", "day"]])
    vitals = vitals[pd.MultiIndex.from_frame(vitals[["stay_id", "day"]]).isin(keys)]
    median = vitals.groupby(["stay_id", "day", "name", "hour"]).valuenum.median().unstack(["name", "hour"])
    median.columns = [f"{name}_{hour:02d}" for name, hour in median.columns]
    return median.reindex(columns=columns()).reset_index()


def main():
    anchors = pd.read_pickle(PRIVATE / "anchors.pkl")[["stay_id", "day", "split"]]
    daily = pd.read_pickle(PRIVATE / "daily.pkl")
    icu = pd.read_csv(E + "icu/icustays.csv.gz", usecols=["stay_id", "intime"], parse_dates=["intime"])
    intime = icu.set_index("stay_id").intime
    table = previous_scores(anchors, daily)
    table["icu_day"] = (table.day + 1).astype(float)
    vitals = pd.read_pickle(PRIVATE / "vitals.pkl")
    table = table.merge(hourly_vitals(anchors, vitals, intime), on=["stay_id", "day"], how="left")
    table = table[["stay_id", "day"] + [f"prev_{organ}" for organ in ORGANS] + ["icu_day"] + columns()]
    table.to_pickle(PRIVATE / "e2.pkl")
    summary = {"anchors": len(table),
               "previous_day_missing_pct": {organ: round(float(table[f"prev_{organ}"].isna().mean()) * 100, 1) for organ in ORGANS},
               "first_icu_day_pct": round(float((table.icu_day == 1).mean()) * 100, 1),
               "hourly_missing_pct": {name: round(float(table[[f"{name}_{h:02d}" for h in range(HOURS)]].isna().to_numpy().mean()) * 100, 1)
                                      for name in VITALS}}
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
