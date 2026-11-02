"""Count candidate downstream labels on train and validation patients. Writes aggregates only."""
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ICD  # noqa: E402

ROOT = HERE.parents[1]
PRIVATE = HERE / "private"
ANCHORS = ROOT / "labels/private/anchors.pkl"
PLAN = ROOT / "waveform/icu_cut/output/plan/plan.sqlite"
CUT = ROOT / "waveform/icu_cut/output/cut/days"
E = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/"
DAY_US, SLOT_US = 86_400_000_000, 10_000_000
READY, ALL_THREE = 1, 7

def share(mask):
    return {"positive": int(mask.sum()), "patients": int(len(mask)), "pct": round(float(mask.mean()) * 100, 1)}


def main():
    anchors = pd.read_pickle(ANCHORS)
    anchors = anchors[anchors.split.isin(["train", "val"])]
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    intime = pd.read_sql("SELECT stay_id, intime_us, outtime_us FROM stays", db)
    db.close()
    first = anchors.sort_values(["subject_id", "stay_id", "day"]).drop_duplicates("subject_id")[["subject_id", "stay_id", "hadm_id", "split"]]
    first = first.merge(intime, on="stay_id")
    report = {"patients": first.split.value_counts().to_dict(), "anchors": anchors.split.value_counts().to_dict(),
              "unit": "first qualifying ICU stay per train or validation patient, unless a row says otherwise"}

    patients = pd.read_csv(E + "hosp/patients.csv.gz")
    admissions = pd.read_csv(E + "hosp/admissions.csv.gz", usecols=["hadm_id", "admittime", "hospital_expire_flag"], parse_dates=["admittime"])
    table = first.merge(patients, on="subject_id", how="left").merge(admissions, on="hadm_id", how="left")
    table["age"] = table.anchor_age + table.admittime.dt.year - table.anchor_year
    exact = table.anchor_age < 89
    report["age"] = {"exact_age_patients": int(exact.sum()), "topcoded_excluded": int((~exact).sum()),
                     "median": float(table.age[exact].median()), "iqr": table.age[exact].quantile([.25, .75]).tolist()}
    report["sex"] = {"female": share(table.gender == "F")}

    omr = pd.read_csv(E + "hosp/omr.csv.gz", parse_dates=["chartdate"])
    omr = omr[omr.result_name.str.startswith("BMI") & omr.subject_id.isin(table.subject_id)].copy()
    omr["bmi"] = pd.to_numeric(omr.result_value, errors="coerce")
    near = omr.merge(table[["subject_id", "admittime"]], on="subject_id")
    near = near[((near.chartdate - near.admittime).abs() <= pd.Timedelta(days=365)) & near.bmi.between(10, 80)]
    report["bmi_outpatient_within_1y"] = {"patients": int(near.subject_id.nunique()), "of": len(table)}

    icd = pd.read_csv(E + "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
    icd = icd[icd.hadm_id.isin(table.hadm_id)]
    groups = {}
    for name, (ten, nine) in ICD.items():
        chosen = icd[((icd.icd_version == 10) & icd.icd_code.str.startswith(ten)) | ((icd.icd_version == 9) & icd.icd_code.str.startswith(nine))]
        groups[name] = share(table.hadm_id.isin(chosen.hadm_id))
    report["icd_this_admission"] = groups
    report["patients_with_any_icd"] = int(table.hadm_id.isin(icd.hadm_id).sum())

    dod = pd.to_datetime(table.dod)
    intime_at = pd.to_datetime(table.intime_us, unit="us")
    report["outcomes"] = {"hospital_death": share(table.hospital_expire_flag == 1),
                          "death_30d_from_icu_admission": share((dod - intime_at) <= pd.Timedelta(days=30)),
                          "death_1y_from_icu_admission": share((dod - intime_at) <= pd.Timedelta(days=365)),
                          "next_24h_fate_anchors": anchors.fate.value_counts().to_dict()}

    chart = pd.read_pickle(PRIVATE / "chart.pkl")
    chart = chart[chart.stay_id.isin(anchors.stay_id)].merge(intime[["stay_id", "intime_us"]], on="stay_id")
    chart["charttime"] = pd.to_datetime(chart.charttime)
    offset = (chart.charttime - pd.Timestamp(0)).dt.total_seconds().mul(1e6).astype("int64") - chart.intime_us
    chart = chart[offset >= 0]
    offset = offset[offset >= 0]
    chart["day"], chart["slot"] = offset // DAY_US, (offset % DAY_US) // SLOT_US
    keep = set(map(tuple, anchors[["stay_id", "day"]].to_numpy().tolist()))
    chart = chart[[pair in keep for pair in zip(chart.stay_id, chart.day)]]
    ready = {}
    for stay, day in keep:
        with np.load(CUT / str(stay) / f"day_{day:03d}_meta.npz", allow_pickle=False) as meta:
            ready[(stay, day)] = (meta["status"] == READY) & (meta["bits"] == ALL_THREE)
    chart["ready"] = [ready[(stay, day)][slot] for stay, day, slot in zip(chart.stay_id, chart.day, chart.slot)]
    chart = chart.merge(anchors[["stay_id", "subject_id"]].drop_duplicates(), on="stay_id")
    counts = {}
    for item, frame in chart.groupby("item"):
        on = frame[frame.ready]
        counts[item] = {"events_in_anchor_days": len(frame), "events_on_ready_slot": len(on), "patients_on_ready_slot": int(on.subject_id.nunique())}
    report["chart_items"] = counts
    for item in ("rhythm", "ectopy", "o2_device", "vent_mode", "breath_pattern"):
        frame = chart[(chart.item == item) & chart.ready]
        values = Counter(frame.value.fillna("").str.strip())
        patients = frame.groupby(frame.value.fillna("").str.strip()).subject_id.nunique()
        report[f"values_{item}"] = [{"value": value, "events": count, "patients": int(patients[value])} for value, count in values.most_common(22)]
    rr = chart[(chart.item == "rr") & chart.ready].valuenum
    report["charted_rr_on_ready"] = {"median": float(rr.median()), "iqr": rr.quantile([.25, .75]).tolist(), "pct_outside_8_30": round(float((~rr.between(8, 30)).mean()) * 100, 1)}
    text = json.dumps(report, indent=2)
    (HERE / "ehr_audit.json").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
