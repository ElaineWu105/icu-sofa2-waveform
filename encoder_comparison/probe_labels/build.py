"""Pick hourly windows on each train or validation patient's first qualifying ICU day and attach labels.

The test split is never read. Windows hold a charted vital sign when the hour has one,
matching arm B's training windows, and never depend on a label value. Writes patient-level
files to the private folder and aggregate counts next to this file.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from common import (ALL_THREE, ANCHORS, CHART, CUT, DAY_US, EHR, ICD, PLAN, PRIVATE, READY, ROW_BYTES, SLOT_US,
                    SLOTS_PER_HOUR)
from resp_labels import map_value

HERE = Path(__file__).resolve().parent
FS = 250
NEAR_US = 1_800_000_000
RHYTHM = {"SR (Sinus Rhythm)": "SR", "ST (Sinus Tachycardia)": "ST", "SB (Sinus Bradycardia)": "SB",
          "AF (Atrial Fibrillation)": "AF", "V Paced": "Paced", "A Paced": "Paced", "AV Paced": "Paced",
          "1st AV (First degree AV Block)": "AVB1", "RBBB (Right Bundle Branch Block)": "RBBB",
          "LBBB (Left Bundle Branch Block)": "LBBB"}
RHYTHM_CLASSES = ("SR", "ST", "SB", "AF", "Paced")
ECTOPY = {"None": "None", "PVC's": "PVC", "Vent. Bigeminy": "PVC", "Vent. Trigeminy": "PVC", "V Quadrigenimy": "PVC",
          "PAC's": "PAC", "Atrial Bigeminy": "PAC", "Atrial Trigeminy": "PAC", "A Quadrigeminy": "PAC"}
BREATHING = {"Regular": 0.0, "Shallow": 1.0, "Irregular": 1.0, "Accessory muscle use/retractions": 1.0,
             "Discoordinate": 1.0, "Prolonged exhalation": 1.0, "Gasping efforts": 1.0, "Cheyne-Stokes": 1.0}
SUPPORT = {"InvasiveVent": 1.0, "None": 0.0, "SupplementalOxygen": 0.0, "HFNC": 0.0, "NonInvasiveVent": 0.0}
NUMERICS = {"hr": (("HR [bpm]",), 20, 250), "rr_monitor": (("RR [rpm]",), 3, 60), "spo2": (("SpO2 [%]",), 50, 100),
            "sbp": (("ABPs [mmHg]", "ARTs [mmHg]"), 40, 260), "dbp": (("ABPd [mmHg]", "ARTd [mmHg]"), 20, 180),
            "qtc": (("QTc [msec]",), 250, 700)}
COLUMNS = {name for columns, _, _ in NUMERICS.values() for name in columns} | {"time"}
VITALS = ("hr", "rr", "spo2")
SUPPORT_ITEMS = ("o2_device", "vent_mode", "vent_mode_h")


def clean_chart(chart):
    """Strip charted text, which often ends in a space, and fail if a label value no longer exists."""
    chart = chart.assign(value=chart.value.str.strip())
    for item, mapping in (("rhythm", RHYTHM), ("ectopy", ECTOPY), ("breath_pattern", BREATHING)):
        missing = set(mapping) - set(chart.value[chart.item == item])
        if missing:
            raise RuntimeError(f"Label values absent from {item}: {sorted(missing)}")
    return chart


def choose_slot(ready, charted):
    """A slot in one hour: the first holding a charted vital, otherwise the middle ready slot."""
    if not len(ready):
        return None, False
    hits = np.intersect1d(ready, charted)
    if len(hits):
        return int(hits[0]), True
    return int(ready[len(ready) // 2]), False


def consensus(values, mapping):
    """The single mapped value, or None when any value is unmapped or two values disagree."""
    mapped = {mapping.get(value) for value in values}
    return next(iter(mapped)) if len(mapped) == 1 else None


def support_label(events):
    """Invasive support from (itemid, value) pairs, or None when unknown, conflicting or tracheostomy."""
    classes = {map_value(int(item), value) for item, value in events}
    return consensus(classes, SUPPORT)


def spectral_rate(values, low=.08, high=1.):
    """Dominant frequency in breaths per minute, the classical baseline for 10-second Resp."""
    values = values - values.mean()
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)), n=2 ** 15)) ** 2
    freq = np.fft.rfftfreq(2 ** 15, 1 / FS)
    keep = (freq >= low) & (freq <= high)
    return float(freq[keep][np.argmax(power[keep])] * 60)


def numeric_label(frame, columns, low, high, left, right):
    """Median of finite in-range values in a window, from the first column holding any."""
    if frame is None:
        return np.nan
    times = frame["time"].to_numpy()
    first, last = np.searchsorted(times, [left, right])
    for column in columns:
        if column in frame:
            values = frame[column].to_numpy()[first:last]
            values = values[np.isfinite(values) & (values >= low) & (values <= high)]
            if len(values):
                return float(np.median(values))
    return np.nan


def load_numerics(record):
    path = Path(record) / "numerics.csv.gz"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path, usecols=lambda column: column in COLUMNS)
    except pd.errors.EmptyDataError:
        # Some records ship an empty numerics file, which is the same as having none.
        return None
    return frame.sort_values("time").reset_index(drop=True)


def read_row(path, row):
    with path.open("rb") as stream:
        np.lib.format.read_magic(stream)
        np.lib.format.read_array_header_1_0(stream)
        stream.seek(stream.tell() + int(row) * ROW_BYTES)
        return np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").reshape(3, 2500)


def chart_labels(events, window_start):
    """Labels from charted items around a window, measured in microseconds from the ICU day start."""
    near = events[np.abs(events.offset.to_numpy() - window_start) <= NEAR_US]
    result = {name: np.nan for name in ("rhythm", "avb1", "rbbb", "lbbb", "pvc", "pac", "support", "breathing", "rr_charted")}
    rhythm = consensus(near.value[near.item == "rhythm"], RHYTHM)
    if rhythm in RHYTHM_CLASSES:
        result["rhythm"] = float(RHYTHM_CLASSES.index(rhythm))
    for name, value in (("avb1", "AVB1"), ("rbbb", "RBBB"), ("lbbb", "LBBB")):
        if rhythm in ("SR", value):
            result[name] = float(rhythm == value)
    ectopy = consensus(near.value[near.item == "ectopy"], ECTOPY)
    for name, value in (("pvc", "PVC"), ("pac", "PAC")):
        if ectopy in ("None", value):
            result[name] = float(ectopy == value)
    support = near[near.item.isin(SUPPORT_ITEMS)]
    if len(support):
        label = support_label(zip(support.itemid, support.value))
        result["support"] = np.nan if label is None else label
    breathing = consensus(near.value[near.item == "breath_pattern"], BREATHING)
    if breathing is not None:
        result["breathing"] = breathing
    inside = events[(events.item == "rr") & (events.offset >= window_start) & (events.offset < window_start + SLOT_US)]
    values = inside.valuenum[inside.valuenum.between(1, 120)].unique()
    if len(values) == 1:
        result["rr_charted"] = float(values[0])
    return result


def patient_day(job):
    """Windows and labels for one ICU day."""
    stay, day, subject, split, start_us, segments, events = job
    folder = CUT / str(stay)
    with np.load(folder / f"day_{day:03d}_meta.npz", allow_pickle=False) as meta:
        status, bits, rows, leads, positions = meta["status"], meta["bits"], meta["row"], meta["lead"], meta["segment_index"]
    usable = np.flatnonzero((status == READY) & (bits == ALL_THREE))
    charted = np.unique(events.offset[events.item.isin(VITALS)].to_numpy() // SLOT_US)
    frames, output = {}, []
    for hour in range(24):
        ready = usable[(usable >= hour * SLOTS_PER_HOUR) & (usable < (hour + 1) * SLOTS_PER_HOUR)]
        slot, has_chart = choose_slot(ready, charted)
        if slot is None:
            continue
        segment = segments[int(positions[slot])]
        record = os.path.dirname(segment["path"])
        if record not in frames:
            frames[record] = load_numerics(record)
        record_start = int(segment["origin_us"]) - int(os.path.basename(segment["path"])) * 1000
        left = (start_us + slot * SLOT_US - record_start) // 1000
        row = {"subject_id": subject, "stay_id": stay, "day": day, "split": split, "hour": hour, "slot": slot,
               "row": int(rows[slot]), "lead": max(int(leads[slot]), 0), "charted_slot": has_chart}
        row.update({name: numeric_label(frames[record], columns, low, high, left, left + SLOT_US // 1000)
                    for name, (columns, low, high) in NUMERICS.items()})
        sbp, dbp = row["sbp"], row["dbp"]
        if np.isfinite(sbp) and np.isfinite(dbp) and dbp >= sbp:
            row["sbp"], row["dbp"] = np.nan, np.nan
        row.update(chart_labels(events, slot * SLOT_US))
        row["rr_spectral"] = spectral_rate(read_row(folder / f"day_{day:03d}_signal.npy", rows[slot])[2])
        output.append(row)
    return output


def stable_order(subject):
    return hashlib.sha256(str(subject).encode()).hexdigest()


def first_days(max_patients):
    """Each patient's earliest qualifying ICU day, on train and validation patients only."""
    anchors = pd.read_pickle(ANCHORS)[["stay_id", "day", "split", "subject_id", "hadm_id", "fate"]]
    anchors = anchors[anchors.split.isin(["train", "val"])]
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    starts = pd.read_sql("SELECT stay_id, day, start_us FROM days WHERE split IN ('train','val')", db)
    db.close()
    days = anchors.merge(starts, on=["stay_id", "day"], validate="one_to_one")
    days = days.sort_values("start_us").drop_duplicates("subject_id").copy()
    if max_patients:
        days["order"] = days.subject_id.map(stable_order)
        days = pd.concat(group.sort_values("order").head(max_patients) for _, group in days.groupby("split"))
    return days.sort_values(["stay_id", "day"]).reset_index(drop=True)


def segment_paths(positions):
    wanted = sorted(positions)
    found = {}
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    for left in range(0, len(wanted), 900):
        part = wanted[left:left + 900]
        marks = ",".join("?" * len(part))
        for index, text in db.execute(f"SELECT segment_index, segment_json FROM segments WHERE segment_index IN ({marks})", part):
            value = json.loads(text)
            found[index] = {"path": value["path"], "origin_us": value["origin_us"]}
    db.close()
    return found


def day_jobs(days):
    """Chart rows and waveform segments per day, measured from each ICU day start."""
    chart = clean_chart(pd.read_pickle(CHART))
    chart = chart[chart.stay_id.isin(days.stay_id)].merge(days[["stay_id", "start_us"]], on="stay_id")
    stamp = (pd.to_datetime(chart.charttime) - pd.Timestamp(0)) // pd.Timedelta(microseconds=1)
    chart["offset"] = stamp - chart.start_us
    chart = chart[(chart.offset >= -NEAR_US) & (chart.offset < DAY_US + NEAR_US)]
    groups = dict(tuple(chart[["stay_id", "offset", "item", "itemid", "value", "valuenum"]].groupby("stay_id")))
    positions, per_day = set(), {}
    for row in days.itertuples():
        with np.load(CUT / str(row.stay_id) / f"day_{row.day:03d}_meta.npz", allow_pickle=False) as meta:
            used = [int(value) for value in np.unique(meta["segment_index"]) if value >= 0]
        per_day[row.stay_id, row.day] = used
        positions.update(used)
    paths = segment_paths(positions)
    empty = chart.iloc[:0][["stay_id", "offset", "item", "itemid", "value", "valuenum"]]
    return [(row.stay_id, row.day, row.subject_id, row.split, row.start_us,
             {position: paths[position] for position in per_day[row.stay_id, row.day]}, groups.get(row.stay_id, empty))
            for row in days.itertuples()]


def patient_labels(days):
    """Demographics, diagnoses and outcomes, one row per patient."""
    patients = pd.read_csv(EHR + "hosp/patients.csv.gz")
    admissions = pd.read_csv(EHR + "hosp/admissions.csv.gz", usecols=["hadm_id", "admittime", "hospital_expire_flag"],
                             parse_dates=["admittime"])
    table = days.merge(patients, on="subject_id", how="left").merge(admissions, on="hadm_id", how="left")
    table["age"] = (table.anchor_age + table.admittime.dt.year - table.anchor_year).where(table.anchor_age < 89)
    table["female"] = (table.gender == "F").astype(float)

    omr = pd.read_csv(EHR + "hosp/omr.csv.gz", parse_dates=["chartdate"])
    omr = omr[omr.result_name.str.startswith("BMI") & omr.subject_id.isin(table.subject_id)].copy()
    omr["bmi"] = pd.to_numeric(omr.result_value, errors="coerce")
    near = omr.merge(table[["subject_id", "admittime"]], on="subject_id")
    near["gap"] = (near.chartdate - near.admittime).abs()
    near = near[(near.gap <= pd.Timedelta(days=365)) & near.bmi.between(10, 80)]
    nearest = near.sort_values(["subject_id", "gap"]).drop_duplicates("subject_id").set_index("subject_id").bmi
    table["bmi"] = table.subject_id.map(nearest)

    icd = pd.read_csv(EHR + "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str})
    icd = icd[icd.hadm_id.isin(table.hadm_id)]
    for name, (ten, nine) in ICD.items():
        chosen = icd[((icd.icd_version == 10) & icd.icd_code.str.startswith(ten)) |
                     ((icd.icd_version == 9) & icd.icd_code.str.startswith(nine))]
        table[f"icd_{name}"] = table.hadm_id.isin(chosen.hadm_id).astype(float)

    day_end = pd.to_datetime(table.start_us + DAY_US, unit="us")
    remaining = pd.to_datetime(table.dod) - day_end
    table["death_hospital"] = (table.hospital_expire_flag == 1).astype(float)
    table["death_24h"] = (table.fate == "died").astype(float)
    table["death_30d"] = (remaining <= pd.Timedelta(days=30)).astype(float)
    table["death_1y"] = (remaining <= pd.Timedelta(days=365)).astype(float)
    keep = ["subject_id", "stay_id", "day", "split", "age", "female", "bmi", "death_hospital", "death_24h", "death_30d", "death_1y"]
    return table[keep + [f"icd_{name}" for name in ICD]]


def counts(frame, skip):
    """Label counts per split; binary labels also report positives."""
    report = {}
    for column in frame.columns.difference(skip):
        known = frame[frame[column].notna()]
        values = known[column]
        entry = {"rows": known.groupby("split").size().to_dict(), "patients": known.groupby("split").subject_id.nunique().to_dict()}
        if set(values.unique()) <= {0.0, 1.0}:
            entry["positive_rows"] = known[values == 1].groupby("split").size().to_dict()
            entry["positive_patients"] = known[values == 1].groupby("split").subject_id.nunique().to_dict()
        report[column] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-patients", type=int, default=0, help="Per split, for a small check run only")
    args = parser.parse_args()
    PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    days = first_days(args.max_patients)
    print("patient days", days.split.value_counts().to_dict(), flush=True)
    jobs = day_jobs(days)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for count, part in enumerate(pool.map(patient_day, jobs, chunksize=4), 1):
            rows.extend(part)
            if count % 200 == 0:
                print("days", count, "of", len(jobs), flush=True)
    windows = pd.DataFrame(rows)
    patients = patient_labels(days)
    if set(windows.split) | set(patients.split) != {"train", "val"}:
        raise RuntimeError("Only train and validation patients may enter this comparison.")
    windows.to_pickle(PRIVATE / "windows.pkl")
    patients.to_pickle(PRIVATE / "patients.pkl")
    skip = ["subject_id", "stay_id", "day", "split", "hour", "slot", "row", "lead", "charted_slot"]
    summary = {"windows": windows.groupby("split").size().to_dict(), "patients": patients.groupby("split").size().to_dict(),
               "windows_on_charted_slot": round(float(windows.charted_slot.mean()), 4),
               "window_labels": counts(windows, skip), "patient_labels": counts(patients, skip)}
    name = "labels_summary.json" if not args.max_patients else "labels_summary_check.json"
    (HERE / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("windows", "patients", "windows_on_charted_slot")}), flush=True)


if __name__ == "__main__":
    main()
