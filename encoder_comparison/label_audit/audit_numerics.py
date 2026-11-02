"""Audit monitor numerics against cut windows on train and validation patients.

Reports field coverage on ready three-channel windows, one per ICU hour, and checks
clock alignment: HR and RR measured from the signal itself should agree with numerics
best at zero lag. Writes aggregates only.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import sqlite3

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WAVEFORM = ROOT / "waveform/icu_cut/output"
PLAN = WAVEFORM / "plan/plan.sqlite"
CUT = WAVEFORM / "cut/days"
FS = 250
ROW_BYTES = 3 * 2500 * 4
SLOT_US, SLOTS_PER_HOUR = 10_000_000, 360
READY, ALL_THREE = 1, 7
DAYS = 800
LAGS_S = (-3600, -300, -60, -30, 0, 30, 60, 300, 3600)
FIELDS = ("HR [bpm]", "RR [rpm]", "SpO2 [%]", "Pulse (SpO2) [bpm]", "PVC [/min]", "QT [msec]", "QTc [msec]", "ST-V [mm]",
          "NBPs [mmHg]", "NBPd [mmHg]", "NBPm [mmHg]", "ABPs [mmHg]", "ABPd [mmHg]", "ABPm [mmHg]",
          "ARTs [mmHg]", "ARTd [mmHg]", "ARTm [mmHg]", "CVPm [mmHg]", "PAPm [mmHg]", "PPV [%]", "Perf [NU]",
          "etCO2 [mmHg]", "awRR [rpm]")
HOLD_US = {"NBPs [mmHg]": 3_600_000_000, "NBPd [mmHg]": 3_600_000_000, "NBPm [mmHg]": 3_600_000_000}


def spectral_rate(values, low, high):
    """Dominant frequency in beats or breaths per minute."""
    values = values - values.mean()
    power = np.abs(np.fft.rfft(values * np.hanning(len(values)), n=2 ** 15)) ** 2
    freq = np.fft.rfftfreq(2 ** 15, 1 / FS)
    keep = (freq >= low) & (freq <= high)
    return float(freq[keep][np.argmax(power[keep])] * 60)


def ecg_rate(values):
    return spectral_rate(np.abs(np.diff(values, prepend=values[0])), .67, 3.5)


def read_row(path, row):
    with path.open("rb") as stream:
        np.lib.format.read_magic(stream)
        np.lib.format.read_array_header_1_0(stream)
        stream.seek(stream.tell() + int(row) * ROW_BYTES)
        return np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").reshape(3, 2500)


def load_numerics(record):
    path = Path(record) / "numerics.csv.gz"
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    frame.columns = [column.strip('"') for column in frame.columns]
    return frame.sort_values("time").reset_index(drop=True)


def window_value(frame, column, left, right):
    if column not in frame:
        return np.nan
    times = frame["time"].to_numpy()
    lo, hi = np.searchsorted(times, [left, right])
    values = frame[column].to_numpy()[lo:hi]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def audit_day(job):
    stay, day, start_us, segments = job
    folder = CUT / str(stay)
    with np.load(folder / f"day_{day:03d}_meta.npz", allow_pickle=False) as meta:
        status, bits, row, segment_index = meta["status"], meta["bits"], meta["row"], meta["segment_index"]
    usable = np.flatnonzero((status == READY) & (bits == ALL_THREE))
    frames, rows = {}, []
    for hour in range(24):
        chosen = usable[(usable >= hour * SLOTS_PER_HOUR) & (usable < (hour + 1) * SLOTS_PER_HOUR)]
        if not len(chosen):
            continue
        slot = int(chosen[len(chosen) // 2])
        segment = segments[int(segment_index[slot])]
        record = os.path.dirname(segment["path"])
        record_start = int(segment["origin_us"]) - int(os.path.basename(segment["path"])) * 1000
        if record not in frames:
            frames[record] = load_numerics(record)
        frame = frames[record]
        result = {"has_file": frame is not None}
        if frame is not None:
            left = (start_us + slot * SLOT_US - record_start) // 1000
            right = left + SLOT_US // 1000
            for field in FIELDS:
                extra = HOLD_US.get(field, 0) // 1000
                result[field] = window_value(frame, field, left - extra, right)
            signal = read_row(folder / f"day_{day:03d}_signal.npy", row[slot])
            result["ecg_rate"], result["resp_rate"] = ecg_rate(signal[0]), spectral_rate(signal[2], .08, 1.)
            for lag in LAGS_S:
                result[f"hr_{lag}"] = window_value(frame, "HR [bpm]", left + lag * 1000, right + lag * 1000)
                result[f"rr_{lag}"] = window_value(frame, "RR [rpm]", left + lag * 1000, right + lag * 1000)
        rows.append(result)
    return rows


def main():
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    days = db.execute("SELECT stay_id, day, start_us, runs_json FROM days WHERE split IN ('train','val')").fetchall()
    random.seed(42)
    days = random.sample(days, DAYS)
    wanted = sorted({position for stay, day, _, _ in days for position in
                     np.unique(np.load(CUT / str(stay) / f"day_{day:03d}_meta.npz", allow_pickle=False)["segment_index"]).tolist()
                     if position >= 0})
    segments = {}
    for left in range(0, len(wanted), 900):
        part = wanted[left:left + 900]
        marks = ",".join("?" * len(part))
        for index, text in db.execute(f"SELECT segment_index, segment_json FROM segments WHERE segment_index IN ({marks})", part):
            segments[index] = {key: value for key, value in json.loads(text).items() if key in ("path", "origin_us")}
    db.close()
    print("segments", len(segments), flush=True)
    jobs = []
    for stay, day, start_us, _ in days:
        meta = np.load(CUT / str(stay) / f"day_{day:03d}_meta.npz", allow_pickle=False)["segment_index"]
        jobs.append((stay, day, start_us, {int(i): segments[int(i)] for i in np.unique(meta) if i >= 0}))
    rows = []
    with ProcessPoolExecutor(8) as pool:
        for count, part in enumerate(pool.map(audit_day, jobs, chunksize=4), 1):
            rows.extend(part)
            if count % 100 == 0:
                print("days", count, flush=True)
    table = pd.DataFrame(rows)
    has = table[table.has_file]
    report = {"days": DAYS, "windows": len(table), "windows_with_numerics_file": round(float(table.has_file.mean()), 4),
              "field_coverage_of_all_windows": {field: round(float(has[field].notna().sum() / len(table)), 4) for field in FIELDS}}
    alignment = {}
    for name, measured, low, high in (("hr", "ecg_rate", 30, 200), ("rr", "resp_rate", 5, 60)):
        lags = {}
        for lag in LAGS_S:
            pair = has[[measured, f"{name}_{lag}"]].dropna()
            pair = pair[pair[f"{name}_{lag}"].between(low, high)]
            lags[str(lag)] = {"n": len(pair), "pearson": round(float(pair.corr().iloc[0, 1]), 4),
                              "median_abs_error": round(float((pair[measured] - pair[f"{name}_{lag}"]).abs().median()), 2)}
        alignment[name] = lags
    report["alignment_signal_vs_numerics_by_lag_seconds"] = alignment
    text = json.dumps(report, indent=2)
    (HERE / "numerics_audit.json").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
