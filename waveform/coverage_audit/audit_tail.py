"""What the waveform past the index end would add, and whether its clock can be trusted.

Rebuilds the header inventory with the index end removed, keeping every other rule, then repeats
the plan logic. The clock check compares monitor HR with charted HR inside the tail at several
lags, with the same records' earlier data as a control. Counts and medians only.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "waveform/icu_cut"))
sys.path.insert(0, str(HERE))
from audit import PLAN, THRESHOLDS, stay_days  # noqa: E402
from common import DAY_US, WAVE_ROOT  # noqa: E402
from indexing import load_wave_records  # noqa: E402
from native import clock_us, inspect_record  # noqa: E402

CHART = ROOT / "encoder_comparison/label_audit/private/chart.pkl"
LAGS = (-21600, -3600, -600, 0, 600, 3600, 21600)


def open_ended(record):
    record = dict(record)
    start = datetime.fromisoformat(record["starttime"])
    end = datetime.fromisoformat(record["endtime"])
    if start.utcoffset() == end.utcoffset():
        record["endtime"] = (start + timedelta(days=3650)).isoformat()
    return record


def inspect(record):
    segments, skips = inspect_record(open_ended(record), WAVE_ROOT)
    return int(record["subject_id"]), segments, dict(skips)


def hr_gaps(job):
    """Absolute monitor minus charted HR by lag, inside and before the record tail."""
    record, tail_start_ms, times_ms, charted = job
    subject = record["subject_id"]
    path = Path(WAVE_ROOT) / "waves" / f"p{subject}"[:4] / f"p{subject}" / record["record_id"] / "numerics.csv.gz"
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path, usecols=lambda column: column in ("time", "HR [bpm]")).dropna().sort_values("time")
    except (ValueError, pd.errors.EmptyDataError):
        return []
    time, value = frame["time"].to_numpy(), frame["HR [bpm]"].to_numpy()
    rows = []
    for moment, heart_rate in zip(times_ms, charted):
        for lag in LAGS:
            left, right = np.searchsorted(time, [moment + lag * 1000 - 30000, moment + lag * 1000 + 30000])
            if right > left:
                rows.append((moment >= tail_start_ms, lag, abs(float(np.median(value[left:right])) - heart_rate)))
    return rows


def main():
    wanted, _ = load_wave_records(WAVE_ROOT)
    records = [record for rows in wanted.values() for record in rows]
    by_subject, skips = defaultdict(list), {}
    with ProcessPoolExecutor(16) as pool:
        for subject, segments, skipped in pool.map(inspect, records, chunksize=20):
            by_subject[subject].extend(segments)
            for key, value in skipped.items():
                skips[key] = skips.get(key, 0) + value
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    stays = db.execute("SELECT stay_id, subject_id, intime_us, outtime_us, split FROM stays").fetchall()
    db.close()
    jobs = [(*stay, by_subject.get(stay[1], [])) for stay in stays]
    with ProcessPoolExecutor(16) as pool:
        results = list(pool.map(stay_days, jobs, chunksize=50))
    report = {"skips": skips, "stays": pd.Series([r[3] for r in results]).value_counts().to_dict()}
    days = [(subject, stay, split, row) for stay, subject, split, _, rows in results for row in rows]
    report["all_three"] = {name: {"days": sum(r[3]["all_three"] >= limit for r in days),
                                  "stays": len({r[1] for r in days if r[3]["all_three"] >= limit}),
                                  "patients": len({r[0] for r in days if r[3]["all_three"] >= limit})} for name, limit in THRESHOLDS.items()}
    report["by_split_80pct"] = {split: {"days": sum(r[3]["all_three"] >= 384 and r[2] == split for r in days),
                                        "patients": len({r[0] for r in days if r[3]["all_three"] >= 384 and r[2] == split})}
                                for split in ("train", "val", "test")}

    chart = pd.read_pickle(CHART)
    chart = chart[(chart.item == "hr") & chart.valuenum.between(20, 250)]
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    owner = dict(db.execute("SELECT stay_id, subject_id FROM stays"))
    db.close()
    chart = chart.assign(subject=chart.stay_id.map(owner), us=(pd.to_datetime(chart.charttime) - pd.Timestamp(0)) // pd.Timedelta(microseconds=1))
    groups = dict(tuple(chart.groupby("subject")[["us", "valuenum"]]))
    random.seed(42)
    random.shuffle(records)
    clock = []
    for record in records:
        origin = clock_us(datetime.fromisoformat(record["starttime"]))
        index_ms = (clock_us(datetime.fromisoformat(record["endtime"])) - origin) // 1000
        rows = groups.get(int(record["subject_id"]))
        if rows is None:
            continue
        master = Path(WAVE_ROOT) / "waves" / f"p{record['subject_id']}"[:4] / f"p{record['subject_id']}" / record["record_id"] / "waves.hea"
        head = next(line for line in master.read_text().split("\n") if line.strip() and not line.startswith("#")).split()
        master_ms = int(float(head[3]) / float(head[2].split("/")[0]) * 1000)
        if master_ms - index_ms < 3_600_000:
            continue
        offset = (rows.us.to_numpy() - origin) // 1000
        tail = np.flatnonzero((offset >= index_ms) & (offset < master_ms))
        early = np.flatnonzero((offset >= 0) & (offset < index_ms))
        if len(tail) < 5:
            continue
        chosen = np.r_[tail[:: max(1, len(tail) // 30)], early[:: max(1, len(early) // 30)]]
        clock.append((record, index_ms, offset[chosen], rows.valuenum.to_numpy()[chosen]))
        if len(clock) == 300:
            break
    with ProcessPoolExecutor(8) as pool:
        gaps = [row for part in pool.map(hr_gaps, clock, chunksize=4) for row in part]
    frame = pd.DataFrame(gaps, columns=["tail", "lag", "gap"])
    report["clock_records"] = len(clock)
    report["hr_median_abs_gap_by_lag"] = {("tail" if tail else "before_index_end"): {str(lag): {"pairs": len(part), "median": round(float(part.gap.median()), 2), "within_5bpm": round(float((part.gap <= 5).mean()), 3)}
                                                                                 for lag, part in group.groupby("lag")} for tail, group in frame.groupby("tail")}
    text = json.dumps(report, indent=2, default=str)
    (HERE / "tail_audit.json").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
