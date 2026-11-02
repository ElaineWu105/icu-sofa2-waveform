"""Plan ICU-grid ten-second windows from WFDB headers, without reading samples.

Each ICU day d of a stay is [intime+24d, intime+24(d+1)). A day is planned when
it can serve as the 24h input of an anchor t=day end, t<outtime, and its headers
could still reach 384 of 480 valid three-minute clips. SOFA availability is not
used here: labels are joined later through (stay_id, day).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import multiprocessing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from common import (CLIPS_PER_DAY, DAY_US, EHR_ROOT, HERE, MAX_BAD_SLOTS, MIN_VALID_CLIPS,
                    SLOTS_PER_CLIP, SLOTS_PER_DAY, SPLIT_CSV, WAVE_ROOT, WINDOW_US,
                    assign_new_split)
from indexing import load_wave_records, timestamp_us
from native import inspect_record
from source import LEAD_ORDER, _choose_channels, _segment_right

HOUR_US = DAY_US // 24
SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS records(subject_id TEXT NOT NULL, record_id TEXT NOT NULL,
  segments_json TEXT NOT NULL, skips_json TEXT NOT NULL, PRIMARY KEY(subject_id, record_id));
CREATE TABLE IF NOT EXISTS stays(stay_id INTEGER PRIMARY KEY, subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL, split TEXT NOT NULL, intime_us INTEGER NOT NULL, outtime_us INTEGER NOT NULL,
  death_us INTEGER);
CREATE TABLE IF NOT EXISTS segments(segment_index INTEGER PRIMARY KEY, segment_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS days(stay_id INTEGER NOT NULL, day INTEGER NOT NULL, subject_id INTEGER NOT NULL,
  split TEXT NOT NULL, start_us INTEGER NOT NULL, fate TEXT NOT NULL, header_clips INTEGER NOT NULL,
  planned_windows INTEGER NOT NULL, runs_json TEXT NOT NULL, PRIMARY KEY(stay_id, day));
"""


def load_stays(ehr_root, wave_subjects):
    """Adult ICU stays of waveform patients, any stay of an admission."""
    stays = pd.read_csv(Path(ehr_root) / "icu/icustays.csv.gz",
                        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    total = len(stays)
    stays = stays[stays.subject_id.isin(wave_subjects) & stays.outtime.notna()].copy()
    patients = pd.read_csv(Path(ehr_root) / "hosp/patients.csv.gz",
                           usecols=["subject_id", "anchor_age", "anchor_year"])
    admissions = pd.read_csv(Path(ehr_root) / "hosp/admissions.csv.gz", usecols=["hadm_id", "deathtime"])
    stays = stays.merge(patients, on="subject_id").merge(admissions, on="hadm_id", how="left")
    age = stays.anchor_age + pd.to_datetime(stays.intime).dt.year - stays.anchor_year
    stays = stays[age >= 18].copy()
    stays["intime_us"] = [timestamp_us(value) for value in stays.intime]
    stays["outtime_us"] = [timestamp_us(value) for value in stays.outtime]
    stays["death_us"] = [None if pd.isna(value) else timestamp_us(value) for value in stays.deathtime]
    stays = stays[stays.outtime_us > stays.intime_us].sort_values(["subject_id", "intime_us"])
    return stays[["stay_id", "subject_id", "hadm_id", "intime_us", "outtime_us", "death_us"]], total


def load_splits(path, subjects):
    frame = pd.read_csv(path, usecols=["subject_id", "split"]).drop_duplicates()
    if frame.subject_id.duplicated().any():
        raise ValueError("A patient has more than one existing split.")
    existing = dict(zip(frame.subject_id, frame.split))
    return {int(subject): existing.get(int(subject), assign_new_split(subject)) for subject in subjects}, set(existing)


def open_ended(record):
    """End a record at its master header, not the index end time, which is sometimes early."""
    start, end = datetime.fromisoformat(record["starttime"]), datetime.fromisoformat(record["endtime"])
    if start.utcoffset() != end.utcoffset():
        return record
    far = start + timedelta(days=3650)
    return {**record, "endtime": far.isoformat(), "end_us": record["start_us"] + int(timedelta(days=3650).total_seconds() * 1e6)}


def inspect_job(job):
    record, wave_root = job
    segments, skips = inspect_record(record, wave_root)
    return record["subject_id"], record["record_id"], segments, dict(skips)


def inventory(db, records, wave_root, workers):
    """Resumable header inventory. Every result is committed by the parent."""
    done = {(row[0], row[1]) for row in db.execute("SELECT subject_id, record_id FROM records")}
    pending = [record for record in records if (record["subject_id"], record["record_id"]) not in done]
    print(f"[inventory] records={len(records)}; pending={len(pending)}", flush=True)
    if not pending:
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(inspect_job, (record, wave_root)) for record in pending]
        for count, future in enumerate(as_completed(futures), 1):
            subject, record_id, segments, skips = future.result()
            db.execute("INSERT INTO records VALUES(?,?,?,?)",
                       (subject, record_id, json.dumps(segments, allow_nan=False), json.dumps(skips)))
            if count % 100 == 0:
                db.commit()
                print(f"[inventory] {count}/{len(pending)}", flush=True)
    db.commit()


def slot_owners(segments, intime, outtime):
    """Segment position for each stay slot; ties follow the 10.14 priority."""
    total = int((outtime - intime) // WINDOW_US)
    owners = np.full(total, -1, dtype=np.int32)
    ranked = []
    for position, segment in enumerate(segments):
        selected = _choose_channels(segment)
        if not selected:
            continue
        ecg = selected.get("ecg")
        rank = LEAD_ORDER.index(ecg["canonical_lead_id"]) if ecg else len(LEAD_ORDER)
        ranked.append(((rank, -len(selected), str(segment["segment_id"])), position))
    for _, position in sorted(ranked):
        segment = segments[position]
        left = max(intime, int(segment["origin_us"]))
        right = min(outtime, _segment_right(segment))
        first = max(0, (left - intime + WINDOW_US - 1) // WINDOW_US)
        last = (right - intime) // WINDOW_US
        if last > first:
            view = owners[first:last]
            view[view < 0] = position
    return owners


def header_clips(owners, complete):
    """Clips that can still pass: >=17 slots owned by one three-channel segment."""
    count = 0
    for clip in range(CLIPS_PER_DAY):
        values = owners[clip * SLOTS_PER_CLIP:(clip + 1) * SLOTS_PER_CLIP]
        values = values[(values >= 0) & complete[np.maximum(values, 0)]]
        if len(values) < SLOTS_PER_CLIP - MAX_BAD_SLOTS:
            continue
        if values.min() == values.max() or np.bincount(values - values.min()).max() >= SLOTS_PER_CLIP - MAX_BAD_SLOTS:
            count += 1
    return count


def runs_of(owners):
    """[[first_slot, stop_slot, segment_position], ...] for planned slots."""
    edges = np.flatnonzero(np.diff(owners)) + 1
    runs = []
    for left, right in zip(np.r_[0, edges], np.r_[edges, len(owners)]):
        if owners[left] >= 0:
            runs.append([int(left), int(right), int(owners[left])])
    return runs


def day_fate(start_us, intime, outtime, death):
    """Fate over the 24h after this input day ends."""
    t = start_us + DAY_US
    if death is not None and death <= t + DAY_US:
        return "died"
    return "left_alive" if outtime < t + DAY_US else "stayed"


def build(db, stays, splits, records, skip_days=frozenset()):
    db.execute("DELETE FROM days")
    db.execute("DELETE FROM segments")
    db.execute("DELETE FROM stays")
    by_subject, skips = defaultdict(list), Counter()
    for subject, _, segments_json, skips_json in db.execute("SELECT * FROM records").fetchall():
        by_subject[int(subject)].extend(json.loads(segments_json))
        skips.update(json.loads(skips_json))
    segment_ids, counts = {}, Counter()
    for stay in stays.itertuples():
        counts["stays"] += 1
        death = None if stay.death_us is None or pd.isna(stay.death_us) else int(stay.death_us)
        db.execute("INSERT INTO stays VALUES(?,?,?,?,?,?,?)", (int(stay.stay_id), int(stay.subject_id),
                   int(stay.hadm_id), splits[int(stay.subject_id)], int(stay.intime_us), int(stay.outtime_us), death))
        segments = [segment for segment in by_subject.get(int(stay.subject_id), [])
                    if int(segment["origin_us"]) < stay.outtime_us and _segment_right(segment) > stay.intime_us]
        days = (stay.outtime_us - stay.intime_us - 1) // DAY_US
        if days < 1:
            counts["stays_without_anchor"] += 1
            continue
        counts["candidate_days"] += days
        if not segments:
            counts["stays_without_segment"] += 1
            continue
        owners = slot_owners(segments, stay.intime_us, stay.outtime_us)
        complete = np.array([len(_choose_channels(segment)) == 3 for segment in segments])
        for day in range(days):
            view = owners[day * SLOTS_PER_DAY:(day + 1) * SLOTS_PER_DAY]
            clips = header_clips(view, complete)
            if clips < MIN_VALID_CLIPS:
                counts["days_below_header_clips"] += 1
                continue
            if (int(stay.stay_id), day) in skip_days:
                counts["days_in_earlier_plan"] += 1
                continue
            runs = []
            for first, stop, position in runs_of(view):
                sid = segments[position]["segment_id"]
                if sid not in segment_ids:
                    segment_ids[sid] = len(segment_ids)
                    db.execute("INSERT INTO segments VALUES(?,?)",
                               (segment_ids[sid], json.dumps(segments[position], allow_nan=False)))
                runs.append([first, stop, segment_ids[sid]])
            start = stay.intime_us + day * DAY_US
            fate = day_fate(start, stay.intime_us, stay.outtime_us, death)
            db.execute("INSERT INTO days VALUES(?,?,?,?,?,?,?,?,?)", (int(stay.stay_id), day, int(stay.subject_id),
                       splits[int(stay.subject_id)], start, fate, clips, int((view >= 0).sum()), json.dumps(runs)))
            counts["planned_days"] += 1
    db.commit()
    return counts, skips


def summarize(db, counts, skips, stays_seen, known_patients):
    one = lambda sql: db.execute(sql).fetchone()[0]
    patients = {row[0] for row in db.execute("SELECT DISTINCT subject_id FROM days")}
    summary = {
        "icu_stays_in_mimic": stays_seen, "attrition": dict(counts),
        "planned_stays": one("SELECT COUNT(DISTINCT stay_id) FROM days"), "planned_patients": len(patients),
        "planned_windows": one("SELECT SUM(planned_windows) FROM days"),
        "patients_with_new_split": len(patients - known_patients),
        "days_by_split": dict(db.execute("SELECT split, COUNT(*) FROM days GROUP BY split").fetchall()),
        "patients_by_split": dict(db.execute("SELECT split, COUNT(DISTINCT subject_id) FROM days GROUP BY split").fetchall()),
        "days_by_fate": dict(db.execute("SELECT fate, COUNT(*) FROM days GROUP BY fate").fetchall()),
        "header_skips": dict(skips),
        "rule": f"header clips >= {MIN_VALID_CLIPS} of {CLIPS_PER_DAY}; at most {MAX_BAD_SLOTS} bad slot per clip",
        "not_used": "SOFA availability, CXR",
    }
    db.execute("INSERT OR REPLACE INTO metadata VALUES('summary', ?)", (json.dumps(summary),))
    db.commit()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", type=Path, default=HERE / "output/plan")
    parser.add_argument("--wave-root", type=Path, default=WAVE_ROOT)
    parser.add_argument("--ehr-root", type=Path, default=EHR_ROOT)
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--ignore-index-end", action="store_true", help="End records at the master header")
    parser.add_argument("--skip-plan", type=Path, help="Leave out days already in this plan database")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.output / "plan.sqlite")
    db.executescript(SCHEMA)
    records_by_subject, record_info = load_wave_records(args.wave_root)
    subjects = {int(subject) for subject in records_by_subject}
    stays, stays_seen = load_stays(args.ehr_root, subjects)
    print(f"[stays] adult stays of waveform patients={len(stays)}; patients={stays.subject_id.nunique()}", flush=True)
    splits, known = load_splits(args.split_csv, stays.subject_id.unique())
    if args.ignore_index_end:
        records_by_subject = {subject: [open_ended(record) for record in rows] for subject, rows in records_by_subject.items()}
    skip_days = frozenset()
    if args.skip_plan:
        earlier = sqlite3.connect(f"file:{args.skip_plan}?mode=ro", uri=True)
        skip_days = frozenset(earlier.execute("SELECT stay_id, day FROM days").fetchall())
        earlier.close()
    wanted = []
    grouped = {subject: frame for subject, frame in stays.groupby("subject_id")}
    for subject, frame in grouped.items():
        lo, hi = frame.intime_us.min(), frame.outtime_us.max()
        for record in records_by_subject[str(subject)]:
            if record["start_us"] < hi and record["end_us"] > lo:
                wanted.append(record)
    inventory(db, wanted, args.wave_root, args.workers)
    counts, skips = build(db, stays, splits, wanted, skip_days)
    summary = summarize(db, counts, skips, stays_seen, known)
    summary["waveform_index"] = record_info
    (args.output / "plan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("attrition", "planned_stays", "planned_patients", "planned_windows",
                                                     "days_by_split", "days_by_fate")}, indent=1), flush=True)
    db.close()


if __name__ == "__main__":
    main()
