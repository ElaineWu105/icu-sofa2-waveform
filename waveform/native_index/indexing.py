"""Cohort identity checks and conservative, compact pre-cutoff waveform index.

The supplied MIMIC local timeline is kept unchanged; aware waveform timestamps
are stripped to their local wall time only for interval lookup. Native header
inspection must refuse counter/UTC-offset risks before any interval is accepted.
No waveform samples, neural features, or research training are read here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

HISTORY_US = 12 * 3600 * 1_000_000
EPOCH = datetime(1970, 1, 1)
SPLITS = {"train", "val", "test"}
PAIR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}\Z")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pairs (
  pair_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, hadm_id TEXT NOT NULL,
  split TEXT NOT NULL, t1_us INTEGER NOT NULL, t2_us INTEGER NOT NULL,
  start_us INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS segments (
  segment_id TEXT PRIMARY KEY, record_id TEXT NOT NULL, subject_id TEXT NOT NULL,
  path TEXT NOT NULL, origin_us INTEGER NOT NULL, frames INTEGER NOT NULL,
  frame_fs REAL NOT NULL, record_end_us INTEGER NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pair_segments (
  pair_id TEXT NOT NULL, segment_id TEXT NOT NULL, left_us INTEGER NOT NULL,
  right_us INTEGER NOT NULL, admission_status TEXT NOT NULL,
  UNIQUE(pair_id, segment_id, left_us, right_us));
CREATE INDEX IF NOT EXISTS pair_segments_pair ON pair_segments(pair_id);
CREATE INDEX IF NOT EXISTS pair_segments_segment ON pair_segments(segment_id);
"""


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _open_csv(path):
    path = Path(path)
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open("r", newline="")


def identifier(value, *, allow_p=False):
    value = str(value).strip()
    if allow_p and value.startswith("p"):
        value = value[1:]
    if not value.isdigit() or int(value) <= 0:
        raise ValueError("Subject/admission identifiers must be positive integers.")
    return str(int(value))


def timestamp_us(value, *, waveform_local=False):
    """Exact microseconds, without host timezone conversions or float epochs."""
    parsed = datetime.fromisoformat(str(value).strip())
    if parsed.tzinfo is not None:
        if not waveform_local:
            raise ValueError("Prepared/ADT timestamps must use the MIMIC timezone-naive timeline.")
        parsed = parsed.replace(tzinfo=None)
    difference = parsed - EPOCH
    return (difference.days * 86400 + difference.seconds) * 1_000_000 + difference.microseconds


def merge_intervals(intervals):
    merged = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((int(left), int(right)))
    return merged


def _subtract_intervals(own, excluded):
    """Subtract other admissions; half-open touching endpoints are unambiguous."""
    excluded = merge_intervals(excluded)
    answer = []
    for left, right in merge_intervals(own):
        cursor = left
        for blocked_left, blocked_right in excluded:
            if blocked_right <= cursor:
                continue
            if blocked_left >= right:
                break
            if blocked_left > cursor:
                answer.append((cursor, min(blocked_left, right)))
            cursor = max(cursor, blocked_right)
            if cursor >= right:
                break
        if cursor < right:
            answer.append((cursor, right))
    return answer


def load_pairs(prepared, pairs_path=None, pair_limit=None):
    """Validate the full CSV before selecting a diagnostic prefix.

    An explicit fixture CSV can be used without dataset.npz. For real prepared
    directories, each identity/split is checked against dataset.npz; the default
    CSV must contain precisely the full prepared cohort. Explicit CSV subsets
    are marked incomplete, and cannot introduce new pairs.
    """
    prepared = Path(prepared)
    path = Path(pairs_path) if pairs_path is not None else prepared / "pairs_index.csv"
    if pair_limit is not None and (isinstance(pair_limit, bool) or int(pair_limit) < 1):
        raise ValueError("pair_limit must be a positive diagnostic limit.")
    rows, seen, patient_splits = [], set(), {}
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        required = {"pair_id", "subject_id", "hadm_id", "split", "t1", "t2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Pair CSV needs pair_id,subject_id,hadm_id,split,t1,t2.")
        for source in reader:
            pair_id = str(source["pair_id"])
            if not PAIR_ID_RE.fullmatch(pair_id) or ".." in pair_id:
                raise ValueError("Unsafe pair_id in pair CSV.")
            if pair_id in seen:
                raise ValueError("Duplicate pair_id in pair CSV.")
            seen.add(pair_id)
            subject = identifier(source["subject_id"])
            admission = identifier(source["hadm_id"])
            split = source["split"]
            if split not in SPLITS:
                raise ValueError("Unknown patient split in pair CSV.")
            if subject in patient_splits and patient_splits[subject] != split:
                raise ValueError("A patient occurs in multiple splits.")
            patient_splits[subject] = split
            t1, t2 = timestamp_us(source["t1"]), timestamp_us(source["t2"])
            if t1 >= t2:
                raise ValueError("Each pair must have t1 < t2.")
            rows.append({"pair_id": pair_id, "subject_id": subject, "hadm_id": admission,
                         "split": split, "t1_us": t1, "t2_us": t2, "start_us": t2 - HISTORY_US})
    if not rows:
        raise ValueError("The pair cohort is empty.")
    info = {"pairs_path": str(path.resolve()), "pairs_sha256": sha256(path),
            "pairs_in_csv": len(rows), "patient_count_csv": len(patient_splits)}
    dataset_path = prepared / "dataset.npz"
    verified_full_cohort = False
    if dataset_path.exists():
        import numpy as np
        with np.load(dataset_path, allow_pickle=False) as dataset:
            required = {"pair_ids", "subject_ids", "hadm_ids", "split"}
            if not required.issubset(dataset.files):
                raise ValueError("Prepared dataset lacks cohort identity arrays.")
            columns = {name: dataset[name] for name in required}
            lengths = {len(column) for column in columns.values()}
            if len(lengths) != 1 or any(column.ndim != 1 for column in columns.values()):
                raise ValueError("Prepared cohort identity arrays have inconsistent shapes.")
            identities = {}
            prepared_splits = {}
            for pair, subject, admission, split in zip(columns["pair_ids"], columns["subject_ids"],
                                                      columns["hadm_ids"], columns["split"]):
                pair, subject, admission, split = str(pair), identifier(subject), identifier(admission), str(split)
                if pair in identities:
                    raise ValueError("Duplicate pair in prepared dataset.")
                if split not in SPLITS or (subject in prepared_splits and prepared_splits[subject] != split):
                    raise ValueError("Prepared patient split is invalid.")
                prepared_splits[subject] = split
                identities[pair] = (subject, admission, split)
        for row in rows:
            expected = identities.get(row["pair_id"])
            actual = (row["subject_id"], row["hadm_id"], row["split"])
            if expected != actual:
                raise ValueError("Pair identity/admission/split does not match prepared dataset.")
        verified_full_cohort = set(identities) == seen
        if pairs_path is None and not verified_full_cohort:
            raise ValueError("Default pair CSV is not the complete prepared cohort.")
        info.update(dataset_path=str(dataset_path.resolve()), dataset_sha256=sha256(dataset_path),
                    prepared_pair_count=len(identities), identity_validation="dataset.npz")
    elif pairs_path is None:
        raise FileNotFoundError("Prepared dataset.npz is required for the default research cohort.")
    else:
        info["identity_validation"] = "explicit_csv_only"
    selected = rows if pair_limit is None else rows[:int(pair_limit)]
    info.update(selected_pair_count=len(selected), split_pair_counts=dict(Counter(r["split"] for r in selected)),
                complete=verified_full_cohort and pair_limit is None,
                diagnostic=pair_limit is not None or not verified_full_cohort)
    return selected, info


def load_wave_records(wave_root):
    wave_root = Path(wave_root)
    path = wave_root / "v0.2_tmp_starttime_index.csv"
    records, seen, skipped = defaultdict(list), set(), Counter()
    total = 0
    with _open_csv(path) as stream:
        reader = csv.DictReader(stream)
        if not {"subject_id", "study_id", "starttime", "endtime"}.issubset(reader.fieldnames or []):
            raise ValueError("Waveform index needs subject_id,study_id,starttime,endtime.")
        for row in reader:
            total += 1
            subject = identifier(row["subject_id"], allow_p=True)
            record_id = str(row["study_id"]).strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
                raise ValueError("Unsafe waveform study identifier.")
            key = subject, record_id
            if key in seen:
                raise ValueError("Duplicate subject/study in waveform index.")
            seen.add(key)
            try:
                left = timestamp_us(row["starttime"], waveform_local=True)
                right = timestamp_us(row["endtime"], waveform_local=True)
            except ValueError:
                skipped["invalid_index_timestamp"] += 1
                continue
            if right <= left:
                skipped["nonpositive_index_interval"] += 1
                continue
            patient_folder = "p" + subject
            record_path = wave_root / "waves" / patient_folder[:4] / patient_folder / record_id / "waves"
            records[subject].append({"record_id": record_id, "study_id": record_id, "subject_id": subject,
                                     "starttime": row["starttime"], "endtime": row["endtime"],
                                     "start_us": left, "end_us": right,
                                     "index_start_us": left, "index_end_us": right,
                                     "record_path": str(record_path), "master_header_path": str(record_path) + ".hea"})
    for subject in records:
        records[subject].sort(key=lambda r: (r["start_us"], r["record_id"]))
    return dict(records), {"path": str(path.resolve()), "sha256": sha256(path),
                           "index_rows": total, "valid_records": sum(map(len, records.values())),
                           "subjects": len(records), "skips": dict(skipped),
                           "clock_policy": "supplied local wall time; native inspection must exclude clock risks"}


class AdmissionIntervals(dict):
    """Interval mapping with source audit and actual admission ownership."""
    def __init__(self):
        super().__init__()
        self.source_info = {}
        self.hadm_owners = {}
        self.ambiguous_hadm_ids = set()
        self.ambiguous_intervals = {}


def _table_path(source_root, name):
    root = Path(source_root)
    for path in (root / "hosp" / (name + ".csv.gz"), root / "hosp" / (name + ".csv"),
                 root / (name + ".csv.gz"), root / (name + ".csv")):
        if path.exists():
            return path
    raise FileNotFoundError("Missing admissions/transfers source: " + name)


def load_admission_intervals(source_root, wanted_subjects):
    wanted = {identifier(subject, allow_p=True) for subject in wanted_subjects}
    result, spans, skipped = AdmissionIntervals(), defaultdict(list), Counter()
    ambiguous_for_wanted = set()
    counts = {}
    for table, left_name, right_name in (("admissions", "admittime", "dischtime"),
                                        ("transfers", "intime", "outtime")):
        path = _table_path(source_root, table)
        rows_read, rows_selected = 0, 0
        with _open_csv(path) as stream:
            reader = csv.DictReader(stream)
            if not {"subject_id", "hadm_id", left_name, right_name}.issubset(reader.fieldnames or []):
                raise ValueError("Missing columns in " + table)
            for row in reader:
                rows_read += 1
                subject = identifier(row["subject_id"])
                if not row["hadm_id"]:
                    if subject in wanted:
                        skipped[table + "_missing_hadm"] += 1
                    continue
                admission = identifier(row["hadm_id"])
                if admission not in result.hadm_owners:
                    result.hadm_owners[admission] = subject
                elif admission in result.ambiguous_hadm_ids:
                    if subject in wanted:
                        ambiguous_for_wanted.add(admission)
                elif result.hadm_owners[admission] != subject:
                    previous = result.hadm_owners[admission]
                    if previous in wanted or subject in wanted:
                        ambiguous_for_wanted.add(admission)
                    result.ambiguous_hadm_ids.add(admission)
                    # Neither conflicting patient is accepted as the owner.
                    # Unrelated source conflicts must not abort the whole cohort.
                    result.hadm_owners[admission] = None
                    skipped[table + "_conflicting_admission_owner"] += 1
                if subject not in wanted:
                    continue
                rows_selected += 1
                if not row[left_name] or not row[right_name]:
                    skipped[table + "_open_or_missing_interval"] += 1
                    continue
                try:
                    left, right = timestamp_us(row[left_name]), timestamp_us(row[right_name])
                except ValueError:
                    skipped[table + "_invalid_interval_timestamp"] += 1
                    continue
                if right <= left:
                    skipped[table + "_nonpositive_interval"] += 1
                    continue
                spans[subject, admission].append((left, right))
        counts[table] = {"path": str(path.resolve()), "sha256": sha256(path),
                         "rows_scanned": rows_read, "rows_for_subjects": rows_selected}
    for key, intervals in spans.items():
        merged = merge_intervals(intervals)
        if key[1] in result.ambiguous_hadm_ids:
            # These spans cannot establish ownership, but they still block any
            # overlapping, otherwise verified admission for the same patient.
            result.ambiguous_intervals[key] = merged
        else:
            result[key] = merged
    result.source_info = {"sources": counts, "interval_keys": len(result), "skips": dict(skipped),
                          "ambiguous_ownership": {
                              "global_conflicting_hadm_count": len(result.ambiguous_hadm_ids),
                              "conflicting_hadm_count_for_requested_subjects": len(ambiguous_for_wanted),
                              "excluded_admission_keys_for_requested_subjects": len(result.ambiguous_intervals),
                              "excluded_closed_intervals_for_requested_subjects": sum(map(len, result.ambiguous_intervals.values()))},
                          "policy": "union of positive, closed admissions/transfers intervals; no inferred end times; conflicting ownership excluded and overlapping spans blocked"}
    return result


def plan_index(pairs, records_by_subject, admissions, inspect_record_fn, wave_root, db):
    """Write compact verified segment spans, preserving the complete pair cohort.

    The caller owns the staging database/transaction. inspect_record_fn must
    return (segment dicts, aggregate skip counts), excluding known clock risks.
    """
    # execute individual statements: executescript would implicitly commit the
    # caller's transaction before its staging index is complete.
    for statement in SCHEMA.split(";"):
        if statement.strip():
            db.execute(statement)
    db.executemany("INSERT INTO pairs VALUES(?,?,?,?,?,?,?)", [
        tuple(row[name] for name in ("pair_id", "subject_id", "hadm_id", "split", "t1_us", "t2_us", "start_us"))
        for row in pairs])
    cache, persisted, matched_pairs, skipped = {}, set(), set(), Counter()
    by_subject = defaultdict(list)
    for (subject, admission), intervals in admissions.items():
        by_subject[identifier(subject, allow_p=True)].append((identifier(admission), intervals))
    ownership_blockers = defaultdict(list)
    for (subject, admission), intervals in getattr(admissions, "ambiguous_intervals", {}).items():
        ownership_blockers[identifier(subject, allow_p=True)].extend(intervals)
    span_count, ambiguous_us, unverified_us = 0, 0, 0
    for pair in pairs:
        subject, admission = identifier(pair["subject_id"]), identifier(pair["hadm_id"])
        left, right = int(pair["start_us"]), int(pair["t2_us"])
        if admission in getattr(admissions, "ambiguous_hadm_ids", set()):
            skipped["pair_admission_ownership_ambiguous"] += 1
            unverified_us += right - left
            continue
        owner = getattr(admissions, "hadm_owners", {}).get(admission)
        if owner is not None and owner != subject:
            raise ValueError("Pair admission belongs to another patient in source ADT.")
        own = [(max(left, a), min(right, b)) for a, b in admissions.get((subject, admission), [])
               if a < right and b > left]
        others = [(max(left, a), min(right, b)) for hadm, intervals in by_subject.get(subject, [])
                  if hadm != admission for a, b in intervals if a < right and b > left]
        others.extend((max(left, a), min(right, b)) for a, b in ownership_blockers.get(subject, [])
                      if a < right and b > left)
        own = merge_intervals(own)
        verified = _subtract_intervals(own, others)
        own_us = sum(b - a for a, b in own)
        verified_us = sum(b - a for a, b in verified)
        ambiguous_us += own_us - verified_us
        unverified_us += right - left - own_us
        if not verified:
            skipped["pair_without_unique_closed_admission_interval"] += 1
            continue
        for record in records_by_subject.get(subject, []):
            record_left = int(record.get("start_us", record.get("index_start_us")))
            record_right = int(record.get("end_us", record.get("index_end_us")))
            if record_left >= right or record_right <= left:
                continue
            key = subject, str(record["record_id"])
            if key not in cache:
                segments, reasons = inspect_record_fn(record, wave_root)
                skipped.update(reasons)
                cache[key] = segments
            for segment in cache[key]:
                if identifier(segment["subject_id"], allow_p=True) != subject:
                    raise ValueError("Native segment patient does not match waveform index.")
                frames, fs = int(segment["frames"]), float(segment["frame_fs"])
                if frames <= 0 or not math.isfinite(fs) or fs <= 0:
                    raise ValueError("Invalid native segment frame metadata.")
                if not segment.get("channels"):
                    continue
                origin = int(segment["origin_us"])
                segment_right = min(int(segment["record_end_us"]), origin + round(frames * 1_000_000 / fs))
                start, end = max(left, record_left, origin), min(right, record_right, segment_right)
                if end <= start:
                    continue
                pieces = [(max(start, a), min(end, b)) for a, b in verified if a < end and b > start]
                if not pieces:
                    continue
                sid = str(segment["segment_id"])
                if sid not in persisted:
                    metadata = {k: v for k, v in segment.items() if k not in {
                        "segment_id", "record_id", "subject_id", "path", "origin_us", "frames", "frame_fs", "record_end_us"}}
                    metadata["index_starttime"] = record.get("starttime")
                    metadata["index_endtime"] = record.get("endtime")
                    db.execute("INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?)", (
                        sid, str(segment["record_id"]), subject, str(segment["path"]), origin,
                        frames, fs, int(segment["record_end_us"]), json.dumps(metadata, sort_keys=True, allow_nan=False)))
                    persisted.add(sid)
                for a, b in pieces:
                    if not left <= a < b <= right:
                        raise AssertionError("Accepted span exceeds the pair observation window.")
                    db.execute("INSERT INTO pair_segments VALUES(?,?,?,?,?)", (pair["pair_id"], sid, a, b, "verified"))
                    span_count += 1
                    matched_pairs.add(pair["pair_id"])
    return {"pairs_all": len(pairs), "pairs_with_verified_segments": len(matched_pairs),
            "missing_pairs": len(pairs) - len(matched_pairs), "matched_segments": len(persisted),
            "verified_spans": span_count, "records_inspected": len(cache),
            "admission_ambiguous_pair_hours": ambiguous_us / 3_600_000_000,
            "admission_unverified_pair_hours": unverified_us / 3_600_000_000,
            "split_pair_counts": dict(Counter(p["split"] for p in pairs)), "skips": dict(skipped),
            "coverage_definition": "verified header segment spans, not decoded or QC-qualified samples",
            "history_hours": 12, "clock_policy": "native inspector refuses nonzero counters and changing offsets"}
