"""Identity, admission uniqueness, and cutoff regression checks; no real scans."""
import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

import indexing

SECOND = 1_000_000
HOUR = 3600 * SECOND


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pair(name="one", subject="1", admission="10", cutoff=12 * HOUR):
    return {"pair_id": name, "subject_id": subject, "hadm_id": admission,
            "split": "train", "t1_us": cutoff - HOUR, "t2_us": cutoff,
            "start_us": cutoff - indexing.HISTORY_US}


class IndexingTests(unittest.TestCase):
    def test_record_started_before_window_cutoff_empty_pair_and_single_inspection(self):
        pairs = [pair(), pair("two", cutoff=13 * HOUR), pair("missing", "2", "20")]
        record = {"record_id": "s1", "subject_id": "1", "start_us": -HOUR,
                  "end_us": 15 * HOUR, "starttime": "1969-12-31 23:00:00",
                  "endtime": "1970-01-01 15:00:00"}
        calls = []
        def inspect(row, root):
            calls.append(row)
            return [{"segment_id": "seg", "record_id": "s1", "subject_id": "1",
                     "path": "/synthetic/segment", "origin_us": -HOUR,
                     "frames": 16 * 3600, "frame_fs": 1., "record_end_us": 15 * HOUR,
                     "channels": [{"modality": "resp", "lead_id": "Resp"}]}], {}
        db = sqlite3.connect(":memory:")
        stats = indexing.plan_index(pairs, {"1": [record]}, {("1", "10"): [(-HOUR, 15 * HOUR)]},
                                    inspect, Path("/synthetic"), db)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats["pairs_all"], 3)
        self.assertEqual(stats["missing_pairs"], 1)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM pairs").fetchone()[0], 3)
        spans = db.execute("SELECT pair_id,left_us,right_us FROM pair_segments ORDER BY pair_id").fetchall()
        self.assertEqual(spans, [("one", 0, 12 * HOUR), ("two", HOUR, 13 * HOUR)])

    def test_ambiguous_admission_span_is_removed_without_dropping_pair(self):
        def inspect(record, root):
            return [{"segment_id": "seg", "record_id": "s1", "subject_id": "1", "path": "/seg",
                     "origin_us": 0, "frames": 12 * 3600, "frame_fs": 1., "record_end_us": 12 * HOUR,
                     "channels": [{"modality": "pleth"}]}], {}
        db = sqlite3.connect(":memory:")
        stats = indexing.plan_index([pair()], {"1": [{"record_id": "s1", "start_us": 0, "end_us": 12 * HOUR}]},
                                    {("1", "10"): [(0, 12 * HOUR)], ("1", "11"): [(5 * HOUR, 7 * HOUR)]},
                                    inspect, "/", db)
        spans = db.execute("SELECT left_us,right_us,admission_status FROM pair_segments ORDER BY left_us").fetchall()
        self.assertEqual(spans, [(0, 5 * HOUR, "verified"), (7 * HOUR, 12 * HOUR, "verified")])
        self.assertEqual(stats["admission_ambiguous_pair_hours"], 2)

    def test_native_clock_rejection_preserves_missing_pair(self):
        db = sqlite3.connect(":memory:")
        stats = indexing.plan_index([pair()], {"1": [{"record_id": "s1", "start_us": 0, "end_us": 12 * HOUR}]},
                                    {("1", "10"): [(0, 12 * HOUR)]},
                                    lambda record, root: ([], {"nonzero_master_counter": 1}), "/", db)
        self.assertEqual(stats["missing_pairs"], 1)
        self.assertEqual(stats["skips"]["nonzero_master_counter"], 1)

    def test_admission_wrong_patient_is_error(self):
        admissions = indexing.AdmissionIntervals()
        admissions.hadm_owners["10"] = "2"
        with self.assertRaisesRegex(ValueError, "another patient"):
            indexing.plan_index([pair()], {}, admissions, lambda record, root: ([], {}), "/", sqlite3.connect(":memory:"))

    def test_half_open_subtraction_keeps_touching_admissions(self):
        self.assertEqual(indexing._subtract_intervals([(0, 10)], [(10, 20)]), [(0, 10)])
        self.assertEqual(indexing._subtract_intervals([(0, 10)], [(-5, 0)]), [(0, 10)])

    def test_pair_csv_checks_full_split_before_diagnostic_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"pair_id": "a", "subject_id": "1", "hadm_id": "10", "split": "train",
                     "t1": "2020-01-01 00:00:00", "t2": "2020-01-01 12:00:00"},
                    {"pair_id": "b", "subject_id": "1", "hadm_id": "10", "split": "test",
                     "t1": "2020-01-02 00:00:00", "t2": "2020-01-02 12:00:00"}]
            path = root / "pairs.csv"
            write_csv(path, rows[0].keys(), rows)
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                indexing.load_pairs(root, path, pair_limit=1)

    def test_pair_identity_must_match_prepared_npz(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {"pair_id": "a", "subject_id": "1", "hadm_id": "10", "split": "train",
                   "t1": "2020-01-01 00:00:00", "t2": "2020-01-01 12:00:00"}
            write_csv(root / "pairs_index.csv", row.keys(), [row])
            np.savez(root / "dataset.npz", pair_ids=["a"], subject_ids=["1"], hadm_ids=["11"], split=["train"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                indexing.load_pairs(root)
            np.savez(root / "dataset.npz", pair_ids=["a"], subject_ids=["1"], hadm_ids=["10"], split=["train"])
            selected, info = indexing.load_pairs(root)
            self.assertEqual(selected[0]["t2_us"] - selected[0]["start_us"], indexing.HISTORY_US)
            self.assertTrue(info["complete"])

    def test_open_admission_is_not_extrapolated_and_closed_transfers_union(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "hosp/admissions.csv", ["subject_id", "hadm_id", "admittime", "dischtime"], [
                {"subject_id": "1", "hadm_id": "10", "admittime": "2020-01-01 00:00:00", "dischtime": ""}])
            write_csv(root / "hosp/transfers.csv", ["subject_id", "hadm_id", "intime", "outtime"], [
                {"subject_id": "1", "hadm_id": "10", "intime": "2020-01-01 01:00:00", "outtime": "2020-01-01 02:00:00"},
                {"subject_id": "1", "hadm_id": "10", "intime": "2020-01-01 02:00:00", "outtime": "2020-01-01 03:00:00"}])
            result = indexing.load_admission_intervals(root, {"1"})
            self.assertEqual(result["1", "10"], [(indexing.timestamp_us("2020-01-01 01:00:00"),
                                                  indexing.timestamp_us("2020-01-01 03:00:00"))])
            self.assertEqual(result.source_info["skips"]["admissions_open_or_missing_interval"], 1)

    def test_waveform_index_normalizes_subject_and_keeps_original_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "v0.2_tmp_starttime_index.csv", ["subject_id", "study_id", "starttime", "endtime"], [
                {"subject_id": "p12345678", "study_id": "s1", "starttime": "2020-01-01 00:00:00-05:00",
                 "endtime": "2020-01-01 01:00:00-05:00"}])
            records, info = indexing.load_wave_records(root)
            record = records["12345678"][0]
            self.assertTrue(record["starttime"].endswith("-05:00"))
            self.assertEqual(record["end_us"] - record["start_us"], HOUR)
            self.assertEqual(Path(record["record_path"]), root / "waves/p123/p12345678/s1/waves")

    def test_unrelated_adt_ownership_conflict_does_not_abort_requested_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "hosp/admissions.csv", ["subject_id", "hadm_id", "admittime", "dischtime"], [
                {"subject_id": "1", "hadm_id": "10", "admittime": "1970-01-01 00:00:00", "dischtime": "1970-01-01 12:00:00"},
                {"subject_id": "2", "hadm_id": "20", "admittime": "1970-01-01 00:00:00", "dischtime": "1970-01-01 12:00:00"}])
            write_csv(root / "hosp/transfers.csv", ["subject_id", "hadm_id", "intime", "outtime"], [
                {"subject_id": "3", "hadm_id": "20", "intime": "1970-01-01 01:00:00", "outtime": "1970-01-01 02:00:00"}])
            result = indexing.load_admission_intervals(root, {"1"})
            self.assertEqual(result["1", "10"], [(0, 12 * HOUR)])
            self.assertIsNone(result.hadm_owners["20"])
            self.assertEqual(result.source_info["ambiguous_ownership"]["global_conflicting_hadm_count"], 1)
            self.assertEqual(result.source_info["ambiguous_ownership"]["conflicting_hadm_count_for_requested_subjects"], 0)

    def test_conflicting_ownership_never_verifies_and_blocks_overlaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / "hosp/admissions.csv", ["subject_id", "hadm_id", "admittime", "dischtime"], [
                {"subject_id": "1", "hadm_id": "10", "admittime": "1970-01-01 00:00:00", "dischtime": "1970-01-01 12:00:00"},
                {"subject_id": "1", "hadm_id": "11", "admittime": "1970-01-01 05:00:00", "dischtime": "1970-01-01 07:00:00"}])
            write_csv(root / "hosp/transfers.csv", ["subject_id", "hadm_id", "intime", "outtime"], [
                {"subject_id": "2", "hadm_id": "11", "intime": "1970-01-01 05:00:00", "outtime": "1970-01-01 07:00:00"}])
            result = indexing.load_admission_intervals(root, {"1"})
            self.assertNotIn(("1", "11"), result)
            self.assertEqual(result.ambiguous_intervals["1", "11"], [(5 * HOUR, 7 * HOUR)])
            def inspect(record, wave_root):
                return [{"segment_id": "seg", "record_id": "s1", "subject_id": "1", "path": "/seg",
                         "origin_us": 0, "frames": 12 * 3600, "frame_fs": 1., "record_end_us": 12 * HOUR,
                         "channels": [{"modality": "resp"}]}], {}
            db = sqlite3.connect(":memory:")
            stats = indexing.plan_index([pair(), pair("ambiguous", admission="11")],
                                        {"1": [{"record_id": "s1", "start_us": 0, "end_us": 12 * HOUR}]},
                                        result, inspect, root, db)
            self.assertEqual(db.execute("SELECT pair_id,left_us,right_us FROM pair_segments ORDER BY left_us").fetchall(),
                             [("one", 0, 5 * HOUR), ("one", 7 * HOUR, 12 * HOUR)])
            self.assertEqual(stats["missing_pairs"], 1)
            self.assertEqual(stats["skips"]["pair_admission_ownership_ambiguous"], 1)


if __name__ == "__main__":
    unittest.main()
