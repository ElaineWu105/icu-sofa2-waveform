"""Real expanded-SPF decoding and compact-source boundary regression checks."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import wfdb

import source

SCHEMA = """
CREATE TABLE pairs(pair_id TEXT,subject_id TEXT,hadm_id TEXT,split TEXT,t1_us INT,t2_us INT,start_us INT);
CREATE TABLE segments(segment_id TEXT,record_id TEXT,subject_id TEXT,path TEXT,origin_us INT,frames INT,
                      frame_fs REAL,record_end_us INT,metadata_json TEXT);
CREATE TABLE pair_segments(pair_id TEXT,segment_id TEXT,left_us INT,right_us INT,admission_status TEXT);
"""


class SourceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.frames = 1875
        self.spf = (4, 2, 1)
        self.digital = [((np.arange(self.frames * count) % 4000) - 2000).astype(np.int64) for count in self.spf]
        self.digital[0][2] = -32768
        self.channels = [
            {"channel_index": 0, "modality": "ecg", "lead_id": "II", "raw_name": "II", "sample_fs": 250.,
             "samples_per_frame": 4, "physical_scale": .001, "unit": "uV", "output_unit": "mV"},
            {"channel_index": 1, "modality": "ppg", "lead_id": "Pleth", "raw_name": "Pleth", "sample_fs": 125.,
             "samples_per_frame": 2, "physical_scale": 1., "unit": "NU"},
            {"channel_index": 2, "modality": "resp", "lead_id": "Resp", "raw_name": "Resp", "sample_fs": 62.5,
             "samples_per_frame": 1, "physical_scale": 1., "unit": "Ohm"}]
        self.write_record()
        self.db_path = self.root / "index.sqlite"
        with sqlite3.connect(self.db_path) as db:
            db.executescript(SCHEMA)
            db.execute("INSERT INTO pairs VALUES(?,?,?,?,?,?,?)", ("one", "1", "10", "train", -1, 30_000_000, 0))
            db.execute("INSERT INTO pairs VALUES(?,?,?,?,?,?,?)", ("empty", "2", "20", "test", -1, 30_000_000, 0))
            self.add_segment(db, "s1", self.channels)
            db.execute("INSERT INTO pair_segments VALUES(?,?,?,?,?)", ("one", "s1", 0, 30_000_000, "verified"))

    def write_record(self):
        record = wfdb.Record(record_name="segment", n_sig=3, fs=62.5, sig_len=self.frames,
                             file_name=["ecg.dat", "ppg.dat", "resp.dat"], fmt=["16"] * 3,
                             samps_per_frame=list(self.spf), skew=[0] * 3, byte_offset=[0] * 3,
                             adc_gain=[200., 10., 5.], baseline=[100, 0, 20], units=["uV", "NU", "Ohm"],
                             adc_res=[16] * 3, adc_zero=[0] * 3,
                             init_value=[int(values[0]) for values in self.digital],
                             checksum=[int(values.sum() % 65536) for values in self.digital],
                             block_size=[0] * 3, sig_name=["II", "Pleth", "Resp"], e_d_signal=self.digital)
        record.wrsamp(expanded=True, write_dir=str(self.root))

    def add_segment(self, db, name, channels, origin=0, frames=None, end=30_000_000):
        db.execute("INSERT INTO segments VALUES(?,?,?,?,?,?,?,?,?)", (
            name, "r1", "1", str(self.root / "segment"), origin, self.frames if frames is None else frames,
            62.5, end, json.dumps({"channels": channels})))

    def test_real_spf_shapes_physical_units_missing_samples_and_read_flags(self):
        with source.SnapshotSource(self.root) as reader:
            job = next(reader.iter_jobs("one"))
            actual_reader = wfdb.rdrecord
            with patch.object(source.wfdb, "rdrecord", wraps=actual_reader) as decode:
                raw = reader.read_window(job)
            self.assertTrue(decode.call_args.kwargs["physical"])
            self.assertFalse(decode.call_args.kwargs["smooth_frames"])
            self.assertEqual(decode.call_args.kwargs["channels"], [0, 1, 2])
            self.assertEqual(raw["lead_id"], "II")
            for index, modality in enumerate(("ecg", "ppg", "resp")):
                self.assertEqual(raw[modality].shape, (source.COUNTS[modality],))
                self.assertEqual(raw[modality].dtype, np.float32)
                self.assertEqual(raw["mask_by_modality"][modality].shape, raw[modality].shape)
                self.assertEqual(raw["mask_by_modality"][modality].dtype, np.bool_)
                self.assertTrue(raw["modality_available"][modality])
                values = self.digital[index][:source.COUNTS[modality]].astype(float)
                values = (values - (100, 0, 20)[index]) / (200, 10, 5)[index] * (.001, 1., 1.)[index]
                values[self.digital[index][:len(values)] == -32768] = np.nan
                np.testing.assert_allclose(raw[modality], values, rtol=1e-6, atol=1e-7, equal_nan=True)
            self.assertFalse(raw["sample_valid"]["ecg"][2])
            self.assertEqual(raw["metadata"]["native_first_sample_offset_us"], {"ecg": 0, "ppg": 0, "resp": 0})

    def test_arbitrary_subframe_cutoff_and_future_modification_do_not_leak(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("UPDATE pairs SET start_us=1000,t2_us=30001000 WHERE pair_id='one'")
        with source.SnapshotSource(self.db_path) as reader:
            job = next(reader.iter_jobs("one"))
            before = reader.read_window(job)
            self.assertEqual(before["metadata"]["native_first_sample_offset_us"],
                             {"ecg": 3000, "ppg": 7000, "resp": 15000})
            for index, modality in enumerate(("ecg", "ppg", "resp")):
                period = int(round(1_000_000 / source.RATES[modality]))
                physical_times = np.arange(len(self.digital[index])) * period
                self.digital[index][physical_times >= job["end_us"]] += 500
                last = before["metadata"]["native_last_sample_offset_us"][modality]
                self.assertLess(last, source.WINDOW_US)
            self.write_record()
            after = reader.read_window(job)
            for modality in source.RATES:
                np.testing.assert_array_equal(before[modality], after[modality])

    def test_does_not_stitch_two_admission_spans_and_keeps_empty_pair(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM pair_segments")
            db.executemany("INSERT INTO pair_segments VALUES(?,?,?,?,?)", [
                ("one", "s1", 0, 9_000_000, "verified"),
                ("one", "s1", 9_000_000, 30_000_000, "verified")])
        with source.SnapshotSource(self.root) as reader:
            self.assertEqual(len(reader.pairs), 2)
            self.assertEqual([job["slot_index"] for job in reader.iter_jobs("one")], [1, 2])
            self.assertEqual(list(reader.iter_jobs("empty")), [])

    def test_segment_boundary_and_gap_cannot_be_crossed(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("DELETE FROM pair_segments")
            db.execute("DELETE FROM segments")
            self.add_segment(db, "early", self.channels, frames=563, end=9_008_000)
            self.add_segment(db, "late", self.channels, origin=10_000_000, frames=1250)
            db.executemany("INSERT INTO pair_segments VALUES(?,?,?,?,?)", [
                ("one", "early", 0, 9_008_000, "verified"),
                ("one", "late", 10_000_000, 30_000_000, "verified")])
        with source.SnapshotSource(self.root) as reader:
            jobs = list(reader.iter_jobs("one"))
            self.assertEqual([job["slot_index"] for job in jobs], [1, 2])
            self.assertTrue(all(job["segment"]["segment_id"] == "late" for job in jobs))

    def test_lead_ii_preferred_and_generic_leads_excluded_without_losing_ppg(self):
        generic = dict(self.channels[0], raw_name="V", lead_id="V")
        lead_i = dict(self.channels[0], channel_index=3, raw_name="I", lead_id="I")
        segment = {"frame_fs": 62.5, "channels": [generic, lead_i, self.channels[0], self.channels[1]]}
        self.assertEqual(source._choose_channels(segment)["ecg"]["canonical_lead_id"], "II")
        segment["channels"] = [generic, dict(generic, raw_name="ECG1"), self.channels[1]]
        chosen = source._choose_channels(segment)
        self.assertNotIn("ecg", chosen)
        self.assertIn("ppg", chosen)

    def test_one_segment_selected_deterministically_when_records_overlap(self):
        with sqlite3.connect(self.db_path) as db:
            self.add_segment(db, "alternate", [dict(self.channels[0], lead_id="I", raw_name="I"), self.channels[1]])
            db.execute("INSERT INTO pair_segments VALUES(?,?,?,?,?)", ("one", "alternate", 0, 30_000_000, "verified"))
        with source.SnapshotSource(self.root) as reader:
            jobs = list(reader.iter_jobs("one"))
            self.assertEqual(len(jobs), 3)
            self.assertTrue(all(job["segment"]["segment_id"] == "s1" for job in jobs))

    def test_only_compact_tables_queried_and_connection_is_read_only(self):
        with source.SnapshotSource(self.root) as reader:
            trace = []
            reader.connection.set_trace_callback(trace.append)
            list(reader.iter_jobs("one"))
            self.assertTrue(all("chunks" not in query and "groups" not in query for query in trace))
            with self.assertRaises(sqlite3.OperationalError):
                reader.connection.execute("DELETE FROM pairs")
            altered = dict(reader.pairs[0], t2_us=40_000_000)
            with self.assertRaisesRegex(ValueError, "does not match"):
                list(reader.iter_jobs(altered))

    def test_worker_does_not_touch_sqlite_and_rejects_future_before_read(self):
        with source.SnapshotSource(self.root) as reader:
            job = next(reader.iter_jobs("one"))
            reader.close()
            raw = reader.read_window(job)
            self.assertEqual(raw["ecg"].shape, (2500,))
            invalid = dict(job, cutoff_us=job["end_us"] - 1)
            with patch.object(source.wfdb, "rdrecord") as decode:
                with self.assertRaisesRegex(source.SourceReadError, "cutoff"):
                    reader.read_window(invalid)
                decode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
