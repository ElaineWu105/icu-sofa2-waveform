"""Native WFDB calibration, clock bounds, SPF and missing-sample regression checks."""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import wfdb

import native


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "waves/p100/p10000001/study"
        self.folder.mkdir(parents=True)
        self.frames = 1888
        self.spf = (4, 2, 1)
        self.digital = [((np.arange(self.frames * spf) % 4000) - 2000).astype(np.int64) for spf in self.spf]
        self.digital[0][12:15] = -32768  # WFDB 16-bit missing-sample sentinel.
        self.gains, self.baselines = (200., 10., 5.), (100, 0, 20)
        record = wfdb.Record(
            record_name="segment", n_sig=3, fs=62.5, counter_freq=1000.,
            sig_len=self.frames, file_name=["nativee.dat", "nativep.dat", "nativer.dat"],
            fmt=["16"] * 3, samps_per_frame=list(self.spf), skew=[0] * 3, byte_offset=[0] * 3,
            adc_gain=list(self.gains), baseline=list(self.baselines), units=["uV", "NU", "Ohm"],
            adc_res=[16] * 3, adc_zero=[0] * 3,
            init_value=[int(values[0]) for values in self.digital],
            checksum=[int(values.sum() % 65536) for values in self.digital], block_size=[0] * 3,
            sig_name=["II", "PLETH", "RESP"], e_d_signal=self.digital,
        )
        record.wrsamp(expanded=True, write_dir=str(self.folder))
        self.master = self.folder / "waves.hea"
        self.master.write_text(f"waves/1 3 62.5/1000(0) {self.frames}\nsegment {self.frames}\n")
        self.index = {"subject_id": "p10000001", "study_id": "study",
                      "starttime": "2020-01-01 00:00:00-05:00", "endtime": "2020-01-01 00:01:00-05:00"}

    def inventory(self):
        segments, skips = native.inspect_record(self.index, self.root)
        self.assertEqual(skips, {})
        self.assertEqual(len(segments), 1)
        return segments[0]

    @staticmethod
    def chunk(segment, channel, left, right):
        return {"chunk_id": f"{channel['index']}_{left}", "segment_id": segment["segment_id"],
                "channel_index": channel["index"], "modality": channel["modality"],
                "lead_id": channel["lead_id"], "sample_fs": channel["sample_fs"],
                "samples_per_frame": channel["spf"], "frame_start": left, "frame_end": right,
                "origin_us": segment["origin_us"] + round(left * 1e6 / segment["frame_fs"]),
                "n_samples": (right - left) * channel["spf"],
                "unit": channel["output_unit"], "rawunit": channel["unit"]}

    def test_inventory_preserves_raw_names_native_rates_and_stable_identity(self):
        segment = self.inventory()
        self.assertEqual(segment["subject_id"], "10000001")
        self.assertEqual(segment["record_id"], "study")
        self.assertEqual([channel["sample_fs"] for channel in segment["channels"]], [250., 125., 62.5])
        self.assertEqual([channel["spf"] for channel in segment["channels"]], [4, 2, 1])
        self.assertEqual([channel["lead_id"] for channel in segment["channels"]], ["II", "PLETH", "RESP"])
        self.assertEqual(segment["segment_id"], self.inventory()["segment_id"])
        self.assertEqual(segment["origin_us"], native.clock_us("2020-01-01 00:00:00"))

    def test_real_native_multispf_decode_calibration_invalid_masks_and_partial_tails(self):
        segment = self.inventory()
        chunks = [self.chunk(segment, channel, left, right)
                  for left, right in ((0, 1875), (1875, self.frames)) for channel in segment["channels"]]
        job = {"group_id": "real_native_fixture", "segment": segment, "frame_start": 0,
               "frame_end": self.frames, "chunks": chunks, "chunk_seconds": 30}
        actual_reader = wfdb.rdrecord
        with patch.object(native.wfdb, "rdrecord", wraps=actual_reader) as reader:
            result = native.decode_group(job)
        self.assertIsNone(result["error"])
        self.assertEqual(reader.call_count, 1)
        self.assertTrue(reader.call_args.kwargs["physical"])
        self.assertFalse(reader.call_args.kwargs["smooth_frames"])
        self.assertEqual(reader.call_args.kwargs["channels"], [0, 1, 2])
        for row, chunk in zip(result["rows"], chunks):
            channel = segment["channels"][chunk["channel_index"]]
            self.assertEqual(row["values"].dtype, np.float32)
            self.assertEqual(row["valid"].dtype, np.bool_)
            self.assertEqual(len(row["values"]), round(30 * channel["sample_fs"]))
            self.assertEqual(row["metadata"], chunk)
            self.assertEqual(row["qc"]["native_samples"], chunk["n_samples"])
            self.assertTrue(np.isnan(row["values"][chunk["n_samples"]:]).all())
            self.assertFalse(row["valid"][chunk["n_samples"]:].any())
            digital = self.digital[channel["index"]][chunk["frame_start"] * channel["spf"]:chunk["frame_end"] * channel["spf"]]
            expected = ((digital.astype(float) - channel["baseline"]) / channel["gain"] * channel["physical_scale"])
            expected[digital == -32768] = np.nan
            np.testing.assert_allclose(row["values"][:len(expected)], expected, rtol=1e-6, atol=1e-7, equal_nan=True)
        ecg = result["rows"][0]
        self.assertFalse(ecg["valid"][12:15].any())
        self.assertTrue(np.isnan(ecg["values"][12:15]).all())
        self.assertEqual(ecg["qc"]["longest_native_gap_samples"], 3)
        self.assertEqual(ecg["qc"]["output_unit"], "mV")

    def test_arbitrary_group_frame_start_never_smooths_native_samples(self):
        segment = self.inventory()
        channel = segment["channels"][0]
        chunk = self.chunk(segment, channel, 17, 117)
        result = native.decode_group({"group_id": "nonzero_start", "segment": segment,
                                      "frame_start": 17, "frame_end": 117, "chunks": [chunk], "chunk_seconds": 30})
        self.assertIsNone(result["error"])
        expected_first = (self.digital[0][17 * 4] - 100) / 200 * .001
        self.assertAlmostEqual(float(result["rows"][0]["values"][0]), expected_first, places=9)
        self.assertEqual(result["rows"][0]["qc"]["native_samples"], 400)

    def test_index_tail_clips_inside_frame_without_valid_future_samples(self):
        segment = self.inventory()
        segment["record_end_us"] = segment["origin_us"] + 30_006_000
        channel = segment["channels"][0]
        chunk = self.chunk(segment, channel, 1875, 1876)
        chunk["n_samples"] = 2  # Native samples at 30.000 and 30.004, cutoff 30.006.
        result = native.decode_group({"group_id": "fractional_tail", "segment": segment,
                                      "frame_start": 1875, "frame_end": 1876, "chunks": [chunk]})
        self.assertIsNone(result["error"])
        self.assertTrue(result["rows"][0]["valid"][:2].all())
        self.assertFalse(result["rows"][0]["valid"][2:].any())
        with patch.object(native.wfdb, "rdrecord") as reader:
            rejected = native.decode_group({"group_id": "future", "segment": segment,
                                           "frame_start": 1875, "frame_end": 1877, "chunks": [chunk]})
        self.assertIn("future frames", rejected["error"])
        reader.assert_not_called()

    def test_gap_layout_offsets_are_preserved_and_frame_total_checked(self):
        self.master.write_text(f"waves/3 3 62.5/1000(0) {self.frames + 125}\nlayout 0\n~ 125\nsegment {self.frames}\n")
        header = wfdb.rdheader(str(self.folder / "segment"))
        header.base_counter = 2000.
        with patch.object(native.wfdb, "rdheader", return_value=header):
            segments, skips = native.inspect_record(self.index, self.root)
        self.assertEqual(skips, {"layout_segment": 1, "gap_segment": 1})
        self.assertEqual(segments[0]["origin_us"], native.clock_us(self.index["starttime"]) + 2_000_000)
        self.master.write_text(f"waves/1 3 62.5/1000(0) {self.frames + 1}\nsegment {self.frames}\n")
        self.assertEqual(native.inspect_record(self.index, self.root), ([], {"master_frame_count_mismatch": 1}))

    def test_clock_risks_and_segment_counter_mismatches_are_explicit_rejections(self):
        changed_offset = dict(self.index, endtime="2020-01-01 00:01:00-04:00")
        self.assertEqual(native.inspect_record(changed_offset, self.root), ([], {"index_utc_offset_change": 1}))
        self.master.write_text(f"waves/1 3 62.5/1000(1000) {self.frames}\nsegment {self.frames}\n")
        self.assertEqual(native.inspect_record(self.index, self.root), ([], {"nonzero_master_counter": 1}))
        self.master.write_text(f"waves/1 3 62.5/1000(0) {self.frames}\nsegment {self.frames}\n")
        header = wfdb.rdheader(str(self.folder / "segment"))
        header.base_counter = 11.
        with patch.object(native.wfdb, "rdheader", return_value=header):
            self.assertEqual(native.inspect_record(self.index, self.root), ([], {"segment_counter_offset_mismatch": 1}))

    def test_channel_classification_keeps_ppg_resp_without_ecg_and_distinct_raw_leads(self):
        self.assertEqual(native.classify_channel("aVR", "anything.dat"), "ecg")
        self.assertEqual(native.classify_channel("ECG2", "anything.dat"), "ecg")
        self.assertEqual(native.classify_channel("unnamed", "nativee.dat"), "ecg")
        self.assertIsNone(native.classify_channel("ART", "nativea.dat"))
        header = wfdb.rdheader(str(self.folder / "segment"))
        header.sig_name[0] = "ART"
        header.file_name[0] = "nativea.dat"
        with patch.object(native.wfdb, "rdheader", return_value=header):
            segments, skips = native.inspect_record(self.index, self.root)
        self.assertEqual(skips, {})
        self.assertEqual([channel["modality"] for channel in segments[0]["channels"]], ["ppg", "resp"])

    def test_unsupported_rate_and_unit_are_counted_and_decode_failure_is_not_empty_success(self):
        header = wfdb.rdheader(str(self.folder / "segment"))
        header.samps_per_frame[1] = 4
        header.units[0] = "NU"
        with patch.object(native.wfdb, "rdheader", return_value=header):
            segments, skips = native.inspect_record(self.index, self.root)
        self.assertEqual(skips, {"unsupported_ecg_unit": 1, "unsupported_native_rate_ppg": 1})
        self.assertEqual([channel["modality"] for channel in segments[0]["channels"]], ["resp"])
        segment = self.inventory()
        chunk = self.chunk(segment, segment["channels"][0], 0, 1875)
        (self.folder / "nativee.dat").unlink()
        result = native.decode_group({"group_id": "missing_data", "segment": segment,
                                      "frame_start": 0, "frame_end": 1875, "chunks": [chunk]})
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["rows"], [])

    def test_qc_flags_never_remove_finite_flat_or_large_amplitudes(self):
        channel = self.inventory()["channels"][0]
        values = np.full(7500, 42., dtype=np.float32)
        valid = np.ones(7500, dtype=bool)
        qc = native.sample_qc(values, valid, 7500, channel)
        self.assertTrue(qc["flat"])
        self.assertTrue(qc["physiological_large"])
        self.assertEqual(qc["equal_fraction"], 1.)
        np.testing.assert_array_equal(values, np.full(7500, 42., dtype=np.float32))
        self.assertTrue(valid.all())


if __name__ == "__main__":
    unittest.main()
