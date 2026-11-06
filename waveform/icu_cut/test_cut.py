"""Synthetic checks for the ICU grid, clip rule, splits and day writer."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import common
from common import ALL_THREE, DAY_US, READY, REJECTED, SLOTS_PER_CLIP, SLOTS_PER_DAY, WINDOW_US, valid_clips
import cut
import plan

CHANNELS = [{"modality": "ecg", "raw_name": "II", "lead_id": "II", "sample_fs": 250., "index": 0,
             "channel_index": 0, "samples_per_frame": 2},
            {"modality": "ppg", "sample_fs": 125., "index": 1, "channel_index": 1, "samples_per_frame": 1},
            {"modality": "resp", "sample_fs": 62.5, "index": 2, "channel_index": 2, "samples_per_frame": 1}]


def segment(name, origin, seconds, channels=CHANNELS):
    return {"segment_id": name, "record_id": "r", "subject_id": "1", "path": "/none", "origin_us": origin,
            "frames": int(seconds * 125), "frame_fs": 125.0, "record_end_us": origin + seconds * 1_000_000,
            "channels": channels}


def arrays():
    return (np.full(SLOTS_PER_DAY, READY, dtype=np.uint8), np.full(SLOTS_PER_DAY, ALL_THREE, dtype=np.uint8),
            np.zeros(SLOTS_PER_DAY, dtype=np.int32), np.ones(SLOTS_PER_DAY, dtype=np.int16))


class ClipRule(unittest.TestCase):
    def test_one_bad_slot_is_allowed_two_are_not(self):
        status, bits, segments, leads = arrays()
        status[3] = REJECTED
        status[SLOTS_PER_CLIP + 2] = status[SLOTS_PER_CLIP + 9] = REJECTED
        clips = valid_clips(status, bits, segments, leads)
        self.assertTrue(clips[0])
        self.assertFalse(clips[1])
        self.assertTrue(clips[2])

    def test_missing_resp_counts_as_bad(self):
        status, bits, segments, leads = arrays()
        bits[:2] = 0b011
        self.assertFalse(valid_clips(status, bits, segments, leads)[0])

    def test_clip_cannot_mix_segments_or_leads(self):
        status, bits, segments, leads = arrays()
        segments[10:18] = 1
        leads[SLOTS_PER_CLIP + 5] = 2
        clips = valid_clips(status, bits, segments, leads)
        self.assertFalse(clips[0])
        self.assertFalse(clips[1])


class Planning(unittest.TestCase):
    def test_grid_is_anchored_at_intime_and_needs_full_windows(self):
        intime = 1_000_000_000
        owners = plan.slot_owners([segment("a", intime + 3_000_000, 35)], intime, intime + 60_000_000)
        self.assertEqual(owners.tolist(), [-1, 0, 0, -1, -1, -1])

    def test_better_lead_wins_a_shared_slot(self):
        lead_i = [dict(CHANNELS[0], raw_name="I", lead_id="I")] + CHANNELS[1:]
        worse, better = segment("a", 0, 30, lead_i), segment("b", 0, 30)
        self.assertEqual(plan.slot_owners([worse, better], 0, 30_000_000).tolist(), [1, 1, 1])

    def test_header_clips_allow_one_gap(self):
        owners = np.zeros(SLOTS_PER_DAY, dtype=np.int32)
        owners[5] = -1
        owners[SLOTS_PER_CLIP + 1:SLOTS_PER_CLIP + 3] = -1
        self.assertEqual(plan.header_clips(owners, np.array([True])), 479)

    def test_runs_skip_gaps(self):
        self.assertEqual(plan.runs_of(np.array([-1, 0, 0, -1, 1])), [[1, 3, 0], [4, 5, 1]])

    def test_fate_uses_the_next_24_hours(self):
        self.assertEqual(plan.day_fate(0, 0, 3 * DAY_US, None), "stayed")
        self.assertEqual(plan.day_fate(0, 0, DAY_US + 5, None), "left_alive")
        self.assertEqual(plan.day_fate(0, 0, 3 * DAY_US, 2 * DAY_US), "died")

    def test_split_is_stable_and_70_15_15(self):
        self.assertEqual(common.assign_split(123), common.assign_split("123"))
        self.assertEqual([common.assign_split(value) for value in (4, 5, 9)], ["train", "val", "test"])
        shares = [common.assign_split(value) for value in range(20000)]
        self.assertAlmostEqual(shares.count("train") / 20000, .7, delta=.02)
        self.assertAlmostEqual(shares.count("val") / 20000, .15, delta=.02)

    def test_cut_refuses_to_write_into_source_data(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(cut, "WAVE_ROOT", Path(folder)):
            with self.assertRaises(SystemExit):
                cut.main(["--output", str(Path(folder) / "cut")])


class DayWriter(unittest.TestCase):
    def test_day_files_map_slots_to_rows_and_record_failures(self):
        def fake(reader, segment_, selected, start, left, right, slot):
            if slot == 1:
                raise ValueError("bad read")
            mask = np.array([slot != 2, False, False])
            value = np.zeros((3, 2500), dtype=np.float32)
            value[0] = slot
            ids = np.array([1 if slot != 2 else -1, -1, -1], dtype=np.int16)
            return {"signal": value, "modality_mask": mask, "channel_ids": ids, "reasons": {"ecg": ["x"] if slot == 2 else []}}

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(cut, "read_slot", fake):
            job = {"stay_id": 9, "day": 0, "split": "train", "fate": "stayed", "start_us": 0, "intime_us": 0,
                   "outtime_us": 2 * DAY_US, "runs": [[0, 4, 0]], "output": folder, "segments": {0: segment("a", 0, 40)}}
            summary = cut.process_day(job)
            signal_path, meta_path = cut.day_paths(folder, 9, 0)
            signal_ = np.load(signal_path)
            with np.load(meta_path) as meta:
                self.assertEqual(meta["status"][:5].tolist(), [READY, common.FAILED, REJECTED, READY, common.NOT_PLANNED])
                self.assertEqual(meta["row"][:4].tolist(), [0, -1, -1, 1])
                self.assertEqual(json.loads(str(meta["summary"]))["failed"], 1)
        self.assertEqual(signal_.shape, (2, 3, 2500))
        self.assertEqual(signal_[1, 0, 0], 3)
        self.assertEqual((summary["ready"], summary["rejected"], summary["failed"]), (2, 1, 1))


class OpenEnded(unittest.TestCase):
    def test_moves_the_end_and_keeps_offset_changes(self):
        record = {"starttime": "2020-07-01 08:00:00-04:00", "endtime": "2020-07-03 08:00:00-04:00", "start_us": 0, "end_us": 5}
        self.assertGreater(plan.open_ended(record)["end_us"], 10**14)
        self.assertEqual(plan.open_ended(record)["endtime"][-6:], "-04:00")
        changed = {**record, "endtime": "2020-12-01 08:00:00-05:00"}
        self.assertEqual(plan.open_ended(changed), changed)


if __name__ == "__main__":
    unittest.main()
