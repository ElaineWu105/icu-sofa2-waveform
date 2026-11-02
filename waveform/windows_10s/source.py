"""Read-only compact-index planning and exact native 10-second WFDB slices.

Only pairs, segments and verified pair_segments are queried. No old waveform
groups/chunks, ready-cache state, source modification, filtering or normalization
is used. read_window has no SQLite access and can run in worker threads.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

import numpy as np
import wfdb

WINDOW_US = 10_000_000
RATES = {"ecg": 250.0, "ppg": 125.0, "resp": 62.5}
COUNTS = {"ecg": 2500, "ppg": 1250, "resp": 625}
LEAD_ORDER = ("II", "I", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
_LEADS = {lead.upper(): lead for lead in LEAD_ORDER}


class SourceReadError(ValueError):
    """An expected source read/clock/count failure; the worker should record it."""


def _lead(channel):
    name = channel.get("raw_name", channel.get("lead_id", ""))
    token = re.sub(r"\s+", "", str(name)).upper()
    return _LEADS.get(token)


def _channel_index(channel):
    return int(channel.get("channel_index", channel.get("index", -1)))


def _choose_channels(segment):
    chosen = {}
    for modality, expected_fs in RATES.items():
        candidates = []
        for original in segment["channels"]:
            declared = original.get("modality")
            if declared == "pleth":
                declared = "ppg"
            if declared != modality:
                continue
            try:
                fs = float(original["sample_fs"])
                index = _channel_index(original)
                spf = int(original.get("samples_per_frame", original.get("spf", 1)))
            except (KeyError, ValueError, TypeError):
                continue
            if index < 0 or spf < 1 or not math.isclose(fs, expected_fs, rel_tol=0., abs_tol=1e-7):
                continue
            if not math.isclose(fs, float(segment["frame_fs"]) * spf, rel_tol=0., abs_tol=1e-7):
                continue
            lead = _lead(original) if modality == "ecg" else None
            if modality == "ecg" and lead is None:
                # V, MCL, ECG1 and other anonymous acquisition names are not
                # relabelled as a diagnostic lead merely to satisfy the model.
                continue
            channel = dict(original, modality=modality, channel_index=index,
                           samples_per_frame=spf, sample_fs=fs)
            if modality == "ecg":
                channel["canonical_lead_id"] = lead
            rank = LEAD_ORDER.index(lead) if lead is not None else 0
            candidates.append((rank, index, channel))
        if candidates:
            chosen[modality] = min(candidates, key=lambda row: (row[0], row[1]))[2]
    return chosen


def _segment_right(segment):
    frequency = Fraction(str(segment["frame_fs"]))
    duration = Fraction(int(segment["frames"]) * 1_000_000, 1) / frequency
    if duration.denominator != 1:
        raise SourceReadError("Segment frame duration is not an integer microsecond.")
    return min(int(segment["record_end_us"]), int(segment["origin_us"]) + int(duration))


class SnapshotSource:
    def __init__(self, source_dir, pair_limit=None):
        path = Path(source_dir).resolve()
        if path.is_dir():
            choices = [path / "sourceindex.sqlite", path / "index.sqlite", path / "source_index.sqlite"]
            path = next((candidate for candidate in choices if candidate.is_file()), choices[0])
        if not path.is_file():
            raise FileNotFoundError("Missing compact waveform index: " + str(path))
        if pair_limit is not None and (isinstance(pair_limit, bool) or int(pair_limit) < 1):
            raise ValueError("pair_limit must be a positive diagnostic limit.")
        self.path = path
        self.connection = sqlite3.connect("file:" + quote(str(path), safe="/") + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        self.stats = Counter()
        self._segments = {}
        try:
            query = "SELECT pair_id,subject_id,hadm_id,split,t1_us,t2_us,start_us FROM pairs ORDER BY rowid"
            parameters = ()
            if pair_limit is not None:
                query += " LIMIT ?"
                parameters = (int(pair_limit),)
            self.pairs = [dict(row) for row in self.connection.execute(query, parameters)]
            self._pairs_by_id = {row["pair_id"]: row for row in self.pairs}
            if len(self._pairs_by_id) != len(self.pairs):
                raise ValueError("Duplicate pairs in source snapshot.")
            patients = {}
            for pair in self.pairs:
                if pair["split"] not in {"train", "val", "test"}:
                    raise ValueError("Invalid patient split in source snapshot.")
                subject = str(pair["subject_id"])
                if subject in patients and patients[subject] != pair["split"]:
                    raise ValueError("A source patient occurs in multiple splits.")
                patients[subject] = pair["split"]
                if int(pair["t1_us"]) >= int(pair["t2_us"]) or int(pair["start_us"]) >= int(pair["t2_us"]):
                    raise ValueError("Invalid pair observation interval.")
            self.pair_limit = pair_limit
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _segment(self, row):
        segment_id = str(row["segment_id"])
        if segment_id not in self._segments:
            segment = {name: row[name] for name in (
                "segment_id", "record_id", "subject_id", "path", "origin_us", "frames", "frame_fs", "record_end_us")}
            metadata = json.loads(row["metadata_json"])
            segment["channels"] = metadata.get("channels", [])
            segment["source_metadata"] = {key: value for key, value in metadata.items() if key != "channels"}
            if not isinstance(segment["channels"], list) or int(segment["frames"]) <= 0 or float(segment["frame_fs"]) <= 0:
                raise ValueError("Invalid native metadata in source snapshot.")
            self._segments[segment_id] = segment
        return self._segments[segment_id]

    def iter_jobs(self, pair):
        """Plan one full segment per logical slot without reading waveform data.

        Logical windows are aligned to this pair's start_us, with 10s stride.
        Candidates are never stitched across segments, gaps, or admission spans.
        """
        pair_id = pair["pair_id"] if isinstance(pair, dict) else str(pair)
        expected = self._pairs_by_id.get(pair_id)
        if expected is None or (isinstance(pair, dict) and any(pair.get(key) != value for key, value in expected.items())):
            raise ValueError("Pair identity/interval does not match this snapshot selection.")
        pair = expected
        start, cutoff = int(pair["start_us"]), int(pair["t2_us"])
        candidates = {}
        query = """SELECT s.*,ps.left_us,ps.right_us FROM pair_segments ps
                   JOIN segments s ON s.segment_id=ps.segment_id
                   WHERE ps.pair_id=? AND ps.admission_status='verified'
                   ORDER BY ps.left_us,s.segment_id"""
        for row in self.connection.execute(query, (pair_id,)):
            segment = self._segment(row)
            if str(segment["subject_id"]) != str(pair["subject_id"]):
                raise ValueError("Source segment patient differs from its pair.")
            selected = _choose_channels(segment)
            if not selected:
                self.stats["segment_span_without_supported_channels"] += 1
                continue
            left = max(start, int(row["left_us"]), int(segment["origin_us"]))
            right = min(cutoff, int(row["right_us"]), _segment_right(segment))
            first = max(0, (left - start + WINDOW_US - 1) // WINDOW_US)
            last = (right - start) // WINDOW_US - 1
            ecg = selected.get("ecg")
            rank = LEAD_ORDER.index(ecg["canonical_lead_id"]) if ecg else len(LEAD_ORDER)
            priority = (rank, -len(selected), str(segment["segment_id"]))
            for slot in range(first, last + 1):
                window_start = start + slot * WINDOW_US
                window_end = window_start + WINDOW_US
                if not left <= window_start < window_end <= right:
                    raise AssertionError("Planned window is not fully within one verified span.")
                if slot not in candidates or priority < candidates[slot][0]:
                    job = {"segment": segment, "selected_channels": selected, "pair_id": pair_id,
                           "slot_index": slot, "start_us": window_start, "end_us": window_end,
                           "pair_start_us": start, "cutoff_us": cutoff, "subject_id": str(pair["subject_id"]),
                           "hadm_id": str(pair["hadm_id"]), "split": pair["split"],
                           "verified_left_us": int(row["left_us"]), "verified_right_us": int(row["right_us"])}
                    candidates[slot] = (priority, job)
        self.stats["pairs_planned"] += 1
        if not candidates:
            self.stats["pairs_without_complete_windows"] += 1
        self.stats["planned_windows"] += len(candidates)
        for slot in sorted(candidates):
            yield candidates[slot][1]

    def read_window(self, job):
        """Decode expanded native samples, then discard all enclosing-frame excess.

        This method is SQLite-free. Decode/count/calibration errors raise
        SourceReadError so the orchestrator can record the failed window.
        Nonfinite samples within a complete slice are retained for downstream QC.
        """
        try:
            segment, selected = job["segment"], job["selected_channels"]
            left, right = int(job["start_us"]), int(job["end_us"])
            origin, frames = int(segment["origin_us"]), int(segment["frames"])
            if right - left != WINDOW_US or not int(job["pair_start_us"]) <= left < right <= int(job["cutoff_us"]):
                raise SourceReadError("Window exceeds its pair cutoff or is not 10 seconds.")
            if not int(job["verified_left_us"]) <= left < right <= int(job["verified_right_us"]):
                raise SourceReadError("Window crosses its verified admission span.")
            if not origin <= left < right <= _segment_right(segment):
                raise SourceReadError("Window crosses its native segment.")
            frame_period = Fraction(1_000_000, 1) / Fraction(str(segment["frame_fs"]))
            if frame_period.denominator != 1:
                raise SourceReadError("Native frame period is not an integer microsecond.")
            frame_period = int(frame_period)
            first_frame = max(0, (left - origin) // frame_period - 1)
            last_frame = min(frames, (right - origin + frame_period - 1) // frame_period + 1)
            modalities = [modality for modality in RATES if modality in selected]
            indices = [_channel_index(selected[modality]) for modality in modalities]
            if not indices or len(indices) != len(set(indices)):
                raise SourceReadError("Missing or duplicate native channel selection.")
            decoded = wfdb.rdrecord(str(segment["path"]), sampfrom=first_frame, sampto=last_frame,
                                    channels=indices, physical=True, smooth_frames=False)
            if decoded.e_p_signal is None or len(decoded.e_p_signal) != len(modalities):
                raise SourceReadError("WFDB did not return expanded physical channels.")
            result = {modality: None for modality in RATES}
            offsets, last_offsets, sample_masks = {}, {}, {}
            channel_details = {}
            for row, modality in enumerate(modalities):
                channel = selected[modality]
                spf = int(channel["samples_per_frame"])
                fs = float(channel["sample_fs"])
                if not math.isclose(fs, RATES[modality], rel_tol=0., abs_tol=1e-7):
                    raise SourceReadError("Unsupported native sample rate.")
                period = Fraction(1_000_000, 1) / Fraction(str(fs))
                if period.denominator != 1:
                    raise SourceReadError("Native sample period is not an integer microsecond.")
                values = np.asarray(decoded.e_p_signal[row], dtype=np.float64)
                if values.ndim != 1 or len(values) != (last_frame - first_frame) * spf:
                    raise SourceReadError("Native expanded sample count differs from WFDB metadata.")
                times = origin + (first_frame * spf + np.arange(len(values), dtype=np.int64)) * int(period)
                keep = (times >= left) & (times < right)
                values, times = values[keep], times[keep]
                if len(values) != COUNTS[modality]:
                    raise SourceReadError("Ten-second slice does not contain the complete native sample count.")
                factor = float(channel.get("physical_scale", 1.))
                if not math.isfinite(factor) or factor <= 0:
                    raise SourceReadError("Invalid physical unit scale.")
                if modality == "ecg" and _lead(channel) is None:
                    raise SourceReadError("Unknown ECG lead cannot be relabelled as a diagnostic lead.")
                with np.errstate(over="ignore", invalid="ignore"):
                    result[modality] = (values * factor).astype(np.float32)
                offsets[modality] = int(times[0] - left)
                last_offsets[modality] = int(times[-1] - left)
                sample_masks[modality] = np.isfinite(result[modality])
                channel_details[modality] = dict(channel)
            result["lead_id"] = selected["ecg"]["canonical_lead_id"] if "ecg" in selected else None
            result["modality_available"] = {modality: result[modality] is not None for modality in RATES}
            result["mask_by_modality"] = sample_masks
            result["sample_valid"] = sample_masks
            result["metadata"] = {
                "pair_id": job["pair_id"], "slot_index": int(job["slot_index"]),
                "record_id": str(segment["record_id"]), "segment_id": str(segment["segment_id"]),
                "source_path": str(segment["path"]), "start_us": left, "end_us": right,
                "cutoff_us": int(job["cutoff_us"]), "lead_id": result["lead_id"],
                "native_first_sample_offset_us": offsets, "native_last_sample_offset_us": last_offsets,
                "channels": channel_details, "admission_status": "verified",
                "decoder_frame_start": first_frame, "decoder_frame_end": last_frame,
                "boundary_policy": "native samples strictly in [start_us,end_us); no cross-segment stitching"}
            return result
        except SourceReadError:
            raise
        except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
            raise SourceReadError("WFDB window read failed: " + str(error)) from error

    def iter_windows(self, pair):
        for job in self.iter_jobs(pair):
            yield self.read_window(job)
