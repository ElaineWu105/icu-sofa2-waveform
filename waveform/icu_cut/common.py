"""Shared constants and reused 10.13/10.14 signal code for the ICU-grid cut.

Reads and cleans waveforms with the unmodified 10.13 inventory and 10.14
window code. Only the time grid, cohort and on-disk layout are new.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPROCESS_DIR = ROOT / "waveform/windows_10s"
INVENTORY_DIR = ROOT / "waveform/native_index"
for _path in (INVENTORY_DIR, PREPROCESS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

WAVE_ROOT = Path(os.environ["MIMIC_WAVEFORM_DIR"])
EHR_ROOT = Path(os.environ["MIMIC_IV_DIR"])

SECOND_US = 1_000_000
WINDOW_US = 10 * SECOND_US
DAY_US = 86_400 * SECOND_US
SLOTS_PER_CLIP = 18
CLIPS_PER_DAY = 480
SLOTS_PER_DAY = SLOTS_PER_CLIP * CLIPS_PER_DAY
MIN_VALID_CLIPS = 384
MAX_BAD_SLOTS = 1
ALL_THREE = 0b111

NOT_PLANNED, READY, REJECTED, FAILED = 0, 1, 2, 3


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def assign_split(subject_id):
    """Stable 70/15/15 by patient, the same on every run and machine."""
    bucket = int(hashlib.sha256(str(int(subject_id)).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def valid_clips(status, modality_bits, segment_index, lead_index, max_bad=MAX_BAD_SLOTS):
    """Boolean[480]: at most max_bad bad slots; good slots share one segment and lead."""
    good = (status == READY) & (modality_bits == ALL_THREE)
    answer = np.zeros(CLIPS_PER_DAY, dtype=bool)
    for clip in range(CLIPS_PER_DAY):
        left, right = clip * SLOTS_PER_CLIP, (clip + 1) * SLOTS_PER_CLIP
        keep = good[left:right]
        if keep.sum() < SLOTS_PER_CLIP - max_bad:
            continue
        segments, leads = segment_index[left:right][keep], lead_index[left:right][keep]
        answer[clip] = segments.min() == segments.max() and leads.min() == leads.max()
    return answer
