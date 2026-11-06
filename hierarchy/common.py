"""Paths, ICU day index and clip reading for the three training levels."""
from collections import OrderedDict
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIVATE = Path(os.environ.get("HIERARCHY_PRIVATE", HERE / "private"))
RUNS = PRIVATE / "runs"
CUT_DIR = ROOT / "waveform/icu_cut/output/cut/days"
ANCHORS = ROOT / "labels/private/anchors.pkl"
# EHR inputs of the level 3 baseline path, from labels/e2/e2.py.
E2_TABLE = ROOT / "labels/private/e2.pkl"
# Arm A, chosen on 2026-11-06 (docs/SETTING.md, "Encoder choice").
ENCODER = ROOT / "finetune/arms/runs/arm_A/selected.pt"
ORGANS = ("brain", "resp", "cv", "liver", "kidney", "hemo")
ROW_BYTES = 3 * 2500 * 4
SLOTS_PER_CLIP, CLIPS_PER_HOUR, CLIPS_PER_DAY = 18, 20, 480
READY, ALL_THREE = 1, 7
PPG, RESP = 12, 13
MIN_HOUR_CLIPS = 16

_spec = importlib.util.spec_from_file_location("icu_cut_common", ROOT / "waveform/icu_cut/common.py")
_cut = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cut)
valid_clips = _cut.valid_clips


def day_folder(stay, day):
    return CUT_DIR / str(stay)


def load_days(splits=("train", "val")):
    """One row per anchor day, never including test unless asked."""
    days = pd.read_pickle(ANCHORS)
    return days[days.split.isin(splits)].sort_values(["stay_id", "day"]).reset_index(drop=True)


class ClipReader:
    """Reads clips from day files, keeping a few day tables open per process."""

    def __init__(self, keep=64):
        self.keep, self.tables = keep, OrderedDict()

    def table(self, stay, day):
        key = stay, day
        if key not in self.tables:
            folder = day_folder(stay, day)
            with np.load(folder / f"day_{day:03d}_meta.npz", allow_pickle=False) as meta:
                self.tables[key] = (folder / f"day_{day:03d}_signal.npy", meta["status"], meta["bits"], meta["row"], meta["lead"])
            while len(self.tables) > self.keep:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def read(self, stay, day, clip):
        """signal[W,3,2500], ids[W,3] and slot positions[W]; a bad slot is dropped, others keep their place."""
        path, status, bits, rows, leads = self.table(stay, day)
        left = clip * SLOTS_PER_CLIP
        slots = np.arange(left, left + SLOTS_PER_CLIP)
        keep = (status[slots] == READY) & (bits[slots] == ALL_THREE)
        chosen, positions = slots[keep], np.flatnonzero(keep)
        signal = np.empty((len(chosen), 3, 2500), dtype=np.float32)
        with path.open("rb") as stream:
            np.lib.format.read_magic(stream)
            np.lib.format.read_array_header_1_0(stream)
            base = stream.tell()
            for index, row in enumerate(rows[chosen]):
                stream.seek(base + int(row) * ROW_BYTES)
                signal[index] = np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").reshape(3, 2500)
        lead = max(int(leads[chosen[0]]), 0)
        ids = np.tile(np.array([lead, PPG, RESP], dtype=np.int64), (len(chosen), 1))
        return signal, ids, positions.astype(np.int64)


def load_arm(path=ENCODER):
    """The chosen fine-tuned 10-second encoder, frozen."""
    from long_model import CSFMEncoder
    bundle = torch.load(path, map_location="cpu", weights_only=False)["encoder"]
    return CSFMEncoder.from_export(bundle).eval()
