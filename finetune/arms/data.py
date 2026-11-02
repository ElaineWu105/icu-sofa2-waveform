"""Hourly window sampling from cut ICU days, optionally with charted text."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset

import common  # noqa: F401  (adds the waveform_cut path)
from common import CUT_DIR, PLAN_PATH, TEXT_DIR

READY = 1
SLOTS_PER_HOUR = 360
TEXT_LENGTH = 64
ROW_BYTES = 3 * 2500 * 4


def planned_days(plan_path, split):
    db = sqlite3.connect(f"file:{plan_path}?mode=ro", uri=True)
    rows = db.execute("SELECT stay_id, day, subject_id FROM days WHERE split=? ORDER BY stay_id, day", (split,)).fetchall()
    db.close()
    return rows


def day_files(cut_dir, stay_id, day):
    folder = Path(cut_dir) / "days" / str(stay_id)
    return folder / f"day_{day:03d}_signal.npy", folder / f"day_{day:03d}_meta.npz"


class HourlyWindows(Dataset):
    """One window per ICU hour, drawn again each epoch.

    Arm A uses any ready window. Arm B uses windows holding a charted snapshot.
    Every patient comes from one split; text is never returned for arm A.
    """

    def __init__(self, split, arm="A", cut_dir=CUT_DIR, plan_path=PLAN_PATH, text_dir=TEXT_DIR,
                 seed=42, fixed=False):
        if arm not in ("A", "B"):
            raise ValueError("arm must be A or B")
        self.split, self.arm, self.seed, self.fixed = split, arm, seed, fixed
        self.cut_dir = Path(cut_dir)
        self.days, self.subjects = [], []
        starts, slots, rows, bits, leads = [0], [], [], [], []
        hour_day, hour_start, hour_stop = [], [], []
        text_index = self._load_text(text_dir) if arm == "B" else None
        for stay_id, day, subject_id in planned_days(plan_path, split):
            _, meta_path = day_files(self.cut_dir, stay_id, day)
            if not meta_path.is_file():
                continue
            with np.load(meta_path, allow_pickle=False) as meta:
                status, row, bit, lead = meta["status"], meta["row"], meta["bits"], meta["lead"]
            usable = np.flatnonzero(status == READY)
            if arm == "B":
                usable = np.intersect1d(usable, np.fromiter(text_index.get((stay_id, day), {}), dtype=np.int64))
            position = len(self.days)
            self.days.append((stay_id, day))
            self.subjects.append(subject_id)
            for hour in range(24):
                chosen = usable[(usable >= hour * SLOTS_PER_HOUR) & (usable < (hour + 1) * SLOTS_PER_HOUR)]
                if not len(chosen):
                    continue
                hour_day.append(position)
                hour_start.append(len(slots))
                slots.extend(chosen.tolist())
                rows.extend(row[chosen].tolist())
                bits.extend(bit[chosen].tolist())
                leads.extend(lead[chosen].tolist())
                hour_stop.append(len(slots))
        if not hour_day:
            raise RuntimeError(f"No usable {split} windows for arm {arm}.")
        self.hour_day = np.array(hour_day, dtype=np.int32)
        self.hour_start = np.array(hour_start, dtype=np.int64)
        self.hour_stop = np.array(hour_stop, dtype=np.int64)
        self.slot = np.array(slots, dtype=np.int16)
        self.row = np.array(rows, dtype=np.int16)
        self.bits = np.array(bits, dtype=np.uint8)
        self.lead = np.array(leads, dtype=np.int16)
        self.text_index = text_index
        self.set_epoch(0)

    def _load_text(self, text_dir):
        stored = np.load(Path(text_dir) / "snapshots.npz", allow_pickle=False)
        self.tokens = stored["tokens"]
        self.attention = stored["attention"]
        self.special = stored["special"]
        index = {}
        for number, (stay_id, day, slot) in enumerate(zip(stored["stay_id"], stored["day"], stored["slot"])):
            index.setdefault((int(stay_id), int(day)), {}).setdefault(int(slot), number)
        return index

    def set_epoch(self, epoch):
        seed = self.seed if self.fixed else self.seed + 1000 + epoch
        rng = np.random.default_rng(seed)
        span = self.hour_stop - self.hour_start
        self.pick = self.hour_start + (rng.random(len(span)) * span).astype(np.int64)

    def __len__(self):
        return len(self.pick)

    def _signal(self, stay_id, day, row):
        """Read exactly one row; memory-mapping random rows wastes disk bandwidth."""
        with day_files(self.cut_dir, stay_id, day)[0].open("rb") as stream:
            np.lib.format.read_magic(stream)
            np.lib.format.read_array_header_1_0(stream)
            stream.seek(stream.tell() + int(row) * ROW_BYTES)
            return np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").reshape(3, 2500).copy()

    def __getitem__(self, index):
        pick = self.pick[index]
        stay_id, day = self.days[self.hour_day[index]]
        signal = self._signal(stay_id, day, self.row[pick])
        bits = int(self.bits[pick])
        mask = np.array([bool(bits & (1 << position)) for position in range(3)])
        channels = np.array([max(int(self.lead[pick]), 0), 12, 13], dtype=np.int64)
        tokens = np.zeros(TEXT_LENGTH, dtype=np.int64)
        attention = np.zeros(TEXT_LENGTH, dtype=bool)
        special = np.ones(TEXT_LENGTH, dtype=bool)
        if self.text_index is not None:
            number = self.text_index[(stay_id, day)][int(self.slot[pick])]
            tokens = self.tokens[number].astype(np.int64)
            attention = self.attention[number].astype(bool)
            special = self.special[number].astype(bool)
        return {"signal": torch.from_numpy(signal), "channel_ids": torch.from_numpy(channels),
                "modality_mask": torch.from_numpy(mask), "input_ids": torch.from_numpy(tokens),
                "attention_mask": torch.from_numpy(attention), "special_tokens_mask": torch.from_numpy(special)}
