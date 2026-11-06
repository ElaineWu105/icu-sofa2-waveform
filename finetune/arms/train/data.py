"""Hourly window sampling from cut ICU days, with text for arm B when a record is near.

See finetune/README.md, sections 3.1 and 3.5 to 3.9.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset

import common  # noqa: F401  (adds the waveform_cut path)
from common import CUT_DIR, PLAN_PATH, TEXT_DIR

READY = 1
SLOTS_PER_HOUR = 360
SLOT_US = 10_000_000
TEXT_LENGTH = 64
ROW_BYTES = 3 * 2500 * 4
MINUTE_US, HOUR_US = 60_000_000, 3_600_000_000
# Allowed time between a window's centre and a text record, by source in prepare_text.SOURCES order.
TOLERANCE_US = np.array([15 * MINUTE_US, 15 * MINUTE_US, 12 * HOUR_US, 24 * HOUR_US], dtype=np.int64)


def selection_patient(subject_id):
    """About 10% of patients, used among train patients only to choose the checkpoint."""
    return int(hashlib.sha256(f"selection-{int(subject_id)}".encode()).hexdigest()[:8], 16) % 10 == 0


def planned_days(plan_path, split):
    db = sqlite3.connect(f"file:{plan_path}?mode=ro", uri=True)
    rows = db.execute("SELECT stay_id, day, subject_id, start_us FROM days WHERE split=? ORDER BY stay_id, day",
                      (split,)).fetchall()
    db.close()
    return rows


def day_files(cut_dir, stay_id, day):
    folder = Path(cut_dir) / "days" / str(stay_id)
    return folder / f"day_{day:03d}_signal.npy", folder / f"day_{day:03d}_meta.npz"


def read_row(cut_dir, stay_id, day, row):
    """Read exactly one window; memory-mapping random rows wastes disk bandwidth."""
    with day_files(cut_dir, stay_id, day)[0].open("rb") as stream:
        np.lib.format.read_magic(stream)
        np.lib.format.read_array_header_1_0(stream)
        stream.seek(stream.tell() + int(row) * ROW_BYTES)
        return np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").reshape(3, 2500).copy()


class TextRecords:
    """Text records by patient and source, sorted by time."""

    def __init__(self, text_dir):
        stored = np.load(Path(text_dir) / "records.npz", allow_pickle=False)
        self.tokens, self.attention, self.special = stored["tokens"], stored["attention"], stored["special"]
        # Value spans of charted text; the arm B text set predates them and has none.
        self.value = stored["value"] if "value" in stored.files else np.zeros_like(self.tokens, dtype=np.int8)
        self.index = {}
        subject, time, source = stored["subject_id"], stored["time_us"], stored["source"]
        order = np.lexsort((time, source, subject))
        bounds = np.flatnonzero(np.diff(subject[order]) | np.diff(source[order])) + 1
        for part in np.split(order, bounds):
            if len(part):
                self.index[(int(subject[part[0]]), int(source[part[0]]))] = (time[part], part)

    def candidates(self, subject_id, time_us):
        """The nearest record of each source within its allowed time, as record numbers."""
        found = []
        for source, tolerance in enumerate(TOLERANCE_US):
            entry = self.index.get((int(subject_id), source))
            if entry is None:
                continue
            times, numbers = entry
            place = np.searchsorted(times, time_us)
            near = [i for i in (place - 1, place) if 0 <= i < len(times)]
            best = min(near, key=lambda i: abs(int(times[i]) - time_us))
            if abs(int(times[best]) - time_us) <= tolerance:
                found.append(int(numbers[best]))
        return found


class HourlyWindows(Dataset):
    """One ready window per ICU hour, drawn again each epoch.

    Both arms use the same windows. Arm B adds the text of one source, drawn at random
    among those with a record near the window, and has no text otherwise (CSFM Eq. 1).
    `selection` keeps ("only") or drops ("exclude") the patients that choose the checkpoint.
    `value_spans` adds each text token's value span number, for masking values only (arm B3).
    """

    def __init__(self, split, arm="A", cut_dir=CUT_DIR, plan_path=PLAN_PATH, text_dir=TEXT_DIR,
                 seed=42, fixed=False, selection=None, value_spans=False):
        if arm not in ("A", "B"):
            raise ValueError("arm must be A or B")
        if selection not in (None, "only", "exclude"):
            raise ValueError("selection must be None, 'only' or 'exclude'")
        self.split, self.arm, self.seed, self.fixed, self.value_spans = split, arm, seed, fixed, value_spans
        self.cut_dir = Path(cut_dir)
        self.days, self.subjects, self.starts = [], [], []
        slots, rows, bits, leads = [], [], [], []
        hour_day, hour_start, hour_stop = [], [], []
        for stay_id, day, subject_id, start_us in planned_days(plan_path, split):
            if selection is not None and selection_patient(subject_id) != (selection == "only"):
                continue
            _, meta_path = day_files(self.cut_dir, stay_id, day)
            if not meta_path.is_file():
                continue
            with np.load(meta_path, allow_pickle=False) as meta:
                status, row, bit, lead = meta["status"], meta["row"], meta["bits"], meta["lead"]
            usable = np.flatnonzero(status == READY)
            position = len(self.days)
            self.days.append((stay_id, day))
            self.subjects.append(subject_id)
            self.starts.append(start_us)
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
            raise RuntimeError(f"No usable {split} windows.")
        self.hour_day = np.array(hour_day, dtype=np.int32)
        self.hour_start = np.array(hour_start, dtype=np.int64)
        self.hour_stop = np.array(hour_stop, dtype=np.int64)
        self.slot = np.array(slots, dtype=np.int16)
        self.row = np.array(rows, dtype=np.int16)
        self.bits = np.array(bits, dtype=np.uint8)
        self.lead = np.array(leads, dtype=np.int16)
        self.text = TextRecords(text_dir) if arm == "B" else None
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.epoch = 0 if self.fixed else epoch
        rng = np.random.default_rng(self.seed if self.fixed else self.seed + 1000 + epoch)
        span = self.hour_stop - self.hour_start
        self.pick = self.hour_start + (rng.random(len(span)) * span).astype(np.int64)

    def __len__(self):
        return len(self.pick)

    def text_for(self, index):
        """Record number of the text given to this window, or None."""
        if self.text is None:
            return None
        position = self.hour_day[index]
        centre = self.starts[position] + int(self.slot[self.pick[index]]) * SLOT_US + SLOT_US // 2
        found = self.text.candidates(self.subjects[position], centre)
        if not found:
            return None
        return found[np.random.default_rng([self.seed, self.epoch, int(index)]).integers(len(found))]

    def __getitem__(self, index):
        pick = self.pick[index]
        stay_id, day = self.days[self.hour_day[index]]
        signal = read_row(self.cut_dir, stay_id, day, self.row[pick])
        bits = int(self.bits[pick])
        mask = np.array([bool(bits & (1 << position)) for position in range(3)])
        channels = np.array([max(int(self.lead[pick]), 0), 12, 13], dtype=np.int64)
        tokens = np.zeros(TEXT_LENGTH, dtype=np.int64)
        attention = np.zeros(TEXT_LENGTH, dtype=bool)
        special = np.ones(TEXT_LENGTH, dtype=bool)
        value = np.zeros(TEXT_LENGTH, dtype=np.int64)
        number = self.text_for(index)
        if number is not None:
            tokens = self.text.tokens[number].astype(np.int64)
            attention = self.text.attention[number].astype(bool)
            special = self.text.special[number].astype(bool)
            value = self.text.value[number].astype(np.int64)
        item = {"signal": torch.from_numpy(signal), "channel_ids": torch.from_numpy(channels),
                "modality_mask": torch.from_numpy(mask), "input_ids": torch.from_numpy(tokens),
                "attention_mask": torch.from_numpy(attention), "special_tokens_mask": torch.from_numpy(special)}
        if self.value_spans:
            item["value_span"] = torch.from_numpy(value)
        return item
