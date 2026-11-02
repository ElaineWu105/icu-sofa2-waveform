"""Exact-equality check: re-cut existing 10.14 windows through cut.read_slot.

Samples ready three-channel windows from the 10.14 index and compares signal,
modality mask and channel IDs with the arrays saved there. Reads only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sqlite3

import numpy as np

from common import HERE, ROOT
import cut
from source import SnapshotSource, _choose_channels

OLD = ROOT / "waveform/windows_10s/output/full"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=HERE / "validation_equivalence.json")
    args = parser.parse_args()
    db = sqlite3.connect(f"file:{OLD / 'index.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    highest = db.execute("SELECT MAX(rowid) FROM windows").fetchone()[0]
    rng = random.Random(args.seed)
    picks = []
    while len(picks) < args.windows:
        row = db.execute("""SELECT window_id, segment_id, start_us, shard_id, shard_row FROM windows
                            WHERE rowid=? AND status='ready' AND mask_bits=7""", (rng.randrange(1, highest + 1),)).fetchone()
        if row:
            picks.append(row)
    reader = object.__new__(SnapshotSource)
    source = SnapshotSource.__new__(SnapshotSource)
    source._segments = {}
    equal, unequal = 0, []
    for row in picks:
        segment_row = db.execute("SELECT * FROM segments WHERE segment_id=?", (row["segment_id"],)).fetchone()
        segment = source._segment(segment_row)
        selected = _choose_channels(segment)
        day_start = row["start_us"]
        result = cut.read_slot(reader, segment, selected, day_start, day_start, day_start + 10_000_000, 0)
        specs = json.loads(db.execute("SELECT arrays_json FROM shards WHERE shard_id=?", (row["shard_id"],)).fetchone()[0])
        same = all(np.array_equal(result[key], np.load(OLD / specs[key]["file"], mmap_mode="r")[row["shard_row"]])
                   for key in ("signal", "modality_mask", "channel_ids"))
        equal += same
        if not same:
            unequal.append(row["window_id"])
    value = {"checked": len(picks), "byte_equal": int(equal), "unequal": len(unequal),
             "compared": ["signal", "modality_mask", "channel_ids"], "source": str(OLD)}
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))
    return 0 if not unequal else 1


if __name__ == "__main__":
    raise SystemExit(main())
