"""Index valid three-minute clips on every anchor day. Train and validation only unless --test is given."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from common import ALL_THREE, CLIPS_PER_DAY, PRIVATE, READY, SLOTS_PER_CLIP, day_folder, load_days, valid_clips


def day_clips(job):
    position, stay, day = job
    with np.load(day_folder(stay, day) / f"day_{day:03d}_meta.npz", allow_pickle=False) as meta:
        status, bits, segments, leads = meta["status"], meta["bits"], meta["segment_index"], meta["lead"]
    clips = np.flatnonzero(valid_clips(status, bits, segments, leads))
    good = (status == READY) & (bits == ALL_THREE)
    windows = good.reshape(CLIPS_PER_DAY, SLOTS_PER_CLIP).sum(1)[clips]
    return position, clips.astype(np.int16), windows.astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--test", action="store_true", help="Also index test days, for the one final run")
    args = parser.parse_args()
    PRIVATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    days = load_days(("train", "val", "test") if args.test else ("train", "val"))
    jobs = [(position, row.stay_id, row.day) for position, row in enumerate(days.itertuples())]
    day, clip, windows = [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for position, clips, counts in pool.map(day_clips, jobs, chunksize=32):
            day.append(np.full(len(clips), position, dtype=np.int32))
            clip.append(clips)
            windows.append(counts)
    np.savez(PRIVATE / "clips.npz", day=np.concatenate(day), clip=np.concatenate(clip), windows=np.concatenate(windows))
    days.to_pickle(PRIVATE / "days.pkl")
    split = days.split.to_numpy()[np.concatenate(day)]
    print(json.dumps({"days": days.split.value_counts().to_dict(), "clips": {name: int((split == name).sum()) for name in np.unique(split)},
                      "clips_with_one_bad_slot": int((np.concatenate(windows) == 17).sum())}), flush=True)


if __name__ == "__main__":
    main()
