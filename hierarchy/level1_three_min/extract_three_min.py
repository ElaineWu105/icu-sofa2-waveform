"""Encode every valid clip of each anchor day into day files of 480 clips. Stopped runs resume by day.

Per day, `<stay>_<day>_windows.npy` holds the 18 window features z_i of every clip [480, 18, 768] in fp16,
zero where a slot or clip is missing; `<stay>_<day>.npz` holds the clip features z-bar [480, 768], the clip mask,
the window mask [480, 18], and for control M the mean CLS of the frozen 10-second encoder over the first
present window of each valid clip. The npz is written last, so its presence marks a finished day.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from clip_model import ClipEncoder, make_config
from clip_model import WINDOWS
from common import CLIPS_PER_DAY, ENCODER, PRIVATE, RUNS, ClipReader, load_arm
import distributed_cpu as transport

OUTPUT = PRIVATE / "three_min"


class DayClips(Dataset):
    def __init__(self, stay, day, clips):
        self.stay, self.day, self.clips, self.reader = stay, day, clips, ClipReader(keep=1)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        values, ids, positions = self.reader.read(self.stay, self.day, int(self.clips[index]))
        return torch.from_numpy(values), torch.from_numpy(ids), torch.from_numpy(positions), int(self.clips[index])


def batches(windows, size):
    """Index batches that hold one window count, since clips with a dropped slot are shorter."""
    for count in (18, 17):
        rows = np.flatnonzero(windows == count)
        for left in range(0, len(rows), size):
            yield rows[left:left + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=str(RUNS / "three_min/best_encoder.pt"))
    parser.add_argument("--checkpoint", default=str(ENCODER))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-days", type=int, help="Stop after this many days, for a short run")
    args = parser.parse_args()
    rank, world, device = transport.setup_distributed(args.device)
    OUTPUT.mkdir(mode=0o700, parents=True, exist_ok=True)
    days = pd.read_pickle(PRIVATE / "days.pkl")
    index = np.load(PRIVATE / "clips.npz")
    starts = np.searchsorted(index["day"], np.arange(len(days) + 1))
    core = ClipEncoder(load_arm(args.checkpoint), make_config())
    core.load_adapter_state_dict(torch.load(args.adapter, map_location="cpu", weights_only=False)["adapter"])
    core = core.to(device).eval()
    done = 0
    for position in range(rank, len(days), world):
        stay, day = int(days.stay_id[position]), int(days.day[position])
        path = OUTPUT / f"{stay}_{day:03d}.npz"
        if path.is_file():
            continue
        clips = index["clip"][starts[position]:starts[position + 1]]
        windows = index["windows"][starts[position]:starts[position + 1]]
        features = np.zeros((CLIPS_PER_DAY, 768), dtype=np.float16)
        window_features = np.zeros((CLIPS_PER_DAY, WINDOWS, 768), dtype=np.float16)
        window_mask = np.zeros((CLIPS_PER_DAY, WINDOWS), dtype=bool)
        ten_second, counted = np.zeros(768, dtype=np.float64), 0
        data = DayClips(stay, day, clips)
        options = dict(batch_sampler=list(batches(windows, args.batch_size)), num_workers=args.workers, pin_memory=True)
        with torch.inference_mode():
            for values, ids, positions, numbers in DataLoader(data, **options):
                values, ids, positions = values.to(device), ids.to(device), positions.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    output = core(values, ids, positions)
                    cls = core.encoder(values[:, 0], ids[:, 0])
                rows, slots = numbers.numpy()[:, None], positions.cpu().numpy()
                features[numbers.numpy()] = output["embedding"].float().cpu().numpy().astype(np.float16)
                window_features[rows, slots] = output["local"].float().cpu().numpy().astype(np.float16)
                window_mask[rows, slots] = True
                ten_second += cls.float().sum(0).cpu().numpy()
                counted += len(cls)
        if not (np.isfinite(features).all() and np.isfinite(window_features).all() and np.isfinite(ten_second).all()):
            raise RuntimeError(f"Nonfinite features on stay {stay} day {day}.")
        large = OUTPUT / f"{stay}_{day:03d}_windows.npy"
        temporary = Path(str(large) + ".tmp.npy")
        np.save(temporary, window_features)
        temporary.replace(large)
        temporary = Path(str(path) + ".tmp.npz")
        np.savez(temporary, features=features, mask=np.isin(np.arange(CLIPS_PER_DAY), clips), window_mask=window_mask,
                 ten_second=(ten_second / max(counted, 1)).astype(np.float32), ten_second_windows=counted)
        temporary.replace(path)
        done += 1
        if done % 20 == 0 and rank == 0:
            print(f"rank 0 finished {done} days", flush=True)
        if args.max_days and done >= args.max_days:
            break
    print(f"rank {rank} done", flush=True)


if __name__ == "__main__":
    main()
