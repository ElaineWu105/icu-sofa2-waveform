"""Encode every valid clip of each anchor day into a day file of 480 slots. Stopped runs resume by day."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from clip_model import ClipEncoder, make_config
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
        data = DayClips(stay, day, clips)
        options = dict(batch_sampler=list(batches(windows, args.batch_size)), num_workers=args.workers, pin_memory=True)
        with torch.inference_mode():
            for values, ids, positions, numbers in DataLoader(data, **options):
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    output = core(values.to(device), ids.to(device), positions.to(device))
                features[numbers.numpy()] = output["embedding"].float().cpu().numpy().astype(np.float16)
        if not np.isfinite(features).all():
            raise RuntimeError(f"Nonfinite features on stay {stay} day {day}.")
        temporary = Path(str(path) + ".tmp.npz")
        np.savez(temporary, features=features, mask=np.isin(np.arange(CLIPS_PER_DAY), clips))
        temporary.replace(path)
        done += 1
        if done % 20 == 0 and rank == 0:
            print(f"rank 0 finished {done} days", flush=True)
        if args.max_days and done >= args.max_days:
            break
    print(f"rank {rank} done", flush=True)


if __name__ == "__main__":
    main()
