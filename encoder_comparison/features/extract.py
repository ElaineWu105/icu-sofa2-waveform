"""Frozen 768-dimensional CLS features for every encoder and input.

One pass over the windows feeds every encoder still to do, so waveforms are read once.
Encoders stay in eval mode with no gradients, no masking and no text. Progress is kept per
encoder: a stopped run resumes from the last saved batch, and an added encoder is extracted
without redoing the others. Features are patient-level private files.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common import ARMS, CUT, ENCODERS, INPUTS, ORIGINAL, PRIVATE, RESP_INPUTS, ROW_BYTES
from model import CSFMEncoder, MAEConfig

FEATURES = PRIVATE / "features"
PPG, RESP = 12, 13


def conditions(encoders=ENCODERS):
    """Original CSFM has no Resp channel, so it never receives one."""
    return [(encoder, name) for encoder in encoders for name in INPUTS
            if not (encoder == "original" and name in RESP_INPUTS)]


def read_progress(path, identity, weights):
    """Windows done per encoder. Older files kept one count for all encoders."""
    saved = json.loads(path.read_text()) if path.is_file() else {"identity": identity, "encoders": {}}
    if "encoders" not in saved:
        stamps = saved["identity"].pop("checkpoints")
        saved = {"identity": saved["identity"],
                 "encoders": {name: {"checkpoint": stamp, "done": saved["done"]} for name, stamp in stamps.items()}}
    if saved["identity"] != identity:
        raise RuntimeError("Saved features come from other windows or settings; move them away first.")
    for name, entry in saved["encoders"].items():
        if name in weights and entry["checkpoint"] != weights[name]:
            raise RuntimeError(f"{name} weights changed after its features were saved; move them away first.")
    return saved


def load_original(cfg):
    """Strict load of the CSFM encoder. The new Resp row stays unused because Resp is never an input here."""
    torch.manual_seed(42)
    encoder = CSFMEncoder(cfg)
    raw = torch.load(ORIGINAL, map_location="cpu", weights_only=False)
    source = raw.get("state_dict", raw)
    heads = ("mlp_head.", "dense_scratch.", "act_", "dense_act_postprocess.", "dense_head.")
    loaded = {key[len("encoder."):]: value for key, value in source.items()
              if key.startswith("encoder.") and not key[len("encoder."):].startswith(heads)}
    state = encoder.state_dict()
    if set(state) - set(loaded) != {"resp_channel_type_embedding"} or set(loaded) - set(state):
        raise RuntimeError("Original CSFM keys do not match the encoder.")
    state.update(loaded)
    encoder.load_state_dict(state, strict=True)
    return encoder


def load_encoder(name, cfg):
    if name == "original":
        return load_original(cfg)
    return CSFMEncoder.from_export(torch.load(ARMS[name], map_location="cpu", weights_only=False)["encoder"])


class Windows(Dataset):
    def __init__(self, frame):
        self.frame = frame[["stay_id", "day", "row", "lead"]].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        stay, day, row, lead = self.frame[index]
        with (CUT / str(stay) / f"day_{day:03d}_signal.npy").open("rb") as stream:
            np.lib.format.read_magic(stream)
            np.lib.format.read_array_header_1_0(stream)
            stream.seek(stream.tell() + int(row) * ROW_BYTES)
            signal = np.frombuffer(stream.read(ROW_BYTES), dtype="<f4").reshape(3, 2500).copy()
        return torch.from_numpy(signal), torch.tensor([lead, PPG, RESP], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=100, help="Batches between progress saves")
    args = parser.parse_args()
    windows = pd.read_pickle(PRIVATE / "windows.pkl")
    if set(windows.split) != {"train", "val"}:
        raise RuntimeError("Features are only extracted for train and validation patients.")
    FEATURES.mkdir(mode=0o700, parents=True, exist_ok=True)
    cfg = MAEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    identity = json.loads(json.dumps({"windows": len(windows), "subjects": int(windows.subject_id.sum()),
                                      "config": asdict(cfg)}))
    weights = {name: os.stat(path).st_mtime_ns for name, path in {"original": ORIGINAL, **ARMS}.items()}
    progress_path = FEATURES / "progress.json"
    saved = read_progress(progress_path, identity, weights)
    done_by = {name: saved["encoders"].get(name, {"done": 0})["done"] for name in ENCODERS}
    todo = [name for name in ENCODERS if done_by[name] < len(windows)]
    if not todo:
        print("features done", flush=True)
        return
    start = min(done_by[name] for name in todo)
    print("extracting", ", ".join(todo), "from window", start, flush=True)
    encoders = {name: load_encoder(name, cfg).to(device).eval().requires_grad_(False) for name in todo}
    shape = (len(windows), cfg.dim)
    stores = {}
    for encoder, name in conditions(todo):
        path = FEATURES / f"{encoder}__{name}.npy"
        stores[encoder, name] = np.lib.format.open_memmap(path, mode="r+" if path.is_file() else "w+", dtype=np.float32, shape=shape)
        os.chmod(path, 0o600)
    remaining = windows.iloc[start:]
    loader = DataLoader(Windows(remaining), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=True, multiprocessing_context="fork" if args.workers else None,
                        prefetch_factor=4 if args.workers else None)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else nullcontext
    done, began = start, time.time()
    with torch.inference_mode():
        for batch, (signal, ids) in enumerate(loader, 1):
            signal, ids = signal.to(device, non_blocking=True), ids.to(device, non_blocking=True)
            size = len(signal)
            for encoder, name in conditions(todo):
                columns = list(INPUTS[name])
                with autocast():
                    output = encoders[encoder](signal[:, columns], ids[:, columns])
                if output.shape != (size, cfg.dim) or not torch.isfinite(output).all():
                    raise RuntimeError(f"Invalid features from {encoder} {name}.")
                stores[encoder, name][done:done + size] = output.float().cpu().numpy()
            done += size
            if batch % args.save_every == 0 or done == len(windows):
                for store in stores.values():
                    store.flush()
                temporary = progress_path.with_suffix(".tmp")
                for name in todo:
                    saved["encoders"][name] = {"checkpoint": weights[name], "done": done}
                temporary.write_text(json.dumps(saved))
                temporary.replace(progress_path)
                rate = (done - start) / (time.time() - began)
                print(f"windows {done} of {len(windows)}, {rate:.0f} per second", flush=True)
    print("features done", flush=True)


if __name__ == "__main__":
    main()
