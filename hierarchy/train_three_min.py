"""Level 1: three-minute SSL over the frozen 10-second encoder. No SOFA loss.

Each epoch draws clean clips from different ICU hours on train patients, up to a cap. Validation uses a fixed
set of validation clips with the same pool size as training, so the two losses compare.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset

from clip_model import ClipEncoder, make_config, ssl_loss
from common import CLIPS_PER_HOUR, ENCODER, PRIVATE, RUNS, ClipReader, load_arm
import distributed_cpu as transport

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


class Clips(Dataset):
    def __init__(self, stays, days, clips):
        self.stays, self.days, self.clips, self.reader = stays, days, clips, None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        if self.reader is None:
            self.reader = ClipReader()
        signal_values, ids, _ = self.reader.read(int(self.stays[index]), int(self.days[index]), int(self.clips[index]))
        return torch.from_numpy(signal_values), torch.from_numpy(ids)


def hourly_sample(day, clip, windows, epoch, seed):
    """One random clean clip per ICU hour, redrawn each epoch, in shuffled order."""
    rng = np.random.default_rng(seed + epoch)
    clean = np.flatnonzero(windows == 18)
    keys = day[clean].astype(np.int64) * 24 + clip[clean] // CLIPS_PER_HOUR
    shuffled = clean[rng.permutation(len(clean))]
    _, first = np.unique(keys[np.searchsorted(clean, shuffled)], return_index=True)
    return rng.permutation(shuffled[first])


def learning_rate(update, total, warmup, base):
    if warmup and update < warmup:
        return base * (update + 1) / warmup
    fraction = min(1., (update - warmup) / max(1, total - warmup))
    return base * .5 * (1 + math.cos(math.pi * fraction))


def trainable(core):
    return {name: value.detach().cpu().clone() for name, value in core.named_parameters() if value.requires_grad}


def restore(core, state):
    named = dict(core.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            named[name].copy_(value)


def pooled_step(core, batch, device):
    values, ids = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        output = core(values, ids)
        teacher = core.teacher_features(values, ids, chunk_size=72)
        local = transport.gather_tensor(output["local"])
        teacher = transport.gather_constant(teacher)
        return ssl_loss(core, {"local": local, "embedding": local.mean(dim=1)}, teacher)


@torch.no_grad()
def validate(core, batches, device):
    core.eval()
    totals = torch.zeros(4, dtype=torch.float64)
    for batch in batches:
        losses = pooled_step(core, batch, device)
        totals += torch.tensor([float(losses[name]) for name in ("loss", "local", "dynamic")] + [1.], dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals)
    core.train()
    return {name: float(totals[index] / totals[3]) for index, name in enumerate(("loss", "local", "dynamic"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ENCODER))
    parser.add_argument("--output", default=str(RUNS / "three_min"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8, help="Per GPU; the pool is GPUs x batch, 16 in the setting")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=float, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-improvement", type=float, default=.001)
    parser.add_argument("--max-train-clips", type=int, default=40000, help="Cap per epoch; 0 uses one clip per ICU hour")
    parser.add_argument("--val-clips", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    os.umask(0o077)
    rank, world, device = transport.setup_distributed(args.device)
    torch.manual_seed(args.seed)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    index = np.load(PRIVATE / "clips.npz")
    table = pd.read_pickle(PRIVATE / "days.pkl")
    stays_by_day, days_by_day = table.stay_id.to_numpy(), table.day.to_numpy()
    split = table.split.to_numpy()[index["day"]]
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero((split == "val") & (index["windows"] == 18))
    val = np.random.default_rng(args.seed).permutation(val)[:args.val_clips]
    per_epoch = args.max_train_clips or len(hourly_sample(index["day"][train], index["clip"][train], index["windows"][train], 0, args.seed))
    updates_per_epoch = max(1, per_epoch // (args.batch_size * world * args.accumulation))
    total, warmup = args.epochs * updates_per_epoch, int(args.warmup_epochs * updates_per_epoch)
    core = ClipEncoder(load_arm(args.checkpoint), make_config()).to(device)
    core.classifier.requires_grad_(False)
    transport.synchronize_trainable_parameters(core)
    parameters = [value for value in core.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    state = {"epoch": 0, "batch": 0, "update": 0, "best": math.inf, "stale": 0}
    last = directory / "last.pt"
    if last.is_file():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        restore(core, saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_last():
        if rank == 0:
            temporary = last.with_suffix(".tmp")
            torch.save({"trainable": trainable(core), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "state": state}, temporary)
            temporary.replace(last)

    def log(value):
        if rank == 0:
            print(json.dumps(value), flush=True)
            with (directory / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")

    def data(rows, offset=0):
        rows = rows[rank::world][offset * args.batch_size:]
        options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
        if args.workers:
            options["prefetch_factor"] = 2
        return DataLoader(Clips(stays_by_day[index["day"][rows]], days_by_day[index["day"][rows]], index["clip"][rows]), **options)

    log({"event": "start", "train_clips_per_epoch": per_epoch, "val_clips": len(val), "world": world, "pool": args.batch_size * world,
         "updates_per_epoch": updates_per_epoch, "resume": state})
    while state["epoch"] < args.epochs and not STOP:
        epoch = state["epoch"]
        began, first_batch = time.time(), state["batch"]
        chosen = train[hourly_sample(index["day"][train], index["clip"][train], index["windows"][train], epoch, args.seed)]
        chosen = chosen[:per_epoch]
        core.train()
        micro = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in data(chosen, state["batch"]):
            losses = pooled_step(core, batch, device)
            if not transport.all_reduce_flag(torch.isfinite(losses["loss"].detach())):
                raise RuntimeError("Nonfinite SSL loss.")
            scaler.scale(losses["loss"] / args.accumulation).backward()
            micro += 1
            state["batch"] += 1
            if micro % args.accumulation:
                continue
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(state["update"], total, warmup, args.lr)
            transport.average_gradients(core)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            state["update"] += 1
            if state["update"] % args.log_every == 0:
                log({"event": "train", "epoch": epoch, "update": state["update"], **{name: float(losses[name]) for name in ("loss", "local", "dynamic")},
                     "clips_per_second": round((state["batch"] - first_batch) * args.batch_size * world / max(1., time.time() - began), 1)})
            stopped = transport.all_reduce_flag(STOP, op=dist.ReduceOp.MAX)
            if stopped or state["update"] % args.save_every == 0:
                save_last()
            if stopped:
                return
        result = validate(core, data(val), device)
        improved = result["loss"] < state["best"] * (1 - args.min_improvement)
        state.update(epoch=epoch + 1, batch=0, stale=0 if improved else state["stale"] + 1, best=min(state["best"], result["loss"]))
        log({"event": "validation", "epoch": epoch, **result, "stale": state["stale"]})
        if improved and rank == 0:
            torch.save({"adapter": core.adapter_state_dict(), "config": make_config().to_dict(), "validation": result, "epoch": epoch,
                        "checkpoint": args.checkpoint}, directory / "best_encoder.pt")
        save_last()
        if state["stale"] >= args.patience and epoch + 1 > args.warmup_epochs:
            break
    log({"event": "finished", "state": state})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
