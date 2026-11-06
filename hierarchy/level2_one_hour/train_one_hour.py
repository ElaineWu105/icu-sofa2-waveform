"""Level 2: one-hour SSL over 20 three-minute clips of 18 window features each, then encode all hours. No SOFA loss.

The loss is masked clip prediction (75% of present clips removed, a light decoder predicts their level 1 clip
features centered within the hour, MSE on masked clips only) plus L_local and L_dyn on the visible clips, all at weight 1. Validation uses
fixed hours and fixed masks; stopping uses the validation masked loss.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import CLIPS_PER_HOUR, PRIVATE, RUNS
from day_data import ONE_HOUR, hour_rows, load_three_min, read_hour
from temporal_model import TemporalConfig, TemporalSSL

LOGGED = ("loss", "masked", "local", "dynamic")
CHECKS = ("copy", "mean", "dynamic_unprojected")


def learning_rate(update, total, warmup, base):
    if warmup and update < warmup:
        return base * (update + 1) / warmup
    fraction = min(1., (update - warmup) / max(1, total - warmup))
    return base * .5 * (1 + math.cos(math.pi * fraction))


class Hours(Dataset):
    """tokens[20,18,768], token mask[20,18] and teacher[20,768] (level 1 clip features) of one hour."""

    def __init__(self, days, features, window_mask, rows):
        self.stays, self.days = days.stay_id.to_numpy(), days.day.to_numpy()
        self.features, self.window_mask, self.rows = features, window_mask, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        day, hour = (int(value) for value in self.rows[index])
        clips = slice(hour * CLIPS_PER_HOUR, (hour + 1) * CLIPS_PER_HOUR)
        tokens = read_hour(self.stays[day], self.days[day], hour)
        return (torch.from_numpy(tokens), torch.from_numpy(self.window_mask[day, clips].copy()),
                torch.from_numpy(self.features[day, clips].astype(np.float32)))


def loader(data, batch_size, workers, shuffle=False, drop_last=False):
    options = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=drop_last, pin_memory=True)
    if workers:
        options["prefetch_factor"] = 4
    return DataLoader(data, **options)


def statistics(days, features, mask, window_mask, rows, seed, vectors=200_000, hours=800, target_hours=20_000):
    """Input normalization from train window features; decoder target normalization from train clip features
    centered within their hour (each clip minus the mean of the hour's present clips)."""
    rng = np.random.default_rng(seed)
    chosen = rows[rng.permutation(len(rows))[:hours]]
    stays, numbers = days.stay_id.to_numpy(), days.day.to_numpy()
    windows = []
    for day, hour in chosen:
        clips = slice(hour * CLIPS_PER_HOUR, (hour + 1) * CLIPS_PER_HOUR)
        windows.append(read_hour(stays[day], numbers[day], hour)[window_mask[day, clips]])
    windows = np.concatenate(windows)[:vectors]
    targets = []
    for day, hour in rows[rng.permutation(len(rows))[:target_hours]]:
        clips = slice(hour * CLIPS_PER_HOUR, (hour + 1) * CLIPS_PER_HOUR)
        values = features[day, clips][mask[day, clips]].astype(np.float32)
        targets.append(values - values.mean(0))
    targets = np.concatenate(targets)
    targets = targets[rng.permutation(len(targets))[:vectors]]
    return ((windows.mean(0), np.maximum(windows.std(0), 1e-6)), (targets.mean(0), np.maximum(targets.std(0), 1e-6)))


def encode_all(model, days, features, mask, window_mask, device, batch_size=256):
    """Every hour with at least one valid clip, unmasked: the 20 clip features z_i, their mask, and the hour feature."""
    ONE_HOUR.mkdir(mode=0o700, parents=True, exist_ok=True)
    count = len(days)
    clip_features = np.lib.format.open_memmap(ONE_HOUR / "clip_features.npy", mode="w+", dtype=np.float16, shape=(count, 24, CLIPS_PER_HOUR, 768))
    hour_features = np.lib.format.open_memmap(ONE_HOUR / "hour_features.npy", mode="w+", dtype=np.float16, shape=(count, 24, 768))
    clip_mask = np.zeros((count, 24, CLIPS_PER_HOUR), dtype=bool)
    rows = hour_rows(mask, 1)
    model.eval()
    data = Hours(days, features, window_mask, rows)
    left = 0
    with torch.inference_mode():
        for tokens, token_mask, _ in loader(data, batch_size, 8):
            part = rows[left:left + len(tokens)]
            left += len(tokens)
            output = model(tokens.to(device), token_mask.to(device))
            clip_features[part[:, 0], part[:, 1]] = output["local"].cpu().numpy().astype(np.float16)
            hour_features[part[:, 0], part[:, 1]] = output["embedding"].cpu().numpy().astype(np.float16)
            clip_mask[part[:, 0], part[:, 1]] = output["mask"].cpu().numpy()
    clip_features.flush()
    hour_features.flush()
    np.savez(ONE_HOUR / "index.npz", stay_id=days.stay_id.to_numpy(), day=days.day.to_numpy(), clip_mask=clip_mask,
             hour_mask=clip_mask.any(2))


def run_epoch_validation(model, batches, device, seed):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    names = LOGGED + CHECKS
    totals, count = np.zeros(len(names)), 0
    with torch.inference_mode():
        for tokens, token_mask, teacher in batches:
            losses = model.masked_step(tokens.to(device), token_mask.to(device), teacher.to(device), generator, references=True)
            totals += [float(losses[name]) for name in names]
            count += 1
    model.train()
    return dict(zip(names, (totals / max(1, count)).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RUNS / "one_hour"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-train-hours", type=int, default=40000, help="Hours per epoch, redrawn each epoch; 0 uses all")
    parser.add_argument("--val-hours", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encode-only", action="store_true", help="Skip training and encode with best.pt")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    directory = Path(args.output)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    days, features, mask, window_mask, _ = load_three_min(windows=True)
    # Training and validation never read test days; encoding covers every indexed day.
    split = days.split.to_numpy()
    eligible = hour_rows(mask)
    rows = {name: eligible[split[eligible[:, 0]] == name] for name in ("train", "val")}
    rng = np.random.default_rng(args.seed)
    rows["val"] = rows["val"][np.sort(rng.permutation(len(rows["val"]))[:args.val_hours])]
    config = TemporalConfig(segments=CLIPS_PER_HOUR, tokens_per_segment=18)
    model = TemporalSSL(config).to(device)

    def log(value):
        print(json.dumps(value), flush=True)
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")

    if not args.encode_only:
        (inputs, targets) = statistics(days, features, mask, window_mask, rows["train"], args.seed)
        model.encoder.set_normalization(*inputs)
        model.decoder.set_target_normalization(*targets)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
        per_epoch = min(len(rows["train"]), args.max_train_hours or len(rows["train"]))
        updates_per_epoch = max(1, per_epoch // args.batch_size)
        total, warmup = args.epochs * updates_per_epoch, args.warmup_epochs * updates_per_epoch
        best, stale, update = math.inf, 0, 0
        val_batches = loader(Hours(days, features, window_mask, rows["val"]), args.batch_size, args.workers)
        log({"event": "start", "train_hours": len(rows["train"]), "hours_per_epoch": per_epoch, "val_hours": len(rows["val"]),
             "config": config.to_dict()})
        for epoch in range(args.epochs):
            model.train()
            chosen = rows["train"][rng.permutation(len(rows["train"]))[:per_epoch]]
            generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
            for tokens, token_mask, teacher in loader(Hours(days, features, window_mask, chosen), args.batch_size, args.workers, drop_last=True):
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(update, total, warmup, args.lr)
                losses = model.masked_step(tokens.to(device), token_mask.to(device), teacher.to(device), generator)
                if not torch.isfinite(losses["loss"]):
                    raise RuntimeError("Nonfinite level 2 loss.")
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                update += 1
                if update % 100 == 0:
                    log({"event": "train", "epoch": epoch, "update": update, **{name: float(losses[name]) for name in LOGGED}})
            result = run_epoch_validation(model, val_batches, device, args.seed)
            improved = result["masked"] < best
            best, stale = min(best, result["masked"]), 0 if improved else stale + 1
            log({"event": "validation", "epoch": epoch, **result, "stale": stale})
            if improved:
                torch.save({"model": model.state_dict(), "config": config.to_dict(), "epoch": epoch, "validation": result}, directory / "best.pt")
            if stale >= args.patience and epoch + 1 > args.warmup_epochs:
                break
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=False)["model"])
    encode_all(model, days, features, mask, window_mask, device)
    log({"event": "encoded", "days": len(days), "folder": str(ONE_HOUR)})


if __name__ == "__main__":
    main()
