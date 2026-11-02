"""Level 2: one-hour SSL over 20 three-minute slots, then encode all hours. No SOFA loss."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import CLIPS_PER_HOUR, PRIVATE, RUNS
from day_data import hour_rows, load_three_min
from temporal_model import TemporalConfig, TemporalSSL


def learning_rate(update, total, warmup, base):
    if warmup and update < warmup:
        return base * (update + 1) / warmup
    fraction = min(1., (update - warmup) / max(1, total - warmup))
    return base * .5 * (1 + math.cos(math.pi * fraction))


def batch_of(features, mask, rows, device):
    start = rows[:, 1] * CLIPS_PER_HOUR
    slots = start[:, None] + np.arange(CLIPS_PER_HOUR)
    values = np.nan_to_num(features[rows[:, 0][:, None], slots].astype(np.float32))
    return torch.from_numpy(values).to(device), torch.from_numpy(mask[rows[:, 0][:, None], slots]).to(device)


def encode_all(model, features, mask, device, batch_size=512):
    """Any nonempty hour is encoded, including those too sparse for SSL."""
    rows = hour_rows(mask, 1)
    output = np.zeros((len(features), 24, 768), dtype=np.float32)
    present = np.zeros((len(features), 24), dtype=bool)
    model.eval()
    with torch.inference_mode():
        for left in range(0, len(rows), batch_size):
            part = rows[left:left + batch_size]
            values, valid = batch_of(features, mask, part, device)
            output[part[:, 0], part[:, 1]] = model.encoder(values, valid).cpu().numpy()
            present[part[:, 0], part[:, 1]] = True
    return output, present


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RUNS / "one_hour"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    directory = Path(args.output)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    days, features, mask = load_three_min()
    split = days.split.to_numpy()
    rows = {name: (lambda chosen: chosen[split[chosen[:, 0]] == name])(hour_rows(mask)) for name in ("train", "val")}
    config = TemporalConfig(sequence_length=CLIPS_PER_HOUR)
    model = TemporalSSL(config).to(device)
    sample = features[split == "train"][mask[split == "train"]]
    sample = sample[np.random.default_rng(args.seed).permutation(len(sample))[:200_000]].astype(np.float32)
    model.encoder.set_normalization(sample.mean(0), np.maximum(sample.std(0), 1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    per_epoch = max(1, len(rows["train"]) // args.batch_size)
    total, warmup = args.epochs * per_epoch, args.warmup_epochs * per_epoch
    best, stale, update = math.inf, 0, 0
    rng = np.random.default_rng(args.seed)
    print(json.dumps({"train_hours": len(rows["train"]), "val_hours": len(rows["val"])}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(rows["train"]))
        for left in range(0, per_epoch * args.batch_size, args.batch_size):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(update, total, warmup, args.lr)
            values, valid = batch_of(features, mask, rows["train"][order[left:left + args.batch_size]], device)
            losses = model.losses(model(values, valid))
            loss = losses["local"] + losses["dynamic"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            update += 1
        model.eval()
        totals, count = np.zeros(2), 0
        with torch.inference_mode():
            for left in range(0, len(rows["val"]) - args.batch_size + 1, args.batch_size):
                values, valid = batch_of(features, mask, rows["val"][left:left + args.batch_size], device)
                losses = model.losses(model(values, valid))
                totals += [float(losses["local"]), float(losses["dynamic"])]
                count += 1
        local, dynamic = totals / max(1, count)
        improved = local + dynamic < best
        best, stale = min(best, local + dynamic), 0 if improved else stale + 1
        print(json.dumps({"epoch": epoch, "train_loss": float(loss), "val_local": local, "val_dynamic": dynamic, "stale": stale}), flush=True)
        if improved:
            torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch}, directory / "best.pt")
        if stale >= args.patience and epoch + 1 > args.warmup_epochs:
            break
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=False)["model"])
    output, present = encode_all(model, features, mask, device)
    np.savez(PRIVATE / "one_hour.npz", features=output, mask=present, stay_id=days.stay_id.to_numpy(), day=days.day.to_numpy())
    print("one-hour features saved", flush=True)


if __name__ == "__main__":
    main()
