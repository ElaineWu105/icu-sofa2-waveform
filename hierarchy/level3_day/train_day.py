"""Level 3: 24-hour model over 24 hours of 20 clip features, with L_SOFA plus 0.1 L_local and 0.1 L_dyn, and its controls.

L_SOFA is the mean over organs of the CORN loss plus the composite BCE (hierarchy/README.md, section 3.6).
Methods, all on identical anchors and labels:

  e0         baseline path on E0 (baseline scores and flags), no waveform
  e2         baseline path on E2 (labels/README.md, section 7.4), no waveform
  m          E2 baseline path plus a waveform correction from the mean frozen 10-second CLS of the day
  hierarchy  E2 baseline path plus a waveform correction from the level 3 day feature
  wave_only  the level 3 day feature alone

Each method and seed stops on the validation L_SOFA and saves its validation predictions; evaluate_day.py then
computes the endpoints with a paired patient bootstrap. Train and validation patients only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common import PRIVATE, RUNS
from day_data import HANDLING, ONE_HOUR, THREE_MIN, context, load_e2, patient_weights, targets
from evaluate_day import evaluate
from sofa_head import SofaHead, composite_terms, corn_terms, cumulative, sofa_loss, sofa_loss_from_terms
from temporal_model import TemporalConfig, TemporalSSL
from train_one_hour import learning_rate

METHODS = {"e0": ("e0", None), "e2": ("e2", None), "m": ("e2", "ten_second"),
           "hierarchy": ("e2", "temporal"), "wave_only": (None, "temporal")}
LAMBDA_LOCAL = LAMBDA_DYN = .1


class DayModel(nn.Module):
    def __init__(self, method, context_dim):
        super().__init__()
        inputs, wave = METHODS[method]
        self.wave = wave
        self.temporal = TemporalSSL(TemporalConfig(segments=24, tokens_per_segment=20, mask_ratio=0.), decoder=False) if wave == "temporal" else None
        self.head = SofaHead(context_dim if inputs else 0, 768 if wave else 0)

    def forward(self, batch):
        output = {}
        wave = None
        if self.wave == "temporal":
            output = self.temporal(batch["tokens"], batch["token_mask"])
            wave = output["embedding"]
        elif self.wave == "ten_second":
            wave = batch["ten_second"]
        output.update(self.head(batch.get("context"), wave))
        return output


def load_day_inputs():
    """Level 2 outputs (memory-mapped), aligned day table, and the control M feature."""
    index = np.load(ONE_HOUR / "index.npz")
    clip_features = np.load(ONE_HOUR / "clip_features.npy", mmap_mode="r")
    hour_features = np.load(ONE_HOUR / "hour_features.npy", mmap_mode="r")
    days = pd.read_pickle(PRIVATE / "days.pkl").set_index(["stay_id", "day"])
    days = days.loc[list(zip(index["stay_id"].tolist(), index["day"].tolist()))].reset_index()
    ten = np.stack([np.load(THREE_MIN / f"{stay}_{day:03d}.npz")["ten_second"] for stay, day in zip(days.stay_id, days.day)])
    return days, clip_features, index["clip_mask"], hour_features, index["hour_mask"], ten


class Split:
    """Rows of one split with their labels and head inputs."""

    def __init__(self, days, rows, handling, e2, context_stats, ten, ten_stats):
        self.rows = rows
        self.labels = targets(days.iloc[rows], handling)
        self.subject = days.subject_id.to_numpy()[rows]
        self.context = {}
        self.stats = {}
        for inputs in ("e0", "e2"):
            self.context[inputs], self.stats[inputs] = context(days.iloc[rows], inputs, e2.iloc[rows].reset_index(drop=True),
                                                               context_stats.get(inputs) if context_stats else None)
        self.ten_stats = ten_stats or (ten[rows].mean(0), np.maximum(ten[rows].std(0), 1e-6))
        self.ten = ((ten[rows] - self.ten_stats[0]) / self.ten_stats[1]).astype(np.float32)
        self.used = np.flatnonzero(self.labels["used"])


def make_batch(split, positions, sources, inputs, device):
    """positions index into split.rows."""
    clip_features, clip_mask, hour_features, hour_mask = sources
    rows = split.rows[positions]
    order = np.argsort(rows)
    unsorted = np.empty_like(order)
    unsorted[order] = np.arange(len(order))
    batch = {"future": torch.from_numpy(split.labels["future"][positions]), "label": torch.from_numpy(split.labels["label"][positions]),
             "event": torch.from_numpy(split.labels["event"][positions]), "event_known": torch.from_numpy(split.labels["event_known"][positions]),
             "ten_second": torch.from_numpy(split.ten[positions])}
    if inputs:
        batch["context"] = torch.from_numpy(split.context[inputs][positions])
    if sources[0] is not None:
        batch["tokens"] = torch.from_numpy(np.asarray(clip_features[rows[order]], dtype=np.float32)[unsorted])
        batch["token_mask"] = torch.from_numpy(clip_mask[rows])
        batch["teacher"] = torch.from_numpy(np.asarray(hour_features[rows[order]], dtype=np.float32)[unsorted])
        batch["hour_mask"] = torch.from_numpy(hour_mask[rows])
    return {key: value.to(device) for key, value in batch.items()}


def step_losses(model, output, batch):
    task = sofa_loss(output, batch["future"], batch["label"], batch["event"], batch["event_known"])
    if model.temporal is None:
        return {**task, "local": torch.zeros((), device=task["task"].device), "dynamic": torch.zeros((), device=task["task"].device),
                "loss": task["task"]}
    ssl = model.temporal.losses(output, batch["teacher"])
    return {**task, **ssl, "loss": task["task"] + LAMBDA_LOCAL * ssl["local"] + LAMBDA_DYN * ssl["dynamic"]}


@torch.no_grad()
def validation_pass(model, split, sources, inputs, device, batch_size=128):
    """L_SOFA over all validation anchors at once, and the predictions."""
    model.eval()
    organ_sum = organ_count = composite_sum = composite_count = 0
    probabilities, composite = [], []
    for left in range(0, len(split.used), batch_size):
        batch = make_batch(split, split.used[left:left + batch_size], sources, inputs, device)
        output = model(batch)
        s, c = corn_terms(output["organ"], batch["future"], batch["label"])
        cs, cc = composite_terms(output["composite"], batch["event"], batch["event_known"])
        organ_sum, organ_count = organ_sum + s, organ_count + c
        composite_sum, composite_count = composite_sum + cs, composite_count + cc
        probabilities.append(cumulative(output["organ"]).cpu().numpy())
        composite.append(torch.sigmoid(output["composite"]).cpu().numpy())
    model.train()
    loss = sofa_loss_from_terms(organ_sum, organ_count, composite_sum, composite_count)
    return {key: float(value) for key, value in loss.items()}, np.concatenate(probabilities), np.concatenate(composite)


def fit(method, seed, train, val, sources, args, device, log):
    inputs, wave = METHODS[method]
    torch.manual_seed(seed)
    model = DayModel(method, train.context["e2" if inputs == "e2" else "e0"].shape[1]).to(device)
    if model.temporal is not None:
        clip_features, clip_mask = sources[0], sources[1]
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(train.rows, min(len(train.rows), 2000), replace=False))
        sample = np.asarray(clip_features[chosen], dtype=np.float32)[clip_mask[chosen]]
        sample = sample[rng.permutation(len(sample))[:200_000]]
        model.temporal.encoder.set_normalization(sample.mean(0), np.maximum(sample.std(0), 1e-6))
    used_sources = sources if model.temporal is not None else (None, None, None, None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    weights = patient_weights(train.subject[train.used])
    probability = weights / weights.sum()
    per_epoch = max(1, len(train.used) // args.batch_size)
    total, warmup = args.epochs * per_epoch, args.warmup_epochs * per_epoch
    rng = np.random.default_rng(seed)
    best, stale, update, chosen_state = np.inf, 0, 0, None
    for epoch in range(args.epochs):
        model.train()
        for _ in range(per_epoch):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(update, total, warmup, args.lr)
            batch = make_batch(train, rng.choice(train.used, args.batch_size, p=probability), used_sources, inputs, device)
            losses = step_losses(model, model(batch), batch)
            if not torch.isfinite(losses["loss"]):
                raise RuntimeError("Nonfinite level 3 loss.")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            update += 1
        result, _, _ = validation_pass(model, val, used_sources, inputs, device)
        improved = result["task"] < best
        best, stale = min(best, result["task"]), 0 if improved else stale + 1
        if improved:
            chosen_state = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
        log({"event": "validation", "method": method, "seed": seed, "epoch": epoch, **{f"val_{k}": v for k, v in result.items()},
             "train_loss": float(losses["loss"]), "train_local": float(losses["local"]), "train_dynamic": float(losses["dynamic"]), "stale": stale})
        if stale >= args.patience and epoch + 1 > args.warmup_epochs:
            break
    model.load_state_dict(chosen_state)
    _, probabilities, composite = validation_pass(model, val, used_sources, inputs, device)
    return model, probabilities, composite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RUNS / "day"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--handling", default="main", choices=HANDLING, help="How death and discharge are scored")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    directory = Path(args.output) / args.handling
    (directory / "predictions").mkdir(mode=0o700, parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    days, clip_features, clip_mask, hour_features, hour_mask, ten = load_day_inputs()
    split = days.split.to_numpy()
    if "test" in split:
        raise RuntimeError("This script reads train and validation only.")
    e2 = load_e2(days)
    missing = e2.icu_day.isna()
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} anchors have no E2 row; rerun labels/run.sh e2.")
    train = Split(days, np.flatnonzero(split == "train"), args.handling, e2, None, ten, None)
    val = Split(days, np.flatnonzero(split == "val"), args.handling, e2, train.stats, ten, train.ten_stats)
    sources = (clip_features, clip_mask, hour_features, hour_mask)
    np.savez(directory / "validation_labels.npz", subject=val.subject[val.used], stay_id=days.stay_id.to_numpy()[val.rows[val.used]],
             day=days.day.to_numpy()[val.rows[val.used]], **{key: value[val.used] for key, value in val.labels.items()})

    def log(value):
        print(json.dumps(value), flush=True)
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")

    log({"event": "start", "handling": args.handling, "train_anchors": len(train.used), "val_anchors": len(val.used),
         "methods": args.methods, "seeds": args.seeds})
    for method in args.methods:
        for seed in args.seeds:
            model, probabilities, composite = fit(method, seed, train, val, sources, args, device, log)
            torch.save({"model": model.state_dict(), "method": method, "seed": seed, "handling": args.handling},
                       directory / f"{method}_seed{seed}.pt")
            np.savez(directory / "predictions" / f"{method}_seed{seed}.npz", cumulative=probabilities, composite=composite)
            log({"event": "saved", "method": method, "seed": seed})
    report = evaluate(directory, args.handling, args.bootstrap)
    log({"event": "endpoints", **{method: entry.get("per_seed") for method, entry in report["methods"].items()}})


if __name__ == "__main__":
    main()
