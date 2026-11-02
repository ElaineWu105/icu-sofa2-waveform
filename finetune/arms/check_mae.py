"""Step 0: does the local MAE forward reproduce the pretrained reconstruction?

Original checkpoint, no Resp input, validation-free waveform windows. Every metric
uses the same masked patches. Patch-normalized targets test whether the checkpoint
was trained on per-patch normalized signals, as its first LayerNorm suggests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import common
from common import CHECKPOINT, HERE
from data import HourlyWindows
from model import CSFMMAE, MAEConfig


def normalize(patches):
    return (patches - patches.mean(-1, keepdim=True)) / (patches.std(-1, keepdim=True, unbiased=False) + 1e-6)


def interpolate(target, visible):
    """Linear interpolation across masked patch positions, per channel."""
    filled = target.clone()
    x = np.arange(target.shape[-2] * target.shape[-1])
    for b in range(target.shape[0]):
        for c in range(target.shape[1]):
            keep = visible[b, c].cpu().numpy()
            if not keep.any():
                continue
            known = np.flatnonzero(np.repeat(keep, target.shape[-1]))
            flat = target[b, c].flatten().cpu().numpy()
            filled[b, c] = torch.from_numpy(np.interp(x, x[known], flat[known]).reshape(target[b, c].shape)).to(target)
    return filled


@torch.no_grad()
def evaluate(model, batches, device):
    totals = {}

    def add(name, value, count):
        low, n = totals.get(name, (0.0, 0))
        totals[name] = (low + float(value) * count, n + count)

    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch["modality_mask"] = batch["modality_mask"].clone()
        batch["modality_mask"][:, 2] = False
        out = model(**batch, return_predictions=True)
        cfg = model.config
        b, c = batch["signal"].shape[:2]
        mask = out["waveform_mask"].reshape(b, c, cfg.num_patches)
        target = batch["signal"].reshape(b, c, cfg.num_patches, cfg.patch_size)
        prediction = out["wave_prediction"]
        filled = interpolate(target, batch["modality_mask"][..., None] & ~mask)
        for position, name in ((0, "ecg"), (1, "ppg")):
            chosen = mask[:, position]
            if not chosen.any():
                continue
            t, p, f = target[:, position][chosen], prediction[:, position][chosen], filled[:, position][chosen]
            count = t.numel()
            add(f"{name}_raw_mse", (p - t).square().mean(), count)
            add(f"{name}_zero_mse", t.square().mean(), count)
            add(f"{name}_interpolation_mse", (f - t).square().mean(), count)
            add(f"{name}_normalized_target_mse", (p - normalize(t)).square().mean(), count)
            add(f"{name}_normalized_zero_mse", normalize(t).square().mean(), count)
            add(f"{name}_scale_free_mse", (normalize(p) - normalize(t)).square().mean(), count)
    return {name: round(low / n, 5) for name, (low, n) in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=HERE / "check_mae.json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    dataset = HourlyWindows("train", "A", fixed=True)
    chosen = np.linspace(0, len(dataset) - 1, min(args.windows, len(dataset)), dtype=int)
    samples = [dataset[int(i)] for i in chosen]
    batches = [{key: torch.stack([s[key] for s in samples[left:left + args.batch_size]]) for key in samples[0]}
               for left in range(0, len(samples), args.batch_size)]
    model = CSFMMAE(MAEConfig())
    report = model.load_pretrained(CHECKPOINT)
    model.to(device).eval()
    result = {"windows": len(samples), "loaded_keys": len(report["loaded_keys"]),
              "initialized_keys": report["initialized_keys"], "metrics": evaluate(model, batches, device)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=1))


if __name__ == "__main__":
    main()
