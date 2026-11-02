"""Four-GPU masked fine-tune of CSFM with a new Resp channel.

Arm A is signal-only; arm B adds charted vital-sign text. Signal loss is MSE on
masked patches against per-patch normalized targets, which is what the pretrained
decoder reconstructs (see check_mae.py). Both arms validate on the same fixed
signal-only windows, so selection never depends on text.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

import common
from common import CHECKPOINT, FINETUNE_DIR, HERE, ROOT, CUT_DIR, PLAN_PATH, TEXT_DIR
from data import HourlyWindows
from model import CSFMMAE, MAEConfig, _sample_mask
from train import WarmupCosine

MODALITIES = ("ecg", "ppg", "resp")


def layer_scale(name, layers, decay):
    """BEiT-style scale, following the CSFM lr_decay rules; new Resp parameters stay at base."""
    if name.startswith("encoder.transformer.layers."):
        layer = int(name.split(".")[3]) + 1
    elif name == "encoder.resp_channel_type_embedding":
        layer = layers + 1
    elif name.startswith("encoder.transformer."):
        layer = layers + 1
    elif name.startswith("encoder."):
        layer = 0
    else:
        layer = layers + 1
    return decay ** (layers + 1 - layer)


def parameter_groups(model, lr, decay, weight_decay):
    layers = model.config.depth
    groups = {}
    for name, parameter in model.named_parameters():
        scale = layer_scale(name, layers, decay)
        decayed = parameter.ndim >= 2 and not any(term in name for term in
                                                  ("pos_embedding", "type_embedding", "cls_token", "mask_token"))
        key = (scale, decayed)
        if key not in groups:
            groups[key] = {"params": [], "lr": lr * scale, "weight_decay": weight_decay if decayed else 0.0,
                           "group_name": f"scale_{scale:.5f}_{'decay' if decayed else 'no_decay'}"}
        groups[key]["params"].append(parameter)
    return list(groups.values())


def normalized(patches):
    """Per-patch target, as MAE norm_pix_loss."""
    mean, variance = patches.mean(-1, keepdim=True), patches.var(-1, keepdim=True)
    return (patches - mean) / (variance + 1e-6).sqrt()


def signal_errors(out, batch, cfg):
    """Sum of squared errors and masked-scalar counts per modality."""
    b, c = batch["signal"].shape[:2]
    mask = out["waveform_mask"].reshape(b, c, cfg.num_patches)
    target = normalized(batch["signal"].reshape(b, c, cfg.num_patches, cfg.patch_size).float())
    squared = (out["wave_prediction"].float() - target).square()
    sums, counts = [], []
    for position in range(3):
        chosen = mask[:, position]
        sums.append(squared[:, position][chosen].sum())
        counts.append(chosen.sum() * cfg.patch_size)
    return torch.stack(sums), torch.stack(counts)


def make_masks(batch, cfg, generator):
    wave = batch["modality_mask"][..., None].expand(-1, -1, cfg.num_patches).flatten(1)
    text = batch["attention_mask"] & ~batch["special_tokens_mask"]
    return (_sample_mask(wave, cfg.waveform_mask_ratio, generator).to(batch["signal"].device),
            _sample_mask(text, cfg.text_mask_ratio, generator).to(batch["signal"].device))


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def order_for(epoch, size, seed):
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def loader_for(dataset, batch_size, workers, indices=None):
    subset = dataset if indices is None else Subset(dataset, indices)
    options = dict(batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)
    if workers:
        options.update(multiprocessing_context="fork", prefetch_factor=4)
    return DataLoader(subset, **options)


@torch.no_grad()
def validate(model, dataset, args, device, rank, world):
    core = model.module if hasattr(model, "module") else model
    core.eval()
    generator = torch.Generator().manual_seed(args.seed + 90210 + rank)
    indices = list(range(rank, len(dataset) if args.val_limit is None else min(args.val_limit, len(dataset)), world))
    totals, counts = torch.zeros(3, device=device, dtype=torch.float64), torch.zeros(3, device=device, dtype=torch.float64)
    for batch in loader_for(dataset, args.batch_size, args.workers, indices):
        batch = to_device(batch, device)
        wave, text = make_masks(batch, core.config, generator)
        with torch.autocast("cuda", dtype=torch.float16):
            out = core(**batch, waveform_mask=wave, text_mask=text, return_predictions=True)
        low, count = signal_errors(out, batch, core.config)
        totals += low.double()
        counts += count.double()
    dist.all_reduce(totals)
    dist.all_reduce(counts)
    core.train()
    values = {f"val_{name}": float(totals[i] / counts[i]) for i, name in enumerate(MODALITIES)}
    values["val_mean"] = float(np.mean([values[f"val_{name}"] for name in MODALITIES]))
    return values


def state(model, optimizer, scheduler, scaler, progress):
    core = model.module
    return {"model": core.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "progress": progress}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def log(path, value):
    value = {"utc": datetime.now(timezone.utc).isoformat(), **value}
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cut-dir", type=Path, default=CUT_DIR)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--text-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64, help="Per GPU; keep GPUs x batch x accumulation at 256")
    parser.add_argument("--accumulation", type=int, default=1, help="Microbatches per optimizer update")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer-decay", type=float, default=.75)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=.05)
    parser.add_argument("--validate-every", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--val-limit", type=int)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    history = args.output / "history.jsonl"

    cfg = MAEConfig()
    model = CSFMMAE(cfg)
    report = model.load_pretrained(args.checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.lr, args.layer_decay, args.weight_decay), betas=(.9, .95))
    data = HourlyWindows("train", args.arm, args.cut_dir, args.plan, args.text_dir, seed=args.seed)
    val_data = HourlyWindows("val", "A", args.cut_dir, args.plan, args.text_dir, seed=args.seed, fixed=True)
    per_epoch = len(data) // (args.batch_size * world * args.accumulation)
    total = per_epoch * args.epochs
    scheduler = WarmupCosine(optimizer, total, args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda")
    progress = {"update": 0, "epoch": 0, "batch": 0, "best_mean": None, "best_resp": None, "stale": 0}
    last = args.output / "last.pt"
    if last.is_file():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        progress = saved["progress"]
        del saved
    ddp = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    generator = torch.Generator().manual_seed(args.seed + 3041 + rank + progress["update"])
    if rank == 0:
        (args.output / "run_config.json").write_text(json.dumps({
            "args": {key: str(value) for key, value in vars(args).items()}, "world": world, "train_windows": len(data),
            "val_windows": len(val_data), "updates_per_epoch": per_epoch, "total_updates": total,
            "warmup_updates": scheduler.warmup_updates, "signal_target": "per-patch normalized, MAE norm_pix_loss",
            "selection": "mean of ECG, PPG, Resp normalized-target MSE on fixed signal-only validation windows",
            "initialized_keys": report["initialized_keys"], "text_tokenizer": "BERT uncased, unconfirmed against CSFM",
            "code": str(HERE), "reused": str(FINETUNE_DIR)}, indent=2) + "\n")
        log(history, {"event": "start", "rank_world": world, "train_windows": len(data), "updates_per_epoch": per_epoch,
                      "resume_update": progress["update"]})
    stop = False
    micro = 0
    stop_file = args.output / "stop"
    while progress["epoch"] < args.epochs and not stop:
        data.set_epoch(progress["epoch"])
        order = order_for(progress["epoch"], len(data), args.seed)
        usable = per_epoch * args.batch_size * world * args.accumulation
        # progress["batch"] counts optimizer updates, so a new GPU count resumes correctly.
        indices = order[:usable][rank::world][progress["batch"] * args.accumulation * args.batch_size:]
        for batch in loader_for(data, args.batch_size, args.workers, indices):
            batch = to_device(batch, device)
            wave, text = make_masks(batch, cfg, generator)
            with torch.autocast("cuda", dtype=torch.float16):
                out = ddp(**batch, waveform_mask=wave, text_mask=text, return_predictions=True)
                sums, counts = signal_errors(out, batch, cfg)
                signal_loss = sums.sum() / counts.sum().clamp_min(1)
                loss = signal_loss + (out["loss_text"] if args.arm == "B" else 0.0 * signal_loss)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            scaler.scale(loss / args.accumulation).backward()
            micro += 1
            if micro % args.accumulation:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= before:
                scheduler.step()
            progress["update"] += 1
            progress["batch"] += 1
            update = progress["update"]
            if rank == 0 and update % 100 == 0:
                log(history, {"event": "train", "update": update, "epoch": progress["epoch"], "loss": float(loss),
                              "signal": float(signal_loss), "text": float(out["loss_text"]),
                              "lr_top": optimizer.param_groups[-1]["lr"]})
            limit = args.max_updates is not None and update >= args.max_updates
            if update % args.validate_every == 0 or limit:
                values = validate(ddp, val_data, args, device, rank, world)
                improved = progress["best_mean"] is None or values["val_mean"] < progress["best_mean"]
                improved_resp = progress["best_resp"] is None or values["val_resp"] < progress["best_resp"]
                if improved:
                    progress["best_mean"] = values["val_mean"]
                if improved_resp:
                    progress["best_resp"] = values["val_resp"]
                active = update > scheduler.warmup_updates
                progress["stale"] = 0 if improved or not active else progress["stale"] + 1
                if rank == 0:
                    core = ddp.module
                    if improved:
                        save(args.output / "best_overall.pt", {"model": core.state_dict(), "encoder": core.export_encoder(),
                                                               "update": update, "metrics": values})
                    if improved_resp:
                        save(args.output / "best_resp.pt", {"model": core.state_dict(), "encoder": core.export_encoder(),
                                                            "update": update, "metrics": values})
                    save(last, state(ddp, optimizer, scheduler, scaler, progress))
                    log(history, {"event": "validation", "update": update, **values, "stale": progress["stale"]})
                dist.barrier()
                stop = limit or (args.patience > 0 and progress["stale"] >= args.patience)
                if stop:
                    break
            if update % 20 == 0:
                requested = torch.tensor(int(rank == 0 and stop_file.exists()), device=device)
                dist.broadcast(requested, 0)
                if requested.item():
                    if rank == 0:
                        save(last, state(ddp, optimizer, scheduler, scaler, progress))
                        stop_file.unlink()
                        log(history, {"event": "stopped", "update": update})
                    dist.barrier()
                    dist.destroy_process_group()
                    return
        if not stop:
            progress["epoch"] += 1
            progress["batch"] = 0
    if rank == 0:
        log(history, {"event": "finished", "update": progress["update"], "best_mean": progress["best_mean"],
                      "best_resp": progress["best_resp"]})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
