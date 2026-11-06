"""Masked fine-tune of CSFM with a new Resp channel, as in finetune/README.md.

Arm A is signal-only; arm B adds text when a record is near the window (CSFM Eq. 1).
Signal loss is MSE on masked patches against per-patch normalized targets, which is
what the pretrained decoder reconstructs (see check_mae.py). Text uses BERT whole word
masking at CSFM's 50%. Training runs a fixed number of updates with cosine decay to
zero. Every 2,500 updates the encoder is saved and the reconstruction error is logged
on held-out train patients; choose_checkpoint.py later chooses one.
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
from model import CSFMMAE, MAEConfig, _sample_mask, normalized
from tokenizer_utils import continuation_ids

MODALITIES = ("ecg", "ppg", "resp")


class WarmupCosine:
    """Schedule optimizer updates, rather than microbatches, with linear warmup."""
    def __init__(self, optimizer, total_updates: int, warmup_ratio: float | None = None,
                 *, warmup_updates: int | None = None):
        if total_updates < 1:
            raise ValueError("total_updates must be positive")
        if (warmup_ratio is None) == (warmup_updates is None):
            raise ValueError("Specify exactly one of warmup_ratio or warmup_updates")
        if warmup_ratio is not None:
            if not math.isfinite(warmup_ratio) or not 0 < warmup_ratio < 1:
                raise ValueError("warmup_ratio must be finite and in (0,1)")
            warmup_updates = max(1, math.ceil(total_updates * warmup_ratio))
        if not 0 <= warmup_updates < total_updates:
            raise ValueError("Warmup must finish before the total optimizer-update budget")
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.warmup_updates = int(warmup_updates)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.completed_updates = 0
        self._apply()

    def factor(self, index: int) -> float:
        if index < self.warmup_updates:
            return (index + 1) / self.warmup_updates
        progress = min(1.0, (index - self.warmup_updates) / max(1, self.total_updates - self.warmup_updates))
        return 0.5 * (1 + math.cos(math.pi * progress))

    def _apply(self):
        factor = self.factor(self.completed_updates)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor

    def step(self):
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        return {"total_updates": self.total_updates, "warmup_updates": self.warmup_updates,
                "base_lrs": self.base_lrs, "completed_updates": self.completed_updates}

    def load_state_dict(self, state):
        if state["total_updates"] != self.total_updates or state["warmup_updates"] != self.warmup_updates:
            raise RuntimeError("Resume scheduler budget differs")
        self.base_lrs = list(state["base_lrs"])
        self.completed_updates = int(state["completed_updates"])
        self._apply()


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


CONTINUATION = continuation_ids()


def word_groups(ids, candidate):
    """Token positions per word: a '##' piece joins the word before it, as in BERT."""
    groups = []
    for position in np.flatnonzero(candidate):
        if groups and ids[position] in CONTINUATION:
            groups[-1].append(int(position))
        else:
            groups.append([int(position)])
    return groups


def whole_word_mask(input_ids, attention_mask, special_tokens_mask, ratio, generator):
    """BERT create_masked_lm_predictions with do_whole_word_mask, at the CSFM text ratio.

    BERT's 15% rate and 20-prediction cap are replaced by CSFM's 50% and no cap. Chosen
    tokens are dropped from the encoder as in CSFM, not replaced 80/10/10 as in BERT.
    """
    ids, valid = input_ids.cpu().numpy(), attention_mask.cpu().bool().numpy()
    candidate = valid & ~special_tokens_mask.cpu().bool().numpy()
    result = torch.zeros(ids.shape, dtype=torch.bool)
    for row in range(len(ids)):
        groups = word_groups(ids[row], candidate[row])
        # BERT counts [CLS] and [SEP] in the sequence length and rounds half to even.
        budget = max(1, int(round(int(valid[row].sum()) * ratio)))
        chosen = 0
        for group in torch.randperm(len(groups), generator=generator).tolist():
            if chosen >= budget:
                break
            if chosen + len(groups[group]) > budget:
                continue
            result[row, groups[group]] = True
            chosen += len(groups[group])
    return result


def value_mask(input_ids, attention_mask, special_tokens_mask, value_span, ratio, generator):
    """Arm B3: in charted text only values are masked, each value whole; field names stay.

    `max(1, round(values x ratio))` values are drawn at random, so a one-token number and a
    multi-token rhythm name are equally likely to be hidden. Text without values (ECG reports)
    keeps whole word masking.
    """
    result = whole_word_mask(input_ids, attention_mask, special_tokens_mask, ratio, generator)
    spans = value_span.cpu().numpy()
    for row in np.flatnonzero((spans > 0).any(1)):
        numbers = np.unique(spans[row][spans[row] > 0])
        count = max(1, int(round(len(numbers) * ratio)))
        result[row] = False
        for pick in torch.randperm(len(numbers), generator=generator)[:count].tolist():
            result[row, torch.from_numpy(spans[row] == numbers[pick])] = True
    return result


def make_masks(batch, cfg, generator, value_span=None):
    """Signal and text masks; `value_span` switches charted text to value-only masking."""
    wave = batch["modality_mask"][..., None].expand(-1, -1, cfg.num_patches).flatten(1)
    if value_span is None:
        text = whole_word_mask(batch["input_ids"], batch["attention_mask"], batch["special_tokens_mask"],
                               cfg.text_mask_ratio, generator)
    else:
        text = value_mask(batch["input_ids"], batch["attention_mask"], batch["special_tokens_mask"], value_span,
                          cfg.text_mask_ratio, generator)
    return (_sample_mask(wave, cfg.waveform_mask_ratio, generator).to(batch["signal"].device),
            text.to(batch["signal"].device))


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
def reconstruction(model, dataset, args, device, rank, world):
    """Masked signal error on the held-out selection patients, signal only. Logged, not used to select."""
    core = model.module if hasattr(model, "module") else model
    core.eval()
    generator = torch.Generator().manual_seed(args.seed + 90210 + rank)
    indices = list(range(rank, len(dataset) if args.check_limit is None else min(args.check_limit, len(dataset)), world))
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
    values = {f"held_out_{name}": float(totals[i] / counts[i]) for i, name in enumerate(MODALITIES)}
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
    parser.add_argument("--text-mask", choices=("words", "values"), default="words",
                        help="words: whole word masking (B, B2); values: in charted text mask values only (B3)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64, help="Per GPU; keep GPUs x batch x accumulation at 256")
    parser.add_argument("--accumulation", type=int, default=1, help="Microbatches per optimizer update")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer-decay", type=float, default=.75)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--updates", type=int, default=60_000, help="Schedule length; the cosine reaches zero here")
    parser.add_argument("--warmup-ratio", type=float, default=.05)
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-updates", type=int, help="Check run: stop early without changing the schedule")
    parser.add_argument("--check-limit", type=int, help="Check run: fewer held-out windows for the logged error")
    args = parser.parse_args()
    if args.arm == "A" and args.text_mask != "words":
        parser.error("--text-mask applies to arm B only")
    values = args.text_mask == "values"

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    history = args.output / "history.jsonl"

    cfg = MAEConfig()
    model = CSFMMAE(cfg)
    report = model.load_pretrained(args.checkpoint)
    model.to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, args.lr, args.layer_decay, args.weight_decay), betas=(.9, .95))
    data = HourlyWindows("train", args.arm, args.cut_dir, args.plan, args.text_dir, seed=args.seed, selection="exclude",
                         value_spans=values)
    held_out = HourlyWindows("train", "A", args.cut_dir, args.plan, args.text_dir, seed=args.seed, fixed=True,
                             selection="only")
    per_epoch = len(data) // (args.batch_size * world * args.accumulation)
    total = args.updates
    scheduler = WarmupCosine(optimizer, total, args.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda")
    progress = {"update": 0, "epoch": 0, "batch": 0}
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
            "held_out_windows": len(held_out), "updates_per_epoch": per_epoch, "total_updates": total,
            "warmup_updates": scheduler.warmup_updates, "signal_target": "per-patch normalized, MAE norm_pix_loss",
            "text_masking": ("half of the values in charted text, each whole, field names kept; ECG reports by whole words"
                             if values else "BERT whole word masking at 50%") + ", masked tokens dropped from the encoder",
            "selection": "choose_checkpoint.py: probes on CLS for held-out train patients, from the checkpoints saved here",
            "initialized_keys": report["initialized_keys"], "text_tokenizer": "BERT uncased, unconfirmed against CSFM",
            "code": str(HERE), "reused": str(FINETUNE_DIR)}, indent=2) + "\n")
        log(history, {"event": "start", "rank_world": world, "train_windows": len(data), "updates_per_epoch": per_epoch,
                      "resume_update": progress["update"]})
    end = total if args.max_updates is None else min(total, args.max_updates)
    stop = progress["update"] >= end
    micro = 0
    stop_file = args.output / "stop"
    while not stop:
        data.set_epoch(progress["epoch"])
        order = order_for(progress["epoch"], len(data), args.seed)
        usable = per_epoch * args.batch_size * world * args.accumulation
        # progress["batch"] counts optimizer updates, so a new GPU count resumes correctly.
        indices = order[:usable][rank::world][progress["batch"] * args.accumulation * args.batch_size:]
        for batch in loader_for(data, args.batch_size, args.workers, indices):
            batch = to_device(batch, device)
            wave, text = make_masks(batch, cfg, generator, batch.pop("value_span", None))
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
                              "text_share": float(batch["attention_mask"].any(1).float().mean()),
                              "lr_top": optimizer.param_groups[-1]["lr"]})
            stop = update >= end
            if update % args.save_every == 0 or stop:
                values = reconstruction(ddp, held_out, args, device, rank, world)
                if rank == 0:
                    core = ddp.module
                    save(args.output / "checkpoints" / f"update_{update:06d}.pt",
                         {"encoder": core.export_encoder(), "update": update, "metrics": values})
                    save(last, state(ddp, optimizer, scheduler, scaler, progress))
                    log(history, {"event": "checkpoint", "update": update, **values})
                dist.barrier()
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
        log(history, {"event": "finished", "update": progress["update"], "total_updates": total})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
