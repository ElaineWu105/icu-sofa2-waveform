"""Level 3: 24-hour model over 24 one-hour slots, with the SOFA-2 task loss plus 0.1 local and 0.1 dynamic SSL."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from common import ORGANS, PRIVATE, RUNS
from day_data import patient_weights, targets
from sofa_head import OrdinalHead, expected_score, sofa_loss
from temporal_model import TemporalConfig, TemporalSSL
from train_one_hour import learning_rate

METHODS = {"hierarchy": (True, True), "baseline": (False, True), "wave_only": (True, False)}


class DayModel(nn.Module):
    def __init__(self, use_wave, use_baseline):
        super().__init__()
        self.temporal = TemporalSSL(TemporalConfig(sequence_length=24))
        self.head = OrdinalHead(use_wave=use_wave, use_baseline=use_baseline)

    def forward(self, values, mask, base, base_mask):
        output = self.temporal(values, mask)
        output["logits"] = self.head(output["embedding"], base, base_mask)
        return output


def table(days, features, mask, keep):
    data = targets(days.iloc[keep])
    return {"x": features[keep], "m": mask[keep], "subject": days.subject_id.to_numpy()[keep], **data}


def tensors(data, rows, device):
    return (torch.from_numpy(data["x"][rows]).to(device), torch.from_numpy(data["m"][rows]).to(device),
            torch.from_numpy(data["base"][rows]).to(device), torch.from_numpy(data["base_mask"][rows]).to(device),
            torch.from_numpy(data["future"][rows]).to(device), torch.from_numpy(data["label"][rows]).to(device))


def predict(model, data, device, batch_size=256):
    model.eval()
    rows, output = np.flatnonzero(data["stayed"]), []
    with torch.inference_mode():
        for left in range(0, len(rows), batch_size):
            values, mask, base, base_mask, _, _ = tensors(data, rows[left:left + batch_size], device)
            output.append(expected_score(model(values, mask, base, base_mask)["logits"]).cpu().numpy())
    return rows, np.concatenate(output)


def metrics(data, rows, score):
    """Change metrics on labeled organs, against predicting no change."""
    delta = score - data["base"][rows]
    truth = data["future"][rows] - data["base"][rows]
    label = data["label"][rows]
    report = {}
    for index, organ in enumerate(ORGANS):
        keep = label[:, index]
        changed = keep & (truth[:, index] != 0)
        report[organ] = {"n": int(keep.sum()), "mae": float(np.abs(delta[keep, index] - truth[keep, index]).mean()),
                         "persistence_mae": float(np.abs(truth[keep, index]).mean()),
                         "mae_changed": float(np.abs(delta[changed, index] - truth[changed, index]).mean())}
        worse = truth[keep, index] >= 1
        report[organ]["worse_auroc"] = float(roc_auc_score(worse, delta[keep, index])) if 0 < worse.sum() < len(worse) else None
    report["macro_mae"] = float(np.mean([report[organ]["mae"] for organ in ORGANS]))
    report["macro_persistence_mae"] = float(np.mean([report[organ]["persistence_mae"] for organ in ORGANS]))
    official = data["official_total"][rows]
    keep = np.isfinite(official)
    total = (delta * data["base_mask"][rows]).sum(1)
    report["total"] = {"n": int(keep.sum()), "mae_official": float(np.abs(total[keep] - official[keep]).mean()),
                       "persistence_mae": float(np.abs(official[keep]).mean())}
    worse = official[keep] >= 2
    if 0 < worse.sum() < len(worse):
        report["total"].update(auroc_worse2=float(roc_auc_score(worse, total[keep])), auprc_worse2=float(average_precision_score(worse, total[keep])),
                               prevalence_worse2=float(worse.mean()))
    return report


def fit(method, train, val, args, device):
    use_wave, use_baseline = METHODS[method]
    torch.manual_seed(args.seed)
    model = DayModel(use_wave, use_baseline).to(device)
    sample = train["x"][train["m"]]
    sample = sample[np.random.default_rng(args.seed).permutation(len(sample))[:200_000]]
    model.temporal.encoder.set_normalization(sample.mean(0), np.maximum(sample.std(0), 1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rows = np.flatnonzero(train["stayed"])
    weights = patient_weights(train["subject"][rows])
    probability = weights / weights.sum()
    per_epoch = max(1, len(rows) // args.batch_size)
    total, warmup = args.epochs * per_epoch, args.warmup_epochs * per_epoch
    rng = np.random.default_rng(args.seed)
    best, stale, update, chosen = np.inf, 0, 0, None
    for epoch in range(args.epochs):
        model.train()
        for _ in range(per_epoch):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(update, total, warmup, args.lr)
            part = rng.choice(rows, args.batch_size, p=probability)
            values, mask, base, base_mask, future, label = tensors(train, part, device)
            output = model(values, mask, base, base_mask)
            task = sofa_loss(output["logits"], future, base, label, torch.ones(len(part), device=device))
            ssl = model.temporal.losses(output) if use_wave else {"local": 0., "dynamic": 0.}
            loss = task["task"] + .1 * ssl["local"] + .1 * ssl["dynamic"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            update += 1
        val_rows = np.flatnonzero(val["stayed"])
        with torch.inference_mode():
            model.eval()
            sums, count = 0., 0
            for left in range(0, len(val_rows), 256):
                values, mask, base, base_mask, future, label = tensors(val, val_rows[left:left + 256], device)
                task = sofa_loss(model(values, mask, base, base_mask)["logits"], future, base, label, torch.ones(len(values), device=device))
                sums += float(task["task"]) * len(values)
                count += len(values)
        value = sums / count
        improved = value < best
        best, stale = min(best, value), 0 if improved else stale + 1
        if improved:
            chosen = {key: tensor.cpu().clone() for key, tensor in model.state_dict().items()}
        print(json.dumps({"method": method, "epoch": epoch, "val_task": value, "stale": stale}), flush=True)
        if stale >= args.patience and epoch + 1 > args.warmup_epochs:
            break
    model.load_state_dict(chosen)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RUNS / "day"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=7)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stored = np.load(PRIVATE / "one_hour.npz")
    days = pd.read_pickle(PRIVATE / "days.pkl").set_index(["stay_id", "day"])
    days = days.loc[list(zip(stored["stay_id"].tolist(), stored["day"].tolist()))].reset_index()
    features, mask = stored["features"], stored["mask"]
    split = days.split.to_numpy()
    if "test" in split:
        raise RuntimeError("This script reads train and validation only.")
    train, val = table(days, features, mask, np.flatnonzero(split == "train")), table(days, features, mask, np.flatnonzero(split == "val"))
    results = {}
    for method in args.methods:
        model = fit(method, train, val, args, device)
        rows, score = predict(model, val, device)
        results[method] = metrics(val, rows, score)
        torch.save({"model": model.state_dict(), "method": method}, directory / f"{method}.pt")
        print(json.dumps({"method": method, "macro_mae": results[method]["macro_mae"], "persistence": results[method]["macro_persistence_mae"]}), flush=True)
        (directory / "validation_metrics.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
