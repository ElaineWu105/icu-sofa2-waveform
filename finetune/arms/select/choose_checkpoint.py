"""Choose the fine-tuned checkpoint with probes on held-out train patients.

finetune/README.md, section 3.9. The windows and labels are the encoder comparison's
(`encoder_comparison/private/windows.pkl`), kept to the train patients that
`data.selection_patient` holds out from fine-tuning. For original CSFM and every saved
checkpoint, frozen CLS features feed linear probes fitted on one half of those patients
and scored on the other, then the halves swap. Each probe's penalty is chosen by
3-fold cross-validation over patients inside the fitting half. A checkpoint qualifies when no ECG or
PPG task is worse than original CSFM by more than the encoder comparison margin. The
qualifying checkpoint with the lowest respiratory rate error is chosen. Text is never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from common import CHECKPOINT, CUT_DIR, ROOT
from data import read_row, selection_patient
from model import CSFMEncoder, CSFMMAE, MAEConfig

INPUTS = {"ecg": (0,), "ppg": (1,), "ecg_ppg": (0, 1), "all3": (0, 1, 2)}
# Task: label column, input, kind, metric, margin against original CSFM (encoder comparison values).
GUARDS = {"rhythm": ("rhythm", "ecg", "multiclass", "macro_f1", .02), "hr": ("hr", "ecg", "regression", "mae", 1.),
          "spo2": ("spo2", "ppg", "regression", "mae", .25), "sbp": ("sbp", "ecg_ppg", "regression", "mae", 1.),
          "dbp": ("dbp", "ecg_ppg", "regression", "mae", 1.)}
TARGET = ("rr_monitor", "all3", "regression", "mae")
REPORTED = {"support": ("support", "all3", "binary", "auroc")}
# Penalties tried for ridge (alpha) and logistic regression (C), chosen inside the fitting half.
ALPHAS = np.logspace(-1, 5, 13)
CS = np.logspace(-4, 0, 5)


class Windows(Dataset):
    def __init__(self, frame, cut_dir):
        self.frame = frame[["stay_id", "day", "row", "lead"]].to_numpy(dtype=np.int64)
        self.cut_dir = cut_dir

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        stay, day, row, lead = self.frame[index]
        signal = read_row(self.cut_dir, stay, day, row)
        return torch.from_numpy(signal), torch.tensor([lead, 12, 13], dtype=torch.long)


def original_encoder():
    model = CSFMMAE(MAEConfig())
    model.load_pretrained(CHECKPOINT)
    return CSFMEncoder.from_export(model.export_encoder())


@torch.no_grad()
def features(encoder, loader, inputs, device):
    encoder = encoder.to(device).eval()
    found = {name: [] for name in inputs}
    for signal, ids in loader:
        signal, ids = signal.to(device), ids.to(device)
        for name in inputs:
            columns = list(INPUTS[name])
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                found[name].append(encoder(signal[:, columns], ids[:, columns]).float().cpu().numpy())
    encoder.cpu()
    return {name: np.concatenate(parts) for name, parts in found.items()}


def probe(kind, penalty):
    if kind == "regression":
        return make_pipeline(StandardScaler(), Ridge(alpha=penalty))
    return make_pipeline(StandardScaler(), LogisticRegression(C=penalty, max_iter=2000))


def measure(kind, metric, model, x, y):
    """The task metric, oriented so that higher is better."""
    if kind == "regression":
        return -mean_absolute_error(y, model.predict(x))
    if metric == "auroc":
        return roc_auc_score(y, model.predict_proba(x)[:, 1])
    return f1_score(y, model.predict(x), average="macro")


def fitted(kind, metric, x, y, groups):
    """Probe whose penalty is chosen by 3-fold cross-validation over patients inside the fitting data."""
    best, best_value = None, -np.inf
    for penalty in ALPHAS if kind == "regression" else CS:
        values = []
        for inner, outer in GroupKFold(3).split(x, y, groups):
            values.append(measure(kind, metric, probe(kind, penalty).fit(x[inner], y[inner]), x[outer], y[outer]))
        if np.mean(values) > best_value:
            best, best_value = penalty, np.mean(values)
    return probe(kind, best).fit(x, y)


def score(kind, metric, x, y, half, groups):
    """Fit on one half of the patients, score the other, both ways; mean of the two."""
    values = []
    for fit in (half, ~half):
        keep_fit, keep_score = fit & np.isfinite(y), ~fit & np.isfinite(y)
        target = y if kind == "regression" else np.nan_to_num(y).astype(int)
        model = fitted(kind, metric, x[keep_fit], target[keep_fit], groups[keep_fit])
        values.append(abs(measure(kind, metric, model, x[keep_score], target[keep_score])))
    return float(np.mean(values))


def evaluate(feature, frame, half, resp):
    """Metrics of one encoder; original CSFM has no Resp channel, so it gets ECG + PPG for Resp tasks."""
    result, groups = {}, frame.subject_id.to_numpy()
    for task, (column, name, kind, metric, *_) in {**GUARDS, "rr_monitor": TARGET, **REPORTED}.items():
        name = name if resp or name != "all3" else "ecg_ppg"
        result[task] = score(kind, metric, feature[name], frame[column].to_numpy(dtype=float), half, groups)
    return result


def constant(frame, half):
    """Median prediction from the fitting half, for scale: a probe should beat it."""
    result = {}
    for task, (column, *_rest) in {**GUARDS, "rr_monitor": TARGET}.items():
        if _rest[1] != "regression":
            continue
        y = frame[column].to_numpy(dtype=float)
        result[task] = float(np.mean([np.nanmean(np.abs(y[~fit] - np.nanmedian(y[fit]))) for fit in (half, ~half)]))
    return result


def qualifies(metrics, reference):
    for task, (_, _, _, metric, margin) in GUARDS.items():
        worse = reference[task] - metrics[task] if metric != "mae" else metrics[task] - reference[task]
        if worse > margin:
            return False
    return True


def choose(candidates, reference):
    """Update with the lowest respiratory rate error among qualifying checkpoints, or None."""
    passing = {update: metrics for update, metrics in candidates.items() if qualifies(metrics, reference)}
    return min(passing, key=lambda update: passing[update]["rr_monitor"]) if passing else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Fine-tuning output folder with checkpoints/")
    parser.add_argument("--windows", type=Path, default=ROOT / "encoder_comparison/private/windows.pkl")
    parser.add_argument("--cut-dir", type=Path, default=CUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    frame = pd.read_pickle(args.windows)
    frame = frame[(frame.split == "train") & frame.subject_id.map(selection_patient)].reset_index(drop=True)
    half = frame.subject_id.map(lambda s: int(hashlib.sha256(f"half-{s}".encode()).hexdigest()[:8], 16) % 2 == 0).to_numpy()
    loader = DataLoader(Windows(frame, args.cut_dir), batch_size=args.batch_size, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("constant", json.dumps(constant(frame, half)), flush=True)
    reference = evaluate(features(original_encoder(), loader, ("ecg", "ppg", "ecg_ppg"), device), frame, half, resp=False)
    print("original", json.dumps(reference), flush=True)
    candidates = {}
    for path in sorted((args.run / "checkpoints").glob("update_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        encoder = CSFMEncoder.from_export(saved["encoder"])
        candidates[saved["update"]] = evaluate(features(encoder, loader, tuple(INPUTS), device), frame, half, resp=True)
        print(saved["update"], json.dumps(candidates[saved["update"]]), flush=True)
    chosen = choose(candidates, reference)
    report = {"windows": len(frame), "patients": int(frame.subject_id.nunique()), "constant": constant(frame, half),
              "original": reference,
              "checkpoints": {str(u): {**m, "qualifies": qualifies(m, reference)} for u, m in candidates.items()},
              "chosen_update": chosen, "rule": __doc__.split("\n\n")[1]}
    (args.run / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    if chosen is None:
        raise SystemExit("No checkpoint keeps the ECG and PPG tasks within the margins; see selection.json.")
    shutil.copy2(args.run / "checkpoints" / f"update_{chosen:06d}.pt", args.run / "selected.pt")
    print("chosen update", chosen, flush=True)


if __name__ == "__main__":
    main()
