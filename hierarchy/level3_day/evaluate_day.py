"""Level 3 endpoints of labels/README.md, section 7, on validation patients, with a paired patient bootstrap.

Reads the validation labels and every method's saved predictions in one handling's run folder and writes
`endpoints.json`. Seeds of one method are averaged into one probability before the bootstrap; per-seed point
values are also reported. Every model is scored on the same bootstrap draws, so differences are paired.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from common import ORGANS, RUNS

LIVER = ORGANS.index("liver")
MIN_EVENTS_SECONDARY = 30
PAIRS = (("hierarchy", "e2"), ("hierarchy", "m"), ("hierarchy", "e0"), ("m", "e2"), ("e2", "e0"))


def auroc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def auprc(y, p):
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else None


def calibration(y, p, bins=10):
    """Slope of a logistic recalibration on logit(p), calibration-in-the-large with slope fixed at 1, Brier, and a
    curve of 10 quantile bins."""
    if not 0 < y.sum() < len(y):
        return None
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    z = logit(clipped)
    slope = float(LogisticRegression(penalty=None).fit(z[:, None], y).coef_[0, 0])
    intercept = 0.
    for _ in range(50):
        q = expit(z + intercept)
        step = (y - q).sum() / max((q * (1 - q)).sum(), 1e-9)
        intercept += step
        if abs(step) < 1e-8:
            break
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    curve = [{"predicted": float(p[which == b].mean()), "observed": float(y[which == b].mean()), "n": int((which == b).sum())}
             for b in range(len(edges) - 1) if (which == b).any()]
    return {"slope": slope, "intercept": float(intercept), "brier": float(np.mean((p - y) ** 2)), "curve": curve}


def organ_event(labels, cumulative, organ, kind):
    """Eligible anchors, events and probabilities for one organ event (labels/README.md, section 7.1)."""
    base = labels["base"][:, organ].astype(int)
    future = labels["future"][:, organ]
    known = labels["label"][:, organ]
    rows = np.arange(len(base))
    if kind == "worse1":
        eligible = known & (base <= 3)
        probability = cumulative[rows, organ, np.clip(base, 0, 3)]
        event = future > base
    elif kind == "worse2":
        eligible = known & (base <= 2)
        probability = cumulative[rows, organ, np.clip(base + 1, 0, 3)]
        event = future >= base + 2
    elif kind == "improve1":
        eligible = known & (base >= 1)
        probability = 1 - cumulative[rows, organ, np.clip(base - 1, 0, 3)]
        event = future < base
    else:
        raise ValueError(kind)
    return eligible, event.astype(float), probability


def primary(labels, cumulative, composite, index=None):
    """Composite AUROC and the macro organ-worsening AUROC (with and without liver) on rows `index`."""
    index = np.arange(len(labels["base"])) if index is None else index
    known = labels["event_known"][index]
    result = {"composite_auroc": auroc(labels["event"][index][known], composite[index][known]) if composite is not None else None}
    per = []
    for organ in range(len(ORGANS)):
        sub = {key: value[index] for key, value in labels.items() if key in ("base", "future", "label")}
        eligible, event, probability = organ_event(sub, cumulative[index], organ, "worse1")
        per.append(auroc(event[eligible], probability[eligible]))
    result["organ_worse1_macro_auroc"] = float(np.mean(per)) if all(value is not None for value in per) else None
    rest = [value for organ, value in enumerate(per) if organ != LIVER]
    result["organ_worse1_macro_auroc_no_liver"] = float(np.mean(rest)) if all(value is not None for value in rest) else None
    return result


def full_report(labels, cumulative, composite, handling):
    report = {}
    if composite is not None:
        known = labels["event_known"]
        y, p = labels["event"][known], composite[known]
        report["composite"] = {"n": int(known.sum()), "events": int(y.sum()), "rate": float(y.mean()), "auroc": auroc(y, p),
                               "auprc": auprc(y, p), "calibration": calibration(y, p)}
    kinds = ["worse1", "worse2"] + (["improve1"] if handling == "locf" else [])
    for kind in kinds:
        per = {}
        for organ, name in enumerate(ORGANS):
            eligible, event, probability = organ_event(labels, cumulative, organ, kind)
            y, p = event[eligible], probability[eligible]
            entry = {"n": int(eligible.sum()), "events": int(y.sum()), "auroc": auroc(y, p), "auprc": auprc(y, p)}
            if kind == "worse1":
                entry["calibration"] = calibration(y, p)
            if kind == "worse2" and y.sum() < MIN_EVENTS_SECONDARY:
                entry.update(auroc=None, auprc=None, note=f"fewer than {MIN_EVENTS_SECONDARY} events")
            per[name] = entry
        values = [entry["auroc"] for entry in per.values()]
        if all(value is not None for value in values):
            per["macro_auroc"] = float(np.mean(values))
            per["macro_auroc_no_liver"] = float(np.mean([value for organ, value in enumerate(values) if organ != LIVER]))
        report[f"organ_{kind}"] = per
    k = np.arange(4)
    truth = labels["future"][..., None] > k
    rps, mae = {}, {}
    expected = cumulative.sum(-1)
    change, true_change = expected - labels["base"], labels["future"] - labels["base"]
    for organ, name in enumerate(ORGANS):
        known = labels["label"][:, organ]
        rps[name] = float(((cumulative[known, organ] - truth[known, organ]) ** 2).mean()) if known.any() else None
        error = np.abs(change[known, organ] - true_change[known, organ])
        changed = known & (true_change[:, organ] != 0)
        mae[name] = {"n": int(known.sum()), "mae": float(error.mean()) if known.any() else None,
                     "mae_changed": float(np.abs(change[changed, organ] - true_change[changed, organ]).mean()) if changed.any() else None}
    report["rps"] = rps
    official = labels["official_total"]
    keep = np.isfinite(official)
    total = (change * labels["base_mask"]).sum(1)
    mae["total"] = {"n": int(keep.sum()), "mae": float(np.abs(total[keep] - official[keep]).mean()),
                    "mae_changed": float(np.abs(total[keep & (official != 0)] - official[keep & (official != 0)]).mean())}
    report["mae"] = mae
    return report


def persistence(labels):
    """P: future equals baseline, as P(future > k) = 1[baseline > k]."""
    return (labels["base"][..., None] > np.arange(4)).astype(float)


def bootstrap(labels, models, draws, seed=42):
    """Primary endpoints per model on shared patient draws, and paired differences."""
    subjects = labels["subject"]
    groups = list(pd.Series(np.arange(len(subjects))).groupby(subjects).indices.values())
    rng = np.random.default_rng(seed)
    values = {name: [] for name in models}
    for _ in range(draws):
        index = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        for name, (cumulative, composite) in models.items():
            values[name].append(primary(labels, cumulative, composite, index))
    result = {}
    metrics = ("composite_auroc", "organ_worse1_macro_auroc", "organ_worse1_macro_auroc_no_liver")

    def interval(series):
        series = np.array([value for value in series if value is not None], dtype=float)
        if not len(series):
            return None
        low, high = np.quantile(series, [.025, .975])
        return {"low": float(low), "high": float(high), "draws": len(series)}

    for name in models:
        result[name] = {metric: interval([draw[metric] for draw in values[name]]) for metric in metrics}
    for left, right in PAIRS:
        if left in models and right in models:
            result[f"{left}_minus_{right}"] = {
                metric: interval([None if a[metric] is None or b[metric] is None else a[metric] - b[metric]
                                  for a, b in zip(values[left], values[right])]) for metric in metrics}
    return result


def evaluate(directory, handling, draws=1000):
    directory = Path(directory)
    with np.load(directory / "validation_labels.npz") as stored:
        labels = {key: stored[key] for key in stored.files}
    predictions = {}
    for path in sorted((directory / "predictions").glob("*_seed*.npz")):
        method = path.stem.rsplit("_seed", 1)[0]
        with np.load(path) as stored:
            predictions.setdefault(method, []).append((int(path.stem.rsplit("_seed", 1)[1]), stored["cumulative"], stored["composite"]))
    report = {"handling": handling, "n": int(len(labels["base"])), "patients": int(len(np.unique(labels["subject"]))), "methods": {}}
    models = {}
    for method, runs in predictions.items():
        cumulative = np.mean([run[1] for run in runs], axis=0)
        composite = np.mean([run[2] for run in runs], axis=0)
        models[method] = (cumulative, composite)
        report["methods"][method] = {"seeds": [run[0] for run in runs],
                                     "per_seed": {str(run[0]): primary(labels, run[1], run[2]) for run in runs},
                                     "seed_mean": full_report(labels, cumulative, composite, handling)}
    report["methods"]["persistence"] = {"seed_mean": {key: value for key, value in full_report(labels, persistence(labels), None, handling).items()
                                                      if key in ("rps", "mae")}}
    report["bootstrap"] = bootstrap(labels, models, draws) if models else {}
    (directory / "endpoints.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RUNS / "day"))
    parser.add_argument("--handling", default="main")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    report = evaluate(Path(args.output) / args.handling, args.handling, args.bootstrap)
    for method, entry in report["methods"].items():
        print(method, json.dumps({key: value for key, value in entry.get("per_seed", {}).items()}), flush=True)


if __name__ == "__main__":
    main()
