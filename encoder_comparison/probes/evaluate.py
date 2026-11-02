"""Compare frozen encoders with same-budget XGBoost probes on validation patients.

Predictors fit on 90% of train patients, thresholds come from the other 10%, and scores
come from validation patients. The test split is never read. Differences use a paired
patient bootstrap, and verdicts apply the margins below, fixed before any result was seen.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from common import ENCODERS, ICD_CARDIAC, ICD_RESP, PRIVATE, RESP_INPUTS, RESULTS
from probe_labels.build import RHYTHM_CLASSES

HERE = Path(__file__).resolve().parent
PREDICTIONS = PRIVATE / "predictions"
FEATURES = PRIVATE / "features"
MAX_WINDOWS_PER_PATIENT = 6
MIN_POSITIVE_FIT, MIN_POSITIVE_VAL = 10, 5
LOWER_IS_BETTER = {"mae"}
XGB = dict(n_estimators=300, max_depth=6, learning_rate=.05, random_state=42)
ALL = ("ecg", "ppg", "ecg_ppg")
RESP_SET = ("ecg_ppg", "all3", "resp")


@dataclass(frozen=True)
class Task:
    name: str
    part: int
    level: str
    kind: str
    columns: tuple
    inputs: tuple
    metric: str
    margin: float = 0.
    gate: bool = True


def icd(names):
    return tuple(f"icd_{name}" for name in names)


TASKS = (
    Task("rhythm", 1, "window", "multiclass", ("rhythm",), ("ecg",), "macro_f1", .02),
    Task("icd_cardiac", 1, "patient", "multilabel", icd(ICD_CARDIAC), ("ecg",), "macro_f1", .02),
    Task("pvc", 1, "window", "binary", ("pvc",), ("ecg",), "auroc", .02),
    Task("pac", 1, "window", "binary", ("pac",), ("ecg",), "auroc", .02),
    Task("avb1", 1, "window", "binary", ("avb1",), ("ecg",), "auroc", .02),
    Task("qtc", 1, "window", "regression", ("qtc",), ("ecg",), "mae", 3.),
    Task("age", 1, "patient", "regression", ("age",), ALL, "mae", .5),
    Task("sex", 1, "patient", "binary", ("female",), ALL, "auroc", .02),
    Task("bmi", 1, "patient", "regression", ("bmi",), ALL, "mae", .25),
    Task("sbp", 1, "window", "regression", ("sbp",), ALL, "mae", 1.),
    Task("dbp", 1, "window", "regression", ("dbp",), ALL, "mae", 1.),
    Task("spo2", 1, "window", "regression", ("spo2",), ("ppg",), "mae", .25),
    Task("death_1y", 1, "patient", "binary", ("death_1y",), ALL, "auroc", .02),
    Task("death_hospital", 1, "patient", "binary", ("death_hospital",), ALL, "auroc", .02),
    Task("death_30d", 1, "patient", "binary", ("death_30d",), ALL, "auroc", .02),
    Task("death_24h", 1, "patient", "binary", ("death_24h",), ALL, "auroc", .02, False),
    Task("rbbb", 1, "window", "binary", ("rbbb",), ("ecg",), "auroc", .02, False),
    Task("lbbb", 1, "window", "binary", ("lbbb",), ("ecg",), "auroc", .02, False),
    Task("hr", 1, "window", "regression", ("hr",), ("ecg",), "mae", 1., False),
    Task("rr_monitor", 2, "window", "regression", ("rr_monitor",), RESP_SET, "mae"),
    Task("support", 2, "window", "binary", ("support",), RESP_SET, "auroc"),
    Task("breathing", 2, "window", "binary", ("breathing",), RESP_SET, "auroc", gate=False),
    Task("rr_charted", 2, "window", "regression", ("rr_charted",), RESP_SET, "mae", gate=False),
    Task("icd_respiratory", 2, "patient", "multilabel", icd(ICD_RESP), RESP_SET, "macro_auroc"),
)


def in_tuning_fold(subject):
    return int(hashlib.sha256(str(subject).encode()).hexdigest(), 16) % 10 == 0


def cap_windows(subjects, limit=MAX_WINDOWS_PER_PATIENT):
    """Up to `limit` evenly spaced rows per patient, never depending on a label value."""
    keep = []
    order = pd.Series(np.arange(len(subjects))).groupby(np.asarray(subjects)).indices
    for rows in order.values():
        keep.extend(rows if len(rows) <= limit else rows[np.linspace(0, len(rows) - 1, limit).astype(int)])
    return np.sort(np.asarray(keep))


def macro_f1(y, labels, classes):
    scores = []
    for value in classes:
        hits = int(((y == value) & (labels == value)).sum())
        total = int((y == value).sum()) + int((labels == value).sum())
        scores.append(2 * hits / total if total else 0.)
    return float(np.mean(scores))


def score(kind, y, prediction, thresholds=None):
    """All metrics for one model; a metric is None when one class is missing."""
    if kind == "regression":
        return {"mae": float(np.abs(y - prediction).mean()), "rmse": float(np.sqrt(np.square(y - prediction).mean()))}
    if kind == "multiclass":
        labels = prediction.argmax(1)
        return {"macro_f1": macro_f1(y, labels, range(prediction.shape[1])), "accuracy": float((labels == y).mean())}
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return {"auroc": None, "auprc": None, "f1": None}
        return {"auroc": float(roc_auc_score(y, prediction)), "auprc": float(average_precision_score(y, prediction)),
                "f1": float(f1_score(y, prediction >= float(np.ravel(thresholds)[0])))}
    if any(len(np.unique(y[:, column])) < 2 for column in range(y.shape[1])):
        return {"macro_f1": None, "macro_auroc": None}
    f1s = [f1_score(y[:, column], prediction[:, column] >= thresholds[column]) for column in range(y.shape[1])]
    aurocs = [roc_auc_score(y[:, column], prediction[:, column]) for column in range(y.shape[1])]
    return {"macro_f1": float(np.mean(f1s)), "macro_auroc": float(np.mean(aurocs))}


def fast_auroc(y, prediction):
    positive = y == 1
    n_positive = int(positive.sum())
    if n_positive in (0, len(y)):
        return None
    ranks = rankdata(prediction)
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * (len(y) - n_positive)))


def binary_f1(y, labels):
    hits = int((y.astype(bool) & labels).sum())
    total = int(y.sum()) + int(labels.sum())
    return 2 * hits / total if total else 0.


def primary(task, y, prediction, thresholds):
    """The task's primary metric alone, for bootstrap draws. Agrees with `score`."""
    if task.metric == "mae":
        return float(np.abs(y - prediction).mean())
    if task.metric == "auroc":
        return fast_auroc(y, prediction)
    if task.kind == "multiclass":
        return macro_f1(y, prediction.argmax(1), range(prediction.shape[1]))
    columns = range(y.shape[1])
    if any(len(np.unique(y[:, column])) < 2 for column in columns):
        return None
    if task.metric == "macro_f1":
        return float(np.mean([binary_f1(y[:, column], prediction[:, column] >= thresholds[column]) for column in columns]))
    return float(np.mean([fast_auroc(y[:, column], prediction[:, column]) for column in columns]))


def fit_predict(kind, x_fit, y_fit, others, threads):
    """Predictions for each array in `others`. One binary model per label when multilabel."""
    if kind == "regression":
        model = XGBRegressor(**XGB, n_jobs=threads).fit(x_fit, y_fit)
        return [model.predict(x).astype(float) for x in others]
    if kind == "multiclass":
        model = XGBClassifier(**XGB, n_jobs=threads, objective="multi:softprob").fit(x_fit, y_fit.astype(int))
        return [model.predict_proba(x).astype(float) for x in others]
    if kind == "binary":
        model = XGBClassifier(**XGB, n_jobs=threads).fit(x_fit, y_fit.astype(int))
        return [model.predict_proba(x)[:, 1].astype(float) for x in others]
    parts = [fit_predict("binary", x_fit, y_fit[:, column], others, threads) for column in range(y_fit.shape[1])]
    return [np.stack([part[index] for part in parts], axis=1) for index in range(len(others))]


def threshold(y, prediction):
    """Threshold with the best F1; ties favor proximity to 0.5, then the lower value."""
    if len(np.unique(y)) != 2:
        return .5
    precision, recall, cuts = precision_recall_curve(y, prediction)
    scores = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                       out=np.zeros_like(cuts), where=(precision[:-1] + recall[:-1]) > 0)
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    return float(min(cuts[best], key=lambda value: (abs(value - .5), value)))


def tune_thresholds(kind, y, prediction):
    if kind == "binary":
        return np.array([threshold(y, prediction)])
    if kind == "multilabel":
        return np.array([threshold(y[:, column], prediction[:, column]) for column in range(y.shape[1])])
    return np.array([.5])


def constant(kind, y_fit, n):
    if kind == "multilabel":
        return np.tile(y_fit.mean(0), (n, 1))
    if kind == "multiclass":
        return np.tile(np.bincount(y_fit.astype(int), minlength=int(y_fit.max()) + 1) / len(y_fit), (n, 1))
    return np.full(n, float(y_fit.mean()))


def load_tables():
    windows = pd.read_pickle(PRIVATE / "windows.pkl").reset_index(drop=True)
    patients = pd.read_pickle(PRIVATE / "patients.pkl").reset_index(drop=True)
    return windows, patients


def task_rows(task, windows, patients):
    """Rows, labels, patients and splits for one task."""
    table = windows if task.level == "window" else patients
    known = np.flatnonzero(table[list(task.columns)].notna().all(axis=1).to_numpy())
    if task.level == "window":
        known = known[cap_windows(table.subject_id.to_numpy()[known])]
    y = table[list(task.columns)].to_numpy(dtype=float)[known]
    return known, y if task.kind == "multilabel" else y[:, 0], table.subject_id.to_numpy()[known], table.split.to_numpy()[known]


def features_for(task, encoder, name, windows, patients, known):
    store = np.load(FEATURES / f"{encoder}__{name}.npy", mmap_mode="r")
    if task.level == "window":
        return np.asarray(store[known], dtype=np.float32)
    frame = pd.DataFrame(np.asarray(store, dtype=np.float32))
    pooled = frame.groupby(windows.subject_id.to_numpy()).mean()
    return pooled.reindex(patients.subject_id.to_numpy()[known]).to_numpy(dtype=np.float32)


def enough(kind, y, fit, val):
    if kind == "regression":
        return True, ""
    if kind == "multiclass":
        low_fit = int(np.bincount(y[fit].astype(int), minlength=len(RHYTHM_CLASSES)).min())
        low_val = int(np.bincount(y[val].astype(int), minlength=len(RHYTHM_CLASSES)).min())
        return low_fit >= MIN_POSITIVE_FIT and low_val >= MIN_POSITIVE_VAL, f"smallest class: fit {low_fit}, validation {low_val}"
    labels = y if y.ndim == 2 else y[:, None]
    if int(labels[fit].sum(0).min()) < MIN_POSITIVE_FIT or int(labels[val].sum(0).min()) < MIN_POSITIVE_VAL:
        return False, f"positives: fit {labels[fit].sum(0).min():.0f}, validation {labels[val].sum(0).min():.0f}"
    return True, ""


def run_task(task, name, windows, patients, threads):
    """Fit every encoder on one task and input, reusing saved predictions."""
    path = PREDICTIONS / f"{task.name}__{name}.npz"
    if path.is_file():
        return path
    known, y, subjects, splits = task_rows(task, windows, patients)
    tuning = np.array([in_tuning_fold(subject) for subject in subjects]) & (splits == "train")
    fit, val = (splits == "train") & ~tuning, splits == "val"
    ok, reason = enough(task.kind, y, fit, val)
    if not ok:
        return reason
    result = {"y": y[val], "subjects": subjects[val], "constant": constant(task.kind, y[fit], int(val.sum())), "fit_rows": int(fit.sum())}
    if task.name.startswith("rr_"):
        result["spectral"] = windows.rr_spectral.to_numpy()[known][val]
    for encoder in ENCODERS:
        if encoder == "original" and name in RESP_INPUTS:
            continue
        x = features_for(task, encoder, name, windows, patients, known)
        tuned, scored = fit_predict(task.kind, x[fit], y[fit], [x[tuning], x[val]], threads)
        result[f"pred_{encoder}"] = scored
        result[f"thr_{encoder}"] = tune_thresholds(task.kind, y[tuning], tuned)
        print(task.name, name, encoder, {key: None if value is None else round(value, 4) for key, value in
                                         score(task.kind, y[val], scored, result[f"thr_{encoder}"]).items()}, flush=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, **result)
    temporary.replace(path)
    return path


def bootstrap(task, data, models, draws, seed=42):
    """Primary metric per model per draw, resampling patients, shared across models."""
    y, subjects = data["y"], data["subjects"]
    groups = list(pd.Series(np.arange(len(subjects))).groupby(subjects).indices.values())
    rng = np.random.default_rng(seed)
    output = {model: [] for model in models}
    for _ in range(draws):
        index = np.concatenate([groups[position] for position in rng.integers(0, len(groups), len(groups))])
        values = {model: primary(task, y[index], data[f"pred_{model}"][index], data[f"thr_{model}"]) for model in models}
        if all(value is not None for value in values.values()):
            for model, value in values.items():
                output[model].append(value)
    return {model: np.asarray(values) for model, values in output.items()}


def gain(metric, reference, candidate):
    """Positive means the candidate is better."""
    return reference - candidate if metric in LOWER_IS_BETTER else candidate - reference


def retention_verdict(point, low, high, margin):
    if point < -margin and high < 0:
        return "lost"
    if low >= -margin:
        return "retained"
    return "inconclusive"


def resp_verdict(low, high):
    if low > 0:
        return "better"
    if high < 0:
        return "worse"
    return "same"


def comparisons(task, name, data, draws):
    """Rows comparing candidates with the reference encoder for this part."""
    pairs = [("original", "epoch17"), ("original", "arm_a"), ("original", "arm_b")]
    if task.part == 2:
        pairs.append(("arm_a", "arm_b"))
    models = sorted({model for pair in pairs for model in pair})
    sampled = bootstrap(task, data, models, draws)
    rows = []
    for reference, candidate in pairs:
        point = gain(task.metric, score(task.kind, data["y"], data[f"pred_{reference}"], data[f"thr_{reference}"])[task.metric],
                     score(task.kind, data["y"], data[f"pred_{candidate}"], data[f"thr_{candidate}"])[task.metric])
        low, high = np.quantile(gain(task.metric, sampled[reference], sampled[candidate]), [.025, .975])
        verdict = retention_verdict(point, low, high, task.margin) if task.part == 1 else resp_verdict(low, high)
        rows.append({"part": task.part, "task": task.name, "input": name, "metric": task.metric, "reference": reference,
                     "candidate": candidate, "gain": round(float(point), 4), "ci_low": round(float(low), 4),
                     "ci_high": round(float(high), 4), "margin": task.margin, "gate": task.gate, "verdict": verdict,
                     "draws": len(sampled[reference])})
    return rows


def metric_rows(task, name, data):
    rows = []
    models = [key[5:] for key in data if key.startswith("pred_")]
    baselines = [("constant", data["constant"], np.array([.5] * max(1, data["y"].shape[1] if data["y"].ndim == 2 else 1)))]
    if "spectral" in data:
        baselines.append(("spectral", data["spectral"], np.array([.5])))
    for model, prediction, thresholds in [(model, data[f"pred_{model}"], data[f"thr_{model}"]) for model in models] + baselines:
        for metric, value in score(task.kind, data["y"], prediction, thresholds).items():
            rows.append({"part": task.part, "task": task.name, "input": name, "model": model, "metric": metric,
                         "value": None if value is None else round(value, 4), "val_rows": len(data["y"]),
                         "val_patients": len(np.unique(data["subjects"])), "fit_rows": int(data["fit_rows"])})
    return rows


def decide(table):
    """Apply the pre-specified rule to the comparison table."""
    lines, passed = [], {}
    gates = table[(table.part == 1) & table.gate]
    resp = table[(table.part == 2) & (table.input == "all3") & (table.reference == "original")]
    for candidate in ("arm_a", "arm_b"):
        rows = gates[gates.candidate == candidate]
        lost = rows[rows.verdict == "lost"]
        unclear = rows[rows.verdict == "inconclusive"]
        mine = resp[resp.candidate == candidate].set_index("task").verdict
        rr, support, icd_resp = (mine.get(task, "missing") for task in ("rr_monitor", "support", "icd_respiratory"))
        improved = "better" in (rr, support) and "worse" not in (rr, support) and "missing" not in (rr, support)
        fair = candidate == "arm_a" or icd_resp in ("same", "better")
        passed[candidate] = lost.empty and improved and fair
        lines += [f"- {candidate}: {'PASS' if passed[candidate] else 'FAIL'}. Lost {len(lost)}, inconclusive {len(unclear)} of {len(rows)} "
                  f"retention checks. All three channels against original CSFM on ECG + PPG: RR {rr}, invasive support {support}, ICD respiratory {icd_resp}."]
        lines += [f"  - lost: {row.task} on {row.input}, {row.metric} gain {row.gain} [{row.ci_low}, {row.ci_high}]" for row in lost.itertuples()]
    versus = table[(table.part == 2) & (table.input == "all3") & (table.reference == "arm_a") & table.task.isin(["rr_monitor", "support"])]
    if all(passed.values()):
        chosen = "arm_b" if (versus.verdict == "better").any() else "arm_a"
    elif any(passed.values()):
        chosen = next(name for name, value in passed.items() if value)
    else:
        chosen = None
    epoch = gates[gates.candidate == "epoch17"]
    lines.append(f"- epoch17 for reference: lost {int((epoch.verdict == 'lost').sum())} of {len(epoch)} retention checks.")
    lines.append(f"- Chosen encoder: {chosen}." if chosen else "- No arm passed, so this needs a decision by the study team.")
    return lines, chosen


def markdown(frame):
    cells = frame.astype(object).where(frame.notna(), "").astype(str)
    rows = [list(frame.columns), ["---"] * frame.shape[1], *cells.to_numpy().tolist()]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def write_report(metrics, table, lines, skipped):
    primary = metrics.merge(pd.DataFrame([{"task": task.name, "metric": task.metric} for task in TASKS]), on=["task", "metric"])
    wide = primary.pivot_table(index=["part", "task", "input", "metric"], columns="model", values="value", sort=False)
    order = [name for name in ("constant", "spectral", *ENCODERS) if name in wide]
    text = ["# Encoder comparison", "", "Validation patients only. See README.md for the method.", "",
            "The Resp reference was changed from epoch 17 to original CSFM after the first results were seen, to match the stated goal.", "",
            "## Decision", "", *lines, "",
            "## Primary metric per task", "", markdown(wide[order].reset_index()), "",
            "## Paired comparisons", "", "`gain` is positive when the candidate is better than the reference.", "",
            markdown(table.drop(columns=["part"])), ""]
    if skipped:
        text += ["## Skipped", "", *[f"- {key}: {value}" for key, value in skipped.items()], ""]
    (RESULTS / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--tasks", nargs="*", help="Only these tasks")
    args = parser.parse_args()
    PREDICTIONS.mkdir(mode=0o700, parents=True, exist_ok=True)
    windows, patients = load_tables()
    if set(windows.split) != {"train", "val"} or set(patients.split) != {"train", "val"}:
        raise RuntimeError("Only train and validation patients may enter this comparison.")
    metrics, rows, skipped = [], [], {}
    for task in TASKS:
        if args.tasks and task.name not in args.tasks:
            continue
        for name in task.inputs:
            result = run_task(task, name, windows, patients, args.threads)
            if isinstance(result, str):
                skipped[f"{task.name} on {name}"] = result
                print("skipped", task.name, name, result, flush=True)
                continue
            with np.load(result) as stored:
                data = {key: stored[key] for key in stored.files}
            if name in RESP_INPUTS:
                # Original CSFM has no Resp channel, so its ECG + PPG result is the reference.
                with np.load(PREDICTIONS / f"{task.name}__ecg_ppg.npz") as stored:
                    data.update(pred_original=stored["pred_original"], thr_original=stored["thr_original"])
            metrics += metric_rows(task, name, data)
            rows += comparisons(task, name, data, args.bootstrap)
            print("compared", task.name, name, flush=True)
    metrics, table = pd.DataFrame(metrics), pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    table.to_csv(RESULTS / "comparisons.csv", index=False)
    lines, chosen = decide(table)
    (RESULTS / "decision.json").write_text(json.dumps({"chosen": chosen, "lines": lines, "skipped": skipped}, indent=2) + "\n")
    write_report(metrics, table, lines, skipped)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
