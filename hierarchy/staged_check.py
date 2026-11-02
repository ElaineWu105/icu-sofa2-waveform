"""Check for signal before training upper levels: current score, past change, then future change.

Mean-pooled three-minute features per ICU day, XGBoost on train patients, validation patients scored.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from common import ORGANS, RUNS
from day_data import load_three_min

XGB = dict(n_estimators=300, max_depth=6, learning_rate=.05, random_state=42)


def pooled(features, mask):
    weights = mask[..., None].astype(np.float32)
    return (np.nan_to_num(features.astype(np.float32)) * weights).sum(1) / weights.sum(1).clip(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", default=str(RUNS / "staged_check.json"))
    args = parser.parse_args()
    days, features, mask = load_three_min()
    x = pooled(features, mask)
    previous = days[["stay_id", "day"] + [f"{organ}_real_base" for organ in ORGANS]].copy()
    previous["day"] += 1
    joined = days.merge(previous, on=["stay_id", "day"], how="left", suffixes=("", "_previous"))
    split = days.split.to_numpy()
    report = {}
    for organ in ORGANS:
        base = joined[f"{organ}_real_base"].to_numpy(dtype=float)
        stages = {"current": base, "past_change": base - joined[f"{organ}_real_base_previous"].to_numpy(dtype=float),
                  "future_change": (joined[f"{organ}_real_next"].to_numpy(dtype=float) - base) * np.where(days.fate == "stayed", 1, np.nan)}
        report[organ] = {}
        for stage, y in stages.items():
            known = np.isfinite(y)
            fit, val = known & (split == "train"), known & (split == "val")
            model = XGBRegressor(**XGB, n_jobs=args.threads).fit(x[fit], y[fit])
            mean = float(y[fit].mean())
            report[organ][stage] = {"n": int(val.sum()), "mae": float(np.abs(model.predict(x[val]) - y[val]).mean()),
                                    "constant_mae": float(np.abs(mean - y[val]).mean())}
        print(json.dumps({organ: report[organ]}), flush=True)
    Path(args.output).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
