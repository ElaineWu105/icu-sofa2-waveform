"""Arm B text: charted vital-sign snapshots on the ICU 10-second grid.

One snapshot is every selected field sharing a charttime in a planned ICU day.
Values are real charted strings; nothing is generated. An item with conflicting
values at one charttime is omitted. Snapshots need HR, RR or SpO2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

import common
from common import PLAN_PATH, TEXT_DIR
from tokenizer_utils import load_tokenizer, tokenizer_contract

CHART = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/icu/chartevents.csv.gz"
ITEMS = os.environ["MIMIC_IV_DIR"].rstrip("/") + "/icu/d_items.csv.gz"
DAY_US, SLOT_US = 86_400_000_000, 10_000_000
# Order sets truncation priority inside the 64-token limit.
FIELDS = (220045, 220048, 224650, 220277, 220210, 229322, 226732, 220181,
          223849, 229314, 223835, 220339, 224685)
REQUIRED = {220045, 220210, 220277}


def snapshot_text(values, labels):
    return " ".join(f"{labels[item]}: {values[item]}." for item in FIELDS if item in values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output", type=Path, default=TEXT_DIR)
    parser.add_argument("--chart", default=CHART)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{args.plan}?mode=ro", uri=True)
    planned = {(stay, day) for stay, day in db.execute("SELECT stay_id, day FROM days")}
    intime = dict(db.execute("SELECT stay_id, intime_us FROM stays"))
    db.close()
    stays = {stay for stay, _ in planned}
    labels = dict(pd.read_csv(ITEMS, usecols=["itemid", "label"]).values)
    groups = defaultdict(lambda: defaultdict(set))
    columns = ["stay_id", "charttime", "itemid", "value"]
    for chunk in pd.read_csv(args.chart, usecols=columns, chunksize=4_000_000, dtype={"value": str}, keep_default_na=False):
        chunk = chunk[chunk.itemid.isin(FIELDS) & chunk.stay_id.isin(stays) & (chunk.value.str.strip() != "")]
        for stay, charttime, item, value in chunk.itertuples(index=False):
            groups[(int(stay), charttime)][int(item)].add(" ".join(value.split()))
    print("[scan] snapshots", len(groups), flush=True)
    rows, texts, counts = [], [], Counter()
    for (stay, charttime), fields in groups.items():
        values = {item: next(iter(found)) for item, found in fields.items() if len(found) == 1}
        counts["conflicting_items"] += len(fields) - len(values)
        if not REQUIRED & set(values):
            counts["without_required_vital"] += 1
            continue
        stamp = pd.Timestamp(charttime)
        offset = int((stamp.value // 1000) - intime[stay])
        if offset < 0:
            continue
        day, slot = offset // DAY_US, (offset % DAY_US) // SLOT_US
        if (stay, day) not in planned:
            counts["outside_planned_day"] += 1
            continue
        rows.append((stay, day, slot))
        texts.append(snapshot_text(values, labels))
    tokenizer = load_tokenizer()
    tokens, attention, special = [], [], []
    for left in range(0, len(texts), 50_000):
        encoded = tokenizer(texts[left:left + 50_000], max_length=64, padding="max_length", truncation=True,
                            return_special_tokens_mask=True)
        tokens.append(np.array(encoded["input_ids"], dtype=np.int32))
        attention.append(np.array(encoded["attention_mask"], dtype=np.bool_))
        special.append(np.array(encoded["special_tokens_mask"], dtype=np.bool_))
    tokens = np.concatenate(tokens)
    rows = np.array(rows, dtype=np.int64)
    np.savez(args.output / "snapshots.npz", stay_id=rows[:, 0], day=rows[:, 1], slot=rows[:, 2],
             tokens=tokens, attention=np.concatenate(attention), special=np.concatenate(special))
    full = sum(int(tokens[left:left + 100_000].shape[0]) for left in range(0, len(tokens), 100_000))
    summary = {"snapshots": len(rows), "stays": len({int(value) for value in rows[:, 0]}),
               "truncated": int((np.concatenate(attention).sum(1) == 64).sum()),
               "median_tokens": float(np.median(np.concatenate(attention).sum(1))), "tokens_checked": full,
               "counts": dict(counts), "fields": {str(item): labels[item] for item in FIELDS},
               "tokenizer": tokenizer_contract()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("snapshots", "stays", "truncated", "median_tokens", "counts")}), flush=True)


if __name__ == "__main__":
    main()
