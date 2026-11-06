"""Arm B text records: charted vital signs, nursing assessments, chest imaging and ECG reports.

`--set all` builds every source (arm B). `--set signal` keeps only text about what a
10-second ECG, PPG and Resp window shows: vital signs, respiratory nursing items and ECG
reports, with no chest imaging (arms B2 and B3).

Each record is one text with a patient and a time. `data.py` pairs a window with the
nearest record of each source within that source's allowed time, then uses one of
them. Text is read for train patients only. Values are real recorded strings; nothing
is generated. A charted item with conflicting values at one charttime is omitted.
See finetune/README.md, sections 3.4 to 3.6.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

import common
from common import PLAN_PATH, TEXT_DIRS
from tokenizer_utils import SPECIAL_IDS, TEXT_LENGTH, continuation_ids, load_tokenizer, tokenizer_contract

EHR = Path(os.environ["MIMIC_IV_DIR"])
NOTE = Path(os.environ["MIMIC_IV_NOTE_DIR"])
ECG = Path(os.environ["MIMIC_IV_ECG_DIR"])
SOURCES = ("vitals", "nursing", "chest", "ecg")
# Order sets truncation priority inside the 64-token limit.
VITALS = (220045, 220048, 224650, 220277, 220210, 229322, 226732, 220181,
          223849, 229314, 223835, 220339, 224685)
REQUIRED = {220045, 220210, 220277}
# Bedside assessments. GCS items are left out because they are SOFA-2 brain inputs.
NURSING = (223985, 223990, 229323, 224723, 223986, 223987, 223988, 223989, 223991, 223992,
           224369, 224370, 224373, 223836, 224389, 226104, 223898, 223910)
# The `signal` set keeps the nursing items about breathing and the airway only.
RESPIRATORY_NURSING = (223985, 223990, 229323, 224723, 223836)
DAY_US = 86_400_000_000
# Sections that can follow IMPRESSION in a radiology report.
AFTER_IMPRESSION = re.compile(r"\n\s*(?:NOTIFICATION|RECOMMENDATIONS?|RECOMMENDATION\(S\)|ADDENDUM|FINDINGS)\s*:", re.I)


def microseconds(values):
    return pd.to_datetime(values).astype("int64") // 1000


def snapshot_parts(values, labels, fields):
    """`<label>: <value>.` per field, and the character span of every value."""
    text, spans = "", []
    for item in fields:
        if item in values:
            text += (" " if text else "") + f"{labels[item]}: "
            spans.append((len(text), len(text) + len(values[item])))
            text += f"{values[item]}."
    return text, spans


def snapshot_text(values, labels, fields):
    return snapshot_parts(values, labels, fields)[0]


def impression(report):
    """The IMPRESSION section, without de-identification blanks."""
    found = re.search(r"IMPRESSION\s*:(.*)", report, flags=re.S | re.I)
    if not found:
        return ""
    text = AFTER_IMPRESSION.split(found.group(1))[0]
    return " ".join(re.sub(r"_{2,}", " ", text).split())


def ecg_report(row):
    lines = [str(value).strip().rstrip(".") for value in row if isinstance(value, str) and value.strip()]
    return ". ".join(lines) + "." if lines else ""


def encode(tokenizer, texts, continuation, spans=None):
    """[CLS] text [SEP] in 64 tokens; a longer text is cut at the last whole word that fits.

    `value[i, k]` is j + 1 when token k lies in the j-th character span of text i, else 0.
    """
    tokens = np.zeros((len(texts), TEXT_LENGTH), dtype=np.int32)
    attention = np.zeros((len(texts), TEXT_LENGTH), dtype=bool)
    value = np.zeros((len(texts), TEXT_LENGTH), dtype=np.int8)
    cut = 0
    for left in range(0, len(texts), 50_000):
        found = tokenizer(list(texts[left:left + 50_000]), add_special_tokens=False, return_offsets_mapping=True)
        for offset, (ids, places) in enumerate(zip(found["input_ids"], found["offset_mapping"])):
            if len(ids) > TEXT_LENGTH - 2:
                end = TEXT_LENGTH - 2
                while end > 0 and ids[end] in continuation:
                    end -= 1
                ids, cut = ids[:end], cut + 1
            row = left + offset
            full = [SPECIAL_IDS["[CLS]"], *ids, SPECIAL_IDS["[SEP]"]]
            tokens[row, :len(full)] = full
            attention[row, :len(full)] = True
            for number, (start, stop) in enumerate(spans[row] if spans is not None else ()):
                for k, (begin, _) in enumerate(places[:len(ids)]):
                    if start <= begin < stop:
                        value[row, k + 1] = number + 1
    special = ~attention
    special[:, 0] = True
    special[np.arange(len(texts)), attention.sum(1) - 1] = True
    return tokens, attention, special, value, cut


def charted(stays, subjects, labels, nursing=NURSING):
    """Vital-sign and nursing snapshots: every selected item sharing a stay and charttime."""
    groups = {"vitals": defaultdict(lambda: defaultdict(set)), "nursing": defaultdict(lambda: defaultdict(set))}
    which = {**{item: "vitals" for item in VITALS}, **{item: "nursing" for item in nursing}}
    columns = ["stay_id", "charttime", "itemid", "value"]
    for chunk in pd.read_csv(EHR / "icu/chartevents.csv.gz", usecols=columns, chunksize=4_000_000,
                             dtype={"value": str}, keep_default_na=False):
        chunk = chunk[chunk.itemid.isin(which) & chunk.stay_id.isin(stays) & (chunk.value.str.strip() != "")]
        for stay, charttime, item, value in chunk.itertuples(index=False):
            groups[which[item]][(int(stay), charttime)][int(item)].add(" ".join(value.split()))
    records, counts, items = [], Counter(), Counter()
    for source, fields in (("vitals", VITALS), ("nursing", nursing)):
        for (stay, charttime), found in groups[source].items():
            values = {item: next(iter(seen)) for item, seen in found.items() if len(seen) == 1}
            counts[f"{source}_conflicting_items"] += len(found) - len(values)
            if source == "vitals" and not REQUIRED & set(values):
                counts["vitals_without_required"] += 1
                continue
            if not values:
                continue
            items.update(f"{source}:{labels[item]}" for item in values)
            records.append((subjects[stay], charttime, source, *snapshot_parts(values, labels, fields)))
    return records, counts, items


def chest_reports(subjects):
    names = pd.read_csv(NOTE / "note/radiology_detail.csv.gz")
    names = names[names.field_name == "exam_name"].drop_duplicates("note_id").set_index("note_id").field_value
    records, counts = [], Counter()
    for chunk in pd.read_csv(NOTE / "note/radiology.csv.gz", usecols=["note_id", "subject_id", "charttime", "text"],
                             chunksize=200_000):
        chunk = chunk[chunk.subject_id.isin(subjects)]
        chunk = chunk[chunk.note_id.map(names).fillna("").str.contains("CHEST", case=False)]
        for subject, charttime, text in chunk[["subject_id", "charttime", "text"]].itertuples(index=False):
            found = impression(text)
            counts["chest_reports"] += 1
            if found:
                records.append((int(subject), charttime, "chest", found, []))
            else:
                counts["chest_without_impression"] += 1
    return records, counts


def ecg_reports(subjects):
    columns = ["subject_id", "ecg_time"] + [f"report_{i}" for i in range(18)]
    table = pd.read_csv(ECG / "machine_measurements.csv", usecols=columns, low_memory=False)
    table = table[table.subject_id.isin(subjects)]
    texts = table[columns[2:]].apply(ecg_report, axis=1)
    keep = texts != ""
    return [(int(s), t, "ecg", x, []) for s, t, x in zip(table.subject_id[keep], table.ecg_time[keep], texts[keep])], \
        Counter(ecg_reports=int(len(table)), ecg_without_text=int((~keep).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--set", choices=tuple(TEXT_DIRS), default="all",
                        help="all: every source (arm B); signal: no chest imaging, respiratory nursing only")
    parser.add_argument("--output", type=Path, help="Defaults to the set's folder in common.TEXT_DIRS")
    args = parser.parse_args()
    args.output = args.output or TEXT_DIRS[args.set]
    nursing = NURSING if args.set == "all" else RESPIRATORY_NURSING
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{args.plan}?mode=ro", uri=True)
    stays = pd.read_sql("SELECT stay_id, subject_id, intime_us, outtime_us FROM stays WHERE split='train'", db)
    db.close()
    subjects = dict(zip(stays.stay_id, stays.subject_id))
    span = stays.groupby("subject_id").agg(lo=("intime_us", "min"), hi=("outtime_us", "max"))
    labels = dict(pd.read_csv(EHR / "icu/d_items.csv.gz", usecols=["itemid", "label"]).values)
    records, counts, items = charted(set(subjects), subjects, labels, nursing)
    print("[charted]", len(records), flush=True)
    reports = (chest_reports, ecg_reports) if args.set == "all" else (ecg_reports,)
    for part, found in (read(set(span.index)) for read in reports):
        records.extend(part)
        counts.update(found)
    table = pd.DataFrame(records, columns=["subject_id", "time", "source", "text", "spans"])
    table["time_us"] = microseconds(table.time)
    lo, hi = table.subject_id.map(span.lo) - DAY_US, table.subject_id.map(span.hi) + DAY_US
    counts["outside_icu_by_more_than_a_day"] = int((~table.time_us.between(lo, hi)).sum())
    table = table[table.time_us.between(lo, hi)].sort_values(["subject_id", "time_us", "source"]).reset_index(drop=True)
    tokens, attention, special, value, cut = encode(load_tokenizer(), table.text.tolist(), continuation_ids(),
                                                    table.spans.tolist())
    np.savez(args.output / "records.npz", subject_id=table.subject_id.to_numpy(np.int64),
             time_us=table.time_us.to_numpy(np.int64),
             source=table.source.map(SOURCES.index).to_numpy(np.int8),
             tokens=tokens, attention=attention, special=special, value=value)
    lengths = attention.sum(1)
    content = attention & ~special
    summary = {"set": args.set,
               "records": {source: int((table.source == source).sum()) for source in SOURCES},
               "patients": int(table.subject_id.nunique()), "cut_to_64_tokens": cut,
               "median_tokens": {source: float(np.median(lengths[table.source.to_numpy() == source]))
                                 for source in SOURCES if (table.source == source).any()},
               "counts": dict(counts), "items": dict(items.most_common()),
               "vital_fields": {str(item): labels[item] for item in VITALS},
               "nursing_fields": {str(item): labels[item] for item in nursing},
               "value_token_share": {source: float((value[mask] > 0).sum() / max(content[mask].sum(), 1))
                                     for source in ("vitals", "nursing")
                                     for mask in [table.source.to_numpy() == source] if mask.any()},
               "tokenizer": tokenizer_contract()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("records", "patients", "cut_to_64_tokens", "median_tokens")}), flush=True)


if __name__ == "__main__":
    main()
