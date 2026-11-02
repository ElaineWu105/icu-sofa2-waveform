"""Where ICU days are lost before the waveform cut, and what relaxed rules would add.

Uses the plan database and the same header logic as plan.py. Reads no samples. Aggregates only.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "waveform/icu_cut"))
from common import DAY_US, MIN_VALID_CLIPS, SLOTS_PER_DAY  # noqa: E402
from plan import header_clips, slot_owners  # noqa: E402
from source import _choose_channels, _segment_right  # noqa: E402

PLAN = ROOT / "waveform/icu_cut/output/plan/plan.sqlite"
THRESHOLDS = {"80%": 384, "70%": 336, "60%": 288, "50%": 240, "30%": 144, "10%": 48, "any": 1}
CHANNELS = {"all_three": {"ecg", "ppg", "resp"}, "ecg_resp": {"ecg", "resp"}, "ecg_ppg": {"ecg", "ppg"}, "ecg": {"ecg"}}


def stay_days(job):
    stay, subject, intime, outtime, split, segments = job
    days = (outtime - intime - 1) // DAY_US
    if days < 1:
        return stay, subject, split, "stay_under_two_grid_days", []
    segments = [s for s in segments if int(s["origin_us"]) < outtime and _segment_right(s) > intime]
    if not segments:
        return stay, subject, split, "no_segment_overlaps_stay", []
    owners = slot_owners(segments, intime, outtime)
    chosen = [set(_choose_channels(s)) for s in segments]
    rows = []
    for day in range(days):
        view = owners[day * SLOTS_PER_DAY:(day + 1) * SLOTS_PER_DAY]
        rows.append({name: header_clips(view, np.array([need <= have for have in chosen])) for name, need in CHANNELS.items()}
                    | {"owned_slots": int((view >= 0).sum())})
    return stay, subject, split, "has_days", rows


def main():
    db = sqlite3.connect(f"file:{PLAN}?mode=ro", uri=True)
    by_subject = defaultdict(list)
    for subject, text in db.execute("SELECT subject_id, segments_json FROM records"):
        by_subject[int(subject)].extend(json.loads(text))
    stays = db.execute("SELECT stay_id, subject_id, intime_us, outtime_us, split FROM stays").fetchall()
    db.close()
    jobs = [(*stay, by_subject.get(stay[1], [])) for stay in stays]
    with ProcessPoolExecutor(16) as pool:
        results = list(pool.map(stay_days, jobs, chunksize=50))
    report = {"stays": Counter(r[3] for r in results), "patients_with_stays": len({r[1] for r in results}),
              "patients_with_a_segment_in_a_stay": len({r[1] for r in results if r[3] == "has_days"})}
    days = [(subject, stay, row) for stay, subject, _, _, rows in results for row in rows]
    report["days_in_stays_with_a_segment"] = len(days)
    report["days_with_no_owned_slot"] = sum(row["owned_slots"] == 0 for _, _, row in days)
    table = {}
    for channels in CHANNELS:
        table[channels] = {}
        for name, limit in THRESHOLDS.items():
            kept = [(subject, stay) for subject, stay, row in days if row[channels] >= limit]
            table[channels][name] = {"days": len(kept), "stays": len({k[1] for k in kept}), "patients": len({k[0] for k in kept})}
    report["kept_by_rule"] = table
    report["check_equals_plan"] = table["all_three"]["80%"]
    edges = [0, 1, 48, 144, 240, 288, 336, 384, 481]
    report["all_three_clip_histogram_days"] = {f"{edges[i]}-{edges[i + 1] - 1}": sum(edges[i] <= row["all_three"] < edges[i + 1] for _, _, row in days)
                                               for i in range(len(edges) - 1)}
    text = json.dumps(report, indent=2, default=dict)
    (HERE / "coverage_audit.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
