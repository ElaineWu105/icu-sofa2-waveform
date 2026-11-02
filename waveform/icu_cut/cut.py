"""Cut planned ICU days into cleaned 10-second ECG/PPG/Resp windows.

Each worker owns one ICU day and writes two files atomically:
  <stay>/day_NNN_signal.npy  float32 [N,3,2500], ready windows in slot order
  <stay>/day_NNN_meta.npz    per-slot status, modality bits, lead, segment, row
The meta file is written last, so its presence marks a finished day. Stop and
rerun the same command to resume; finished days are never decoded again.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import contextlib
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import time

import numpy as np

from common import (CLIPS_PER_DAY, DAY_US, FAILED, HERE, INVENTORY_DIR, MIN_VALID_CLIPS, NOT_PLANNED,
                    PREPROCESS_DIR, READY, REJECTED, SLOTS_PER_DAY, WINDOW_US, sha256, valid_clips)
from processing import canonical_settings, process_window
from source import LEAD_ORDER, SnapshotSource, _choose_channels

MIN_FREE_BYTES = 200 * 2**30
SOURCES = {name: path for name, path in (
    ("10.14/processing.py", PREPROCESS_DIR / "processing.py"), ("10.14/source.py", PREPROCESS_DIR / "source.py"),
    ("10.13/native.py", INVENTORY_DIR / "native.py"), ("10.13/indexing.py", INVENTORY_DIR / "indexing.py"),
    ("common.py", HERE / "common.py"), ("plan.py", HERE / "plan.py"), ("cut.py", HERE / "cut.py"))}


def day_paths(output, stay_id, day):
    folder = Path(output) / "days" / str(stay_id)
    return folder / f"day_{day:03d}_signal.npy", folder / f"day_{day:03d}_meta.npz"


def read_slot(reader, segment, selected, day_start, stay_left, stay_right, slot):
    start = day_start + slot * WINDOW_US
    job = {"segment": segment, "selected_channels": selected, "pair_id": "icu", "slot_index": slot,
           "start_us": start, "end_us": start + WINDOW_US, "pair_start_us": day_start,
           "cutoff_us": day_start + DAY_US, "verified_left_us": stay_left, "verified_right_us": stay_right}
    return process_window(reader.read_window(job))


def process_day(job):
    """Decode one ICU day. Expected read failures become FAILED slots."""
    reader = object.__new__(SnapshotSource)
    status = np.full(SLOTS_PER_DAY, NOT_PLANNED, dtype=np.uint8)
    bits = np.zeros(SLOTS_PER_DAY, dtype=np.uint8)
    lead = np.full(SLOTS_PER_DAY, -1, dtype=np.int16)
    segment_index = np.full(SLOTS_PER_DAY, -1, dtype=np.int32)
    row = np.full(SLOTS_PER_DAY, -1, dtype=np.int16)
    signals, reasons, errors = [], Counter(), Counter()
    for first, stop, index in job["runs"]:
        segment = job["segments"][index]
        selected = _choose_channels(segment)
        for slot in range(first, stop):
            segment_index[slot] = index
            try:
                result = read_slot(reader, segment, selected, job["start_us"], job["intime_us"], job["outtime_us"], slot)
            except Exception as error:
                status[slot] = FAILED
                errors[f"{type(error).__name__}: {error}"[:120]] += 1
                continue
            for modality, values in result["reasons"].items():
                reasons.update(f"{modality}:{value}" for value in values)
            mask = result["modality_mask"]
            bits[slot] = sum(1 << position for position, present in enumerate(mask) if present)
            if not bits[slot]:
                status[slot] = REJECTED
                continue
            status[slot] = READY
            lead[slot] = result["channel_ids"][0]
            row[slot] = len(signals)
            signals.append(result["signal"])
    signal_path, meta_path = day_paths(job["output"], job["stay_id"], job["day"])
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.stack(signals) if signals else np.zeros((0, 3, 2500), dtype=np.float32)
    temporary = signal_path.with_suffix(f".{os.getpid()}.tmp.npy")
    np.save(temporary, array.astype(np.float32), allow_pickle=False)
    temporary.replace(signal_path)
    clips = int(valid_clips(status, bits, segment_index, lead).sum())
    summary = {"stay_id": job["stay_id"], "day": job["day"], "split": job["split"], "fate": job["fate"],
               "planned": int((status != NOT_PLANNED).sum()), "ready": len(signals),
               "all_three": int(((status == READY) & (bits == 7)).sum()), "rejected": int((status == REJECTED).sum()),
               "failed": int((status == FAILED).sum()), "valid_clips": clips,
               "reasons": dict(reasons), "errors": dict(errors)}
    temporary = meta_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(temporary, status=status, bits=bits, lead=lead, segment_index=segment_index, row=row,
             summary=np.array(json.dumps(summary)))
    temporary.replace(meta_path)
    return summary


def load_jobs(plan_path, output, stay_limit=None):
    db = sqlite3.connect(f"file:{plan_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    stays = {row["stay_id"]: row for row in db.execute("SELECT * FROM stays")}
    query = "SELECT * FROM days ORDER BY stay_id, day"
    days = db.execute(query).fetchall()
    if stay_limit is not None:
        chosen = sorted({row["stay_id"] for row in days})[:stay_limit]
        days = [row for row in days if row["stay_id"] in set(chosen)]
    segments = {}

    def segment(index):
        if index not in segments:
            segments[index] = json.loads(db.execute("SELECT segment_json FROM segments WHERE segment_index=?", (index,)).fetchone()[0])
        return segments[index]

    for row in days:
        runs = json.loads(row["runs_json"])
        yield {"stay_id": row["stay_id"], "day": row["day"], "split": row["split"], "fate": row["fate"],
               "start_us": row["start_us"], "intime_us": stays[row["stay_id"]]["intime_us"],
               "outtime_us": stays[row["stay_id"]]["outtime_us"], "runs": runs, "output": str(output),
               "segments": {index: segment(index) for _, _, index in runs}}
    db.close()


def has_failures(meta_path):
    with np.load(meta_path, allow_pickle=False) as meta:
        return json.loads(str(meta["summary"]))["failed"] > 0


def report(output, plan_path):
    """Authoritative totals, read back from finished day files."""
    db = sqlite3.connect(f"file:{plan_path}?mode=ro", uri=True)
    planned = dict(db.execute("SELECT split, COUNT(*) FROM days GROUP BY split").fetchall())
    subjects = {row[0]: row[1] for row in db.execute("SELECT stay_id, subject_id FROM stays")}
    db.close()
    totals, reasons, errors = Counter(), Counter(), Counter()
    splits, fates, histogram, passing = Counter(), Counter(), Counter(), set()
    for path in Path(output).glob("days/*/*_meta.npz"):
        with np.load(path, allow_pickle=False) as meta:
            summary = json.loads(str(meta["summary"]))
        totals["days"] += 1
        for key in ("planned", "ready", "all_three", "rejected", "failed"):
            totals[key] += summary[key]
        reasons.update(summary["reasons"])
        errors.update(summary["errors"])
        histogram[min(summary["valid_clips"] // 48, 10)] += 1
        if summary["valid_clips"] >= MIN_VALID_CLIPS:
            totals["days_passing"] += 1
            splits[summary["split"]] += 1
            fates[summary["fate"]] += 1
            passing.add(subjects[summary["stay_id"]])
    value = {"complete": totals["days"] == sum(planned.values()), "planned_days": sum(planned.values()),
             "planned_days_by_split": planned, "totals": dict(totals),
             "passing_days_by_split": dict(splits), "passing_days_by_fate": dict(fates),
             "passing_patients": len(passing), "clip_coverage_histogram_tenths": dict(sorted(histogram.items())),
             "rejection_reasons": dict(reasons.most_common()), "read_errors": dict(errors.most_common(20)),
             "rule": f"valid_clips >= {MIN_VALID_CLIPS} of {CLIPS_PER_DAY}"}
    temporary = Path(output) / "manifest.tmp"
    existing = json.loads((Path(output) / "manifest.json").read_text()) if (Path(output) / "manifest.json").is_file() else {}
    existing.update(report=value, report_utc=datetime.now(timezone.utc).isoformat())
    temporary.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(Path(output) / "manifest.json")
    return value


def definition(args):
    return {"window_seconds": 10, "grid": "slot k = intime + 10k seconds, ICU days only",
            "settings": canonical_settings(), "lead_order": list(LEAD_ORDER),
            "source_sha256": {name: sha256(path) for name, path in SOURCES.items()},
            "versions": {name: version(name) for name in ("numpy", "scipy", "wfdb", "neurokit2", "PyWavelets")},
            "plan_sha256": sha256(args.plan), "stay_limit": args.limit_stays}


def stop(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def _worker_init():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    request = definition(args)
    manifest = args.output / "manifest.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text()).get("request")
        if previous is not None and previous != request:
            raise ValueError("Plan, code, versions or limits changed. Use a new --output.")
    manifest.write_text(json.dumps({"request": request}, indent=2) + "\n")
    context = multiprocessing.get_context("spawn")
    started, finished, skipped, launched = time.monotonic(), 0, 0, 0
    reported = started
    futures = set()
    limit_reached = False
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context, initializer=_worker_init) as pool:
        try:
            for job in load_jobs(args.plan, args.output, args.limit_stays):
                _, meta_path = day_paths(args.output, job["stay_id"], job["day"])
                if meta_path.is_file() and not (args.retry_failed and has_failures(meta_path)):
                    skipped += 1
                    continue
                if args.max_days is not None and launched >= args.max_days:
                    limit_reached = True
                    break
                if shutil.disk_usage(args.output).free < MIN_FREE_BYTES:
                    raise OSError("Less than 200 GiB free; finished days are safe to resume.")
                futures.add(pool.submit(process_day, job))
                launched += 1
                while len(futures) >= args.workers * 2:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        finished += 1
                    if time.monotonic() - reported >= 300:
                        reported = time.monotonic()
                        rate = finished / (reported - started)
                        print(f"[cut] finished={finished}; skipped={skipped}; days_per_hour={rate * 3600:.0f}", flush=True)
            for future in futures:
                future.result()
                finished += 1
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    value = report(args.output, args.plan)
    print(f"[done] this_run={finished}; skipped={skipped}; limit_reached={limit_reached}", flush=True)
    print(json.dumps(value["totals"]), flush=True)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plan", type=Path, default=HERE / "output/plan/plan.sqlite")
    parser.add_argument("--output", type=Path, default=HERE / "output/cut")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit-stays", type=int, help="Diagnostic: first N stays; use a separate --output")
    parser.add_argument("--max-days", type=int, help="Diagnostic: cap on days decoded in this invocation")
    parser.add_argument("--retry-failed", action="store_true", help="Decode days with read failures again")
    parser.add_argument("--report-only", action="store_true", help="Recount finished days; decode nothing")
    args = parser.parse_args(argv)
    args.plan, args.output = args.plan.resolve(), args.output.resolve()
    if args.output.is_relative_to(Path("/home/public")):
        parser.error("Do not write into clinical sources.")
    if args.report_only:
        print(json.dumps(report(args.output, args.plan), indent=1, ensure_ascii=False))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    with (args.output / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another cut process is using this output.") from None
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
