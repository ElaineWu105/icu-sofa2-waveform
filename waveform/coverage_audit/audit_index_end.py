"""Compare each record's master-header span with the index end time, and count data past the index end."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "waveform/icu_cut"))
from common import WAVE_ROOT  # noqa: E402
from indexing import load_wave_records  # noqa: E402


def span(record):
    subject = record["subject_id"]
    folder = Path(WAVE_ROOT) / "waves" / f"p{subject}"[:4] / f"p{subject}" / record["record_id"]
    lines = (folder / "waves.hea").read_text().split("\n")
    lines = [line for line in lines if line.strip() and not line.startswith("#")]
    head = lines[0].split()
    fs = float(head[2].split("/")[0])
    index_seconds = (record["end_us"] - record["start_us"]) / 1e6
    cursor = kept = past = 0.
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        name, length = parts[0], int(parts[1]) / fs
        if name != "~" and name != "waves_layout":
            kept += length
            past += max(0., cursor + length - max(cursor, index_seconds)) if cursor + length > index_seconds else 0.
        cursor += length
    return index_seconds, cursor, kept, past


def main():
    wanted = load_wave_records(WAVE_ROOT)
    wanted = wanted[0] if isinstance(wanted, tuple) else wanted
    records = [record for rows in wanted.values() for record in rows]
    with ThreadPoolExecutor(16) as pool:
        rows = np.array(list(pool.map(span, records)))
    index_seconds, master_seconds, data_seconds, past = rows.T
    ratio = master_seconds / index_seconds
    report = {"records": len(rows), "index_hours": round(index_seconds.sum() / 3600), "master_hours": round(master_seconds.sum() / 3600),
              "data_hours": round(data_seconds.sum() / 3600), "data_hours_past_index_end": round(past.sum() / 3600),
              "share_of_data_past_index_end": round(float(past.sum() / data_seconds.sum()), 4),
              "records_with_data_past_index_end": int((past > 0).sum()), "records_over_1h_past": int((past > 3600).sum()),
              "master_over_index_ratio": {k: round(float(np.quantile(ratio, q)), 3) for k, q in (("p1", .01), ("median", .5), ("p99", .99), ("max", 1.))}}
    text = json.dumps(report, indent=2)
    (HERE / "index_end_audit.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
