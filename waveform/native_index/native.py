"""Native WFDB inventory and lossless physical-sample decoding.

Supplied wall clocks are retained without converting them to UTC. Native
sampling rates, lead names, gaps and physical amplitudes are preserved; this
module does not filter, interpolate, resample or normalize any waveform.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import wfdb


RATES = {"ecg": 250.0, "ppg": 125.0, "resp": 62.5}
KNOWN_LEADS = {"I", "II", "III", "AVR", "AVL", "AVF", "V", *{f"V{i}" for i in range(1, 10)}}
EPOCH = datetime(1970, 1, 1)


def clock_us(value):
    stamp = datetime.fromisoformat(str(value)).replace(tzinfo=None)
    elapsed = stamp - EPOCH
    return (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds


def classify_channel(raw_name, data_file):
    """Return a modality while retaining the original acquisition lead name."""
    token = re.sub(r"[\s_-]+", "", str(raw_name).strip()).upper()
    if "PLETH" in token or token.startswith("PLTH") or token in {"PPG", "PULSEWAVE"}:
        return "ppg"
    if token in {"RESP", "RESPIRATION", "RESPIRATORY"}:
        return "resp"
    if token in KNOWN_LEADS or re.fullmatch(r"ECG\d*", token):
        return "ecg"
    # The local waveform acquisition stores ECG-family channels in *e.dat.
    # Their raw names remain separate identities, including unnamed leads.
    if str(data_file).lower().endswith("e.dat") and token not in {"ART", "ABP", "CVP", "PAP", "ICP"}:
        return "ecg"
    return None


def frequency_counter(token):
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    match = re.fullmatch(rf"({number})(?:/({number}))?(?:\(({number})\))?", str(token))
    if not match:
        raise ValueError("invalid WFDB frequency/counter")
    frame_fs = float(match[1])
    counter_fs = float(match[2]) if match[2] is not None else frame_fs
    counter = float(match[3]) if match[3] is not None else 0.0
    if not all(map(math.isfinite, (frame_fs, counter_fs, counter))) or frame_fs <= 0 or counter_fs <= 0:
        raise ValueError("invalid WFDB frequency/counter")
    return frame_fs, counter / counter_fs


def _header_lines(path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _identifier(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def inspect_record(record, wave_root, max_counter_error=0.01):
    """Inventory selected native channels; reject ambiguous clocks explicitly."""
    skips = Counter()
    if not math.isfinite(max_counter_error) or max_counter_error < 0:
        raise ValueError("max_counter_error must be finite and nonnegative")
    try:
        subject = str(record["subject_id"]).removeprefix("p")
        study = str(record["study_id"])
        if not subject.isdigit() or not re.fullmatch(r"[A-Za-z0-9_.-]+", study) or study in {".", ".."}:
            raise ValueError("unsafe subject/study identifier")
        start = datetime.fromisoformat(str(record["starttime"]))
        end = datetime.fromisoformat(str(record["endtime"]))
        if start.utcoffset() != end.utcoffset():
            return [], {"index_utc_offset_change": 1}
        origin_us, index_end_us = clock_us(start), clock_us(end)
        if index_end_us <= origin_us:
            raise ValueError("invalid supplied index interval")
    except (KeyError, TypeError, ValueError, OverflowError):
        return [], {"invalid_record_index": 1}
    prefixed = "p" + subject
    master_path = Path(wave_root) / "waves" / prefixed[:4] / prefixed / study / "waves.hea"
    if not master_path.is_file():
        return [], {"missing_master_header": 1}
    try:
        lines = _header_lines(master_path)
        first = lines[0].split()
        frame_fs, master_counter_seconds = frequency_counter(first[2])
        frames = int(first[3])
        if frames <= 0 or int(first[1]) <= 0:
            raise ValueError("invalid master frame/channel count")
        if abs(master_counter_seconds) > 1e-9:
            return [], {"nonzero_master_counter": 1}
        if "/" in first[0]:
            segment_count = int(first[0].rsplit("/", 1)[1])
            descriptions = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError("invalid master segment descriptor")
                name, length = parts[0], int(parts[1])
                if length < 0 or (name != "~" and (Path(name).name != name or name in {".", ".."})):
                    raise ValueError("invalid master segment descriptor")
                descriptions.append((name, length))
            if len(descriptions) != segment_count:
                return [], {"master_segment_count_mismatch": 1}
        else:
            descriptions = [(master_path.stem, frames)]
        if sum(length for _, length in descriptions) != frames:
            return [], {"master_frame_count_mismatch": 1}
        record_end_us = min(index_end_us, origin_us + round(frames * 1e6 / frame_fs))
    except (OSError, IndexError, TypeError, ValueError, OverflowError):
        return [], {"invalid_master_header": 1}
    segments, cursor = [], 0
    for name, length in descriptions:
        offset = cursor / frame_fs
        segment_origin_us = origin_us + round(offset * 1e6)
        cursor += length
        if name == "~":
            skips["gap_segment"] += 1
            continue
        if length == 0:
            skips["layout_segment"] += 1
            continue
        if segment_origin_us >= record_end_us:
            skips["segment_outside_index"] += 1
            continue
        path = master_path.parent / name
        if not Path(str(path) + ".hea").is_file():
            skips["missing_segment_header"] += 1
            continue
        try:
            header = wfdb.rdheader(str(path))
            if int(header.sig_len) != length or not math.isclose(float(header.fs), frame_fs, rel_tol=0., abs_tol=1e-7):
                skips["master_segment_frame_mismatch"] += 1
                continue
            counter_error = None
            if header.base_counter is not None:
                counter_fs = float(header.counter_freq or header.fs)
                counter = float(header.base_counter)
                if not math.isfinite(counter_fs) or counter_fs <= 0 or not math.isfinite(counter):
                    skips["invalid_segment_counter"] += 1
                    continue
                counter_error = counter / counter_fs - master_counter_seconds - offset
                if abs(counter_error) > max_counter_error:
                    skips["segment_counter_offset_mismatch"] += 1
                    continue
            channels = []
            for index, raw_name in enumerate(header.sig_name):
                data_file = str(header.file_name[index])
                modality = classify_channel(raw_name, data_file)
                if modality is None:
                    continue
                spf = int(header.samps_per_frame[index] or 1)
                sample_fs = frame_fs * spf
                if spf < 1 or not math.isclose(sample_fs, RATES[modality], rel_tol=0., abs_tol=1e-7):
                    skips["unsupported_native_rate_" + modality] += 1
                    continue
                unit = str(header.units[index])
                factor = {"mV": 1., "uV": .001, "µV": .001, "μV": .001, "V": 1000.}.get(unit) if modality == "ecg" else 1.
                if factor is None:
                    skips["unsupported_ecg_unit"] += 1
                    continue
                gain, baseline = float(header.adc_gain[index]), float(header.baseline[index])
                if not math.isfinite(gain) or gain <= 0 or not math.isfinite(baseline):
                    skips["invalid_channel_calibration"] += 1
                    continue
                if not (path.parent / data_file).is_file():
                    # Retain the expected channel. Its decode must produce an
                    # explicit failure rather than an invented empty waveform.
                    skips["missing_data_file"] += 1
                channels.append({
                    "index": index, "channel_index": index, "modality": modality,
                    "lead_id": str(raw_name), "raw_name": str(raw_name),
                    "sample_fs": float(sample_fs), "spf": spf, "samples_per_frame": spf,
                    "unit": unit, "output_unit": "mV" if modality == "ecg" else unit,
                    "physical_scale": factor, "gain": gain, "adc_gain": gain,
                    "baseline": int(baseline), "fmt": str(header.fmt[index]),
                    "format": str(header.fmt[index]), "file_name": data_file,
                })
            if not channels:
                skips["no_selected_signal_channels"] += 1
                continue
            segment = {
                "record_id": study, "subject_id": subject, "path": str(path),
                "origin_us": segment_origin_us, "frames": length, "frame_fs": float(frame_fs),
                "record_end_us": record_end_us, "channels": channels,
                "counter_error_seconds": counter_error,
            }
            segment["segment_id"] = _identifier([str(path), segment_origin_us, length, float(frame_fs)])
            segments.append(segment)
        except (OSError, ValueError, RuntimeError, IndexError, TypeError, AttributeError, OverflowError):
            skips["invalid_segment_header"] += 1
    return segments, dict(skips)


def _longest_true(mask):
    if not len(mask):
        return 0
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max(initial=0))


def sample_qc(values, valid, native_count, channel):
    observed = values[valid].astype(np.float64)
    adjacent = valid[1:] & valid[:-1]
    equal = (values[1:] == values[:-1]) & adjacent
    std = float(observed.std()) if len(observed) else None
    ecg_large = bool(np.any(np.abs(observed) > 10.)) if channel["modality"] == "ecg" else None
    rate = float(channel["sample_fs"])
    longest_gap = _longest_true(~valid)
    return {
        "mean": float(observed.mean()) if len(observed) else None, "std": std,
        "min": float(observed.min()) if len(observed) else None,
        "max": float(observed.max()) if len(observed) else None,
        "equal_fraction": float(equal.sum() / adjacent.sum()) if adjacent.any() else None,
        "valid_samples": int(valid.sum()), "native_samples": int(native_count),
        "padded_samples": int(len(values) - native_count), "valid_fraction": float(valid.mean()),
        "longest_gap_samples": longest_gap, "longest_gap_seconds": longest_gap / rate,
        "longest_native_gap_samples": _longest_true(~valid[:native_count]),
        "flat": bool(std is not None and len(observed) >= 2 and std < 1e-8),
        "physiological_large": ecg_large, "ecg_large_amplitude_threshold_mV": 10. if ecg_large is not None else None,
        "calibration": "wfdb_physical_gain_baseline_then_unit_conversion",
        "gain": channel["gain"], "baseline": channel["baseline"],
        "native_unit": channel["unit"], "output_unit": channel["output_unit"],
    }


def decode_group(job):
    """Decode a contiguous frame group once and return native fixed-length rows."""
    group_id = job.get("group_id")
    try:
        segment = job["segment"]
        begin, stop = int(job["frame_start"]), int(job["frame_end"])
        if begin < 0 or stop <= begin or stop > int(segment["frames"]):
            raise ValueError("group frame bounds outside segment")
        real_frame_end = math.ceil((int(segment["record_end_us"]) - int(segment["origin_us"]))
                                   * float(segment["frame_fs"]) / 1e6)
        if stop > real_frame_end:
            raise ValueError("group includes future frames outside the index/header interval")
        chunk_seconds = float(job.get("chunk_seconds", 30))
        if not math.isfinite(chunk_seconds) or chunk_seconds <= 0:
            raise ValueError("invalid chunk duration")
        by_index = {int(channel["index"]): channel for channel in segment["channels"]}
        chunks = job["chunks"]
        if not chunks:
            raise ValueError("group has no chunks")
        selected = sorted({int(chunk.get("channel_index", chunk.get("channel", {}).get("index", -1))) for chunk in chunks})
        if set(selected) - set(by_index):
            raise ValueError("chunk selects a channel absent from segment")
        decoded = wfdb.rdrecord(
            segment["path"], sampfrom=begin, sampto=stop, channels=selected,
            physical=True, smooth_frames=False,
        )
        expanded = decoded.e_p_signal
        if expanded is None or len(expanded) != len(selected):
            raise ValueError("decoder did not return native expanded channels")
        streams = {}
        for position, index in enumerate(selected):
            channel = by_index[index]
            spf = int(channel["spf"])
            physical = np.asarray(expanded[position])
            if physical.ndim != 1 or len(physical) != (stop - begin) * spf:
                raise ValueError("native expanded sample count mismatch")
            with np.errstate(over="ignore", invalid="ignore"):
                streams[index] = (physical * float(channel["physical_scale"])).astype(np.float32)
        rows = []
        for chunk in chunks:
            index = int(chunk.get("channel_index", chunk.get("channel", {}).get("index", -1)))
            channel = by_index[index]
            spf, rate = int(channel["spf"]), float(channel["sample_fs"])
            if not math.isclose(rate, RATES[channel["modality"]], rel_tol=0., abs_tol=1e-7):
                raise ValueError("unsupported native sampling rate")
            if "sample_fs" in chunk and not math.isclose(float(chunk["sample_fs"]), rate, rel_tol=0., abs_tol=1e-7):
                raise ValueError("chunk sampling rate differs from channel")
            if "samples_per_frame" in chunk and int(chunk["samples_per_frame"]) != spf:
                raise ValueError("chunk samples_per_frame differs from channel")
            left, right = int(chunk["frame_start"]), int(chunk["frame_end"])
            if left < begin or right > stop or right <= left:
                raise ValueError("chunk frame bounds outside decoded group")
            count = round(chunk_seconds * rate)
            if not math.isclose(chunk_seconds * rate, count, rel_tol=0., abs_tol=1e-7):
                raise ValueError("chunk duration does not span an integral native sample count")
            raw = streams[index][(left - begin) * spf:(right - begin) * spf]
            if len(raw) > count:
                raise ValueError("chunk exceeds fixed output duration")
            step_us = round(1e6 / rate)
            if not math.isclose(step_us * rate, 1e6, rel_tol=0., abs_tol=1e-7):
                raise ValueError("native sample period is not an integer microsecond")
            origin_us = int(segment["origin_us"]) + round(left * 1e6 / float(segment["frame_fs"]))
            if "origin_us" in chunk and int(chunk["origin_us"]) != origin_us:
                raise ValueError("chunk origin differs from native frame clock")
            cutoff_us = int(segment["record_end_us"])
            if "actual_end_us" in chunk:
                cutoff_us = min(cutoff_us, int(chunk["actual_end_us"]))
            native_count = max(0, min(len(raw), -((origin_us - cutoff_us) // step_us)))
            if "n_samples" in chunk:
                declared = int(chunk["n_samples"])
                if declared < 0 or declared > native_count:
                    raise ValueError("chunk n_samples exceeds real index/header bounds")
                native_count = declared
            values = np.full(count, np.nan, dtype=np.float32)
            values[:native_count] = raw[:native_count]
            valid = np.isfinite(values)
            values[~valid] = np.nan
            rows.append({"chunk_id": chunk["chunk_id"], "values": values, "valid": valid,
                         "qc": sample_qc(values, valid, native_count, channel), "metadata": dict(chunk)})
        return {"group_id": group_id, "rows": rows, "error": None}
    except Exception as error:
        return {"group_id": group_id, "rows": [], "error": f"{type(error).__name__}: {error}"}
