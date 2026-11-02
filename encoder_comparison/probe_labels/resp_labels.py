"""Conservative respiratory-support labels from already-extracted chart events.

No waveform or original chartevents file is scanned here.  This is a source-label
audit, not a reproduction of MIMIC's ventilation concept: the cached source has
one oxygen-device item and two ventilator-mode items, whereas the official
concept joins four oxygen-device slots and has ICU-stay information.

Reference: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/ventilation.sql
Unknown/conflicting observations remain in the timeline as barriers.  They must
not silently become "None" or be removed before interval construction.
"""
from __future__ import annotations

import bisect
import collections
import csv
import functools
import gzip
import hashlib
import json
from pathlib import Path


OXYGEN_DEVICE = 226732
VENT_MODE = 223849
HAMILTON_MODE = 229314
RESP_ITEMS = frozenset((OXYGEN_DEVICE, VENT_MODE, HAMILTON_MODE))
DEFAULT_MAX_GAP_US = 2 * 60 * 60 * 1_000_000
OFFICIAL_SQL = "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/ventilation.sql"

DEVICE_CLASSES = {
    "Tracheostomy tube": "Tracheostomy", "Trach mask": "Tracheostomy",
    "Endotracheal tube": "InvasiveVent",
    "Bipap mask": "NonInvasiveVent", "CPAP mask": "NonInvasiveVent",
    "High flow nasal cannula": "HFNC",
    "Non-rebreather": "SupplementalOxygen", "Face tent": "SupplementalOxygen",
    "Aerosol-cool": "SupplementalOxygen", "Venti mask": "SupplementalOxygen",
    "Medium conc mask": "SupplementalOxygen", "Ultrasonic neb": "SupplementalOxygen",
    "Vapomist": "SupplementalOxygen", "Oxymizer": "SupplementalOxygen",
    "High flow neb": "SupplementalOxygen", "Nasal cannula": "SupplementalOxygen",
    "None": "None",
}
INVASIVE_MODES = frozenset((
    "(S) CMV", "APRV", "APRV/Biphasic+ApnPress", "APRV/Biphasic+ApnVol",
    "APV (cmv)", "Ambient", "Apnea Ventilation", "CMV", "CMV/ASSIST",
    "CMV/ASSIST/AutoFlow", "CMV/AutoFlow", "CPAP/PPS", "CPAP/PSV",
    "CPAP/PSV+Apn TCPL", "CPAP/PSV+ApnPres", "CPAP/PSV+ApnVol", "MMV",
    "MMV/AutoFlow", "MMV/PSV", "MMV/PSV/AutoFlow", "P-CMV", "PCV+",
    "PCV+/PSV", "PCV+Assist", "PRES/AC", "PRVC/AC", "PRVC/SIMV", "PSV/SBT",
    "SIMV", "SIMV/AutoFlow", "SIMV/PRES", "SIMV/PSV", "SIMV/PSV/AutoFlow",
    "SIMV/VOL", "SYNCHRON MASTER", "SYNCHRON SLAVE", "VOL/AC",
))
HAMILTON_INVASIVE = frozenset((
    "APRV", "APV (cmv)", "Ambient", "(S) CMV", "P-CMV", "SIMV",
    "APV (simv)", "P-SIMV", "VS", "ASV",
))
HAMILTON_NONINVASIVE = frozenset(("DuoPaP", "NIV", "NIV-ST"))


def map_value(itemid: int, value: str) -> str | None:
    """Map explicit documentation only; absent/ambiguous values are unknown."""
    value = value.strip()
    if itemid == OXYGEN_DEVICE:
        return DEVICE_CLASSES.get(value)
    if itemid == VENT_MODE and value in INVASIVE_MODES:
        return "InvasiveVent"
    if itemid == HAMILTON_MODE:
        if value in HAMILTON_INVASIVE:
            return "InvasiveVent"
        if value in HAMILTON_NONINVASIVE:
            return "NonInvasiveVent"
    return None


def _load_value_map(value_path):
    raw = json.loads(Path(value_path).read_text(encoding="utf-8"))
    result = {}
    for item, digest, value in raw["values"]:
        if int(item) not in RESP_ITEMS:
            continue
        # Cached values are stripped by the extraction pipeline before hashing.
        if hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
            raise ValueError("Cached respiratory value hash mismatch")
        key = int(item), digest
        if key in result and result[key] != value:
            raise ValueError("Conflicting cached respiratory value")
        result[key] = value
    return result


def _load_cohort(pairs_path):
    admissions = {}
    patient_splits = {}
    with open(pairs_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            subject, hadm = int(row["subject_id"]), int(row["hadm_id"])
            split = row["split"]
            if split not in ("train", "validation", "val", "test"):
                raise ValueError("Unexpected cohort split")
            # Preserve caller's split naming rather than invent a new partition.
            if subject in patient_splits and patient_splits[subject] != split:
                raise ValueError("Patient is present in multiple splits")
            patient_splits[subject] = split
            admissions[(subject, hadm)] = split
    return admissions


def _resolve_observations(observations):
    """Resolve concurrent chart records without assuming a clinical priority.

    Explicit tracheostomy plus an invasive mode is compatible.  Other differing
    known classes are excluded, even if official SQL would select a priority.
    Unknown generic modes such as SPONT cannot overrule an explicit interface;
    an unknown oxygen-device record does invalidate device-based interpretation.
    """
    if any(not row.get("value_recovered", True) for row in observations):
        return None, "missing_value_cache"
    known = {row["mapped"] for row in observations if row["mapped"] is not None}
    unknown_device = any(row["itemid"] == OXYGEN_DEVICE and row["mapped"] is None
                         for row in observations)
    if len(known) > 1:
        if known == {"Tracheostomy", "InvasiveVent"} and not unknown_device:
            return "Tracheostomy", "compatible_tracheostomy_invasive_mode"
        return None, "conflicting_classes"
    if unknown_device:
        return None, "unknown_device"
    if len(known) == 1:
        return next(iter(known)), "explicit_source_label"
    return None, "unknown_mode_or_missing_value_cache"


def build_resp_candidates(events_path, pairs_path, value_path):
    """Return sorted timestamp labels (including barriers) and aggregate counts.

    Each source row is included once, irrespective of overlapping CXR pairs.
    The events cache itself is restricted to the original CXR history cohort.
    store_us is the latest store time contributing to that timestamp label.
    """
    value_map = _load_value_map(value_path)
    admissions = _load_cohort(pairs_path)
    grouped = collections.defaultdict(list)
    seen = set()
    raw_item_counts = collections.Counter()
    unknown_value_counts = collections.Counter()
    duplicate_rows = 0
    opener = gzip.open if str(events_path).endswith(".gz") else open
    with opener(events_path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            item = int(row["itemid"])
            if item not in RESP_ITEMS:
                continue
            subject, hadm = int(row["subject_id"]), int(row["hadm_id"])
            split = admissions.get((subject, hadm))
            if split is None:
                continue
            source_key = str(row["source_row"])
            if source_key in seen:
                duplicate_rows += 1
                continue
            seen.add(source_key)
            raw_item_counts[str(item)] += 1
            digest = row["value_hash"]
            value = value_map.get((item, digest))
            if value is None:
                unknown_value_counts[str(item)] += 1
            grouped[(subject, hadm, int(row["chart_us"]))].append({
                "itemid": item, "mapped": map_value(item, value) if value is not None else None,
                "store_us": int(row["store_us"]), "source_event_key": source_key,
                "value_recovered": value is not None,
            })

    candidates = []
    counts = collections.Counter()
    reasons = collections.Counter()
    split_counts = collections.defaultdict(collections.Counter)
    patients = collections.defaultdict(set)
    for (subject, hadm, chart_us), observations in sorted(grouped.items()):
        label, reason = _resolve_observations(observations)
        split = admissions[(subject, hadm)]
        candidate = {
            "subject_id": subject, "hadm_id": hadm, "split": split,
            "chart_us": chart_us, "store_us": max(r["store_us"] for r in observations),
            "label": label, "source_event_key": ",".join(sorted(
                (r["source_event_key"] for r in observations), key=int)),
            "source_items": sorted({r["itemid"] for r in observations}), "reason": reason,
            "observations": observations,
            "has_any_known_observation": any(r["mapped"] is not None for r in observations),
        }
        candidates.append(candidate)
        category = label if label is not None else "UNKNOWN"
        counts[category] += 1
        reasons[reason] += 1
        split_counts[split][category] += 1
        patients[category].add(subject)

    summary = {
        "label_kind": "derived_source_labels_not_lung_disease_diagnoses",
        "official_reference": OFFICIAL_SQL,
        "official_ventilation_concept_fully_reproduced": False,
        "cohort_scope": "already-extracted events within CXR-history cohort; no original chartevents rescan",
        "source_items": sorted(RESP_ITEMS), "source_event_counts": dict(raw_item_counts),
        "unrecoverable_cached_value_events_by_item": dict(unknown_value_counts),
        "duplicate_source_rows_dropped": duplicate_rows,
        "timestamp_labels_including_barriers": len(candidates),
        "timestamp_counts_by_label": dict(counts),
        "patient_counts_by_label": {k: len(v) for k, v in sorted(patients.items())},
        "timestamp_counts_by_split": {k: dict(v) for k, v in sorted(split_counts.items())},
        "resolution_reasons": dict(reasons),
        "limitations": [
            "Single cached oxygen-device item is not the official four-slot oxygen_delivery table.",
            "No ICU stay_id is available in this cache; grouping is by subject and hospital admission.",
            "SPONT, Standby, unqualified CPAP, T-piece, Other, missing values are unknown, never None.",
            "Concurrent incompatible classes are excluded instead of official priority resolution.",
            "Point documentation does not independently certify the status of every second in a 3-minute window.",
        ],
    }
    return CandidateList(candidates), summary


class CandidateList(list):
    """List-compatible timestamp candidates with fast per-admission lookup."""

    def __init__(self, values):
        super().__init__(values)
        self._groups = collections.defaultdict(list)
        for row in self:
            self._groups[int(row["subject_id"]), int(row["hadm_id"])].append(row)
        self._times = {}
        for key, rows in self._groups.items():
            rows.sort(key=lambda r: r["chart_us"])
            self._times[key] = [r["chart_us"] for r in rows]
        self._visible_history = functools.lru_cache(maxsize=256)(self._build_visible_history)

    def _build_visible_history(self, key, cutoff_us):
        visible = []
        for row in self._groups.get(key, ()):
            if cutoff_us is not None and int(row["chart_us"]) >= cutoff_us:
                continue
            observations = row.get("observations")
            if observations is None:
                # Permit synthetic already-resolved records in boundary tests.
                if int(row["store_us"]) >= 0 and (
                        cutoff_us is None or int(row["store_us"]) < cutoff_us):
                    visible.append(row)
                continue
            available = [r for r in observations if int(r["store_us"]) >= 0 and (
                cutoff_us is None or int(r["store_us"]) < cutoff_us)]
            if not available:
                continue
            label, reason = _resolve_observations(available)
            snapshot = dict(row)
            snapshot.update({
                "label": label, "reason": reason,
                "store_us": max(r["store_us"] for r in available),
                "source_event_key": ",".join(sorted(
                    (r["source_event_key"] for r in available), key=int)),
                "source_items": sorted({r["itemid"] for r in available}),
            })
            visible.append(snapshot)
        return [r["chart_us"] for r in visible], visible

    def classify_point_interval(self, subject, hadm, start, end, cutoff_us=None):
        """Require a known nonconflicting observation inside [start, end).

        This is a point-label count only; it does not establish stable state for
        all 180 seconds.  Any available unknown or conflict is a barrier.
        """
        if end <= start:
            raise ValueError("Invalid interval")
        key = int(subject), int(hadm)
        times, rows = self._visible_history(key, cutoff_us)
        left, right = bisect.bisect_left(times, start), bisect.bisect_left(times, end)
        selected = rows[left:right]
        if not selected:
            return {"label": None, "reason": "no_available_label_inside_block"}
        labels = {r["label"] for r in selected}
        if None in labels:
            return {"label": None, "reason": "unknown_or_conflict_inside_block"}
        if len(labels) != 1:
            return {"label": None, "reason": "different_states_inside_block"}
        return {
            "label": selected[0]["label"], "reason": "point_documentation_inside_block",
            "left_chart_us": selected[0]["chart_us"],
            "right_chart_us": selected[-1]["chart_us"],
            "source_event_keys": [r["source_event_key"] for r in selected],
        }

    def classify_interval(self, subject, hadm, start, end,
                          max_gap_us=DEFAULT_MAX_GAP_US, cutoff_us=None):
        """Classify full window using available, two-sided, same-state records.

        All records from the last observation at/before start through the first
        observation at/after end must agree.  Intermediate observations are
        allowed; the TOTAL span of the bracketing records is <= max_gap_us.
        Availability is resolved per raw observation before conflicts. Storage times
        must be strictly before the caller's CXR prediction cutoff.
        """
        if end <= start or max_gap_us <= 0:
            raise ValueError("Invalid interval or maximum gap")
        key = int(subject), int(hadm)
        times, rows = self._visible_history(key, cutoff_us)
        left = bisect.bisect_right(times, start) - 1
        right = bisect.bisect_left(times, end)
        if left < 0 or right >= len(rows):
            return {"label": None, "reason": "not_two_sided_bracketed"}
        selected = rows[left:right + 1]
        evidence = {
            "left_chart_us": selected[0]["chart_us"],
            "right_chart_us": selected[-1]["chart_us"],
            "left_source_event_key": selected[0]["source_event_key"],
            "right_source_event_key": selected[-1]["source_event_key"],
            "max_store_us": max(r["store_us"] for r in selected),
        }
        labels = {r["label"] for r in selected}
        if None in labels:
            return {"label": None, "reason": "unknown_or_conflict_inside_bracket", **evidence}
        if len(labels) != 1:
            return {"label": None, "reason": "state_change_inside_bracket", **evidence}
        total_span = selected[-1]["chart_us"] - selected[0]["chart_us"]
        evidence["bracket_span_us"] = total_span
        if total_span > max_gap_us:
            return {"label": None, "reason": "bracket_span_too_long", **evidence}
        return {"label": selected[0]["label"], "reason": "same_state_two_sided_bracket", **evidence}


def adjacent_stable_intervals(candidates, max_gap_us=DEFAULT_MAX_GAP_US):
    """Yield same-state bracketing intervals; unknowns and changes break chains.

    Caller must additionally require the actual 3-minute waveform to be complete
    and quality eligible.  Bounds are chart timestamps and are inclusive for
    bracketing [window_start, window_end].  A two-hour threshold is an audit
    choice, intentionally stricter than the official concept's 14-hour rule.
    """
    if max_gap_us <= 0:
        raise ValueError("max_gap_us must be positive")
    previous = None
    for current in sorted(candidates, key=lambda r: (
            int(r["subject_id"]), int(r["hadm_id"]), int(r["chart_us"]))):
        if previous is not None:
            same_admission = (previous["subject_id"], previous["hadm_id"]) == (
                current["subject_id"], current["hadm_id"])
            duration = current["chart_us"] - previous["chart_us"]
            if (same_admission and previous["label"] is not None
                    and previous["label"] == current["label"]
                    and 0 < duration <= max_gap_us):
                yield {
                    "subject_id": current["subject_id"], "hadm_id": current["hadm_id"],
                    "split": current["split"], "label": current["label"],
                    "start_us": previous["chart_us"], "end_us": current["chart_us"],
                    "store_us": max(previous["store_us"], current["store_us"]),
                    "left_event_key": previous["source_event_key"],
                    "right_event_key": current["source_event_key"],
                }
        previous = current


def classify_interval(candidates, start_us, end_us, max_gap_us=DEFAULT_MAX_GAP_US):
    """Classify one window from one admission, requiring full two-sided support.

    candidates should contain exactly one subject/admission.  A window crossing
    multiple adjacent equal-state intervals is allowed, but no unknown, change,
    or overlong chart gap may occur inside it.
    """
    if end_us <= start_us:
        raise ValueError("Invalid interval")
    groups = {(r["subject_id"], r["hadm_id"]) for r in candidates}
    if len(groups) > 1:
        raise ValueError("classify_interval requires exactly one admission")
    ordered = sorted(candidates, key=lambda r: r["chart_us"])
    times = [r["chart_us"] for r in ordered]
    left = bisect.bisect_right(times, start_us) - 1
    right = bisect.bisect_left(times, end_us)
    if left < 0 or right >= len(ordered):
        return None, "not_bracketed"
    selected = ordered[left:right + 1]
    labels = {r["label"] for r in selected}
    if None in labels:
        return None, "unknown_or_conflict_inside_bracket"
    if len(labels) != 1:
        return None, "state_change_inside_bracket"
    if any(b["chart_us"] - a["chart_us"] > max_gap_us
           for a, b in zip(selected, selected[1:])):
        return None, "chart_gap_too_long"
    return selected[0]["label"], "same_state_two_sided_bracket"
