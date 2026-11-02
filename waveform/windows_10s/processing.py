"""Deterministic cleaning of complete, already clipped native 10-second windows.

ECG, PPG and Resp share a 250 Hz, 2500-sample representation for a joint encoder.
Resp is an added channel with ID 13: the future CSFM encoder must extend its
channel embeddings from 13 to 14 rows and train this new respiratory embedding.
No pretrained Resp weights are claimed. Missing/rejected rows contain zeros
only with a false modality mask and channel ID -1.
"""

from __future__ import annotations

import math

import neurokit2 as nk
import numpy as np
from scipy.signal import resample


MODALITIES = ("ecg", "ppg", "resp")
NATIVE_RATES = {"ecg": 250.0, "ppg": 125.0, "resp": 62.5}
NATIVE_LENGTHS = {"ecg": 2500, "ppg": 1250, "resp": 625}
OUTPUT_RATES = {"ecg": 250.0, "ppg": 250.0, "resp": 250.0}
OUTPUT_LENGTHS = {"ecg": 2500, "ppg": 2500, "resp": 2500}
PPG_CHANNEL_ID = 12
RESP_CHANNEL_ID = 13
REQUIRED_ENCODER_CHANNELS = 14
CLIPPING_EXTREME_FRACTION = 0.10
CLIPPING_PLATEAU_SECONDS = 0.10
ECG_CHANNEL_IDS = {
    "I": 0, "II": 1, "III": 2, "AVR": 3, "AVL": 4, "AVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11,
}
DEFAULT_SETTINGS = {
    "ecg_method": "neurokit", "ecg_powerline": 50.0,
    "ppg_method": "elgendi", "resp_method": "khodadad2018",
    "normalization_epsilon": 1e-8, "min_std": 1e-8,
}


def canonical_settings(settings=None):
    """Validate the explicit recipe before any channel is processed."""
    supplied = {} if settings is None else dict(settings)
    unknown = set(supplied) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError("Unknown processing settings: " + ", ".join(sorted(unknown)))
    value = {**DEFAULT_SETTINGS, **supplied}
    for key in ("ecg_method", "ppg_method", "resp_method"):
        if value[key] != DEFAULT_SETTINGS[key]:
            raise ValueError(f"{key} must be {DEFAULT_SETTINGS[key]!r} for this processing recipe")
    for key in ("ecg_powerline", "normalization_epsilon", "min_std"):
        value[key] = float(value[key])
        if not math.isfinite(value[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 < value["ecg_powerline"] < 125:
        raise ValueError("ecg_powerline must be positive and below the 250 Hz Nyquist frequency")
    if value["normalization_epsilon"] <= 0 or value["min_std"] < 0:
        raise ValueError("normalization_epsilon must be positive and min_std nonnegative")
    return value


def _stats(values, valid=None):
    array = np.asarray(values, dtype=np.float64)
    observed = np.isfinite(array)
    if valid is not None:
        observed &= valid
    selected = array[observed]
    return {
        "samples": int(array.size), "observed_samples": int(observed.sum()),
        "mean": float(selected.mean()) if len(selected) else None,
        "std": float(selected.std()) if len(selected) else None,
        "min": float(selected.min()) if len(selected) else None,
        "max": float(selected.max()) if len(selected) else None,
    }


def _clean(values, modality, settings):
    if modality == "ecg":
        return nk.ecg_clean(values, sampling_rate=250, method=settings["ecg_method"],
                            powerline=settings["ecg_powerline"])
    if modality == "ppg":
        return nk.ppg_clean(resample(values, 2500), sampling_rate=250,
                            method=settings["ppg_method"])
    return nk.rsp_clean(resample(values, 2500), sampling_rate=250, method=settings["resp_method"])


def _longest_run(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max(initial=0))


def _raw_audit(values, valid, stats, sample_fs, min_std):
    """Flag possible clipping without asserting ADC saturation or rejecting data.

    The heuristic requires nonflat data, >=10% of observed values at ONE raw
    min/max, and a contiguous >=100ms plateau at that same extreme. These
    sample-equality flags never impose a physiological amplitude range.
    """
    observed = np.isfinite(values) & valid
    adjacent = observed[:-1] & observed[1:]
    repeated = (values[:-1] == values[1:]) & adjacent
    count = int(observed.sum())
    result = {
        "repeated_step_fraction": float(repeated.sum() / adjacent.sum()) if adjacent.any() else None,
        "raw_extreme_fraction": None, "possible_clipping": False,
        "clipping_heuristic": True, "clipping_adc_certified": False,
    }
    if not count:
        return result
    minimum = observed & (values == stats["min"])
    maximum = observed & (values == stats["max"])
    fractions = (float(minimum.sum() / count), float(maximum.sum() / count))
    result["raw_extreme_fraction"] = max(fractions)
    sustained = math.ceil(sample_fs * CLIPPING_PLATEAU_SECONDS)
    nonflat = stats["std"] is not None and stats["std"] > min_std
    result["possible_clipping"] = bool(nonflat and any(
        fraction >= CLIPPING_EXTREME_FRACTION and _longest_run(extreme) >= sustained
        for fraction, extreme in zip(fractions, (minimum, maximum))
    ))
    return result


def process_window(raw, settings=None):
    """Return joint float32 signal[3,2500], masks, typed channel IDs and audit data.

    ``raw`` contains one-dimensional ecg[2500], ppg[1250], resp[625] arrays or
    None. Optional mask_by_modality contains bool arrays of the same shapes.
    ECG lead_id must identify a standard lead; V/ECG1 are never guessed as V1/I.
    The caller must perform exact time/ADT/cutoff selection BEFORE this call.
    Accepted rows have the true ECG ID, PPG ID 12 and new Resp ID 13. The latter
    requires a 14-row CSFM channel embedding and has no pretrained Resp weights.
    """
    settings = canonical_settings(settings)
    if not isinstance(raw, dict):
        raise TypeError("raw must be a dictionary of clipped native signal arrays")
    result = {
        "signal": np.zeros((3, 2500), dtype=np.float32),
        "modality_mask": np.zeros(3, dtype=bool),
        "channel_ids": np.full(3, -1, dtype=np.int16),
        "stats": {}, "qc": {}, "reasons": {},
        "source_metadata": raw.get("source_metadata", raw.get("metadata")),
    }
    masks = raw.get("mask_by_modality") or {}
    if not isinstance(masks, dict):
        raise TypeError("mask_by_modality must map modality names to bool sample masks")
    lead = str(raw.get("lead_id", "")).strip().upper()
    for position, modality in enumerate(MODALITIES):
        reasons = result["reasons"][modality] = []
        stats = result["stats"][modality] = {"raw": None, "clean": None}
        qc = result["qc"][modality] = {
            "accepted": False, "input_present": raw.get(modality) is not None,
            "native_rate": NATIVE_RATES[modality], "native_samples_expected": NATIVE_LENGTHS[modality],
            "output_rate": OUTPUT_RATES[modality],
            "output_samples": OUTPUT_LENGTHS[modality],
        }
        if raw.get(modality) is None:
            reasons.append("missing")
            continue
        try:
            values = np.asarray(raw[modality], dtype=np.float64).copy()
        except (TypeError, ValueError, OverflowError):
            reasons.append("invalid_values")
            continue
        qc["input_shape"] = list(values.shape)
        if values.shape != (NATIVE_LENGTHS[modality],):
            reasons.append("invalid_native_shape")
            stats["raw"] = _stats(values)
            continue
        valid = np.ones(values.shape, dtype=bool)
        if masks.get(modality) is not None:
            mask = np.asarray(masks[modality])
            if mask.shape != values.shape or mask.dtype != np.bool_:
                reasons.append("invalid_sample_mask")
                stats["raw"] = _stats(values)
                continue
            valid = mask.copy()
        finite = np.isfinite(values)
        qc["finite_samples"] = int(finite.sum())
        qc["valid_samples"] = int((valid & finite).sum())
        qc["complete_sample_mask"] = bool(valid.all())
        stats["raw"] = _stats(values, valid)
        qc.update(_raw_audit(values, valid, stats["raw"], NATIVE_RATES[modality], settings["min_std"]))
        if not valid.all():
            reasons.append("incomplete_sample_mask")
        if not finite.all():
            reasons.append("nonfinite_samples")
        if modality == "ecg" and lead not in ECG_CHANNEL_IDS:
            reasons.append("unsupported_ecg_lead")
        raw_std = stats["raw"]["std"]
        qc["flat_raw"] = bool(raw_std is not None and raw_std <= settings["min_std"])
        if qc["flat_raw"]:
            reasons.append("near_zero_raw_variance")
        if reasons:
            continue
        try:
            cleaned = np.asarray(_clean(values, modality, settings), dtype=np.float64)
        except Exception as error:
            reasons.append("cleaning_failed")
            qc["cleaning_error"] = f"{type(error).__name__}: {error}"
            continue
        if cleaned.shape != (OUTPUT_LENGTHS[modality],):
            reasons.append("unexpected_clean_shape")
            continue
        stats["clean"] = _stats(cleaned)
        if not np.isfinite(cleaned).all():
            reasons.append("nonfinite_clean_samples")
            continue
        clean_std = stats["clean"]["std"]
        qc["flat_clean"] = bool(clean_std <= settings["min_std"])
        if qc["flat_clean"]:
            reasons.append("near_zero_clean_variance")
            continue
        normalized = (cleaned - stats["clean"]["mean"]) / (clean_std + settings["normalization_epsilon"])
        normalized = normalized.astype(np.float32)
        if not np.isfinite(normalized).all():
            reasons.append("nonfinite_normalized_samples")
            continue
        stats["normalized"] = _stats(normalized)
        result["signal"][position] = normalized
        result["modality_mask"][position] = True
        qc["accepted"] = True
        result["channel_ids"][position] = (ECG_CHANNEL_IDS[lead] if modality == "ecg"
                                           else PPG_CHANNEL_ID if modality == "ppg"
                                           else RESP_CHANNEL_ID)
    return result
