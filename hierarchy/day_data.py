"""Day-level arrays for the temporal levels, loaded from the three-minute and one-hour caches."""
import numpy as np
import pandas as pd

from common import CLIPS_PER_DAY, CLIPS_PER_HOUR, MIN_HOUR_CLIPS, ORGANS, PRIVATE


def load_three_min():
    """features[D,480,768] and mask[D,480] in days.pkl order. Unfinished days are dropped."""
    days = pd.read_pickle(PRIVATE / "days.pkl")
    keep, features, masks = [], [], []
    for position, row in enumerate(days.itertuples()):
        path = PRIVATE / "three_min" / f"{row.stay_id}_{row.day:03d}.npz"
        if path.is_file():
            with np.load(path) as stored:
                features.append(stored["features"])
                masks.append(stored["mask"])
            keep.append(position)
    return days.iloc[keep].reset_index(drop=True), np.stack(features), np.stack(masks)


def hour_rows(mask, minimum=MIN_HOUR_CLIPS):
    """(day, hour) pairs with at least `minimum` valid clips."""
    counts = mask.reshape(-1, 24, CLIPS_PER_HOUR).sum(2)
    return np.argwhere(counts >= minimum)


def targets(days):
    """Real scores at both ends, masks, and the official-rule total."""
    base = days[[f"{organ}_real_base" for organ in ORGANS]].to_numpy(dtype=np.float32)
    future = days[[f"{organ}_real_next" for organ in ORGANS]].to_numpy(dtype=np.float32)
    stayed = days.fate.to_numpy() == "stayed"
    label = np.isfinite(base) & np.isfinite(future) & stayed[:, None]
    return {"base": np.nan_to_num(base), "base_mask": np.isfinite(base), "future": np.nan_to_num(future), "label": label,
            "official_total": days.total_delta_full.to_numpy(dtype=np.float32), "stayed": stayed}


def patient_weights(subjects):
    """Each patient carries the same total weight, whatever their number of anchors."""
    counts = pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(dtype=np.float64)
    return 1 / counts
