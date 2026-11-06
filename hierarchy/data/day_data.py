"""Day-level arrays for the temporal levels, SOFA-2 targets, and the EHR inputs of the level 3 head."""
import numpy as np
import pandas as pd

from common import CLIPS_PER_DAY, CLIPS_PER_HOUR, E2_TABLE, MIN_HOUR_CLIPS, ORGANS, PRIVATE

HANDLING = ("main", "locf", "partial_discharge", "no_death", "stayed_only")
THREE_MIN = PRIVATE / "three_min"
ONE_HOUR = PRIVATE / "one_hour"
VITALS = ("heart_rate", "resp_rate", "spo2", "mbp")
HOURS = 24


def load_three_min(windows=False):
    """Clip features[D,480,768] and mask[D,480] in days.pkl order; with windows=True also window_mask[D,480,18]
    and the control M feature[D,768]. Unfinished days are dropped."""
    days = pd.read_pickle(PRIVATE / "days.pkl")
    keep, features, masks, window_masks, ten = [], [], [], [], []
    for position, row in enumerate(days.itertuples()):
        path = THREE_MIN / f"{row.stay_id}_{row.day:03d}.npz"
        if path.is_file():
            with np.load(path) as stored:
                features.append(stored["features"])
                masks.append(stored["mask"])
                if windows:
                    window_masks.append(stored["window_mask"])
                    ten.append(stored["ten_second"])
            keep.append(position)
    days = days.iloc[keep].reset_index(drop=True)
    if windows:
        return days, np.stack(features), np.stack(masks), np.stack(window_masks), np.stack(ten)
    return days, np.stack(features), np.stack(masks)


def window_path(stay, day):
    return THREE_MIN / f"{stay}_{day:03d}_windows.npy"


def read_hour(stay, day, hour):
    """The 20 clips x 18 window features of one hour, as float32."""
    stored = np.load(window_path(stay, day), mmap_mode="r")
    return np.asarray(stored[hour * CLIPS_PER_HOUR:(hour + 1) * CLIPS_PER_HOUR], dtype=np.float32)


def hour_rows(mask, minimum=MIN_HOUR_CLIPS):
    """(day, hour) pairs with at least `minimum` valid clips."""
    counts = mask.reshape(-1, 24, CLIPS_PER_HOUR).sum(2)
    return np.argwhere(counts >= minimum)


def targets(days, handling="main"):
    """Scores at both ends, masks, the official-rule total and the composite event.

    No anchor is dropped for what happens next, which is unknown at prediction time. After ICU exit,
    de Grooth 2017 lists 24 on death and 0 on discharge (main), or the last observation (locf).
    The other handlings are sensitivity analyses. The composite is total change >= 2 or death
    (labels/README.md, section 7.2), on anchors with an official total.
    """
    base = days[[f"{organ}_real_base" for organ in ORGANS]].to_numpy(dtype=np.float32)
    future = days[[f"{organ}_real_next" for organ in ORGANS]].to_numpy(dtype=np.float32)
    base_total = days[[f"{organ}_full_base" for organ in ORGANS]].sum(axis=1).to_numpy(dtype=np.float32)
    total = days[[f"{organ}_full_next" for organ in ORGANS]].sum(axis=1).to_numpy(dtype=np.float32) - base_total
    fate = days.fate.to_numpy()
    stayed, died, left = fate == "stayed", fate == "died", fate == "left_alive"
    used = stayed | (left & (handling != "stayed_only")) | (died & (handling not in ("stayed_only", "no_death")))
    if handling == "locf":
        future[died | left] = base[died | left]
        total[died | left] = 0
    else:
        future[died] = 4
        total[died] = 24 - base_total[died]
        if handling != "partial_discharge":
            future[left] = 0
            total[left] = -base_total[left]
    total[~used] = np.nan
    label = np.isfinite(base) & np.isfinite(future) & used[:, None]
    known = np.isfinite(total)
    event = known & ((np.nan_to_num(total, nan=-np.inf) >= 2) | died)
    return {"base": np.nan_to_num(base), "base_mask": np.isfinite(base), "future": np.nan_to_num(future), "label": label,
            "official_total": total, "used": used, "died": died, "event": event.astype(np.float32), "event_known": known}


def patient_weights(subjects):
    """Each patient carries the same total weight, whatever their number of anchors."""
    counts = pd.Series(subjects).map(pd.Series(subjects).value_counts()).to_numpy(dtype=np.float64)
    return 1 / counts


def e2_columns():
    return ([f"prev_{organ}" for organ in ORGANS] + ["icu_day"]
            + [f"{name}_{hour:02d}" for name in VITALS for hour in range(HOURS)])


def load_e2(days):
    """E2 inputs of labels/README.md, section 7.4, aligned to days; rows missing from the table stay NaN."""
    table = pd.read_pickle(E2_TABLE)
    return days[["stay_id", "day"]].merge(table, on=["stay_id", "day"], how="left")[e2_columns()]


def context(days, inputs="e2", e2=None, stats=None):
    """Head inputs: scores scaled to 0..1 and zero when missing, with missing flags.

    E0 is the six baseline scores and flags. E2 adds the previous day's scores and flags, ICU day, and the
    96 hourly median vital signs and flags. Continuous E2 values are standardized with train statistics
    (`stats`, computed when None). Returns the array and the statistics.
    """
    base = days[[f"{organ}_real_base" for organ in ORGANS]].to_numpy(dtype=np.float32)
    parts = [np.nan_to_num(base / 4), np.isfinite(base)]
    if inputs == "e2":
        if e2 is None:
            e2 = load_e2(days)
        previous = e2[[f"prev_{organ}" for organ in ORGANS]].to_numpy(dtype=np.float32)
        continuous = e2[["icu_day"] + [f"{name}_{hour:02d}" for name in VITALS for hour in range(HOURS)]].to_numpy(dtype=np.float32)
        if stats is None:
            stats = {"mean": np.nanmean(continuous, axis=0), "std": np.nanstd(continuous, axis=0)}
            stats["mean"] = np.nan_to_num(stats["mean"])
            stats["std"] = np.where(np.isfinite(stats["std"]) & (stats["std"] > 0), stats["std"], 1.)
        standardized = (continuous - stats["mean"]) / stats["std"]
        vitals = np.isfinite(continuous[:, 1:])
        parts += [np.nan_to_num(previous / 4), np.isfinite(previous), np.nan_to_num(standardized), vitals]
    elif inputs != "e0":
        raise ValueError("inputs must be e0 or e2")
    return np.concatenate([part.astype(np.float32) for part in parts], axis=1), stats
