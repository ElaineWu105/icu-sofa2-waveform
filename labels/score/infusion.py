"""SOFA-2 note j: a vasopressor is scored only if given by continuous IV infusion for at least 1 hour."""
import numpy as np

MINIMUM_MINUTES = 60.
DAY_MINUTES = 1440.


def day_doses(fragments, day):
    """Highest dose in each hour of one ICU day, from runs of one drug that qualify, or None if none do.

    fragments are (start, end, dose, known_day), in minutes from ICU admission. Touching or
    overlapping fragments form one run; any gap ends it. Only what is known at the day's end
    counts: records stored by then, and run time up to then.
    """
    left, right = day * DAY_MINUTES, (day + 1) * DAY_MINUTES
    usable = sorted(item for item in fragments if item[3] <= day and item[0] < right)
    doses, run, end = None, [], None

    def close():
        nonlocal doses
        if not run or min(end, right) - run[0][0] < MINIMUM_MINUTES:
            return
        for start, stop, dose, _ in run:
            lo, hi = max(start, left), min(stop, right)
            if hi <= lo:
                continue
            doses = np.zeros(24) if doses is None else doses
            first, last = int((lo - left) // 60), int(np.ceil((hi - left) / 60))
            doses[first:last] = np.maximum(doses[first:last], dose)

    for item in usable:
        if run and item[0] > end:
            close()
            run, end = [], None
        run.append(item)
        end = item[1] if end is None else max(end, item[1])
    close()
    return doses
