import gzip
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from probe_labels.build import (BREATHING, ECTOPY, RHYTHM, RHYTHM_CLASSES, chart_labels, choose_slot, clean_chart, consensus,
                          load_numerics, numeric_label, spectral_rate, support_label)


def events(rows):
    return pd.DataFrame(rows, columns=["offset", "item", "itemid", "value", "valuenum"])


class BuildTests(unittest.TestCase):
    def test_choose_slot_prefers_charted_then_middle(self):
        self.assertEqual(choose_slot(np.array([1, 2, 3]), np.array([3, 9])), (3, True))
        self.assertEqual(choose_slot(np.array([1, 2, 3]), np.array([9])), (2, False))
        self.assertEqual(choose_slot(np.array([], dtype=int), np.array([9])), (None, False))

    def test_consensus_rejects_conflict_and_unknown(self):
        self.assertEqual(consensus(["SR (Sinus Rhythm)"] * 2, RHYTHM), "SR")
        self.assertIsNone(consensus(["SR (Sinus Rhythm)", "AF (Atrial Fibrillation)"], RHYTHM))
        self.assertIsNone(consensus(["SR (Sinus Rhythm)", "unknown"], RHYTHM))
        self.assertIsNone(consensus([], RHYTHM))

    def test_support_label(self):
        self.assertEqual(support_label([(226732, "Endotracheal tube")]), 1.0)
        self.assertEqual(support_label([(226732, "Nasal cannula")]), 0.0)
        self.assertEqual(support_label([(226732, "None")]), 0.0)
        self.assertIsNone(support_label([(226732, "Tracheostomy tube")]))
        self.assertIsNone(support_label([(226732, "Nasal cannula"), (226732, "Endotracheal tube")]))

    def test_chart_labels_use_near_window_only(self):
        rows = [(0, "rhythm", 220048, "AF (Atrial Fibrillation)", np.nan), (0, "ectopy", 224650, "None", np.nan),
                (3_600_000_000, "ectopy", 224650, "PVC's", np.nan), (5_000_000, "rr", 220210, "18", 18.),
                (0, "breath_pattern", 229322, "Shallow", np.nan)]
        result = chart_labels(events(rows), 0)
        self.assertEqual(result["rhythm"], RHYTHM_CLASSES.index("AF"))
        self.assertTrue(np.isnan(result["avb1"]))
        self.assertEqual((result["pvc"], result["pac"]), (0., 0.))
        self.assertEqual(result["rr_charted"], 18.)
        self.assertEqual(result["breathing"], 1.)

    def test_bundle_branch_labels_use_sinus_negatives(self):
        result = chart_labels(events([(0, "rhythm", 220048, "SR (Sinus Rhythm)", np.nan)]), 0)
        self.assertEqual((result["avb1"], result["rbbb"], result["lbbb"]), (0., 0., 0.))
        result = chart_labels(events([(0, "rhythm", 220048, "RBBB (Right Bundle Branch Block)", np.nan)]), 0)
        self.assertEqual(result["rbbb"], 1.)
        self.assertTrue(np.isnan(result["rhythm"]))
        self.assertTrue(np.isnan(result["lbbb"]))

    def test_numeric_label_range_and_fallback(self):
        frame = pd.DataFrame({"time": [0, 500, 1000, 20000], "ABPs [mmHg]": [np.nan, 999, np.nan, 90], "ARTs [mmHg]": [110, 120, 130, 90]})
        self.assertEqual(numeric_label(frame, ("ABPs [mmHg]", "ARTs [mmHg]"), 40, 260, 0, 10000), 120)
        self.assertTrue(np.isnan(numeric_label(frame, ("HR [bpm]",), 20, 250, 0, 10000)))
        self.assertTrue(np.isnan(numeric_label(None, ("HR [bpm]",), 20, 250, 0, 10000)))

    def test_clean_chart_strips_values_and_rejects_absent_ones(self):
        rows = [("rhythm", key + " ") for key in RHYTHM] + [("ectopy", key) for key in ECTOPY] + [("breath_pattern", key) for key in BREATHING]
        chart = pd.DataFrame(rows, columns=["item", "value"])
        self.assertTrue(set(RHYTHM) <= set(clean_chart(chart).value))
        with self.assertRaises(RuntimeError):
            clean_chart(chart[chart.value != "V Paced "])

    def test_empty_or_missing_numerics_mean_no_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(load_numerics(folder))
            with gzip.open(Path(folder) / "numerics.csv.gz", "wb"):
                pass
            self.assertIsNone(load_numerics(folder))

    def test_spectral_rate_finds_breathing(self):
        time = np.arange(2500) / 250
        self.assertAlmostEqual(spectral_rate(np.sin(2 * np.pi * .3 * time)), 18, delta=.5)


if __name__ == "__main__":
    unittest.main()
