"""Joint 250 Hz ECG/PPG/Resp cleaning, strict masks and new typed channel IDs."""

import unittest
from unittest.mock import patch

import neurokit2 as nk
import numpy as np
from scipy.signal import resample

from processing import (DEFAULT_SETTINGS, ECG_CHANNEL_IDS, REQUIRED_ENCODER_CHANNELS,
                        canonical_settings, process_window)


def synthetic_window():
    ecg_t = np.arange(2500) / 250
    ppg_t = np.arange(1250) / 125
    resp_t = np.arange(625) / 62.5
    return {
        "ecg": (np.sin(2 * np.pi * 1.3 * ecg_t) + .2 * np.sin(2 * np.pi * 20 * ecg_t)).astype(np.float32),
        "ppg": (3 + np.sin(2 * np.pi * 1.3 * ppg_t) + .3 * np.sin(2 * np.pi * 2.6 * ppg_t)).astype(np.float32),
        "resp": (5 + np.sin(2 * np.pi * .2 * resp_t) + .1 * np.sin(2 * np.pi * .4 * resp_t)).astype(np.float32),
        "lead_id": "II", "source_metadata": {"source": "synthetic engineering fixture"},
    }


class ProcessingTests(unittest.TestCase):
    def test_real_cleaning_shapes_normalization_determinism_and_input_preservation(self):
        raw = synthetic_window()
        originals = {key: raw[key].copy() for key in ("ecg", "ppg", "resp")}
        first, second = process_window(raw), process_window(raw)
        self.assertEqual(first["signal"].shape, (3, 2500))
        self.assertEqual(first["signal"].dtype, np.float32)
        self.assertEqual(first["modality_mask"].dtype, np.bool_)
        self.assertTrue(first["modality_mask"].all())
        self.assertEqual(first["channel_ids"].dtype, np.int16)
        np.testing.assert_array_equal(first["channel_ids"], [1, 12, 13])
        self.assertEqual(REQUIRED_ENCODER_CHANNELS, 14)
        self.assertEqual(first["source_metadata"], raw["source_metadata"])
        np.testing.assert_array_equal(first["signal"], second["signal"])
        for legacy_key in ("ecg_ppg", "resp", "ecg_channel_id"):
            self.assertNotIn(legacy_key, first)
        for key in originals:
            np.testing.assert_array_equal(raw[key], originals[key])
            self.assertTrue(first["qc"][key]["accepted"])
            self.assertEqual(first["reasons"][key], [])
            self.assertIsNotNone(first["stats"][key]["raw"])
            self.assertIsNotNone(first["stats"][key]["clean"])
        for output in first["signal"]:
            self.assertTrue(np.isfinite(output).all())
            self.assertAlmostEqual(float(output.mean()), 0., places=6)
            self.assertAlmostEqual(float(output.std()), 1., places=5)

    def test_ppg_matches_official_fft_resampling_and_neurokit_recipe(self):
        raw = synthetic_window()
        upsampled = resample(raw["ppg"].astype(np.float64), 2500)
        clean = nk.ppg_clean(upsampled, sampling_rate=250, method="elgendi")
        expected = ((clean - clean.mean()) / (clean.std() + 1e-8)).astype(np.float32)
        result = process_window(raw)
        np.testing.assert_array_equal(result["signal"][1], expected)
        self.assertAlmostEqual(result["stats"]["ppg"]["raw"]["mean"], float(raw["ppg"].astype(np.float64).mean()))
        self.assertAlmostEqual(result["stats"]["ppg"]["clean"]["std"], float(clean.std()))

    def test_resp_fft_resampling_and_cleaning_at250_matches_independent_reference(self):
        raw = synthetic_window()
        upsampled = resample(raw["resp"].astype(np.float64), 2500)
        clean = nk.rsp_clean(upsampled, sampling_rate=250, method="khodadad2018")
        expected = ((clean - clean.mean()) / (clean.std() + 1e-8)).astype(np.float32)
        result = process_window(raw)
        np.testing.assert_array_equal(result["signal"][2], expected)
        self.assertTrue(result["modality_mask"][2])
        self.assertEqual(result["channel_ids"][2], 13)
        self.assertEqual(result["stats"]["resp"]["raw"]["samples"], 625)
        self.assertEqual(result["stats"]["resp"]["clean"]["samples"], 2500)
        self.assertEqual(result["qc"]["resp"]["native_rate"], 62.5)
        self.assertEqual(result["qc"]["resp"]["output_rate"], 250.)

    def test_incomplete_or_nonfinite_input_is_rejected_before_neurokit_can_fill_gaps(self):
        raw = synthetic_window()
        raw["ecg"][42] = np.nan
        raw["ppg"][7] = np.inf
        raw["mask_by_modality"] = {"resp": np.ones(625, dtype=bool)}
        raw["mask_by_modality"]["resp"][12] = False
        with patch("processing.nk.ecg_clean") as ecg, patch("processing.nk.ppg_clean") as ppg, patch("processing.nk.rsp_clean") as resp:
            result = process_window(raw)
        ecg.assert_not_called()
        ppg.assert_not_called()
        resp.assert_not_called()
        self.assertFalse(result["modality_mask"].any())
        np.testing.assert_array_equal(result["channel_ids"], [-1, -1, -1])
        self.assertTrue((result["signal"] == 0).all())
        self.assertIn("nonfinite_samples", result["reasons"]["ecg"])
        self.assertIn("nonfinite_samples", result["reasons"]["ppg"])
        self.assertIn("incomplete_sample_mask", result["reasons"]["resp"])

    def test_missing_flat_and_partial_native_channels_keep_false_masks_and_raw_stats(self):
        raw = {"ecg": np.zeros(2500, np.float32), "ppg": None,
               "resp": np.ones(624, np.float32), "lead_id": "II"}
        result = process_window(raw)
        self.assertFalse(result["modality_mask"].any())
        self.assertIn("near_zero_raw_variance", result["reasons"]["ecg"])
        self.assertEqual(result["stats"]["ecg"]["raw"]["std"], 0.)
        self.assertTrue(result["qc"]["ecg"]["flat_raw"])
        self.assertEqual(result["reasons"]["ppg"], ["missing"])
        self.assertIn("invalid_native_shape", result["reasons"]["resp"])

    def test_unknown_lead_is_not_guessed_but_other_modalities_still_work(self):
        for name in ("V", "ECG1", "ECG2", "V7", "unknown"):
            result = process_window(dict(synthetic_window(), lead_id=name))
            np.testing.assert_array_equal(result["modality_mask"], [False, True, True])
            np.testing.assert_array_equal(result["channel_ids"], [-1, 12, 13])
            self.assertIn("unsupported_ecg_lead", result["reasons"]["ecg"])
        for name, expected in (("aVR", 3), ("V1", 6), ("V6", 11)):
            result = process_window(dict(synthetic_window(), lead_id=name))
            self.assertEqual(result["channel_ids"][0], expected)
        self.assertEqual(set(ECG_CHANNEL_IDS.values()), set(range(12)))

    def test_cleaner_failures_and_nonfinite_or_flat_outputs_do_not_invent_signal(self):
        raw = synthetic_window()
        with patch("processing.nk.ecg_clean", side_effect=ValueError("synthetic failure")), \
             patch("processing.nk.ppg_clean", return_value=np.full(2500, np.nan)), \
             patch("processing.nk.rsp_clean", return_value=np.ones(2500)):
            result = process_window(raw)
        self.assertFalse(result["modality_mask"].any())
        self.assertIn("cleaning_failed", result["reasons"]["ecg"])
        self.assertIn("nonfinite_clean_samples", result["reasons"]["ppg"])
        self.assertIn("near_zero_clean_variance", result["reasons"]["resp"])
        self.assertEqual(result["stats"]["resp"]["clean"]["std"], 0.)

    def test_explicit_filter_arguments_and_both_ppg_resp_resampling(self):
        raw = synthetic_window()
        with patch("processing.nk.ecg_clean", wraps=nk.ecg_clean) as ecg, \
             patch("processing.nk.ppg_clean", wraps=nk.ppg_clean) as ppg, \
             patch("processing.nk.rsp_clean", wraps=nk.rsp_clean) as resp:
            result = process_window(raw)
        self.assertTrue(result["modality_mask"].all())
        self.assertEqual(ecg.call_args.kwargs, {"sampling_rate": 250, "method": "neurokit", "powerline": 50.})
        self.assertEqual(ppg.call_args.kwargs, {"sampling_rate": 250, "method": "elgendi"})
        self.assertEqual(resp.call_args.kwargs, {"sampling_rate": 250, "method": "khodadad2018"})
        self.assertEqual(len(resp.call_args.args[0]), 2500)
        np.testing.assert_array_equal(resp.call_args.args[0], resample(raw["resp"].astype(np.float64), 2500))
        np.testing.assert_array_equal(ppg.call_args.args[0], resample(raw["ppg"].astype(np.float64), 2500))

    def test_resp_only_window_has_no_invented_ecg_ppg_ids_or_values(self):
        raw = dict(synthetic_window(), ecg=None, ppg=None)
        result = process_window(raw)
        np.testing.assert_array_equal(result["modality_mask"], [False, False, True])
        np.testing.assert_array_equal(result["channel_ids"], [-1, -1, 13])
        self.assertTrue((result["signal"][:2] == 0).all())
        self.assertGreater(float(np.abs(result["signal"][2]).sum()), 0.)

    def test_settings_and_mask_shape_validation(self):
        self.assertEqual(canonical_settings(), DEFAULT_SETTINGS)
        for settings in ({"min_std": -1}, {"normalization_epsilon": 0}, {"ecg_powerline": np.nan},
                         {"ppg_method": "none"}, {"unknown": True}):
            with self.assertRaises(ValueError):
                process_window(synthetic_window(), settings)
        raw = synthetic_window()
        raw["mask_by_modality"] = {"ecg": np.ones(2499, bool)}
        result = process_window(raw)
        self.assertIn("invalid_sample_mask", result["reasons"]["ecg"])
        self.assertFalse(result["modality_mask"][0])

    def test_sustained_raw_extreme_plateau_is_audit_only_and_remains_accepted(self):
        normal = process_window(synthetic_window())
        for modality in ("ecg", "ppg", "resp"):
            self.assertFalse(normal["qc"][modality]["possible_clipping"])
        raw = synthetic_window()
        raw["ecg"][:300] = 2.0  # 12% at one extreme, continuous 1.2-second plateau.
        result = process_window(raw)
        self.assertTrue(result["modality_mask"][0])
        self.assertTrue(result["qc"]["ecg"]["possible_clipping"])
        self.assertTrue(result["qc"]["ecg"]["clipping_heuristic"])
        self.assertFalse(result["qc"]["ecg"]["clipping_adc_certified"])
        self.assertGreater(result["qc"]["ecg"]["repeated_step_fraction"], .10)
        self.assertAlmostEqual(result["qc"]["ecg"]["raw_extreme_fraction"], .12)
        self.assertEqual(result["reasons"]["ecg"], [])
        self.assertEqual(result["stats"]["ecg"]["raw"]["max"], 2.)

    def test_isolated_extreme_repeats_and_flat_signal_do_not_certify_clipping(self):
        raw = synthetic_window()
        raw["ecg"][::5] = 2.0  # 20% at the extreme, no sustained plateau.
        result = process_window(raw)
        self.assertTrue(result["modality_mask"][0])
        self.assertGreaterEqual(result["qc"]["ecg"]["raw_extreme_fraction"], .10)
        self.assertFalse(result["qc"]["ecg"]["possible_clipping"])
        raw["ecg"][:] = 2.0
        flat = process_window(raw)
        self.assertFalse(flat["modality_mask"][0])
        self.assertTrue(flat["qc"]["ecg"]["flat_raw"])
        self.assertFalse(flat["qc"]["ecg"]["possible_clipping"])
        self.assertEqual(flat["qc"]["ecg"]["raw_extreme_fraction"], 1.)
        self.assertEqual(flat["qc"]["ecg"]["repeated_step_fraction"], 1.)
        self.assertIn("near_zero_raw_variance", flat["reasons"]["ecg"])


if __name__ == "__main__":
    unittest.main()
