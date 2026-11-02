import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import common
from day_data import hour_rows, patient_weights, targets
from extract_three_min import batches
from long_model import CSFMEncoder, MAEConfig
from clip_model import ClipEncoder, make_config, ssl_loss
from sofa_head import OrdinalHead, expected_score, ordinal_targets, sofa_loss
from temporal_model import TemporalConfig, TemporalSSL
from train_day import DayModel, metrics
from train_three_min import hourly_sample


class Head(unittest.TestCase):
    def test_ordinal_targets(self):
        self.assertEqual(ordinal_targets(torch.tensor([0., 2., 4.])).tolist(), [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])

    def test_expected_score_stays_in_range(self):
        values = expected_score(torch.tensor([[50., 50., 50., 50.], [-50., -50., -50., -50.]]))
        self.assertAlmostEqual(float(values[0]), 4, places=4)
        self.assertAlmostEqual(float(values[1]), 0, places=4)

    def test_waveform_starts_at_zero_correction(self):
        head = OrdinalHead()
        base, known = torch.ones(2, 6), torch.ones(2, 6, dtype=torch.bool)
        first = head(torch.zeros(2, 768), base, known)
        second = head(torch.randn(2, 768), base, known)
        self.assertTrue(torch.equal(first, second))

    def test_baseline_control_ignores_waveform(self):
        head = OrdinalHead(use_wave=False)
        base, known = torch.ones(2, 6), torch.ones(2, 6, dtype=torch.bool)
        nn_weights = next(head.wave.parameters())
        torch.nn.init.ones_(nn_weights)
        self.assertTrue(torch.equal(head(torch.zeros(2, 768), base, known), head(torch.randn(2, 768), base, known)))

    def test_wave_only_ignores_baseline(self):
        head = OrdinalHead(use_baseline=False).eval()
        torch.nn.init.ones_(head.wave[-1].weight)
        known = torch.ones(2, 6, dtype=torch.bool)
        embedding = torch.randn(2, 768)
        self.assertTrue(torch.equal(head(embedding, torch.ones(2, 6), known), head(embedding, 3 * torch.ones(2, 6), known)))

    def test_unlabeled_organ_never_reaches_the_loss(self):
        logits = torch.zeros(2, 6, 4, requires_grad=True)
        future, base = torch.ones(2, 6), torch.ones(2, 6)
        label = torch.zeros(2, 6, dtype=torch.bool)
        label[:, 0] = True
        sofa_loss(logits, future, base, label, torch.ones(2))["task"].backward()
        self.assertTrue((logits.grad[:, 0] != 0).any())
        self.assertTrue((logits.grad[:, 1:] == 0).all())

    def test_perfect_prediction_has_lower_loss(self):
        future, base = torch.tensor([[3., 0, 0, 0, 0, 0]]), torch.zeros(1, 6)
        label = torch.ones(1, 6, dtype=torch.bool)
        good = (ordinal_targets(future) * 2 - 1) * 20
        bad = -good
        self.assertLess(float(sofa_loss(good, future, base, label, torch.ones(1))["task"]), float(sofa_loss(bad, future, base, label, torch.ones(1))["task"]))


class Data(unittest.TestCase):
    def test_hour_rows_need_sixteen_clips(self):
        mask = np.zeros((1, 480), dtype=bool)
        mask[0, :16] = True
        mask[0, 20:35] = True
        self.assertEqual(hour_rows(mask).tolist(), [[0, 0]])
        self.assertEqual(hour_rows(mask, 1).tolist(), [[0, 0], [0, 1]])

    def test_patient_weights_equal_patient_totals(self):
        weights = patient_weights(np.array([1, 1, 1, 2]))
        self.assertAlmostEqual(weights[:3].sum(), weights[3])

    def test_targets_mask_missing_and_non_stayed(self):
        columns = {}
        for organ in common.ORGANS:
            columns[f"{organ}_real_base"] = [1., np.nan, 1.]
            columns[f"{organ}_real_next"] = [2., 2., 2.]
        days = pd.DataFrame({**columns, "fate": ["stayed", "stayed", "died"], "total_delta_full": [1., 1., np.nan]})
        data = targets(days)
        self.assertEqual(data["label"][:, 0].tolist(), [True, False, False])

    def test_hourly_sample_gives_one_clean_clip_per_hour(self):
        day = np.zeros(45, dtype=np.int32)
        clip = np.arange(45, dtype=np.int16)
        windows = np.full(45, 18, dtype=np.int8)
        windows[20:40] = 17
        chosen = hourly_sample(day, clip, windows, 0, 1)
        self.assertEqual(sorted((clip[chosen] // 20).tolist()), [0, 2])
        self.assertTrue((windows[chosen] == 18).all())

    def test_batches_never_mix_window_counts(self):
        windows = np.array([18, 17, 18, 18, 17])
        groups = list(batches(windows, 2))
        self.assertTrue(all(len(set(windows[group])) == 1 for group in groups))
        self.assertEqual(sorted(int(i) for group in groups for i in group), [0, 1, 2, 3, 4])

    def test_clip_reader_drops_a_bad_slot_and_keeps_position(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "7").mkdir()
            status = np.zeros(8640, dtype=np.uint8)
            status[:18] = common.READY
            status[3] = 2
            bits = np.full(8640, 7, dtype=np.uint8)
            rows = np.cumsum(status == common.READY).astype(np.int16) - 1
            lead = np.zeros(8640, dtype=np.int16)
            signal = np.arange(17, dtype=np.float32)[:, None, None] * np.ones((17, 3, 2500), dtype=np.float32)
            np.save(root / "7/day_000_signal.npy", signal)
            np.savez(root / "7/day_000_meta.npz", status=status, bits=bits, row=rows, lead=lead)
            with mock.patch.object(common, "CUT_DIRS", (root,)):
                values, ids, positions = common.ClipReader().read(7, 0, 0)
        self.assertEqual(values.shape, (17, 3, 2500))
        self.assertEqual(positions.tolist(), [0, 1, 2] + list(range(4, 18)))
        self.assertEqual(values[3, 0, 0], 3)
        self.assertEqual(ids[0].tolist(), [0, 12, 13])


class Models(unittest.TestCase):
    def small(self):
        return CSFMEncoder(MAEConfig(depth=1, heads=2, dim_head=8, mlp_dim=16, dim=16))

    def test_positions_match_the_plain_forward(self):
        core = ClipEncoder(self.small(), make_config()).eval()
        signal, ids = torch.randn(1, 18, 3, 2500) * 0 + 1, torch.tensor([0, 12, 13]).repeat(1, 18, 1)
        signal = signal[:, :2]
        ids = ids[:, :2]
        with torch.no_grad():
            plain = core(signal, ids)["embedding"]
            placed = core(signal, ids, torch.tensor([[0, 1]]))["embedding"]
        self.assertTrue(torch.allclose(plain, placed, atol=1e-5))

    def test_a_dropped_slot_uses_its_true_position(self):
        core = ClipEncoder(self.small(), make_config()).eval()
        signal, ids = torch.randn(1, 2, 3, 2500), torch.tensor([0, 12, 13]).repeat(1, 2, 1)
        with torch.no_grad():
            first = core(signal, ids, torch.tensor([[0, 1]]))["embedding"]
            second = core(signal, ids, torch.tensor([[0, 5]]))["embedding"]
        self.assertFalse(torch.allclose(first, second))

    def test_ssl_loss_reaches_prompts_only_through_trainable_parts(self):
        core = ClipEncoder(self.small(), make_config())
        core.classifier.requires_grad_(False)
        signal, ids = torch.randn(2, 2, 3, 2500), torch.tensor([0, 12, 13]).repeat(2, 2, 1)
        output = core(signal, ids)
        ssl_loss(core, output, core.teacher_features(signal, ids))["loss"].backward()
        self.assertIsNotNone(core.prompts.grad)
        self.assertTrue(all(value.grad is None for value in core.encoder.parameters()))

    def test_temporal_level_runs_with_missing_slots(self):
        model = TemporalSSL(TemporalConfig(sequence_length=20))
        values, mask = torch.randn(4, 20, 768), torch.rand(4, 20) > .3
        losses = model.losses(model(values, mask))
        self.assertTrue(torch.isfinite(losses["local"] + losses["dynamic"]))

    def test_day_model_output_shape(self):
        model = DayModel(True, True)
        output = model(torch.randn(3, 24, 768), torch.rand(3, 24) > .2, torch.zeros(3, 6), torch.ones(3, 6, dtype=torch.bool))
        self.assertEqual(output["logits"].shape, (3, 6, 4))

    def test_metrics_use_labeled_organs_only(self):
        n = 20
        data = {"base": np.zeros((n, 6), np.float32), "future": np.ones((n, 6), np.float32), "label": np.ones((n, 6), bool),
                "base_mask": np.ones((n, 6), bool), "official_total": np.full(n, 6., np.float32)}
        data["label"][:, 1] = False
        data["label"][0, 1] = True
        report = metrics(data, np.arange(n), np.ones((n, 6)))
        self.assertEqual(report["brain"]["mae"], 0)
        self.assertEqual(report["resp"]["n"], 1)
        self.assertEqual(report["brain"]["persistence_mae"], 1)
        self.assertEqual(report["total"]["mae_official"], 0)


if __name__ == "__main__":
    unittest.main()
