"""Synthetic checks for the target, learning rates and hourly sampling."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

import common  # noqa: F401
from data import HourlyWindows, SLOTS_PER_HOUR
import finetune
from model import CSFMMAE, MAEConfig
import prepare_text


SMALL = MAEConfig(dim=32, depth=2, heads=2, dim_head=8, mlp_dim=64, decoder_dim=16, decoder_depth=1,
                  decoder_heads=2, decoder_dim_head=4, decoder_mlp_dim=32, dropout=0., emb_dropout=0.)


class Target(unittest.TestCase):
    def test_normalized_target_matches_norm_pix(self):
        patch = torch.tensor([[1., 2., 3., 4.]])
        expected = (patch - 2.5) / (patch.var(-1, keepdim=True) + 1e-6).sqrt()
        self.assertTrue(torch.allclose(finetune.normalized(patch), expected))

    def test_flat_patch_stays_finite(self):
        self.assertTrue(torch.isfinite(finetune.normalized(torch.ones(1, 25))).all())

    def test_signal_errors_count_masked_valid_modalities_only(self):
        model = CSFMMAE(SMALL)
        batch = {"signal": torch.randn(2, 3, 2500), "channel_ids": torch.tensor([[1, 12, 13]] * 2),
                 "modality_mask": torch.tensor([[True, True, False]] * 2), "input_ids": torch.zeros(2, 64, dtype=torch.long),
                 "attention_mask": torch.zeros(2, 64, dtype=torch.bool), "special_tokens_mask": torch.ones(2, 64, dtype=torch.bool)}
        wave, text = finetune.make_masks(batch, SMALL, torch.Generator().manual_seed(0))
        out = model(**batch, waveform_mask=wave, text_mask=text, return_predictions=True)
        sums, counts = finetune.signal_errors(out, batch, SMALL)
        self.assertEqual(counts[2].item(), 0)
        self.assertEqual(counts[0].item(), 2 * 75 * 25)
        self.assertEqual(out["text_count"].item(), 0)
        (sums.sum() / counts.sum()).backward()
        self.assertIsNotNone(model.encoder.ts_to_patch_embedding[2].weight.grad)


class LearningRates(unittest.TestCase):
    def test_layer_decay_order_and_new_parameters(self):
        scale = lambda name: finetune.layer_scale(name, 12, .75)
        self.assertAlmostEqual(scale("decoder.layers.0.0.to_qkv.weight"), 1.0)
        self.assertAlmostEqual(scale("encoder.resp_channel_type_embedding"), 1.0)
        self.assertAlmostEqual(scale("encoder.transformer.layers.11.0.to_qkv.weight"), .75)
        self.assertAlmostEqual(scale("encoder.transformer.layers.0.0.to_qkv.weight"), .75 ** 12)
        self.assertAlmostEqual(scale("encoder.ts_pos_embedding"), .75 ** 13)

    def test_every_parameter_is_in_exactly_one_group(self):
        model = CSFMMAE(SMALL)
        groups = finetune.parameter_groups(model, 1e-4, .75, .05)
        grouped = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters()})


class Windows(unittest.TestCase):
    def make(self, folder):
        folder = Path(folder)
        db = sqlite3.connect(folder / "plan.sqlite")
        db.execute("CREATE TABLE days(stay_id, day, subject_id, split)")
        db.execute("INSERT INTO days VALUES(1, 0, 7, 'train')")
        db.commit()
        db.close()
        days = folder / "cut/days/1"
        days.mkdir(parents=True)
        status = np.zeros(8640, dtype=np.uint8)
        row = np.full(8640, -1, dtype=np.int16)
        bits = np.zeros(8640, dtype=np.uint8)
        lead = np.full(8640, -1, dtype=np.int16)
        slots = [0, 5, SLOTS_PER_HOUR + 1]
        status[slots], row[slots], bits[slots], lead[slots] = 1, [0, 1, 2], [7, 3, 7], 1
        np.save(days / "day_000_signal.npy", np.stack([np.full((3, 2500), i, dtype=np.float32) for i in range(3)]))
        np.savez(days / "day_000_meta.npz", status=status, row=row, bits=bits, lead=lead)
        return folder

    def test_one_window_per_hour_and_mask_from_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make(folder)
            data = HourlyWindows("train", "A", root / "cut", root / "plan.sqlite")
            self.assertEqual(len(data), 2)
            item = data[1]
            self.assertTrue(item["modality_mask"].all())
            self.assertEqual(item["signal"][0, 0].item(), 2)
            self.assertEqual(item["channel_ids"].tolist(), [1, 12, 13])
            self.assertFalse(item["attention_mask"].any())

    def test_arm_b_keeps_only_charted_slots(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make(folder)
            tokens = np.zeros((1, 64), dtype=np.int32)
            tokens[0, :3] = [101, 2000, 102]
            attention = np.zeros((1, 64), dtype=bool)
            attention[0, :3] = True
            special = ~attention
            special[0, 0] = special[0, 2] = True
            (root / "text").mkdir()
            np.savez(root / "text/snapshots.npz", stay_id=[1], day=[0], slot=[5], tokens=tokens, attention=attention, special=special)
            data = HourlyWindows("train", "B", root / "cut", root / "plan.sqlite", root / "text")
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["signal"][0, 0].item(), 1)
            self.assertEqual(data[0]["input_ids"][1].item(), 2000)


class Text(unittest.TestCase):
    def test_snapshot_uses_field_order_and_real_values(self):
        labels = {220045: "Heart Rate", 220210: "Respiratory Rate"}
        self.assertEqual(prepare_text.snapshot_text({220210: "18", 220045: "88"}, labels),
                         "Heart Rate: 88. Respiratory Rate: 18.")


if __name__ == "__main__":
    unittest.main()
