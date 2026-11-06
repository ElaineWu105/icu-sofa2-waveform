"""Synthetic checks for the target, learning rates, sampling, text and checkpoint choice."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import torch

import common  # noqa: F401
import data
from data import HourlyWindows, SLOTS_PER_HOUR
import finetune
from model import CSFMMAE, MAEConfig
import prepare_text
import choose_checkpoint as selection
from tokenizer_utils import load_tokenizer


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


class Schedule(unittest.TestCase):
    def test_warmup_counts_optimizer_updates(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        schedule = finetune.WarmupCosine(optimizer, total_updates=20, warmup_ratio=0.10)
        self.assertEqual(schedule.warmup_updates, 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        schedule.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        restored = finetune.WarmupCosine(optimizer, total_updates=20, warmup_ratio=0.10)
        restored.load_state_dict(schedule.state_dict())
        self.assertEqual(restored.completed_updates, 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)


class Windows(unittest.TestCase):
    MINUTE, HOUR = 60_000_000, 3_600_000_000

    def make(self, folder, subject=7):
        folder = Path(folder)
        db = sqlite3.connect(folder / "plan.sqlite")
        db.execute("CREATE TABLE days(stay_id, day, subject_id, split, start_us)")
        db.execute("INSERT INTO days VALUES(1, 0, ?, 'train', 0)", (subject,))
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

    def text(self, root, subject=7):
        """Vital signs 10 minutes after slot 5, a chest report 14 hours later, an ECG report 20 hours earlier."""
        tokens = np.zeros((3, 64), dtype=np.int32)
        tokens[:, :3] = [[101, 2000, 102], [101, 2001, 102], [101, 2002, 102]]
        attention = tokens > 0
        special = ~attention
        special[:, [0, 2]] = True
        (root / "text").mkdir()
        times = [55_000_000 + 10 * self.MINUTE, 14 * self.HOUR, -20 * self.HOUR]
        np.savez(root / "text/records.npz", subject_id=[subject] * 3, time_us=times, source=np.array([0, 2, 3], np.int8),
                 tokens=tokens, attention=attention, special=special)

    def test_one_window_per_hour_and_mask_from_bits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make(folder)
            windows = HourlyWindows("train", "A", root / "cut", root / "plan.sqlite")
            self.assertEqual(len(windows), 2)
            item = windows[1]
            self.assertTrue(item["modality_mask"].all())
            self.assertEqual(item["signal"][0, 0].item(), 2)
            self.assertEqual(item["channel_ids"].tolist(), [1, 12, 13])
            self.assertFalse(item["attention_mask"].any())

    def test_arm_b_has_arm_a_windows_and_text_only_within_each_allowed_time(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.make(folder)
            self.text(root)
            a = HourlyWindows("train", "A", root / "cut", root / "plan.sqlite", root / "text")
            b = HourlyWindows("train", "B", root / "cut", root / "plan.sqlite", root / "text")
            self.assertEqual(a.pick.tolist(), b.pick.tolist())
            first = set()
            for epoch in range(20):
                b.set_epoch(epoch)
                first.add(b.text_for(0))
                self.assertEqual(b.text_for(1), 2)
            self.assertEqual(first, {0, 2})
            b.set_epoch(0)
            self.assertIn(b[0]["input_ids"][1].item(), (2000, 2002))
            self.assertNotIn("value_span", b[0])
            spans = HourlyWindows("train", "B", root / "cut", root / "plan.sqlite", root / "text", value_spans=True)
            self.assertFalse(spans[0]["value_span"].any())

    def test_selection_patients_are_held_out(self):
        for subject, held_out in ((3, True), (1, False)):
            with tempfile.TemporaryDirectory() as folder:
                root = self.make(folder, subject)
                self.assertEqual(data.selection_patient(subject), held_out)
                only = lambda: HourlyWindows("train", "A", root / "cut", root / "plan.sqlite", selection="only")
                exclude = lambda: HourlyWindows("train", "A", root / "cut", root / "plan.sqlite", selection="exclude")
                kept, dropped = (only, exclude) if held_out else (exclude, only)
                self.assertEqual(len(kept()), 2)
                with self.assertRaises(RuntimeError):
                    dropped()


class WholeWord(unittest.TestCase):
    def encode(self, texts):
        tokenizer = load_tokenizer()
        out = tokenizer(texts, max_length=64, padding="max_length", truncation=True, return_special_tokens_mask=True,
                        return_tensors="pt")
        return out["input_ids"], out["attention_mask"].bool(), out["special_tokens_mask"].bool(), tokenizer

    def test_pieces_join_the_word_before_them(self):
        ids, valid, special, tokenizer = self.encode(["o2 saturation pulseoxymetry: 96."])
        groups = finetune.word_groups(ids[0].numpy(), (valid & ~special)[0].numpy())
        words = [" ".join(tokenizer.convert_ids_to_tokens(ids[0, group])) for group in groups]
        self.assertEqual(words, ["o ##2", "sat ##uration", "pulse ##ox ##yme ##try", ":", "96", "."])

    def test_masks_whole_words_within_the_budget(self):
        ids, valid, special, _ = self.encode(["o2 saturation pulseoxymetry: 96. respiratory rate: 18."] * 50)
        mask = finetune.whole_word_mask(ids, valid, special, .5, torch.Generator().manual_seed(0))
        budget = round(int(valid[0].sum()) * .5)
        groups = finetune.word_groups(ids[0].numpy(), (valid & ~special)[0].numpy())
        for row in mask:
            self.assertLessEqual(int(row.sum()), budget)
            self.assertFalse((row & special[0]).any())
            for group in groups:
                self.assertIn(int(row[group].sum()), (0, len(group)))
        self.assertGreater(len({tuple(row.tolist()) for row in mask}), 1)

    def test_value_mask_hides_whole_values_and_keeps_field_names(self):
        tokenizer = load_tokenizer()
        text, spans = prepare_text.snapshot_parts(
            {220045: "113", 220048: "AF (Atrial Fibrillation)", 220210: "19"},
            {220045: "Heart Rate", 220048: "Heart Rhythm", 220210: "Respiratory Rate"}, prepare_text.VITALS)
        tokens, attention, special, value, _ = prepare_text.encode(
            tokenizer, [text, "Sinus rhythm. Normal ECG."], prepare_text.continuation_ids(), [spans, []])
        pieces = tokenizer.convert_ids_to_tokens(tokens[0][value[0] > 0])
        self.assertEqual(pieces, ["113", "af", "(", "at", "##rial", "fi", "##bri", "##llation", ")", "19"])
        ids, valid = torch.from_numpy(tokens).long(), torch.from_numpy(attention)
        special, value = torch.from_numpy(special), torch.from_numpy(value).long()
        seen = set()
        for seed in range(30):
            mask = finetune.value_mask(ids, valid, special, value, .5, torch.Generator().manual_seed(seed))
            self.assertFalse((mask[0] & (value[0] == 0)).any())
            for number in (1, 2, 3):
                self.assertIn(int(mask[0][value[0] == number].sum()), (0, int((value[0] == number).sum())))
            self.assertEqual(sum(bool(mask[0][value[0] == number].any()) for number in (1, 2, 3)), 2)
            self.assertTrue(mask[1].any())
            seen.add(tuple(mask[0].tolist()))
        self.assertGreater(len(seen), 1)

    def test_no_text_masks_nothing(self):
        empty = torch.zeros(2, 64, dtype=torch.long)
        mask = finetune.whole_word_mask(empty, empty.bool(), torch.ones(2, 64, dtype=torch.bool), .5, torch.Generator())
        self.assertFalse(mask.any())


class Text(unittest.TestCase):
    def test_snapshot_uses_field_order_and_real_values(self):
        labels = {220045: "Heart Rate", 220210: "Respiratory Rate"}
        self.assertEqual(prepare_text.snapshot_text({220210: "18", 220045: "88"}, labels, prepare_text.VITALS),
                         "Heart Rate: 88. Respiratory Rate: 18.")
        text, spans = prepare_text.snapshot_parts({220210: "18", 220045: "88"}, labels, prepare_text.VITALS)
        self.assertEqual([text[a:b] for a, b in spans], ["88", "18"])

    def test_impression_section_only(self):
        report = "FINDINGS: lines.\n\nIMPRESSION:\n\nSmall ___ effusion.\n\nNOTIFICATION: called"
        self.assertEqual(prepare_text.impression(report), "Small effusion.")
        self.assertEqual(prepare_text.impression("FINDINGS: none"), "")

    def test_long_text_is_cut_at_a_word_boundary(self):
        tokenizer = load_tokenizer()
        tokens, attention, special, _, cut = prepare_text.encode(
            tokenizer, ["word " * 61 + "pulseoxymetry"], prepare_text.continuation_ids())
        pieces = tokenizer.convert_ids_to_tokens(tokens[0][attention[0]])
        self.assertEqual(cut, 1)
        self.assertEqual(pieces[0], "[CLS]")
        self.assertEqual(pieces[-2:], ["word", "[SEP]"])
        self.assertEqual(special[0].tolist(), (~attention[0]).tolist()[:0] + [True] + [False] * 61 + [True, True])


class Selection(unittest.TestCase):
    def test_lowest_rr_among_checkpoints_that_hold_ecg_and_ppg(self):
        reference = {"rhythm": .80, "hr": 4.0, "spo2": 2.5, "sbp": 14.0, "dbp": 8.0, "rr_monitor": 4.3}
        good = dict(reference, rr_monitor=3.5)
        better_rr_but_worse_hr = dict(reference, hr=5.2, rr_monitor=3.0)
        slightly_worse = dict(reference, rhythm=.79, rr_monitor=3.8)
        candidates = {2500: good, 5000: better_rr_but_worse_hr, 7500: slightly_worse}
        self.assertEqual(selection.choose(candidates, reference), 2500)
        self.assertFalse(selection.qualifies(better_rr_but_worse_hr, reference))
        self.assertIsNone(selection.choose({5000: better_rr_but_worse_hr}, reference))


if __name__ == "__main__":
    unittest.main()
