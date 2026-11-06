import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import common
from clip_model import ClipEncoder, check_terms, draw_windows, make_config, masked_forward, ssl_loss
from day_data import context, hour_rows, patient_weights, targets
from evaluate_day import calibration, organ_event, persistence, primary
from extract_three_min import batches
from long_model import CSFMEncoder, MAEConfig
from sofa_head import SofaHead, corn_terms, cumulative, expected_score, sofa_loss
from temporal_model import MaskedDecoder, TemporalConfig, TemporalSSL, draw_visible, reference_errors, visible_count
from train_day import DayModel
from train_three_min import hourly_sample


class Head(unittest.TestCase):
    def test_cumulative_probabilities_never_increase(self):
        values = cumulative(torch.randn(50, 6, 4) * 3)
        self.assertTrue((values[..., 1:] <= values[..., :-1] + 1e-7).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_readme_example(self):
        conditional = torch.tensor([.95, .84, .375, .167])
        values = cumulative(torch.logit(conditional))
        self.assertTrue(torch.allclose(values, torch.tensor([.95, .798, .29925, .04997]), atol=1e-4))

    def test_expected_score_stays_in_range(self):
        self.assertAlmostEqual(float(expected_score(torch.full((1, 4), 50.))), 4, places=4)
        self.assertAlmostEqual(float(expected_score(torch.full((1, 4), -50.))), 0, places=4)

    def test_corn_trains_threshold_k_only_above_k_minus_one(self):
        future = torch.tensor([[0.] * 6, [2.] * 6])
        label = torch.ones(2, 6, dtype=torch.bool)
        _, count = corn_terms(torch.zeros(2, 6, 4), future, label)
        # Score 0: only k = 0 is eligible. Score 2: k = 0, 1, 2 are eligible.
        self.assertEqual(count.tolist(), [4] * 6)

    def test_unlabeled_organ_never_reaches_the_loss(self):
        output = {"organ": torch.zeros(2, 6, 4, requires_grad=True), "composite": torch.zeros(2, requires_grad=True)}
        label = torch.zeros(2, 6, dtype=torch.bool)
        label[:, 0] = True
        sofa_loss(output, torch.ones(2, 6), label, torch.zeros(2), torch.ones(2, dtype=torch.bool))["task"].backward()
        self.assertTrue((output["organ"].grad[:, 0] != 0).any())
        self.assertTrue((output["organ"].grad[:, 1:] == 0).all())

    def test_composite_is_skipped_without_an_official_total(self):
        output = {"organ": torch.zeros(1, 6, 4), "composite": torch.zeros(1, requires_grad=True)}
        loss = sofa_loss(output, torch.zeros(1, 6), torch.ones(1, 6, dtype=torch.bool), torch.ones(1), torch.zeros(1, dtype=torch.bool))
        self.assertEqual(float(loss["composite"]), 0)

    def test_perfect_prediction_has_lower_loss(self):
        future, label = torch.tensor([[3., 0, 0, 0, 0, 0]]), torch.ones(1, 6, dtype=torch.bool)
        good = (future[..., None] > torch.arange(4)).float() * 40 - 20
        loss = lambda logits: float(sofa_loss({"organ": logits, "composite": torch.zeros(1)}, future, label, torch.zeros(1),
                                              torch.ones(1, dtype=torch.bool))["task"])
        self.assertLess(loss(good), loss(-good))

    def test_waveform_starts_at_zero_correction(self):
        head = SofaHead(12)
        context_values = torch.randn(2, 12)
        first = head(context_values, torch.zeros(2, 768))
        second = head(context_values, torch.randn(2, 768))
        self.assertTrue(torch.equal(first["organ"], second["organ"]))
        self.assertEqual(first["organ"].shape, (2, 6, 4))
        self.assertEqual(first["composite"].shape, (2,))

    def test_baseline_control_has_no_waveform_path(self):
        self.assertIsNone(SofaHead(12, 0).wave)

    def test_wave_only_reads_no_context(self):
        head = SofaHead(0, 768).eval()
        torch.nn.init.ones_(head.wave[-1].weight)
        output = head(None, torch.randn(2, 768))
        self.assertEqual(output["organ"].shape, (2, 6, 4))


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

    def anchors(self):
        columns = {}
        for organ in common.ORGANS:
            columns[f"{organ}_real_base"] = [1., np.nan, 1., 1.]
            columns[f"{organ}_real_next"] = [2., 2., 3., np.nan]
            columns[f"{organ}_full_base"] = [1., 1., 1., 1.]
            columns[f"{organ}_full_next"] = [2., 2., 3., 0.]
        return pd.DataFrame({**columns, "fate": ["stayed", "stayed", "died", "left_alive"]})

    def test_main_scores_death_24_and_discharge_0(self):
        data = targets(self.anchors())
        self.assertEqual(data["label"][:, 0].tolist(), [True, False, True, True])
        self.assertEqual(data["future"][:, 0].tolist(), [2., 2., 4., 0.])
        self.assertEqual(data["official_total"].tolist(), [6., 6., 18., -6.])
        self.assertEqual(data["died"].tolist(), [False, False, True, False])
        self.assertEqual(data["event"].tolist(), [1., 1., 1., 0.])

    def test_locf_keeps_the_last_score_and_death_stays_an_event(self):
        data = targets(self.anchors(), "locf")
        self.assertEqual(data["future"][:, 0].tolist(), [2., 2., 1., 1.])
        self.assertEqual(data["official_total"].tolist(), [6., 6., 0., 0.])
        self.assertEqual(data["event"].tolist(), [1., 1., 1., 0.])

    def test_partial_discharge_uses_the_scores_before_leaving(self):
        data = targets(self.anchors(), "partial_discharge")
        self.assertEqual(data["label"][:, 0].tolist(), [True, False, True, False])
        self.assertEqual(data["official_total"][3], -6.)

    def test_sensitivity_analyses_leave_anchors_out(self):
        self.assertEqual(targets(self.anchors(), "no_death")["used"].tolist(), [True, True, False, True])
        data = targets(self.anchors(), "stayed_only")
        self.assertEqual(data["used"].tolist(), [True, True, False, False])
        self.assertEqual(data["label"][:, 0].tolist(), [True, False, False, False])
        self.assertTrue(np.isnan(data["official_total"][2:]).all())
        self.assertEqual(data["event_known"].tolist(), [True, True, False, False])

    def test_e2_context_has_flags_and_train_statistics(self):
        days = self.anchors()
        columns = [f"prev_{organ}" for organ in common.ORGANS] + ["icu_day"] + [
            f"{name}_{hour:02d}" for name in ("heart_rate", "resp_rate", "spo2", "mbp") for hour in range(24)]
        e2 = pd.DataFrame(np.ones((4, len(columns))), columns=columns)
        e2.iloc[0, 0] = np.nan
        e2.loc[1, "heart_rate_05"] = np.nan
        values, stats = context(days, "e2", e2)
        self.assertEqual(values.shape, (4, 12 + 12 + 1 + 96 + 96))
        self.assertEqual(values[0, 12], 0)
        self.assertEqual(values[0, 18], 0)
        self.assertEqual(values[1, 18], 1)
        flags = values[:, 12 + 12 + 1 + 96:]
        self.assertEqual(flags[1, 5], 0)
        self.assertEqual(flags[0, 5], 1)
        again, _ = context(days, "e2", e2, stats)
        self.assertTrue(np.array_equal(values, again))
        self.assertEqual(context(days, "e0")[0].shape, (4, 12))

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
            with mock.patch.object(common, "CUT_DIR", root):
                values, ids, positions = common.ClipReader().read(7, 0, 0)
        self.assertEqual(values.shape, (17, 3, 2500))
        self.assertEqual(positions.tolist(), [0, 1, 2] + list(range(4, 18)))
        self.assertEqual(values[3, 0, 0], 3)
        self.assertEqual(ids[0].tolist(), [0, 12, 13])


class Masking(unittest.TestCase):
    def test_visible_counts_match_the_readme(self):
        self.assertEqual(visible_count(18), 4)
        self.assertEqual(visible_count(20), 5)
        self.assertEqual(visible_count(1), 1)
        self.assertEqual(visible_count(torch.tensor([18, 20, 16, 1])).tolist(), [4, 5, 4, 1])

    def test_draw_visible_uses_present_segments_only(self):
        present = torch.zeros(3, 20, dtype=torch.bool)
        present[0] = True
        present[1, :8] = True
        index, valid = draw_visible(present, .75, torch.Generator().manual_seed(0))
        self.assertEqual(valid.sum(1).tolist(), [5, 2, 0])
        self.assertTrue(present[1][index[1][valid[1]]].all())
        chosen = index[0][valid[0]]
        self.assertTrue((chosen[1:] > chosen[:-1]).all())

    def test_draw_windows_gives_four_of_eighteen(self):
        visible = draw_windows(5, generator=torch.Generator().manual_seed(1))
        self.assertEqual(visible.shape, (5, 4))
        self.assertTrue((visible[:, 1:] > visible[:, :-1]).all())

    def test_decoder_marks_visible_segments_and_ignores_padding(self):
        decoder = MaskedDecoder(6, 8, 16, 1, 2, 8)
        index = torch.tensor([[1, 4], [2, 0]])
        valid = torch.tensor([[True, True], [True, False]])
        present = torch.ones(2, 6, dtype=torch.bool)
        prediction, shown = decoder(torch.randn(2, 2, 8), index, valid, present)
        self.assertEqual(prediction.shape, (2, 6, 8))
        self.assertEqual(shown.tolist(), [[False, True, False, False, True, False], [False, False, True, False, False, False]])

    def test_targets_are_centered_within_the_recording(self):
        decoder = MaskedDecoder(3, 2, 8, 1, 2, 2)
        teacher = torch.tensor([[[1., 5.], [3., 5.], [100., 100.]]])
        present = torch.tensor([[True, True, False]])
        target = decoder.standardize(teacher, present)
        self.assertEqual(target[0, :2].tolist(), [[-1., 0.], [1., 0.]])
        # A constant shift of the whole recording leaves the target unchanged.
        self.assertTrue(torch.equal(decoder.standardize(teacher + 7, present)[0, :2], target[0, :2]))

    def test_reference_errors_copy_the_nearest_visible_segment(self):
        target = torch.arange(5.)[None, :, None].expand(1, 5, 2).clone()
        index, valid = torch.tensor([[0, 4]]), torch.tensor([[True, True]])
        masked = torch.tensor([[False, True, True, True, False]])
        errors = reference_errors(target, index, valid, masked)
        # Slot 1 copies 0, slot 2 ties and copies 0, slot 3 copies 4: squared errors 1, 4, 1.
        self.assertAlmostEqual(float(errors["copy"]), 2.)
        self.assertAlmostEqual(float(errors["mean"]), (1 + 0 + 1) / 3)


class Models(unittest.TestCase):
    def small(self):
        return CSFMEncoder(MAEConfig(depth=1, heads=2, dim_head=8, mlp_dim=16, dim=16))

    def test_positions_match_the_plain_forward(self):
        core = ClipEncoder(self.small(), make_config()).eval()
        signal, ids = torch.ones(1, 2, 3, 2500), torch.tensor([0, 12, 13]).repeat(1, 2, 1)
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

    def test_level_one_masked_step_trains_adapter_parts_only(self):
        core = ClipEncoder(self.small(), make_config())
        core.classifier.requires_grad_(False)
        signal, ids = torch.randn(2, 18, 3, 2500), torch.tensor([0, 12, 13]).repeat(2, 18, 1)
        teacher = core.teacher_features(signal, ids)
        visible = draw_windows(2, generator=torch.Generator().manual_seed(0))
        step = masked_forward(core, signal, ids, teacher, visible)
        self.assertEqual(step["output"]["local"].shape, (2, 4, 16))
        self.assertEqual(int(step["masked_slots"].sum()), 2 * 14)
        losses = ssl_loss(core, step["output"], step["visible_teacher"])
        (step["masked"] + losses["loss"]).backward()
        self.assertIsNotNone(core.prompts.grad)
        self.assertIsNotNone(core.decoder.mask_token.grad)
        self.assertTrue(all(value.grad is None for value in core.encoder.parameters()))
        checks = check_terms(core, step, visible)
        self.assertTrue(all(torch.isfinite(value) for value in checks.values()))

    def test_level_two_masked_step_with_missing_windows_and_clips(self):
        model = TemporalSSL(TemporalConfig(segments=20, tokens_per_segment=18, input_dim=16, output_dim=16, model_dim=16, heads=2,
                                           projection_dim=8, decoder_width=16))
        tokens, mask = torch.randn(3, 20, 18, 16), torch.rand(3, 20, 18) > .1
        mask[0, 10:] = False
        teacher = torch.randn(3, 20, 16)
        losses = model.masked_step(tokens, mask, teacher, torch.Generator().manual_seed(0), references=True)
        self.assertTrue(all(torch.isfinite(losses[name]) for name in ("loss", "masked", "local", "dynamic", "copy", "mean")))
        losses["loss"].backward()
        self.assertIsNotNone(model.encoder.within.grad)

    def test_level_two_encoding_masks_absent_clips(self):
        model = TemporalSSL(TemporalConfig(segments=4, tokens_per_segment=3, input_dim=8, output_dim=8, model_dim=8, heads=2,
                                           projection_dim=4, decoder_width=8)).eval()
        tokens, mask = torch.randn(2, 4, 3, 8), torch.ones(2, 4, 3, dtype=torch.bool)
        mask[1] = False
        mask[0, 2] = False
        output = model(tokens, mask)
        self.assertEqual(output["mask"].tolist(), [[True, True, False, True], [False] * 4])
        self.assertTrue((output["local"][0, 2] == 0).all())
        self.assertTrue((output["embedding"][1] == 0).all())

    def test_day_model_methods(self):
        batch = {"tokens": torch.randn(3, 24, 20, 768), "token_mask": torch.rand(3, 24, 20) > .2, "context": torch.randn(3, 217),
                 "ten_second": torch.randn(3, 768)}
        for method in ("e0", "e2", "m", "hierarchy", "wave_only"):
            model = DayModel(method, 12 if method == "e0" else 217).eval()
            values = dict(batch, context=batch["context"][:, :12]) if method == "e0" else batch
            output = model(values)
            self.assertEqual(output["organ"].shape, (3, 6, 4), method)
        self.assertIsNone(DayModel("e2", 217).temporal)


class Endpoints(unittest.TestCase):
    def labels(self):
        base = np.array([[2.] * 6, [4.] * 6, [0.] * 6, [1.] * 6])
        future = np.array([[3.] * 6, [4.] * 6, [0.] * 6, [3.] * 6])
        return {"base": base, "future": future, "label": np.ones((4, 6), bool), "base_mask": np.ones((4, 6), bool)}

    def test_worsening_needs_room_to_worsen(self):
        labels = self.labels()
        cum = np.tile(np.array([.95, .80, .30, .05]), (4, 6, 1))
        eligible, event, probability = organ_event(labels, cum, 0, "worse1")
        self.assertEqual(eligible.tolist(), [True, False, True, True])
        self.assertEqual(event[eligible].tolist(), [1., 0., 1.])
        self.assertAlmostEqual(probability[0], .30)
        eligible, event, probability = organ_event(labels, cum, 0, "worse2")
        self.assertEqual(eligible.tolist(), [True, False, True, True])
        self.assertEqual(event[eligible].tolist(), [0., 0., 1.])
        self.assertAlmostEqual(probability[0], .05)
        eligible, _, probability = organ_event(labels, cum, 0, "improve1")
        self.assertEqual(eligible.tolist(), [True, True, False, True])
        self.assertAlmostEqual(probability[0], .20)

    def test_persistence_is_the_baseline(self):
        self.assertEqual(persistence({"base": np.array([[2.]])}).tolist(), [[[1., 1., 0., 0.]]])

    def test_primary_on_a_perfect_model(self):
        n = 40
        rng = np.random.default_rng(0)
        base = rng.integers(0, 4, (n, 6)).astype(float)
        future = np.clip(base + rng.integers(-1, 2, (n, 6)), 0, 4)
        cum = (future[..., None] > np.arange(4)).astype(float)
        event = rng.random(n) > .5
        labels = {"base": base, "future": future, "label": np.ones((n, 6), bool), "event": event.astype(float),
                  "event_known": np.ones(n, bool)}
        result = primary(labels, cum, event.astype(float))
        self.assertEqual(result["composite_auroc"], 1)
        self.assertEqual(result["organ_worse1_macro_auroc"], 1)

    def test_calibration_of_calibrated_probabilities(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(.05, .95, 20000)
        y = (rng.random(20000) < p).astype(float)
        result = calibration(y, p)
        self.assertAlmostEqual(result["slope"], 1, delta=.1)
        self.assertAlmostEqual(result["intercept"], 0, delta=.1)


if __name__ == "__main__":
    unittest.main()
