"""Small, deterministic checks of masking, target isolation, and training gradients."""
import copy
import unittest
from dataclasses import replace

import torch

from model import Attention, CSFMEncoder, CSFMMAE, MAEConfig


class MAETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.cfg = MAEConfig(signal_size=20, patch_size=5, text_len=8, dim=16,
            depth=2, heads=2, dim_head=8, mlp_dim=32, decoder_dim=12,
            decoder_depth=1, decoder_heads=2, decoder_dim_head=6,
            decoder_mlp_dim=24, vocab_size=32, dropout=0.0, emb_dropout=0.0,
            special_token_ids=(0, 1, 2))
        self.model = CSFMMAE(self.cfg).eval()
        self.signal = torch.randn(2, 3, 20)
        self.ids = torch.tensor([[1, 12, 13], [6, 12, 13]])
        self.text = torch.tensor([[1, 4, 5, 6, 7, 2, 0, 0], [1, 8, 9, 10, 11, 12, 2, 0]])
        self.valid = self.text != 0

    def run_model(self, **kwargs):
        params = dict(signal=self.signal, channel_ids=self.ids, input_ids=self.text,
                      attention_mask=self.valid, return_predictions=True)
        params.update(kwargs)
        if 'waveform_mask' not in params and 'generator' not in params:
            params['generator'] = torch.Generator().manual_seed(19)
        return self.model(**params)

    def test_exact_mask_counts_special_exclusion_and_visible_only_encoder(self):
        out = self.run_model()
        self.assertEqual(out['waveform_mask'].sum(1).tolist(), [9, 9])
        self.assertEqual(out['text_mask'].sum(1).tolist(), [2, 2])
        self.assertFalse((out['text_mask'] & ((self.text <= 2) | ~self.valid)).any())
        expected = 12 - out['waveform_mask'].sum(1) + self.valid.sum(1) - out['text_mask'].sum(1)
        self.assertEqual(out['packed_valid'].sum(1).tolist(), expected.tolist())
        self.assertEqual(out['encoded'].shape[1], 1 + int(expected.max()))

    def test_masked_targets_cannot_leak_to_encoder_or_predictions(self):
        out = self.run_model()
        changed = self.signal.reshape(2, 3, 4, 5).clone()
        hidden = out['waveform_mask'].reshape(2, 3, 4)
        # A per-patch shift would vanish under the normalized target, so change the shape.
        changed[hidden] += 100 * torch.randn_like(changed[hidden])
        changed_text = self.text.clone()
        changed_text[out['text_mask']] = 29
        other = self.run_model(signal=changed.reshape(2, 3, 20), input_ids=changed_text,
            waveform_mask=out['waveform_mask'], text_mask=out['text_mask'])
        torch.testing.assert_close(out['encoded'], other['encoded'], rtol=0, atol=0)
        torch.testing.assert_close(out['wave_prediction'], other['wave_prediction'], rtol=0, atol=0)
        torch.testing.assert_close(out['text_logits'], other['text_logits'], rtol=0, atol=0)
        self.assertNotEqual(out['loss_wave'].item(), other['loss_wave'].item())

    def test_signal_loss_uses_per_patch_normalized_targets(self):
        out = self.run_model()
        mask = out['waveform_mask'].reshape(2, 3, 4)
        target = self.signal.reshape(2, 3, 4, 5)
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        expected = (out['wave_prediction'] - target)[mask].square().mean()
        torch.testing.assert_close(out['loss_wave'], expected)

    def test_padding_values_have_no_effect(self):
        out = self.run_model()
        changed = self.text.clone()
        changed[~self.valid] = 27
        other = self.run_model(input_ids=changed,
            waveform_mask=out['waveform_mask'], text_mask=out['text_mask'])
        torch.testing.assert_close(out['wave_prediction'], other['wave_prediction'], rtol=0, atol=0)
        torch.testing.assert_close(out['text_logits'], other['text_logits'], rtol=0, atol=0)
        torch.testing.assert_close(out['loss'], other['loss'], rtol=0, atol=0)

    def test_old_and_new_encoder_and_decoder_receive_gradients(self):
        self.model.train()
        out = self.run_model()
        out['loss'].backward()
        params = dict(self.model.named_parameters())
        for name in ['encoder.resp_channel_type_embedding', 'resp_decoder_ts_type_embeddings',
                     'encoder.ts_to_patch_embedding.2.weight',
                     'encoder.transformer.layers.0.0.to_qkv.weight',
                     'encoder.text_to_embedding.0.weight', 'to_seg.13.weight']:
            self.assertIsNotNone(params[name].grad, name)
            self.assertGreater(params[name].grad.abs().sum().item(), 0, name)
        self.assertIs(self.model.patch_to_emb[1], self.model.encoder.ts_to_patch_embedding[2])
        self.assertIs(self.model.text_to_emb, self.model.encoder.text_to_embedding)

    def test_missing_channels_are_not_targets_or_context(self):
        present = torch.tensor([[True, True, False], [True, True, False]])
        ids = self.ids.clone()
        ids[:, 2] = -1
        out = self.run_model(channel_ids=ids, modality_mask=present)
        self.assertFalse(out['waveform_mask'].reshape(2, 3, 4)[:, 2].any())
        changed = self.signal.clone()
        changed[:, 2] = float('nan')
        other = self.run_model(signal=changed, channel_ids=ids, modality_mask=present,
            waveform_mask=out['waveform_mask'], text_mask=out['text_mask'])
        self.assertTrue(torch.isfinite(other['loss']))
        torch.testing.assert_close(out['loss'], other['loss'], rtol=0, atol=0)
        self.assertEqual(out['resp_count'].item(), 0)

    def test_strict_roundtrip_and_combined_encoder_export(self):
        other = CSFMMAE(self.cfg).eval()
        other.load_state_dict(copy.deepcopy(self.model.state_dict()), strict=True)
        exported = self.model.export_encoder()
        self.assertEqual(exported['encoder']['ts_channel_type_embedding'].shape, (1, 14, 16))
        self.assertNotIn('resp_channel_type_embedding', exported['encoder'])
        restored_encoder = CSFMEncoder.from_export(exported).eval()
        torch.testing.assert_close(self.model.encoder(self.signal, self.ids),
                                   restored_encoder(self.signal, self.ids), rtol=0, atol=0)
        torch.testing.assert_close(self.model.encoder(self.signal, self.ids),
                                   other.encoder(self.signal, self.ids), rtol=0, atol=0)
        torch.testing.assert_close(self.run_model()['loss'],
            other(self.signal, self.ids, self.text, self.valid,
                  generator=torch.Generator().manual_seed(19))['loss'], rtol=0, atol=0)

    def test_optimizer_groups_cover_parameters_once_and_use_new_lr(self):
        groups = self.model.parameter_groups()
        all_ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {id(p) for p in self.model.parameters()})
        by_id = {id(p): g for g in groups for p in g['params']}
        for name, p in self.model.named_parameters():
            self.assertEqual(by_id[id(p)]['lr'], 1e-4 if name in self.model.new_parameter_names() else 1e-5)

    def test_sdpa_matches_public_attention_math_without_padding(self):
        layer = Attention(16, 2, 8, 0).eval()
        x = torch.randn(2, 5, 16)
        q, k, v = (z.reshape(2, 5, 2, 8).transpose(1, 2)
            for z in layer.to_qkv(layer.norm(x)).chunk(3, dim=-1))
        expected = ((q @ k.transpose(-2, -1)) * (8 ** -0.5)).softmax(-1) @ v
        expected = layer.to_out(expected.transpose(1, 2).reshape(2, 5, 16))
        torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)

    def test_strict_pretrained_loader_rejects_missing_and_unknown_keys(self):
        from unittest.mock import patch
        original = {k: v.clone() for k, v in self.model.state_dict().items()
                    if k not in self.model.new_parameter_names()}
        with patch('model.torch.load', return_value=original):
            report = self.model.load_pretrained('synthetic_pretrained.pth')
        self.assertEqual(len(report['initialized_keys']), 4)
        broken = dict(original)
        del broken['decoder.norm.weight']
        with patch('model.torch.load', return_value=broken), self.assertRaisesRegex(RuntimeError, 'missing='):
            self.model.load_pretrained('synthetic_pretrained.pth')
        broken = dict(original, unknown_weight=torch.zeros(1))
        with patch('model.torch.load', return_value=broken), self.assertRaisesRegex(RuntimeError, 'unexpected='):
            self.model.load_pretrained('synthetic_pretrained.pth')

    def test_special_tokens_mask_excludes_additional_special_ids(self):
        special = self.text == 4
        out = self.run_model(special_tokens_mask=special)
        self.assertFalse((out['text_mask'] & special).any())

    def test_autocast_decoder_restore_preserves_gradients(self):
        self.model.train()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = self.run_model()
        self.assertTrue(torch.isfinite(out['loss']))
        out['loss'].backward()
        self.assertGreater(float(self.model.enc_to_dec.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
