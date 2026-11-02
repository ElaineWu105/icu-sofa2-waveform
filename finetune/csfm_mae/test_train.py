"""CPU checks for effective-batch math, warmup, leakage guards, and exact resume."""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import copy
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import Dataset

from model import CSFMMAE, MAEConfig
from train import (Metrics, TrainConfig, WarmupCosine, accumulation_step,
                   model_inputs, prepare_masks, run_training, validate_patient_splits, parser)


class ToyDataset(Dataset):
    def __init__(self, count, subject_offset=0):
        generator = torch.Generator().manual_seed(718 + subject_offset)
        self.signal = torch.randn(count, 3, 20, generator=generator)
        self.subject_ids = [subject_offset + i // 2 for i in range(count)]
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        length = 1 + index % 5
        text = torch.tensor([1] + list(range(3, 3 + length)) + [2] + [0] * (6 - length))
        value = torch.tensor([False] + [True] * length + [False] * (7 - length))
        return {'signal': self.signal[index].clone(), 'channel_ids': torch.tensor([1, 12, 13]),
                'modality_mask': torch.ones(3, dtype=torch.bool), 'input_ids': text,
                'attention_mask': text != 0, 'special_tokens_mask': text <= 2,
                'value_mask': value, 'sample_id': torch.tensor(index),
                'subject_id': torch.tensor(self.subject_ids[index])}


def toy_model_config(dropout=0.0):
    return MAEConfig(signal_size=20, patch_size=5, text_len=8, dim=16, depth=2,
        heads=2, dim_head=8, mlp_dim=32, decoder_dim=12, decoder_depth=1,
        decoder_heads=2, decoder_dim_head=6, decoder_mlp_dim=24, vocab_size=32,
        dropout=dropout, emb_dropout=dropout, special_token_ids=(0, 1, 2))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.device = torch.device('cpu')

    def test_accumulation_is_global_masked_mean_with_unequal_text_and_tail(self):
        from torch.utils.data._utils.collate import default_collate
        torch.manual_seed(9)
        cfg = toy_model_config()
        model = CSFMMAE(cfg).train()
        reference = copy.deepcopy(model)
        data = ToyDataset(5)
        batches = [default_collate([data[0], data[1]]),
                   default_collate([data[2], data[3]]), default_collate([data[4]])]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        scheduler = WarmupCosine(optimizer, 10, 0.2)
        reference_scheduler = WarmupCosine(reference_optimizer, 10, 0.2)
        metrics = Metrics()
        step = accumulation_step(model, optimizer, scheduler,
            torch.amp.GradScaler('cuda', enabled=False), batches, self.device,
            torch.Generator().manual_seed(123), metrics)
        generator = torch.Generator().manual_seed(123)
        masks = [prepare_masks(batch, cfg, generator) for batch in batches]
        combined = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        combined_masks = (torch.cat([mask[0] for mask in masks]), torch.cat([mask[1] for mask in masks]))
        loss = reference(**model_inputs(combined, self.device, combined_masks))['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        reference_optimizer.step()
        reference_scheduler.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                       rtol=1e-5, atol=2e-7, msg=name)
        self.assertTrue(step['updated'])
        self.assertEqual(scheduler.completed_updates, 1)
        self.assertEqual(metrics.samples, 5)
        self.assertNotEqual(int(masks[0][1].sum()), int(masks[1][1].sum()))

    def test_warmup_counts_optimizer_updates(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        schedule = WarmupCosine(optimizer, total_updates=20, warmup_ratio=0.10)
        self.assertEqual(schedule.warmup_updates, 2)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.5)
        schedule.step()
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.0)
        saved = schedule.state_dict()
        restored = WarmupCosine(optimizer, total_updates=20, warmup_ratio=0.10)
        restored.load_state_dict(saved)
        self.assertEqual(restored.completed_updates, 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.0)

    def test_resume_matches_uninterrupted_with_dropout_and_partial_epoch(self):
        datasets = {'train': ToyDataset(9), 'val': ToyDataset(4, subject_offset=100)}
        manifest = {'counts': {'train': 9, 'val': 4}, 'training_sqlite_sha256': 'synthetic-v1',
                    'input_signature': 'synthetic-v1', 'tokenizer_contract': {'synthetic_test_only': True}}
        model_cfg = toy_model_config(dropout=0.1)
        with tempfile.TemporaryDirectory(prefix='csfm_trainer_test_') as tmp:
            base = TrainConfig(prepared_dir='injected-synthetic-only', output_dir=str(Path(tmp) / 'whole'),
                checkpoint=None, device='cpu', epochs=3, batch_size=2, accumulation=2,
                workers=0, warmup_epochs=None, warmup_ratio=0.4, diagnostic_samples=0,
                save_every_updates=2, log_every_updates=1000)
            with redirect_stdout(io.StringIO()):
                whole_result = run_training(base, model_config=model_cfg, datasets=datasets,
                                            prepared_manifest=manifest)
                partial = replace(base, output_dir=str(Path(tmp) / 'resumed'), max_updates=2)
                stopped = run_training(partial, model_config=model_cfg, datasets=datasets,
                                       prepared_manifest=manifest)
                resumed = run_training(replace(partial, max_updates=None), model_config=model_cfg,
                                       datasets=datasets, prepared_manifest=manifest)
            self.assertEqual(stopped['status'], 'max_updates')
            self.assertEqual(stopped['next_batch'], 4)
            self.assertEqual(resumed['status'], 'complete')
            self.assertEqual(whole_result['global_update'], resumed['global_update'])
            whole = torch.load(Path(base.output_dir) / 'last.pt', weights_only=True)
            restored = torch.load(Path(partial.output_dir) / 'last.pt', weights_only=True)
            self.assertEqual(whole['scheduler'], restored['scheduler'])
            self.assertEqual(whole['history'], restored['history'])
            for name in whole['model']:
                torch.testing.assert_close(whole['model'][name], restored['model'][name], rtol=0, atol=0, msg=name)
            self.assertTrue((Path(partial.output_dir) / 'best_encoder.pt').exists())
            self.assertTrue((Path(partial.output_dir) / 'metrics.csv').exists())
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'incompatible'):
                run_training(replace(partial, epochs=4), model_config=model_cfg,
                             datasets=datasets, prepared_manifest=manifest)

    def test_default_fixed_seven_epoch_warmup_converts_to_optimizer_updates(self):
        import json
        datasets = {'train': ToyDataset(5), 'val': ToyDataset(2, subject_offset=100)}
        manifest = {'counts': {'train': 5, 'val': 2}, 'training_sqlite_sha256': 'toy-fixed-warmup',
                    'input_signature': 'toy-fixed-warmup'}
        with tempfile.TemporaryDirectory(prefix='csfm_fixed_warmup_') as tmp:
            cfg = TrainConfig(prepared_dir='injected', output_dir=tmp, checkpoint=None,
                device='cpu', batch_size=2, accumulation=2, workers=0, diagnostic_samples=0,
                max_updates=1, log_every_updates=1000)
            with redirect_stdout(io.StringIO()):
                result = run_training(cfg, model_config=toy_model_config(), datasets=datasets,
                                      prepared_manifest=manifest)
            description = json.loads((Path(tmp) / 'run_config.json').read_text())
            self.assertEqual(cfg.epochs, 100)
            self.assertEqual(cfg.early_stopping_patience, 10)
            self.assertEqual(description['updates_per_epoch'], 2)
            self.assertEqual(result['warmup_updates'], 14)
            self.assertEqual(result['total_updates'], 200)
            self.assertEqual(description['warmup_epochs_equivalent'], 7.0)
            self.assertEqual(result['early_stopping_counter'], 0)

    def test_early_stop_patience_ten_after_warmup_persists_and_resumes_exactly(self):
        from unittest.mock import patch
        datasets = {'train': ToyDataset(2), 'val': ToyDataset(2, subject_offset=100)}
        manifest = {'counts': {'train': 2, 'val': 2}, 'training_sqlite_sha256': 'toy-stop',
                    'input_signature': 'toy-stop'}
        model_cfg = toy_model_config(dropout=0.1)
        with tempfile.TemporaryDirectory(prefix='csfm_early_stop_') as tmp:
            base = TrainConfig(prepared_dir='injected', output_dir=str(Path(tmp) / 'whole'),
                checkpoint=None, device='cpu', batch_size=2, accumulation=1, workers=0,
                diagnostic_samples=0, log_every_updates=1000)
            with redirect_stdout(io.StringIO()), patch('train.evaluate', return_value={'loss': 1.0}):
                whole_result = run_training(base, model_config=model_cfg, datasets=datasets,
                                            prepared_manifest=manifest)
                partial_cfg = replace(base, output_dir=str(Path(tmp) / 'resumed'), max_updates=11)
                partial_result = run_training(partial_cfg, model_config=model_cfg, datasets=datasets,
                                              prepared_manifest=manifest)
                partial = torch.load(Path(partial_cfg.output_dir) / 'last.pt', weights_only=True)
                resumed_result = run_training(replace(partial_cfg, max_updates=None), model_config=model_cfg,
                                              datasets=datasets, prepared_manifest=manifest)
            self.assertEqual(whole_result['status'], 'early_stopped')
            self.assertEqual(whole_result['epoch_completed'], 17)
            self.assertEqual(whole_result['best_epoch'], 1)
            self.assertEqual(partial_result['epoch_completed'], 11)
            self.assertEqual(partial['early_stopping_counter'], 4)
            self.assertFalse(partial['early_stopped'])
            self.assertEqual(resumed_result['status'], 'early_stopped')
            self.assertEqual(resumed_result['epoch_completed'], 17)
            self.assertEqual(resumed_result['early_stopping_counter'], 10)
            whole = torch.load(Path(base.output_dir) / 'last.pt', weights_only=True)
            resumed = torch.load(Path(partial_cfg.output_dir) / 'last.pt', weights_only=True)
            self.assertEqual(whole['history'], resumed['history'])
            self.assertEqual([row['early_stopping_counter'] for row in whole['history'][:7]], [0] * 7)
            self.assertEqual([row['early_stopping_counter'] for row in whole['history'][7:]], list(range(1, 11)))
            self.assertTrue(whole['early_stopped'])
            for name in whole['model']:
                torch.testing.assert_close(whole['model'][name], resumed['model'][name], rtol=0, atol=0, msg=name)
            with (redirect_stdout(io.StringIO()), patch('train.evaluate', side_effect=AssertionError('Unexpected validation')),
                  patch('train.accumulation_step', side_effect=AssertionError('Unexpected optimizer step'))):
                again = run_training(replace(partial_cfg, max_updates=None), model_config=model_cfg,
                                     datasets=datasets, prepared_manifest=manifest)
            self.assertEqual(again['status'], 'early_stopped')
            self.assertEqual(again['global_update'], 17)

    def test_early_stopping_can_be_disabled_and_cli_warmup_is_exclusive(self):
        from unittest.mock import patch
        datasets = {'train': ToyDataset(2), 'val': ToyDataset(2, subject_offset=100)}
        manifest = {'counts': {'train': 2, 'val': 2}, 'training_sqlite_sha256': 'toy-disabled',
                    'input_signature': 'toy-disabled'}
        with tempfile.TemporaryDirectory(prefix='csfm_no_stop_') as tmp:
            cfg = TrainConfig(prepared_dir='injected', output_dir=tmp, checkpoint=None,
                device='cpu', epochs=4, warmup_epochs=0, early_stopping_patience=0,
                batch_size=2, accumulation=1, workers=0, diagnostic_samples=0,
                log_every_updates=1000)
            with redirect_stdout(io.StringIO()), patch('train.evaluate', return_value={'loss': 1.0}):
                result = run_training(cfg, model_config=toy_model_config(), datasets=datasets,
                                      prepared_manifest=manifest)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['epoch_completed'], 4)
            self.assertEqual(result['early_stopping_counter'], 0)
            self.assertEqual(result['warmup_updates'], 0)
        cli = parser().parse_args(['--prepared-dir', 'x', '--output-dir', 'y'])
        self.assertEqual(cli.epochs, 100)
        self.assertEqual(cli.early_stopping_patience, 10)
        self.assertIsNone(cli.warmup_epochs)
        self.assertIsNone(cli.warmup_ratio)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(['--prepared-dir', 'x', '--output-dir', 'y',
                                 '--warmup-epochs', '7', '--warmup-ratio', '0.1'])

    def test_patient_overlap_is_rejected(self):
        datasets = {'train': ToyDataset(3), 'val': ToyDataset(2)}
        with self.assertRaisesRegex(RuntimeError, 'Patient split leakage'):
            validate_patient_splits('not-used', datasets)


if __name__ == '__main__':
    unittest.main()
