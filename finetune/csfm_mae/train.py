"""Train only the ten-second CSFM waveform/text MAE; never evaluates test data.

A larger effective batch is computed by summing masked-sample/token losses across
microbatches, including the final short accumulation window. Checkpoints resume
at completed optimizer boundaries with deterministic epoch order and RNG states.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal as signal_module
import sqlite3
import time
import uuid
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from dataset import WaveTextDataset, file_sha256, load_prepared_manifest
from model import CSFMMAE, MAEConfig, _sample_mask


@dataclass
class TrainConfig:
    prepared_dir: str
    output_dir: str
    checkpoint: str | None
    device: str = 'cuda:0'
    epochs: int = 100
    batch_size: int = 16
    accumulation: int = 16
    workers: int = 4
    warmup_epochs: float | None = 7.0
    warmup_ratio: float | None = None
    early_stopping_patience: int = 10
    lr: float = 1e-5
    new_lr: float = 1e-4
    text_loss_weight: float = 1.0
    weight_decay: float = 0.05
    seed: int = 42
    resume: str = 'auto'
    save_every_updates: int = 100
    log_every_updates: int = 10
    diagnostic_samples: int = 512
    max_open_shards: int = 16
    max_updates: int | None = None
    deterministic: bool = False


def atomic_json(path: Path, value: Any):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        os.chmod(temporary, 0o600)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as handle:
        os.chmod(temporary, 0o600)
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit(value: dict):
    print(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **value},
                     sort_keys=True, allow_nan=False), flush=True)


def identity_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def capture_rng(device: torch.device, mask_generator: torch.Generator) -> dict:
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'name': state[0], 'keys': torch.tensor(state[1].astype(np.int64)),
                      'position': state[2], 'has_gauss': state[3], 'cached_gaussian': state[4]},
            'torch_cpu': torch.get_rng_state(),
            # Do not initialize or collect RNG from the other seven GPUs.
            'torch_device': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            'mask_generator': mask_generator.get_state()}


def restore_rng(state: dict, device: torch.device, mask_generator: torch.Generator):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state['name'], numpy_state['keys'].numpy().astype(np.uint32),
                         numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda' and state['torch_device'] is not None:
        torch.cuda.set_rng_state(state['torch_device'], device)
    mask_generator.set_state(state['mask_generator'])


class WarmupCosine:
    """Schedule optimizer updates, rather than microbatches, with linear warmup."""
    def __init__(self, optimizer, total_updates: int, warmup_ratio: float | None = None,
                 *, warmup_updates: int | None = None):
        if total_updates < 1:
            raise ValueError('total_updates must be positive')
        if (warmup_ratio is None) == (warmup_updates is None):
            raise ValueError('Specify exactly one of warmup_ratio or warmup_updates')
        if warmup_ratio is not None:
            if not math.isfinite(warmup_ratio) or not 0 < warmup_ratio < 1:
                raise ValueError('warmup_ratio must be finite and in (0,1)')
            warmup_updates = max(1, math.ceil(total_updates * warmup_ratio))
        if not 0 <= warmup_updates < total_updates:
            raise ValueError('Warmup must finish before the total optimizer-update budget')
        self.optimizer = optimizer
        self.total_updates = total_updates
        self.warmup_updates = int(warmup_updates)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.completed_updates = 0
        self._apply()

    def factor(self, index: int) -> float:
        if index < self.warmup_updates:
            return (index + 1) / self.warmup_updates
        progress = min(1.0, (index - self.warmup_updates) / max(1, self.total_updates - self.warmup_updates))
        return 0.5 * (1 + math.cos(math.pi * progress))

    def _apply(self):
        factor = self.factor(self.completed_updates)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base * factor

    def step(self):
        self.completed_updates += 1
        self._apply()

    def state_dict(self):
        return {'total_updates': self.total_updates, 'warmup_updates': self.warmup_updates,
                'base_lrs': self.base_lrs, 'completed_updates': self.completed_updates}

    def load_state_dict(self, state):
        if state['total_updates'] != self.total_updates or state['warmup_updates'] != self.warmup_updates:
            raise RuntimeError('Resume scheduler budget differs')
        self.base_lrs = list(state['base_lrs'])
        self.completed_updates = int(state['completed_updates'])
        self._apply()


def prepare_masks(batch: dict[str, Tensor], cfg: MAEConfig,
                  generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Generate CPU masks before accumulation to know the exact global denominators."""
    modalities = batch['modality_mask'].bool()
    eligible_wave = modalities[..., None].expand(-1, -1, cfg.num_patches).flatten(1)
    eligible_text = batch['attention_mask'].bool() & ~batch['special_tokens_mask'].bool()
    for token in cfg.special_token_ids:
        eligible_text &= batch['input_ids'] != token
    return (_sample_mask(eligible_wave, cfg.waveform_mask_ratio, generator),
            _sample_mask(eligible_text, cfg.text_mask_ratio, generator))


def model_inputs(batch: dict[str, Tensor], device: torch.device,
                 masks: tuple[Tensor, Tensor]) -> dict:
    keys = ('signal', 'channel_ids', 'modality_mask', 'input_ids',
            'attention_mask', 'special_tokens_mask')
    result = {key: batch[key].to(device, non_blocking=device.type == 'cuda') for key in keys}
    result['waveform_mask'] = masks[0].to(device, non_blocking=device.type == 'cuda')
    result['text_mask'] = masks[1].to(device, non_blocking=device.type == 'cuda')
    return result


class Metrics:
    def __init__(self, state: dict | None = None):
        self.sums = {name: 0.0 for name in ('wave', 'text', 'ecg', 'ppg', 'resp')}
        self.counts = {name: 0 for name in self.sums}
        self.text_correct = self.value_correct = self.value_count = self.samples = 0
        if state:
            self.__dict__.update(state)

    def update(self, out: dict, batch: dict | None = None):
        for name in self.sums:
            count = int(out[f'{name}_count'].item())
            self.sums[name] += float(out[f'loss_{name}'].detach().item()) * count
            self.counts[name] += count
        self.text_correct += int(out['text_correct'].item())
        if batch is not None:
            self.samples += len(batch['signal'])
            if 'text_logits' in out:
                targets = batch['input_ids'].to(out['text_mask'].device)[out['text_mask']]
                chosen = batch['value_mask'].to(out['text_mask'].device)[out['text_mask']].bool()
                self.value_correct += int(((out['text_logits'].argmax(-1) == targets) & chosen).sum().item())
                self.value_count += int(chosen.sum().item())

    def state_dict(self):
        return dict(self.__dict__)

    def values(self, text_weight: float) -> dict:
        result = {f'loss_{key}': self.sums[key] / self.counts[key] if self.counts[key] else 0.0
                  for key in self.sums}
        result['loss'] = result['loss_wave'] + text_weight * result['loss_text']
        result.update(samples=self.samples, text_accuracy=self.text_correct / self.counts['text'] if self.counts['text'] else None,
                      value_accuracy=self.value_correct / self.value_count if self.value_count else None,
                      wave_targets=self.counts['wave'], text_targets=self.counts['text'], value_targets=self.value_count)
        return result


def accumulation_step(model: CSFMMAE, optimizer, scheduler: WarmupCosine,
                      scaler, batches: list[dict], device: torch.device,
                      mask_generator: torch.Generator, metrics: Metrics) -> dict:
    """Exact masked-point/token mean over the whole effective batch, including tails."""
    cfg = model.config
    masks = [prepare_masks(batch, cfg, mask_generator) for batch in batches]
    total_wave = sum(int(mask[0].sum()) * cfg.patch_size for mask in masks)
    total_text = sum(int(mask[1].sum()) for mask in masks)
    optimizer.zero_grad(set_to_none=True)
    amp = device.type == 'cuda'
    for batch, mask in zip(batches, masks):
        context = torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext()
        with context:
            out = model(**model_inputs(batch, device, mask))
            loss = (out['loss_wave'] * (int(mask[0].sum()) * cfg.patch_size / max(1, total_wave))
                    + cfg.text_loss_weight * out['loss_text'] * (int(mask[1].sum()) / max(1, total_text)))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite MAE loss; the last completed optimizer checkpoint remains available')
        scaler.scale(loss).backward()
        metrics.update(out, batch)
    scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    if not scaler.is_enabled() and not math.isfinite(grad_norm):
        raise RuntimeError('Nonfinite full-precision gradient norm; refusing an invalid optimizer step')
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = not scaler.is_enabled() or scaler.get_scale() >= old_scale
    if updated:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return {'updated': updated, 'grad_norm': grad_norm, 'amp_scale': scaler.get_scale(),
            'wave_targets': total_wave, 'text_targets': total_text}


def make_loader(dataset: Dataset, cfg: TrainConfig, epoch: int | None,
                *, validation: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed + 100000 + (epoch or 0))
    order = None if validation else torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.seed + int(epoch))).tolist()
    options = dict(batch_size=cfg.batch_size, sampler=order, shuffle=False,
                   num_workers=cfg.workers, pin_memory=str(cfg.device).startswith('cuda'),
                   drop_last=False, generator=generator)
    if cfg.workers:
        options.update(multiprocessing_context='spawn', prefetch_factor=2)
    return DataLoader(dataset, **options)


class TrainingInterrupted(Exception):
    pass


@torch.no_grad()
def evaluate(model: CSFMMAE, loader: DataLoader, device: torch.device,
             seed: int, stop_requested=None) -> dict:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    metrics = Metrics()
    for batch in loader:
        if stop_requested is not None and stop_requested():
            raise TrainingInterrupted()
        mask = prepare_masks(batch, model.config, generator)
        context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
        with context:
            out = model(**model_inputs(batch, device, mask), return_predictions=True)
        if not torch.isfinite(out['loss']):
            raise RuntimeError('Nonfinite validation loss')
        metrics.update(out, batch)
    return metrics.values(model.config.text_loss_weight)


@torch.no_grad()
def diagnostics(model: CSFMMAE, dataset: Dataset, cfg: TrainConfig,
                device: torch.device, stop_requested=None) -> dict:
    """Fixed validation subset; correct/no/shuffled context with identical wave masks."""
    if cfg.diagnostic_samples <= 0:
        return {}
    from torch.utils.data._utils.collate import default_collate
    count = min(cfg.diagnostic_samples, len(dataset))
    # Spread across the validation cohort instead of selecting only its first patient.
    selected = np.linspace(0, len(dataset) - 1, num=count, dtype=int).tolist()
    all_data = default_collate([dataset[i] for i in selected])
    generator = torch.Generator().manual_seed(cfg.seed + 87321)
    wave_mask, text_mask = prepare_masks(all_data, model.config, generator)
    donors = None
    if 'subject_id' in all_data:
        subjects = all_data['subject_id'].tolist()
        candidate = []
        for i in range(count):
            donor = next((j for j in range(i + 1, i + count) if subjects[j % count] != subjects[i]), None)
            if donor is None:
                candidate = []
                break
            candidate.append(donor % count)
        if candidate:
            donors = torch.tensor(candidate, dtype=torch.long)
    result = {'samples': count, 'shuffled_text_available': donors is not None}
    for mode in ['paired', 'no_text'] + (['shuffled_text'] if donors is not None else []):
        metrics = Metrics()
        for left in range(0, count, cfg.batch_size):
            if stop_requested is not None and stop_requested():
                raise TrainingInterrupted()
            right = min(count, left + cfg.batch_size)
            batch = {key: value[left:right].clone() for key, value in all_data.items()}
            wmask, tmask = wave_mask[left:right], text_mask[left:right]
            if mode == 'no_text':
                batch['input_ids'].zero_()
                batch['attention_mask'].zero_()
                batch['special_tokens_mask'].fill_(True)
                batch['value_mask'].zero_()
                tmask = torch.zeros_like(tmask)
            elif mode == 'shuffled_text':
                chosen = donors[left:right]
                for key in ('input_ids', 'attention_mask', 'special_tokens_mask', 'value_mask'):
                    batch[key] = all_data[key][chosen]
                tmask = text_mask[chosen]
            context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
            with context:
                out = model(**model_inputs(batch, device, (wmask, tmask)), return_predictions=True)
            metrics.update(out, batch)
        values = metrics.values(model.config.text_loss_weight)
        result[mode] = {key: values[key] for key in ('loss_wave', 'loss_ecg', 'loss_ppg', 'loss_resp', 'loss_text', 'value_accuracy')}
    return result


def validate_patient_splits(prepared_dir: str, datasets: dict | None = None):
    if datasets is not None:
        train_subjects = getattr(datasets['train'], 'subject_ids', None)
        val_subjects = getattr(datasets['val'], 'subject_ids', None)
        if train_subjects is not None and val_subjects is not None and set(train_subjects) & set(val_subjects):
            raise RuntimeError('Patient split leakage between train and validation')
        return
    connection = sqlite3.connect(Path(prepared_dir, 'training.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    try:
        collision = connection.execute('SELECT 1 FROM samples GROUP BY subject_id HAVING count(DISTINCT split)>1 LIMIT 1').fetchone()
    finally:
        connection.close()
    if collision:
        raise RuntimeError('Patient split leakage in prepared data')


def save_metrics(output: Path, history: list[dict]):
    json_path = output / 'metrics.jsonl'
    temporary = json_path.with_name(json_path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as handle:
        os.chmod(temporary, 0o600)
        for record in history:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, json_path)
    rows = []
    for record in history:
        row = {key: value for key, value in record.items() if not isinstance(value, dict)}
        for section in ('train', 'val'):
            row.update({f'{section}_{key}': value for key, value in record.get(section, {}).items()})
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    temporary = output / ('metrics.csv.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', newline='') as handle:
        os.chmod(temporary, 0o600)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / 'metrics.csv')


def _critical_identity(cfg: TrainConfig, model_config: MAEConfig, manifest: dict,
                       checkpoint_sha256: str | None) -> dict:
    runtime_only = {'output_dir', 'prepared_dir', 'checkpoint', 'device', 'workers', 'resume',
                    'save_every_updates', 'log_every_updates', 'diagnostic_samples',
                    'max_open_shards', 'max_updates'}
    return {'train': {key: value for key, value in asdict(cfg).items() if key not in runtime_only},
            'model': asdict(model_config), 'checkpoint_sha256': checkpoint_sha256,
            'training_sqlite_sha256': manifest.get('training_sqlite_sha256'),
            'prepared_input_signature': manifest.get('input_signature'),
            'tokenizer_contract': manifest.get('tokenizer_contract'),
            'split_counts': manifest['counts'],
            'code_sha256': {name: file_sha256(Path(__file__).with_name(name))
                            for name in ('model.py', 'dataset.py', 'train.py')}}


def run_training(cfg: TrainConfig, *, model_config: MAEConfig | None = None,
                 datasets: dict[str, Dataset] | None = None,
                 prepared_manifest: dict | None = None) -> dict:
    """Test injection is Python-only; CLI always loads the audited prepared dataset."""
    if min(cfg.epochs, cfg.batch_size, cfg.accumulation) < 1:
        raise ValueError('Epochs/batch/accumulation must be positive')
    if cfg.warmup_epochs is None and cfg.warmup_ratio is None:
        cfg = replace(cfg, warmup_epochs=7.0)
    if cfg.warmup_epochs is not None and cfg.warmup_ratio is not None:
        raise ValueError('warmup_epochs and warmup_ratio are mutually exclusive')
    if cfg.warmup_ratio is not None:
        if not math.isfinite(cfg.warmup_ratio) or not 0 < cfg.warmup_ratio < 1:
            raise ValueError('warmup_ratio must be finite and in (0,1)')
    elif not math.isfinite(cfg.warmup_epochs) or not 0 <= cfg.warmup_epochs < cfg.epochs:
        raise ValueError('warmup_epochs must be finite, nonnegative and less than total epochs')
    if cfg.early_stopping_patience < 0:
        raise ValueError('early_stopping_patience must be nonnegative; zero disables it')
    if cfg.workers < 0 or cfg.save_every_updates < 1 or cfg.log_every_updates < 1:
        raise ValueError('Invalid worker/save/log setting')
    if cfg.max_updates is not None and cfg.max_updates < 1:
        raise ValueError('max_updates must be positive')
    for name, value in [('lr', cfg.lr), ('new_lr', cfg.new_lr)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    for name, value in [('weight_decay', cfg.weight_decay), ('text_loss_weight', cfg.text_loss_weight)]:
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 <= cfg.seed < 2**32 or cfg.diagnostic_samples < 0 or cfg.max_open_shards < 1:
        raise ValueError('Invalid seed, diagnostic budget, or shard cache size')
    output = Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    with (output / 'train.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another trainer owns this output directory') from error
        return _run_locked(cfg, output, model_config, datasets, prepared_manifest)


def _run_locked(cfg, output, model_config, datasets, prepared_manifest):
    manifest = prepared_manifest if prepared_manifest is not None else load_prepared_manifest(cfg.prepared_dir)
    validate_patient_splits(cfg.prepared_dir, datasets)
    train_data = datasets['train'] if datasets is not None else WaveTextDataset(cfg.prepared_dir, 'train', cfg.max_open_shards)
    val_data = datasets['val'] if datasets is not None else WaveTextDataset(cfg.prepared_dir, 'val', cfg.max_open_shards)
    if len(train_data) != manifest['counts']['train'] or len(val_data) != manifest['counts']['val']:
        raise RuntimeError('Prepared split counts differ from loaded datasets')
    contract = manifest.get('tokenizer_contract', {})
    specials = contract.get('special_token_ids', {})
    model_config = model_config or MAEConfig(special_token_ids=tuple(specials.values()) if isinstance(specials, dict) and specials else MAEConfig().special_token_ids)
    model_config = replace(model_config, text_loss_weight=cfg.text_loss_weight)
    device = torch.device(cfg.device)
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeError(f'Unavailable training device: {cfg.device}')
    if cfg.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        # manual_seed affects only the currently selected CUDA device.
        torch.cuda.manual_seed(cfg.seed)
    checkpoint_hash = file_sha256(cfg.checkpoint) if cfg.checkpoint is not None else None
    if cfg.checkpoint is None and datasets is None:
        raise RuntimeError('A pretrained CSFM checkpoint is required')
    identity = _critical_identity(cfg, model_config, manifest, checkpoint_hash)
    fingerprint = identity_hash(identity)
    config_path, last_path = output / 'run_config.json', output / 'last.pt'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous['identity_fingerprint'] != fingerprint:
            raise RuntimeError('Output contains an incompatible run; use a new --output-dir. Existing files were preserved.')
    elif any((output / name).exists() for name in ('last.pt', 'best_encoder.pt', 'metrics.jsonl', 'metrics.csv', 'progress.json')):
        raise RuntimeError('Training artifacts exist without their run identity; select a new output directory')
    if last_path.exists() and cfg.resume == 'none':
        raise RuntimeError('Output already has a checkpoint; --resume none requires a new output directory')
    resume_path = last_path if cfg.resume == 'auto' and last_path.exists() else None
    if cfg.resume not in ('auto', 'none'):
        resume_path = Path(cfg.resume).resolve()
        if not resume_path.exists():
            raise RuntimeError('Explicit resume checkpoint does not exist')
    saved = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path else None
    if saved and saved['identity_fingerprint'] != fingerprint:
        raise RuntimeError('Resume checkpoint model/data/optimizer identity differs')
    if saved and saved['best_metric'] is not None and not (output / 'best_encoder.pt').exists():
        source_best = resume_path.parent / 'best_encoder.pt'
        if not source_best.exists():
            raise RuntimeError('Resume checkpoint references a missing best encoder artifact')
        best_bundle = torch.load(source_best, map_location='cpu', weights_only=True)
        if best_bundle.get('identity_fingerprint') != fingerprint:
            raise RuntimeError('Resume best encoder artifact identity differs')
        atomic_torch(output / 'best_encoder.pt', best_bundle)
        del best_bundle
    model = CSFMMAE(model_config)
    if saved:
        pretrained_report = saved['pretrained_load_report']
        model.load_state_dict(saved['model'], strict=True)
    elif cfg.checkpoint is not None:
        pretrained_report = model.load_pretrained(cfg.checkpoint)
    else:
        pretrained_report = {'synthetic_test_only': True, 'initialized_all': True}
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameter_groups(cfg.lr, cfg.new_lr, cfg.weight_decay), betas=(0.9, 0.95))
    batches_per_epoch = math.ceil(len(train_data) / cfg.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / cfg.accumulation)
    scheduler = (WarmupCosine(optimizer, cfg.epochs * updates_per_epoch, cfg.warmup_ratio)
                 if cfg.warmup_ratio is not None else
                 WarmupCosine(optimizer, cfg.epochs * updates_per_epoch,
                              warmup_updates=math.ceil(cfg.warmup_epochs * updates_per_epoch)))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    mask_generator = torch.Generator().manual_seed(cfg.seed + 3041)
    epoch, next_batch, global_update = 0, 0, 0
    best_metric, best_epoch = None, None
    early_stopping_counter, early_stopped = 0, False
    history, epoch_metrics = [], Metrics()
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        epoch, next_batch, global_update = saved['epoch'], saved['next_batch'], saved['global_update']
        history = saved['history']
        epoch_metrics = Metrics(saved['epoch_metrics'])
        best_metric, best_epoch = saved['best_metric'], saved['best_epoch']
        early_stopping_counter = saved.get('early_stopping_counter', 0)
        early_stopped = saved.get('early_stopped', False)
        restore_rng(saved['rng'], device, mask_generator)
        # Drop the extra CPU state copies before any GPU forward.
        del saved
    run_description = {'identity_fingerprint': fingerprint, 'identity': identity,
                       'configuration': asdict(cfg), 'model_config': asdict(model_config),
                       'pretrained_load_report': pretrained_report,
                       'tokenizer_contract': contract,
                       'original_csfm_token_mapping_verified': contract.get('original_csfm_mapping_verified', False),
                       'total_updates': scheduler.total_updates, 'warmup_updates': scheduler.warmup_updates,
                       'batches_per_epoch': batches_per_epoch, 'updates_per_epoch': updates_per_epoch,
                       'warmup_epochs_equivalent': scheduler.warmup_updates / updates_per_epoch,
                       'early_stopping': {'patience': cfg.early_stopping_patience,
                                          'metric': 'full validation total MAE loss',
                                          'min_delta': 0.0,
                                          'best_selection': 'All epochs including warmup',
                                          'counting': 'Only full validations strictly after the warmup update budget'},
                       'signal_loss': 'Mean squared L2 over masked, originally valid scalar samples',
                       'text_loss': 'Mean cross entropy over masked nonspecial content tokens',
                       'resume_boundary': 'Completed accumulation/optimizer boundary',
                       'scope': '10-second CSFM MAE only; no CXR or 3-minute training',
                       'code_sha256': {name: file_sha256(Path(__file__).with_name(name))
                                       for name in ('model.py', 'dataset.py', 'train.py')}}
    atomic_json(config_path, run_description)
    save_metrics(output, history)
    emit({'stage': 'initialized', 'resumed': bool(resume_path), 'epoch': epoch + 1,
          'global_update': global_update, 'train_samples': len(train_data), 'val_samples': len(val_data),
          'total_updates': scheduler.total_updates, 'warmup_updates': scheduler.warmup_updates,
          'effective_batch': cfg.batch_size * cfg.accumulation,
          'early_stopping_patience': cfg.early_stopping_patience,
          'early_stopping_counter': early_stopping_counter, 'early_stopped': early_stopped,
          'original_csfm_token_mapping_verified': run_description['original_csfm_token_mapping_verified']})
    stop = {'requested': False, 'signal': None}
    previous_handlers = {}
    def request_stop(signum, frame):
        stop['requested'], stop['signal'] = True, signum
    for signum in (signal_module.SIGINT, signal_module.SIGTERM):
        previous_handlers[signum] = signal_module.signal(signum, request_stop)

    def checkpoint():
        payload = {'format_version': 2, 'identity_fingerprint': fingerprint,
                   'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                   'epoch': epoch, 'next_batch': next_batch, 'global_update': global_update,
                   'best_metric': best_metric, 'best_epoch': best_epoch,
                   'early_stopping_counter': early_stopping_counter,
                   'early_stopped': early_stopped,
                   'history': history, 'epoch_metrics': epoch_metrics.state_dict(),
                   'rng': capture_rng(device, mask_generator),
                   'pretrained_load_report': pretrained_report}
        atomic_torch(last_path, payload)

    started = time.monotonic()
    status = 'early_stopped' if early_stopped else 'running'
    try:
        if not resume_path:
            checkpoint()
        while epoch < cfg.epochs and not early_stopped:
            if stop['requested'] or (cfg.max_updates is not None and global_update >= cfg.max_updates):
                status = 'interrupted' if stop['requested'] else 'max_updates'
                break
            model.train()
            loader = make_loader(train_data, cfg, epoch)
            iterator = iter(loader)
            for _ in range(next_batch):
                next(iterator)
            while next_batch < batches_per_epoch:
                group = []
                for _ in range(min(cfg.accumulation, batches_per_epoch - next_batch)):
                    group.append(next(iterator))
                update = accumulation_step(model, optimizer, scheduler, scaler, group,
                                           device, mask_generator, epoch_metrics)
                next_batch += len(group)
                global_update += int(update['updated'])
                if global_update % cfg.log_every_updates == 0 or next_batch == batches_per_epoch:
                    progress = {'status': 'running', 'epoch': epoch + 1, 'epochs': cfg.epochs,
                                'next_batch': next_batch, 'batches_per_epoch': batches_per_epoch,
                                'global_update': global_update, 'total_updates': scheduler.total_updates,
                                'elapsed_seconds_this_invocation': time.monotonic() - started,
                                'metrics': epoch_metrics.values(model.config.text_loss_weight),
                                'lr_groups': {g['group_name']: g['lr'] for g in optimizer.param_groups},
                                'amp_scale': update['amp_scale'],
                                'optimizer_step_skipped': not update['updated'],
                                'early_stopping_counter': early_stopping_counter,
                                'early_stopping_patience': cfg.early_stopping_patience}
                    atomic_json(output / 'progress.json', progress)
                    emit({'stage': 'train', **progress})
                at_limit = cfg.max_updates is not None and global_update >= cfg.max_updates
                should_save = (update['updated'] and global_update > 0 and global_update % cfg.save_every_updates == 0) or stop['requested'] or at_limit
                if should_save:
                    checkpoint()
                if stop['requested'] or (at_limit and next_batch < batches_per_epoch):
                    status = 'interrupted' if stop['requested'] else 'max_updates'
                    break
            if status != 'running':
                break
            try:
                val = evaluate(model, make_loader(val_data, cfg, None, validation=True), device,
                               cfg.seed + 90210, stop_requested=lambda: stop['requested'])
                diag = diagnostics(model, val_data, cfg, device,
                                   stop_requested=lambda: stop['requested'])
            except TrainingInterrupted:
                status = 'interrupted'
                checkpoint()
                break
            metric = val['loss']
            improved = best_metric is None or metric < best_metric
            if improved:
                best_metric, best_epoch = metric, epoch + 1
                bundle = model.export_encoder()
                bundle.update(epoch=best_epoch, global_update=global_update,
                              val_metrics=val, identity_fingerprint=fingerprint,
                              tokenizer_contract=contract, pretrained_load_report=pretrained_report)
                atomic_torch(output / 'best_encoder.pt', bundle)
            # The validation exactly at the warmup boundary still belongs to
            # warmup. With seven complete warmup epochs, epoch 8 counts first.
            early_stopping_active = (cfg.early_stopping_patience > 0
                                     and global_update > scheduler.warmup_updates)
            if improved or not early_stopping_active:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            early_stopped = (early_stopping_active
                             and early_stopping_counter >= cfg.early_stopping_patience)
            history.append({'epoch': epoch + 1, 'global_update': global_update,
                            'early_stopping_active': early_stopping_active,
                            'early_stopping_counter': early_stopping_counter,
                            'early_stopped': early_stopped,
                            'train': epoch_metrics.values(model.config.text_loss_weight),
                            'val': val, 'diagnostics': diag, 'best': improved})
            save_metrics(output, history)
            emit({'stage': 'validation', **history[-1]})
            epoch, next_batch, epoch_metrics = epoch + 1, 0, Metrics()
            checkpoint()
        if early_stopped:
            status = 'early_stopped'
        elif epoch >= cfg.epochs:
            status = 'complete'
        else:
            checkpoint()
        final = {'status': status, 'epoch_completed': epoch, 'next_batch': next_batch,
                 'global_update': global_update, 'best_epoch': best_epoch,
                 'best_val_loss': best_metric, 'total_updates': scheduler.total_updates,
                 'warmup_updates': scheduler.warmup_updates,
                 'early_stopping_counter': early_stopping_counter,
                 'early_stopping_patience': cfg.early_stopping_patience,
                 'early_stopped': early_stopped,
                 'elapsed_seconds_this_invocation': time.monotonic() - started}
        atomic_json(output / 'progress.json', final)
        emit({'stage': 'finished', **final})
        return final
    finally:
        for signum, handler in previous_handlers.items():
            signal_module.signal(signum, handler)
        for dataset in (train_data, val_data):
            close = getattr(dataset, 'close', None)
            if close:
                close()


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--prepared-dir', required=True)
    result.add_argument('--output-dir', required=True)
    result.add_argument('--checkpoint', default=str(root / 'pretrained_model_for_share' / 'CSFM_base.pth'))
    result.add_argument('--device', default='cuda:0')
    for name, default in [('epochs', 100), ('batch-size', 16), ('accumulation', 16), ('workers', 4),
                          ('seed', 42), ('save-every-updates', 100), ('log-every-updates', 10),
                          ('diagnostic-samples', 512), ('max-open-shards', 16),
                          ('early-stopping-patience', 10)]:
        result.add_argument('--' + name, type=int, default=default)
    warmup = result.add_mutually_exclusive_group()
    warmup.add_argument('--warmup-epochs', type=float, default=None,
                        help='Fixed warmup duration in epochs (default: 7)')
    warmup.add_argument('--warmup-ratio', type=float, default=None,
                        help='Legacy alternative: warmup as a fraction of all planned updates')
    for name, default in [('lr', 1e-5), ('new-lr', 1e-4),
                          ('text-loss-weight', 1.0), ('weight-decay', 0.05)]:
        result.add_argument('--' + name, type=float, default=default)
    result.add_argument('--max-updates', type=int)
    result.add_argument('--resume', default='auto', help='auto, none, or a matching full MAE checkpoint path')
    result.add_argument('--deterministic', action='store_true')
    return result


def main():
    args = parser().parse_args()
    if args.warmup_epochs is None and args.warmup_ratio is None:
        args.warmup_epochs = 7.0
    run_training(TrainConfig(**vars(args)))


if __name__ == '__main__':
    main()
