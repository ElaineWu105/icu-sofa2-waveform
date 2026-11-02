"""Read the compact text/pair index and existing waveform shards without copying them."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from torch.utils.data import Dataset


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_prepared_manifest(directory: str | Path, verify_database: bool = True) -> dict:
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise RuntimeError('Prepared-data manifest is incomplete')
    database = directory / 'training.sqlite'
    if verify_database and file_sha256(database) != manifest['training_sqlite_sha256']:
        raise RuntimeError('Prepared training.sqlite does not match its manifest')
    return manifest


class WaveTextDataset(Dataset):
    """One existing 10-second ECG/PPG/Resp window plus a 64-token EHR snapshot."""

    def __init__(self, directory: str | Path, split: str, max_open_shards: int = 16):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        self.directory = Path(directory).resolve()
        self.split = split
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards = OrderedDict()
        connection = sqlite3.connect((self.directory / 'training.sqlite').as_uri() + '?mode=ro', uri=True)
        try:
            self.samples = list(connection.execute('''SELECT sample_id,shard_row,signal_path,
                mask_path,channel_ids_path,text_id,subject_id FROM samples WHERE split=? ORDER BY sample_id''', (split,)))
            required = {row[5] for row in self.samples}
            self.texts = {}
            for text_id, token_ids, attention, special, values in connection.execute(
                    'SELECT text_id,token_ids,attention_mask,special_mask,value_mask FROM texts'):
                if text_id not in required:
                    continue
                arrays = (np.frombuffer(token_ids, dtype='<i4').astype(np.int64),
                          np.frombuffer(attention, dtype=np.uint8).astype(bool),
                          np.frombuffer(special, dtype=np.uint8).astype(bool),
                          np.frombuffer(values, dtype=np.uint8).astype(bool))
                if any(array.shape != (64,) for array in arrays):
                    raise RuntimeError('Prepared text must have exactly 64 positions')
                if not (arrays[1] & ~arrays[2] & arrays[3]).any():
                    raise RuntimeError('Prepared text contains no visible clinical-value token')
                self.texts[text_id] = arrays
            if set(self.texts) != required:
                raise RuntimeError('Samples reference missing prepared text')
        finally:
            connection.close()
        if not self.samples:
            raise RuntimeError(f'No {split} samples in prepared dataset')

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_shards'] = OrderedDict()
        return state

    @staticmethod
    def _close(arrays):
        for array in arrays:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()

    def close(self):
        while self._shards:
            _, arrays = self._shards.popitem(last=False)
            self._close(arrays)

    def _load(self, paths):
        if paths in self._shards:
            self._shards.move_to_end(paths)
            return self._shards[paths]
        opened = []
        try:
            for path in paths:
                opened.append(np.load(path, mmap_mode='r', allow_pickle=False))
        except Exception:
            self._close(opened)
            raise
        arrays = tuple(opened)
        if arrays[0].ndim != 3 or arrays[0].shape[1:] != (3, 2500):
            self._close(arrays)
            raise RuntimeError('Unexpected waveform shard shape')
        if arrays[1].shape != arrays[2].shape or arrays[1].shape != (arrays[0].shape[0], 3):
            self._close(arrays)
            raise RuntimeError('Waveform mask/channel shard shapes disagree')
        self._shards[paths] = arrays
        while len(self._shards) > self.max_open_shards:
            _, old = self._shards.popitem(last=False)
            self._close(old)
        return arrays

    def __getitem__(self, index):
        sample_id, row, signal_path, mask_path, ids_path, text_id, subject_id = self.samples[index]
        signal_array, mask_array, id_array = self._load((signal_path, mask_path, ids_path))
        if not isinstance(row, int) or not 0 <= row < signal_array.shape[0]:
            raise RuntimeError('Prepared waveform shard row is out of bounds')
        signal = np.array(signal_array[row], dtype=np.float32, copy=True)
        modalities = np.array(mask_array[row], dtype=bool, copy=True)
        channel_ids = np.array(id_array[row], dtype=np.int64, copy=True)
        if not modalities.all() or not np.isfinite(signal).all():
            raise RuntimeError('Primary prepared sample must contain three finite, valid modalities')
        if not (0 <= channel_ids[0] <= 11 and channel_ids[1] == 12 and channel_ids[2] == 13):
            raise RuntimeError('Primary sample has invalid or reordered channel IDs')
        tokens, attention, special, value_mask = self.texts[text_id]
        return {'signal': signal, 'channel_ids': channel_ids, 'modality_mask': modalities,
                'input_ids': tokens, 'attention_mask': attention,
                'special_tokens_mask': special, 'value_mask': value_mask,
                'sample_id': np.int64(sample_id), 'subject_id': np.int64(subject_id)}
