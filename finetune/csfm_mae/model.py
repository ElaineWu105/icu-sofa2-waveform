"""CSFM-compatible 10 s multimodal MAE for ECG/PPG/Resp and EHR text.

Attention/FFN and encoder embedding layout follow the public CSFM implementation:
https://github.com/guxiao0822/Cardiac-Sensing-FM/blob/main/network/model.py
The unpublished MAE forward is reconstructed explicitly (not a bitwise reproduction).
Signal loss is mean squared L2 error over masked, originally valid samples; text
loss is mean CE over masked content tokens. Their weight is configurable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MAEConfig:
    signal_size: int = 2500
    patch_size: int = 25
    text_len: int = 64
    dim: int = 768
    depth: int = 12
    heads: int = 12
    dim_head: int = 64
    mlp_dim: int = 3072
    decoder_dim: int = 256
    decoder_depth: int = 3
    decoder_heads: int = 8
    decoder_dim_head: int = 64
    decoder_mlp_dim: int = 1024
    vocab_size: int = 30522
    pretrained_channels: int = 13
    dropout: float = 0.1
    emb_dropout: float = 0.1
    waveform_mask_ratio: float = 0.75
    text_mask_ratio: float = 0.5
    text_loss_weight: float = 1.0
    special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103)

    @property
    def num_patches(self) -> int:
        return self.signal_size // self.patch_size


class Patchify(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], x.shape[-1] // self.patch_size, self.patch_size)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float):
        super().__init__()
        self.heads, self.dim_head, self.dropout_p = heads, dim_head, dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        b, n, _ = x.shape
        q, k, v = (z.reshape(b, n, self.heads, self.dim_head).transpose(1, 2)
                   for z in self.to_qkv(self.norm(x)).chunk(3, dim=-1))
        # True means an available key, unlike the public model's unused mask API.
        # This prevents padded text/packed slots from affecting real tokens.
        allowed = valid[:, None, None, :] if valid is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
            dropout_p=self.dropout_p if self.training else 0.0)
        return self.to_out(y.transpose(1, 2).reshape(b, n, self.heads * self.dim_head))


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int,
                 mlp_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([nn.ModuleList([
            Attention(dim, heads, dim_head, dropout), FeedForward(dim, mlp_dim, dropout)
        ]) for _ in range(depth)])

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        for attn, ff in self.layers:
            x = x + attn(x, valid)
            x = x + ff(x)
        return self.norm(x)


class CSFMEncoder(nn.Module):
    """Core shared CSFM encoder; diagnostic downstream dense heads are excluded."""
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, d))
        self.ts_channel_type_embedding = nn.Parameter(torch.randn(1, cfg.pretrained_channels, d))
        self.resp_channel_type_embedding = nn.Parameter(torch.randn(1, 1, d))
        self.text_type_embedding = nn.Parameter(torch.randn(1, 1, d))
        self.ts_pos_embedding = nn.Parameter(torch.randn(1, cfg.num_patches, d))
        self.text_pos_embedding = nn.Parameter(torch.randn(1, cfg.text_len, d))
        self.ts_to_patch_embedding = nn.Sequential(Patchify(cfg.patch_size),
            nn.LayerNorm(cfg.patch_size), nn.Linear(cfg.patch_size, d), nn.LayerNorm(d))
        self.text_to_embedding = nn.Sequential(nn.Embedding(cfg.vocab_size, d), nn.LayerNorm(d))
        self.dropout = nn.Dropout(cfg.emb_dropout)
        self.transformer = Transformer(d, cfg.depth, cfg.heads, cfg.dim_head,
                                       cfg.mlp_dim, cfg.dropout)

    def channel_embeddings(self) -> Tensor:
        return torch.cat((self.ts_channel_type_embedding, self.resp_channel_type_embedding), dim=1)

    def embed(self, signal: Tensor, channel_ids: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        wave = self.ts_to_patch_embedding(signal)
        wave = wave + self.channel_embeddings()[0, channel_ids, None, :]
        wave = wave + self.ts_pos_embedding[:, None]
        text = self.text_to_embedding(input_ids)
        text = text + self.text_pos_embedding[:, :input_ids.shape[1]] + self.text_type_embedding
        return wave.flatten(1, 2), text

    def export_state_dict(self) -> dict[str, Tensor]:
        result = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                  if k != 'resp_channel_type_embedding'}
        result['ts_channel_type_embedding'] = self.channel_embeddings().detach().cpu().clone()
        return result

    @classmethod
    def from_export(cls, bundle: dict[str, Any]) -> 'CSFMEncoder':
        cfg = MAEConfig(**bundle['config'])
        encoder = cls(cfg)
        state = dict(bundle['encoder'])
        combined = state['ts_channel_type_embedding']
        if combined.shape[1] != cfg.pretrained_channels + 1:
            raise ValueError('Export must include the new Resp channel row')
        state['ts_channel_type_embedding'] = combined[:, :cfg.pretrained_channels]
        state['resp_channel_type_embedding'] = combined[:, cfg.pretrained_channels:]
        encoder.load_state_dict(state, strict=True)
        return encoder

    def forward(self, signal: Tensor, channel_ids: Tensor,
                modality_mask: Tensor | None = None) -> Tensor:
        """Waveform-only CLS embedding with no training-time masking."""
        b, c, _ = signal.shape
        valid = torch.ones((b, c), dtype=torch.bool, device=signal.device) if modality_mask is None else modality_mask.bool()
        safe_ids = channel_ids.masked_fill(~valid, 0)
        safe_signal = torch.where(valid[..., None], signal, torch.zeros_like(signal))
        wave = self.ts_to_patch_embedding(safe_signal)
        wave = wave + self.channel_embeddings()[0, safe_ids, None, :] + self.ts_pos_embedding[:, None]
        tokens = torch.cat((self.cls_token.expand(b, -1, -1), wave.flatten(1, 2)), dim=1)
        keys = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=signal.device),
                          valid[..., None].expand(-1, -1, self.cfg.num_patches).flatten(1)), dim=1)
        return self.transformer(self.dropout(tokens), keys)[:, 0]


def _sample_mask(eligible: Tensor, ratio: float, generator: torch.Generator | None) -> Tensor:
    result = torch.zeros_like(eligible)
    for i in range(eligible.shape[0]):
        ids = eligible[i].nonzero(as_tuple=False).flatten()
        n = min(ids.numel(), max(1, int(ids.numel() * ratio))) if ratio > 0 else 0
        if n:
            random_device = generator.device if generator is not None else ids.device
            order = torch.randperm(ids.numel(), generator=generator, device=random_device).to(ids.device)
            result[i, ids[order[:n]]] = True
    return result


def _pack(tokens: Tensor, visible: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Gather only visible tokens; returned positions restore them in the decoder."""
    b, _, d = tokens.shape
    counts = visible.sum(dim=1)
    width = int(counts.max().item())
    packed = tokens.new_zeros((b, width, d))
    valid = torch.zeros((b, width), dtype=torch.bool, device=tokens.device)
    positions = torch.zeros((b, width), dtype=torch.long, device=tokens.device)
    for i in range(b):
        ids = visible[i].nonzero(as_tuple=False).flatten()
        packed[i, :ids.numel()] = tokens[i, ids]
        valid[i, :ids.numel()] = True
        positions[i, :ids.numel()] = ids
    return packed, valid, positions


class CSFMMAE(nn.Module):
    def __init__(self, config: MAEConfig | None = None):
        super().__init__()
        self.config = config or MAEConfig()
        cfg = self.config
        if cfg.signal_size % cfg.patch_size:
            raise ValueError('signal_size must be divisible by patch_size')
        self.encoder = CSFMEncoder(cfg)
        d = cfg.decoder_dim
        self.decoder_ts_type_embeddings = nn.Parameter(torch.randn(1, cfg.pretrained_channels, d))
        self.resp_decoder_ts_type_embeddings = nn.Parameter(torch.randn(1, 1, d))
        self.decoder_text_type_embeddings = nn.Parameter(torch.randn(1, 1, d))
        self.decoder_ts_pos_embedding = nn.Parameter(torch.randn(1, cfg.num_patches, d))
        self.decoder_text_pos_embedding = nn.Parameter(torch.randn(1, cfg.text_len, d))
        self.mask_token_ts = nn.Parameter(torch.randn(d))
        self.mask_token_text = nn.Parameter(torch.randn(d))
        # Preserve the original MAE checkpoint's parameter sharing and alias names.
        self.patch_to_emb = nn.Sequential(*list(self.encoder.ts_to_patch_embedding.children())[1:])
        self.text_to_emb = self.encoder.text_to_embedding
        self.enc_to_dec = nn.Linear(cfg.dim, d)
        self.decoder = Transformer(d, cfg.decoder_depth, cfg.decoder_heads,
                                  cfg.decoder_dim_head, cfg.decoder_mlp_dim, cfg.dropout)
        self.to_seg = nn.ModuleList([nn.Linear(d, cfg.patch_size)
                                    for _ in range(cfg.pretrained_channels + 1)])
        self.decoder_text_output = nn.Linear(d, cfg.vocab_size)

    def new_parameter_names(self) -> set[str]:
        return {'encoder.resp_channel_type_embedding', 'resp_decoder_ts_type_embeddings',
                f'to_seg.{self.config.pretrained_channels}.weight',
                f'to_seg.{self.config.pretrained_channels}.bias'}

    def load_pretrained(self, path: str | Path) -> dict[str, Any]:
        """Load every required MAE tensor strictly; explicitly account for unused heads."""
        raw = torch.load(path, map_location='cpu', weights_only=True)
        source = raw.get('state_dict', raw)
        target = self.state_dict()
        new = self.new_parameter_names()
        excluded_prefixes = ('encoder.mlp_head.', 'encoder.dense_scratch.',
                             'encoder.act_', 'encoder.dense_act_postprocess.', 'encoder.dense_head.')
        missing = sorted(set(target) - set(source) - new)
        unexpected = sorted(k for k in source if k not in target and not k.startswith(excluded_prefixes))
        mismatched = sorted(k for k in target if k in source and target[k].shape != source[k].shape)
        if missing or unexpected or mismatched:
            raise RuntimeError(f'Checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={mismatched}')
        for source_alias, core in [('patch_to_emb.', 'encoder.ts_to_patch_embedding.'),
                                   ('text_to_emb.', 'encoder.text_to_embedding.')]:
            for name in source:
                if name.startswith(source_alias):
                    suffix = name[len(source_alias):]
                    if source_alias == 'patch_to_emb.':
                        index, rest = suffix.split('.', 1)
                        suffix = f'{int(index)+1}.{rest}'
                    other = core + suffix
                    if other not in source or not torch.equal(source[name], source[other]):
                        raise RuntimeError(f'Checkpoint embedding aliases disagree: {name}, {other}')
        initialized = sorted(new)
        target.update({k: source[k] for k in target if k in source})
        # Match the scale of existing channel rows without claiming Resp is a known modality.
        for name, oldname in [('encoder.resp_channel_type_embedding', 'encoder.ts_channel_type_embedding'),
                              ('resp_decoder_ts_type_embeddings', 'decoder_ts_type_embeddings')]:
            target[name].normal_(mean=float(source[oldname].mean()), std=float(source[oldname].std()))
        self.load_state_dict(target, strict=True)
        return {'checkpoint': str(Path(path).resolve()),
                'loaded_keys': sorted(k for k in target if k in source),
                'initialized_keys': initialized,
                'excluded_unused_downstream_head_keys': sorted(k for k in source if k not in target),
                'shared_patch_embedding': True, 'shared_text_embedding': True,
                'forward_provenance': 'Reconstructed MAE forward from paper plus public encoder and full checkpoint',
                'signal_loss_reduction': 'mean squared L2 over masked valid scalar samples',
                'text_loss_reduction': 'mean CE over masked content tokens'}

    def forward(self, signal: Tensor, channel_ids: Tensor, input_ids: Tensor,
                attention_mask: Tensor, modality_mask: Tensor | None = None,
                special_tokens_mask: Tensor | None = None,
                generator: torch.Generator | None = None,
                waveform_mask: Tensor | None = None, text_mask: Tensor | None = None,
                return_predictions: bool = False) -> dict[str, Tensor]:
        cfg = self.config
        b, c, length = signal.shape
        if length != cfg.signal_size or channel_ids.shape != (b, c):
            raise ValueError('Invalid signal size or channel_ids shape')
        if input_ids.shape != attention_mask.shape or input_ids.shape[0] != b or input_ids.shape[1] > cfg.text_len:
            raise ValueError('Invalid text shape')
        p, t = cfg.num_patches, input_ids.shape[1]
        modalities = torch.ones((b, c), dtype=torch.bool, device=signal.device) if modality_mask is None else modality_mask.bool()
        if modalities.shape != (b, c):
            raise ValueError('Invalid modality mask shape')
        if ((channel_ids[modalities] < 0) | (channel_ids[modalities] > cfg.pretrained_channels)).any():
            raise ValueError('Valid channels must use real ECG/PPG/Resp IDs')
        ids = channel_ids.masked_fill(~modalities, 0)
        safe_signal = torch.where(modalities[..., None], signal, torch.zeros_like(signal))
        text_valid = attention_mask.bool()
        safe_text = input_ids.masked_fill(~text_valid, 0)
        wave_valid = modalities[..., None].expand(-1, -1, p).flatten(1)
        content = text_valid.clone()
        if special_tokens_mask is not None:
            if special_tokens_mask.shape != input_ids.shape:
                raise ValueError('Invalid special_tokens_mask shape')
            content &= ~special_tokens_mask.bool()
        for special_id in cfg.special_token_ids:
            content &= safe_text != special_id
        wmask = _sample_mask(wave_valid, cfg.waveform_mask_ratio, generator) if waveform_mask is None else waveform_mask.reshape(b, c*p).bool()
        tmask = _sample_mask(content, cfg.text_mask_ratio, generator) if text_mask is None else text_mask.bool()
        if (wmask & ~wave_valid).any() or (tmask & ~content).any():
            raise ValueError('Mask targets must be originally valid wave or nonspecial text tokens')
        wave, text = self.encoder.embed(safe_signal, ids, safe_text)
        embedded = torch.cat((wave, text), dim=1)
        visible = torch.cat((wave_valid & ~wmask, text_valid & ~tmask), dim=1)
        packed, packed_valid, positions = _pack(embedded, visible)
        cls = self.encoder.cls_token.expand(b, -1, -1)
        encoded_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=signal.device), packed_valid), dim=1)
        encoded = self.encoder.transformer(self.encoder.dropout(torch.cat((cls, packed), dim=1)), encoded_valid)
        latent = self.enc_to_dec(encoded)
        restored = torch.cat((self.mask_token_ts.expand(b, c*p, -1),
                              self.mask_token_text.expand(b, t, -1)), dim=1).clone()
        for i in range(b):
            n = int(packed_valid[i].sum().item())
            restored[i, positions[i, :n]] = latent[i, 1:1+n].to(restored.dtype)
        channel_emb = torch.cat((self.decoder_ts_type_embeddings, self.resp_decoder_ts_type_embeddings), dim=1)
        wave_pos = channel_emb[0, ids, None, :] + self.decoder_ts_pos_embedding[:, None]
        text_pos = self.decoder_text_type_embeddings + self.decoder_text_pos_embedding[:, :t]
        restored = restored + torch.cat((wave_pos.expand(-1, -1, p, -1).flatten(1, 2), text_pos.expand(b, -1, -1)), dim=1)
        decoder_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=signal.device), wave_valid, text_valid), dim=1)
        decoded = self.decoder(torch.cat((latent[:, :1], restored), dim=1), decoder_valid)[:, 1:]
        wave_features = decoded[:, :c*p].reshape(b, c, p, cfg.decoder_dim)
        prediction = signal.new_zeros((b, c, p, cfg.patch_size))
        for channel in range(cfg.pretrained_channels + 1):
            chosen = modalities & (ids == channel)
            if chosen.any():
                prediction[chosen] = self.to_seg[channel](wave_features[chosen]).to(prediction.dtype)
        target = safe_signal.reshape(b, c, p, cfg.patch_size)
        error = (prediction.float() - target.float()).square()
        mask3d = wmask.reshape(b, c, p)
        masked_error = error[mask3d]
        zero = decoded.float().sum() * 0
        loss_wave = masked_error.mean() if masked_error.numel() else zero
        text_logits = self.decoder_text_output(decoded[:, c*p:][tmask]).float()
        loss_text = F.cross_entropy(text_logits, safe_text[tmask]) if tmask.any() else zero
        result = {'loss': loss_wave + cfg.text_loss_weight * loss_text,
                  'loss_wave': loss_wave, 'loss_text': loss_text,
                  'wave_count': mask3d.sum() * cfg.patch_size,
                  'text_count': tmask.sum(),
                  'text_correct': (text_logits.argmax(-1) == safe_text[tmask]).sum() if tmask.any() else tmask.sum()}
        for name, eligible in [('ecg', ids < 12), ('ppg', ids == 12), ('resp', ids == 13)]:
            selected = mask3d & eligible[..., None] & modalities[..., None]
            values = error[selected]
            result[f'loss_{name}'] = values.mean() if values.numel() else zero
            result[f'{name}_count'] = selected.sum() * cfg.patch_size
        if return_predictions:
            result.update(wave_prediction=prediction, text_logits=text_logits,
                          waveform_mask=wmask, text_mask=tmask,
                          encoded=encoded, packed_valid=packed_valid, positions=positions)
        return result

    def parameter_groups(self, lr_existing: float = 1e-5, lr_new: float = 1e-4,
                         weight_decay: float = 0.05) -> list[dict[str, Any]]:
        groups: dict[tuple[bool, bool], dict[str, Any]] = {}
        new = self.new_parameter_names()
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            decay = parameter.ndim >= 2 and not any(term in name for term in
                ('pos_embedding', 'type_embedding', 'cls_token', 'mask_token'))
            key = (name in new, decay)
            if key not in groups:
                groups[key] = {'params': [], 'lr': lr_new if key[0] else lr_existing,
                               'weight_decay': weight_decay if decay else 0.0,
                               'group_name': ('new_resp' if key[0] else 'pretrained') + ('_decay' if decay else '_no_decay')}
            groups[key]['params'].append(parameter)
        return list(groups.values())

    def export_encoder(self) -> dict[str, Any]:
        return {'format_version': 1, 'config': asdict(self.config),
                'encoder': self.encoder.export_state_dict(),
                'channel_ids': {'ecg': list(range(12)), 'ppg': 12, 'resp': 13},
                'note': 'Core waveform encoder only; no MAE decoder or downstream prediction head.'}
