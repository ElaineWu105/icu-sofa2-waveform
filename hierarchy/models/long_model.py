"""Frozen-CSFM 180 s extension following arXiv:2605.16975, Eqs. 1--4.

The paper specifies the extension and objectives, but not the full numerical
configuration or official implementation. Defaults below are OUR implementation
choices, not reported paper hyperparameters. In particular: deep prompts replace
the previous layer's prompts; the local/global readout is mean waveform tokens;
both teacher and local-student similarities use negative Euclidean distance;
high-affinity windows have the largest mean off-diagonal teacher affinity.

Equation 3 explicitly supervises the high-affinity group only. The surrounding
paper text also discusses both groups without giving a complete second-group
objective; we implement the displayed equation rather than inventing that loss.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


_SOURCE = Path(__file__).resolve().parents[2] / "finetune/csfm_mae/model.py"
_MODULE_NAME = "_csfm_3min_original_model"
if _MODULE_NAME not in sys.modules:
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load existing CSFM implementation: {_SOURCE}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = _MODULE
    _SPEC.loader.exec_module(_MODULE)
_MODULE = sys.modules[_MODULE_NAME]
CSFMEncoder = _MODULE.CSFMEncoder
MAEConfig = _MODULE.MAEConfig


@dataclass(frozen=True)
class LongConfig:
    """Numerical defaults are engineering choices, not paper-reported values."""
    max_windows: int = 18
    prompt_tokens: int = 8
    prompt_std: float = 0.02
    projection_dim: int = 256
    temperature: float = 0.1
    lambda_local: float = 0.1
    lambda_dyn: float = 0.1
    high_affinity_fraction: float = 0.5
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        for name in ("max_windows", "prompt_tokens", "projection_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        for name in ("prompt_std", "temperature", "lambda_local", "lambda_dyn",
                     "high_affinity_fraction"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.max_windows < 1 or self.prompt_tokens < 1 or self.projection_dim < 1:
            raise ValueError("Window, prompt and projection dimensions must be positive")
        if self.prompt_std <= 0 or self.temperature <= 0:
            raise ValueError("Prompt std and contrastive temperature must be positive")
        if not 0 < self.high_affinity_fraction <= 1:
            raise ValueError("high_affinity_fraction must be in (0, 1]")
        if self.lambda_local < 0 or self.lambda_dyn < 0:
            raise ValueError("Auxiliary loss weights must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LongConfig":
        return cls(**values)


def load_e17(path: str | Path, device: str | torch.device = "cpu") -> CSFMEncoder:
    """Load the existing strict 14-channel export without changing any weight."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or "encoder" not in bundle or "config" not in bundle:
        raise ValueError("Expected an exported E17 bundle with encoder and config")
    model = CSFMEncoder.from_export(bundle)
    return model.to(device).eval()


def pairwise_similarity(features: Tensor) -> Tensor:
    """Negative L2, one example permitted by paper Sec. 3.2; computed in FP32."""
    values = features.float()
    return -torch.cdist(values, values, p=2)


def rank_local_loss(student: Tensor, teacher: Tensor, temperature: float = 0.1) -> Tensor:
    """Eq. 2 with strict teacher ranking, ties excluded from each other's Omega.

    For each anchor i, apply the paper's 1/K normalization over j != i; then
    average anchors. Hence the final denominator is K*K (the printed equation
    omits an outer anchor average). This differs from averaging K*(K-1) pairs.
    Sorting + prefix log-sums uses O(K^2) storage and O(K^2 log K) work, avoiding
    a K^3 anchor/positive/negative tensor. Teacher similarities carry no gradient.
    """
    if student.ndim != 2 or teacher.ndim != 2 or student.shape[0] != teacher.shape[0]:
        raise ValueError("Expected student[K,D1] and teacher[K,D2]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    k = student.shape[0]
    if k < 2:
        return student.sum() * 0.0
    student_sim = pairwise_similarity(student) / temperature
    with torch.no_grad():
        teacher_sim = pairwise_similarity(teacher.detach())
        teacher_sim.fill_diagonal_(float("inf"))
        # The +inf diagonal is sorted last, so self never enters Omega.
        order = torch.argsort(teacher_sim, dim=1, stable=True)[:, : k - 1]
        sorted_teacher = teacher_sim.gather(1, order)
        positions = torch.arange(k - 1, device=student.device).expand(k, -1)
        first = torch.ones((k, 1), dtype=torch.bool, device=student.device)
        starts = torch.cat((first, sorted_teacher[:, 1:] > sorted_teacher[:, :-1]), dim=1)
        group_start = torch.where(starts, positions, 0).cummax(dim=1).values
    sorted_logits = student_sim.gather(1, order)
    prefix = torch.logcumsumexp(sorted_logits, dim=1)
    lower = prefix.gather(1, (group_start - 1).clamp_min(0))
    lower = lower.masked_fill(group_start == 0, float("-inf"))
    terms = torch.logaddexp(sorted_logits, lower) - sorted_logits
    return terms.sum() / (k * k)


def select_high_affinity(teacher: Tensor, fraction: float) -> Tensor:
    """Select teacher-consistent windows independently inside each recording.

    The fraction and mean-affinity ranking are explicit implementation choices;
    the paper does not specify these numerical/selection details. Ties resolve
    in temporal order through a stable sort.
    """
    if teacher.ndim != 3 or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Expected teacher[B,M,D] and fraction in (0, 1]")
    with torch.no_grad():
        b, windows, _ = teacher.shape
        if windows < 1:
            raise ValueError("At least one window is required")
        affinity = -torch.cdist(teacher.detach().float(), teacher.detach().float(), p=2)
        scores = affinity.sum(dim=-1) / max(windows - 1, 1)
        count = max(1, math.ceil(windows * fraction))
        return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :count]


def dynamics_loss(projected_local: Tensor, global_embedding: Tensor,
                  teacher: Tensor, fraction: float) -> Tensor:
    """Displayed Eq. 3: cosine distance to stopgrad(student global feature)."""
    selected = select_high_affinity(teacher, fraction)
    indices = selected[..., None].expand(-1, -1, projected_local.shape[-1])
    local = projected_local.gather(1, indices).float()
    target = global_embedding.detach().float()[:, None, :].expand_as(local)
    return (1.0 - F.cosine_similarity(local, target, dim=-1)).mean()


class LongCSFM(nn.Module):
    """Joint attention over all waveform patches, with a completely frozen E17.

    Input is [B,M,3,L] with channel order ECG/PPG/Resp. Actual ECG lead IDs are
    preserved (0..11); PPG=12 and Resp=13. No text is provided to this model.
    The same frozen backbone computes independent 10 s teacher features.
    """
    def __init__(self, encoder: CSFMEncoder, config: LongConfig | None = None):
        super().__init__()
        self.config = config or LongConfig()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        cfg, d = self.config, encoder.cfg.dim
        self.global_pos = nn.Parameter(torch.empty(cfg.max_windows, d))
        self.prompts = nn.Parameter(torch.empty(encoder.cfg.depth, cfg.prompt_tokens, d))
        nn.init.normal_(self.global_pos, std=cfg.prompt_std)
        nn.init.normal_(self.prompts, std=cfg.prompt_std)
        hidden = cfg.projection_dim
        self.g_local = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.g_dyn = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.classifier = nn.Linear(d, 1)

    @property
    def backbone(self) -> CSFMEncoder:
        """Alias without registering/duplicating the frozen state dictionary."""
        return self.encoder

    def train(self, mode: bool = True) -> "LongCSFM":
        super().train(mode)
        # Frozen backbone dropout must not inject teacher/student inconsistency.
        self.encoder.eval()
        return self

    def _validate(self, signal: Tensor, channel_ids: Tensor) -> tuple[int, int, int, int]:
        if signal.ndim != 4:
            raise ValueError("Expected signal[B,M,3,signal_size]")
        b, windows, channels, length = signal.shape
        if b < 1 or not 1 <= windows <= self.config.max_windows:
            raise ValueError("Invalid batch size or number of 10 s windows")
        if channels != 3 or length != self.encoder.cfg.signal_size:
            raise ValueError("Expected exactly ECG/PPG/Resp at the encoder signal size")
        if channel_ids.shape != (b, windows, channels) or channel_ids.dtype != torch.long:
            raise ValueError("Expected int64 channel_ids[B,M,3]")
        if ((channel_ids[..., 0] < 0) | (channel_ids[..., 0] > 11)).any():
            raise ValueError("ECG lead IDs must remain in 0..11")
        if (channel_ids[..., 1] != 12).any() or (channel_ids[..., 2] != 13).any():
            raise ValueError("Channel order must be actual ECG lead, PPG=12, Resp=13")
        return b, windows, channels, length

    def _wave_embedding(self, signal: Tensor, channel_ids: Tensor) -> Tensor:
        wave = self.encoder.ts_to_patch_embedding(signal)
        wave = wave + self.encoder.channel_embeddings()[0, channel_ids, None, :]
        return wave + self.encoder.ts_pos_embedding[:, None, None, :, :]

    def forward(self, signal: Tensor, channel_ids: Tensor) -> dict[str, Tensor]:
        b, windows, channels, _ = self._validate(signal, channel_ids)
        wave = self._wave_embedding(signal, channel_ids)
        wave = wave + self.global_pos[None, :windows, None, None, :]
        wave = wave.flatten(1, 3)
        tokens = torch.cat((self.encoder.cls_token.expand(b, -1, -1), wave), dim=1)
        for depth, (attn, ff) in enumerate(self.encoder.transformer.layers):
            prompt = self.prompts[depth][None].expand(b, -1, -1)
            tokens = torch.cat((prompt, tokens), dim=1)

            def block(values: Tensor, attention: nn.Module = attn,
                      feed_forward: nn.Module = ff) -> Tensor:
                values = values + attention(values)
                return values + feed_forward(values)

            if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
            # Replace prompts at each layer; never grow the sequence by depth.
            tokens = tokens[:, self.config.prompt_tokens:]
        tokens = self.encoder.transformer.norm(tokens)
        local = tokens[:, 1:].reshape(b, windows, channels * self.encoder.cfg.num_patches, -1).mean(2)
        embedding = local.mean(dim=1)
        return {"embedding": embedding, "local": local,
                "logits": self.classifier(embedding).squeeze(-1)}

    @torch.no_grad()
    def teacher_features(self, signal: Tensor, channel_ids: Tensor,
                         chunk_size: int = 18) -> Tensor:
        """Independent frozen 10 s features, mean-readout matched to the student."""
        b, windows, channels, length = self._validate(signal, channel_ids)
        if chunk_size < 1:
            raise ValueError("Teacher chunk size must be positive")
        self.encoder.eval()
        flat_signal = signal.reshape(b * windows, channels, length)
        flat_ids = channel_ids.reshape(b * windows, channels)
        features = []
        for start in range(0, b * windows, chunk_size):
            wave = self.encoder.ts_to_patch_embedding(flat_signal[start:start + chunk_size])
            wave = wave + self.encoder.channel_embeddings()[0, flat_ids[start:start + chunk_size], None, :]
            wave = wave + self.encoder.ts_pos_embedding[:, None, :, :]
            n = wave.shape[0]
            tokens = torch.cat((self.encoder.cls_token.expand(n, -1, -1), wave.flatten(1, 2)), dim=1)
            features.append(self.encoder.transformer(tokens)[:, 1:].mean(dim=1))
        return torch.cat(features, dim=0).reshape(b, windows, -1)

    def loss(self, outputs: dict[str, Tensor], teacher: Tensor, label: Tensor) -> dict[str, Tensor]:
        if label.shape != outputs["logits"].shape:
            raise ValueError("Label shape must match the [B] AF logits")
        if teacher.shape != outputs["local"].shape:
            raise ValueError("Teacher shape must match student [B,M,D] local features")
        task = F.binary_cross_entropy_with_logits(outputs["logits"].float(), label.float())
        projected = self.g_local(outputs["local"])
        local = rank_local_loss(projected.flatten(0, 1), teacher.flatten(0, 1), self.config.temperature)
        dyn = dynamics_loss(self.g_dyn(outputs["local"]), outputs["embedding"], teacher,
                            self.config.high_affinity_fraction)
        total = task + self.config.lambda_local * local + self.config.lambda_dyn * dyn
        return {"loss": total, "task": task, "local": local, "dyn": dyn}

    def adapter_state_dict(self) -> dict[str, Tensor]:
        """Small export; reconstruction additionally requires the original E17."""
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()
                if not name.startswith("encoder.")}

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.named_parameters()
                if value.requires_grad}

    def trainable_state(self) -> dict[str, Tensor]:
        return self.trainable_state_dict()

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = {name: value for name, value in self.named_parameters() if value.requires_grad}
        if set(state) != set(expected):
            raise ValueError(f"Trainable keys mismatch: missing={sorted(set(expected) - set(state))}, "
                             f"unexpected={sorted(set(state) - set(expected))}")
        for name, value in state.items():
            if value.shape != expected[name].shape:
                raise ValueError(f"Trainable tensor shape mismatch for {name}: "
                                 f"{tuple(value.shape)} != {tuple(expected[name].shape)}")
        with torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value)

    def load_adapter_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = {name for name in self.state_dict() if not name.startswith("encoder.")}
        if set(state) != expected:
            raise ValueError(f"Adapter keys mismatch: missing={sorted(expected - set(state))}, "
                             f"unexpected={sorted(set(state) - expected)}")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not name.startswith("encoder.") for name in result.missing_keys):
            raise RuntimeError(f"Unexpected adapter load mismatch: {result}")
