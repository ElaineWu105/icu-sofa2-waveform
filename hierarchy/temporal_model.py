"""Temporal local/dynamic SSL and the final six-organ SOFA delta head.

No AF head, AF labels, or legacy AF model is imported. Frozen independent
input features are the teachers. A 256-dimensional temporal transformer emits
contextualized 768d tokens *before* SSL projections and global pooling, so its
portable 768d readout is actually trained by the self-supervised objectives.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ORGANS = ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal")
METHODS = ("hierarchical", "baseline", "mean", "wave_only")
ENCODER_FORMAT = "sofa-multiscale-temporal-encoder-v1"


@dataclass(frozen=True)
class TemporalConfig:
    sequence_length: int = 10
    input_dim: int = 768
    model_dim: int = 256
    output_dim: int = 768
    depth: int = 2
    heads: int = 8
    projection_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.1
    high_affinity_fraction: float = 0.5

    def __post_init__(self):
        if min(self.sequence_length, self.input_dim, self.model_dim, self.output_dim,
               self.depth, self.heads, self.projection_dim) < 1:
            raise ValueError("Temporal dimensions must be positive")
        if self.model_dim % self.heads:
            raise ValueError("Temporal hidden dimension must divide into attention heads")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not 0 < self.high_affinity_fraction <= 1:
            raise ValueError("High-affinity fraction must be in (0,1]")


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    safe = values.masked_fill(~mask[..., None], 0)
    return safe.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)


def rank_local_loss(student: Tensor, teacher: Tensor, temperature: float = .1) -> Tensor:
    """The established strict-rank local objective, including teacher ties.

    This is the same sorting/prefix-logsumexp formula as the previous extension:
    teacher similarities are negative L2, strictly lower similarities enter the
    denominator, diagonal/self and equal-similarity peers are excluded, and the
    final reduction is sum/K^2. It imports no previous encoder or task weights.
    """
    if student.ndim != 2 or teacher.ndim != 2 or len(student) != len(teacher):
        raise ValueError("Expected student[K,D1] and teacher[K,D2]")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive")
    count = len(student)
    if count < 2:
        return student.sum() * 0
    logits = -torch.cdist(student.float(), student.float(), p=2) / temperature
    with torch.no_grad():
        similarities = -torch.cdist(teacher.detach().float(), teacher.detach().float(), p=2)
        similarities.fill_diagonal_(float("inf"))
        order = torch.argsort(similarities, dim=1, stable=True)[:, :count - 1]
        ordered = similarities.gather(1, order)
        positions = torch.arange(count - 1, device=student.device).expand(count, -1)
        starts = torch.cat((torch.ones((count, 1), device=student.device, dtype=torch.bool),
                            ordered[:, 1:] > ordered[:, :-1]), dim=1)
        group_start = torch.where(starts, positions, 0).cummax(dim=1).values
    ordered_logits = logits.gather(1, order)
    prefix = torch.logcumsumexp(ordered_logits, dim=1)
    lower = prefix.gather(1, (group_start - 1).clamp_min(0))
    lower = lower.masked_fill(group_start == 0, float("-inf"))
    return (torch.logaddexp(ordered_logits, lower) - ordered_logits).sum() / (count * count)


def dynamics_loss(projected: Tensor, global_feature: Tensor, teacher: Tensor,
                  mask: Tensor, fraction: float = .5) -> Tensor:
    """Within each nonempty example, align selected locals to stopgrad(global).

    High-affinity ranking considers only valid independent teacher vectors.
    Each example contributes one mean, so examples with more valid slots do not
    automatically receive a larger consistency weight. This is consistency, not
    a supervised prediction of physiological derivatives or future states.
    """
    if projected.shape[:2] != mask.shape or teacher.shape[:2] != mask.shape:
        raise ValueError("Dynamics inputs and mask disagree")
    if not 0 < fraction <= 1:
        raise ValueError("Invalid high-affinity fraction")
    losses = []
    for index in range(len(projected)):
        valid = mask[index].nonzero(as_tuple=False).flatten()
        if not len(valid):
            continue
        with torch.no_grad():
            independent = teacher[index].index_select(0, valid).detach().float()
            affinities = -torch.cdist(independent, independent, p=2)
            scores = affinities.sum(dim=-1) / max(len(valid) - 1, 1)
            number = max(1, math.ceil(len(valid) * fraction))
            chosen = valid.index_select(0, torch.argsort(scores, descending=True, stable=True)[:number])
        local = projected[index].index_select(0, chosen).float()
        target = global_feature[index].detach().float()[None].expand_as(local)
        losses.append((1 - F.cosine_similarity(local, target, dim=-1)).mean())
    return torch.stack(losses).mean() if losses else projected.sum() * 0


class TemporalEncoder(nn.Module):
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.register_buffer("input_mean", torch.zeros(config.input_dim))
        self.register_buffer("input_std", torch.ones(config.input_dim))
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.positions = nn.Parameter(torch.empty(1, config.sequence_length, config.model_dim))
        nn.init.normal_(self.positions, std=.02)
        layer = nn.TransformerEncoderLayer(config.model_dim, config.heads, 4 * config.model_dim,
                                            config.dropout, activation="gelu", batch_first=True,
                                            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.depth, enable_nested_tensor=False)
        # Project EACH contextual local before both SSL losses and pooling. A
        # global-only projection used exclusively as a detached target would
        # otherwise never learn in the middle stage.
        self.output_projection = nn.Sequential(nn.LayerNorm(config.model_dim),
                                               nn.Linear(config.model_dim, config.output_dim),
                                               nn.LayerNorm(config.output_dim))

    def set_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.input_mean.dtype, device=self.input_mean.device)
        std = torch.as_tensor(std, dtype=self.input_std.dtype, device=self.input_std.device)
        if mean.shape != self.input_mean.shape or std.shape != self.input_std.shape:
            raise ValueError("Input normalization shape differs")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid input normalization")
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def normalize(self, features: Tensor, mask: Tensor) -> Tensor:
        c = self.config
        if features.ndim != 3 or features.shape[1:] != (c.sequence_length, c.input_dim):
            raise ValueError("Temporal input shape differs from configured physical slots")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Expected a boolean physical-slot mask")
        safe = features.detach().masked_fill(~mask[..., None], 0)
        return ((safe - self.input_mean) / self.input_std).masked_fill(~mask[..., None], 0)

    def encode_normalized(self, teacher: Tensor, mask: Tensor) -> dict[str, Tensor]:
        tokens = self.input_projection(teacher) + self.positions
        active = mask.any(dim=1)
        contextual = torch.zeros_like(tokens)
        if active.any():
            indices = active.nonzero(as_tuple=False).flatten()
            selected_mask = mask.index_select(0, indices)
            encoded = self.transformer(tokens.index_select(0, indices), src_key_padding_mask=~selected_mask)
            encoded = encoded.masked_fill(~selected_mask[..., None], 0)
            contextual = contextual.index_copy(0, indices, encoded)
        local = self.output_projection(contextual).masked_fill(~mask[..., None], 0)
        return {"local": local, "embedding": masked_mean(local, mask),
                "teacher": teacher.detach(), "mask": mask}

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        return self.encode_normalized(self.normalize(features, mask), mask)["embedding"]


class TemporalSSL(nn.Module):
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.encoder = TemporalEncoder(config)
        self.local_projector = nn.Sequential(nn.Linear(config.output_dim, config.projection_dim),
                                             nn.GELU(), nn.Linear(config.projection_dim, config.projection_dim))
        self.dynamic_projector = nn.Sequential(nn.Linear(config.output_dim, config.projection_dim),
                                               nn.GELU(), nn.Linear(config.projection_dim, config.output_dim))

    def forward(self, features, mask):
        return self.encoder.encode_normalized(self.encoder.normalize(features, mask), mask)

    def losses(self, outputs):
        mask = outputs["mask"]
        local = self.local_projector(outputs["local"])
        rank = rank_local_loss(local[mask], outputs["teacher"][mask], self.config.temperature)
        dynamic = dynamics_loss(self.dynamic_projector(outputs["local"]), outputs["embedding"],
                                outputs["teacher"], mask, self.config.high_affinity_fraction)
        return {"local": rank, "dynamic": dynamic}


class DeltaPredictor(nn.Module):
    def __init__(self, config: TemporalConfig, method="hierarchical"):
        super().__init__()
        if method not in METHODS:
            raise ValueError("Unknown downstream control")
        self.config, self.method = config, method
        if method in ("hierarchical", "wave_only"):
            self.temporal = TemporalSSL(config)
        else:
            self.register_buffer("input_mean", torch.zeros(config.input_dim))
            self.register_buffer("input_std", torch.ones(config.input_dim))
        self.context = nn.Sequential(nn.Linear(13, 128), nn.GELU(), nn.LayerNorm(128))
        self.head = nn.Sequential(nn.Linear(config.output_dim + 128, 256), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(256, len(ORGANS)))

    def set_normalization(self, mean, std):
        if hasattr(self, "temporal"):
            self.temporal.encoder.set_normalization(mean, std)
        else:
            self.input_mean.copy_(torch.as_tensor(mean, dtype=self.input_mean.dtype))
            self.input_std.copy_(torch.as_tensor(std, dtype=self.input_std.dtype))

    def forward(self, features, mask, baseline, baseline_mask, delta_hours):
        if baseline.shape != (len(features), 6) or baseline_mask.shape != baseline.shape:
            raise ValueError("Expected baseline [B,6] and its mask")
        if baseline_mask.dtype != torch.bool:
            raise ValueError("Baseline mask must be boolean")
        if hasattr(self, "temporal"):
            result = self.temporal(features, mask)
            embedding = result["embedding"]
        elif self.method == "mean":
            safe = features.detach().masked_fill(~mask[..., None], 0)
            normalized = ((safe - self.input_mean) / self.input_std).masked_fill(~mask[..., None], 0)
            embedding = masked_mean(normalized, mask)
            result = {"embedding": embedding}
        else:
            embedding = torch.zeros((len(features), self.config.output_dim), device=features.device, dtype=features.dtype)
            result = {"embedding": embedding}
        if self.method == "wave_only":
            score, known = torch.zeros_like(baseline), torch.zeros_like(baseline_mask)
        else:
            score, known = baseline.masked_fill(~baseline_mask, 0) / 4, baseline_mask
        context = self.context(torch.cat((score, known.to(score.dtype),
                                          delta_hours.reshape(-1, 1).to(score.dtype) / 24), dim=-1))
        result["delta"] = self.head(torch.cat((embedding, context), dim=-1))
        return result

    def ssl_losses(self, outputs):
        if hasattr(self, "temporal"):
            return self.temporal.losses(outputs)
        zero = outputs["delta"].sum() * 0
        return {"local": zero, "dynamic": zero}


def masked_macro_huber(prediction, target, mask, organ_denominators=None, sample_multiplier=1.):
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("Expected matching predictions, targets and masks")
    safe = torch.where(mask, target, prediction.detach())
    pointwise = F.smooth_l1_loss(prediction, safe, beta=1, reduction="none").masked_fill(~mask, 0)
    denominator = mask.sum(0).to(prediction.dtype) if organ_denominators is None else organ_denominators.to(prediction)
    valid = denominator > 0
    if not valid.any():
        return prediction.sum() * 0
    return (pointwise.sum(0)[valid] / denominator[valid]).mean() * sample_multiplier


def export_encoder(model, stage, normalization_info, source, model_checkpoint_sha256=None):
    encoder = model.encoder if isinstance(model, TemporalSSL) else model.temporal.encoder
    return {"format": ENCODER_FORMAT, "stage": stage,
            "objective": "local_dynamic_only" if stage == "middle" else "local_dynamic_sofa_delta",
            "no_af_task": True, "model_config": asdict(encoder.config),
            "encoder_state": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
            "normalization": normalization_info, "source": source,
            "model_checkpoint_sha256": model_checkpoint_sha256}


def load_encoder(path: str | Path, device="cpu") -> TemporalEncoder:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format") != ENCODER_FORMAT or bundle.get("no_af_task") is not True:
        raise ValueError("Expected a new multiscale temporal SSL encoder")
    if bundle.get("objective") not in ("local_dynamic_only", "local_dynamic_sofa_delta"):
        raise ValueError("Unexpected temporal training objective")
    encoder = TemporalEncoder(TemporalConfig(**bundle["model_config"]))
    encoder.load_state_dict(bundle["encoder_state"], strict=True)
    encoder.requires_grad_(False)
    return encoder.to(device).eval()
