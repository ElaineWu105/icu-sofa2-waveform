"""Temporal levels 2 and 3, and the masked-segment decoder shared with level 1.

A recording is a sequence of segments, and each segment keeps its inner tokens: an hour is
20 clips of 18 window features, a day is 24 hours of 20 clip features. Token t of segment s
gets a within-segment position plus a global segment position, as in Eq. 1 of Tang et al.
2026. The segment feature is the mean of its present output tokens, and the recording
feature is the mean of present segment features.

Level 2 trains with masked segment prediction (MAE: 75% of present segments removed, a light
decoder, MSE on masked segments only) plus the local and dynamic losses. Level 3 uses the same
encoder without masking, under the SOFA-2 loss. Frozen lower-level features are the teachers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

MASK_RATIO = .75


@dataclass(frozen=True)
class TemporalConfig:
    segments: int = 20
    tokens_per_segment: int = 18
    input_dim: int = 768
    model_dim: int = 256
    output_dim: int = 768
    depth: int = 2
    heads: int = 8
    projection_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.1
    high_affinity_fraction: float = 0.5
    mask_ratio: float = MASK_RATIO
    decoder_width: int = 256
    decoder_depth: int = 2

    def __post_init__(self):
        if min(self.segments, self.tokens_per_segment, self.input_dim, self.model_dim, self.output_dim,
               self.depth, self.heads, self.projection_dim, self.decoder_width, self.decoder_depth) < 1:
            raise ValueError("Temporal dimensions must be positive")
        if self.model_dim % self.heads or self.decoder_width % self.heads:
            raise ValueError("Hidden dimensions must divide into attention heads")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be positive")
        if not 0 < self.high_affinity_fraction <= 1:
            raise ValueError("High-affinity fraction must be in (0,1]")
        if not 0 <= self.mask_ratio < 1:
            raise ValueError("Mask ratio must be in [0,1)")

    def to_dict(self):
        return asdict(self)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    safe = values.masked_fill(~mask[..., None], 0)
    return safe.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)


def visible_count(present: int | Tensor, ratio: float = MASK_RATIO):
    """max(1, round(present x (1 - ratio))): 4 of 18 and 5 of 20 at 75%."""
    if isinstance(present, Tensor):
        return torch.round(present.float() * (1 - ratio)).long().clamp_min(1).minimum(present.long().clamp_min(1))
    return max(1, round(present * (1 - ratio)))


def draw_visible(present: Tensor, ratio: float, generator: torch.Generator | None = None):
    """Random visible segments among present ones, independently per row.

    Returns index[B,V] into the segments, sorted, and valid[B,V] for rows with fewer than V
    visible segments. Rows without any present segment get one invalid entry.
    """
    scores = torch.rand(present.shape, generator=generator).to(present.device)
    scores = scores.masked_fill(~present, 2.)
    order = scores.argsort(dim=1)
    counts = torch.where(present.any(1), visible_count(present.sum(1), ratio), torch.zeros_like(present[:, 0], dtype=torch.long))
    width = max(1, int(counts.max()))
    index = order[:, :width]
    valid = torch.arange(width, device=present.device)[None] < counts[:, None]
    # Sort the valid entries by time; invalid entries stay last, so valid keeps its order.
    key = torch.where(valid, index, index + present.shape[1])
    return index.gather(1, key.argsort(dim=1)), valid


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


def high_affinity(teacher: Tensor, mask: Tensor, fraction: float) -> list[Tensor]:
    """Per row, the valid segments with the highest mean teacher affinity (negative L2)."""
    chosen = []
    with torch.no_grad():
        for index in range(len(teacher)):
            valid = mask[index].nonzero(as_tuple=False).flatten()
            if not len(valid):
                chosen.append(valid)
                continue
            independent = teacher[index].index_select(0, valid).detach().float()
            scores = (-torch.cdist(independent, independent, p=2)).sum(dim=-1) / max(len(valid) - 1, 1)
            number = max(1, math.ceil(len(valid) * fraction))
            chosen.append(valid.index_select(0, torch.argsort(scores, descending=True, stable=True)[:number]))
    return chosen


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
    for index, chosen in enumerate(high_affinity(teacher, mask, fraction)):
        if not len(chosen):
            continue
        local = projected[index].index_select(0, chosen).float()
        target = global_feature[index].detach().float()[None].expand_as(local)
        losses.append((1 - F.cosine_similarity(local, target, dim=-1)).mean())
    return torch.stack(losses).mean() if losses else projected.sum() * 0


def unprojected_dynamics(local: Tensor, global_feature: Tensor, teacher: Tensor, mask: Tensor, fraction: float) -> Tensor:
    """L_dyn without its projector: a check that the projector has not solved the term alone."""
    with torch.no_grad():
        return dynamics_loss(local, global_feature, teacher, mask, fraction)


def centered(target: Tensor, present: Tensor) -> Tensor:
    """Each segment's teacher minus the mean teacher of the present segments of its recording.

    Adjacent segments are nearly alike, so an uncentered target is met by the mean of the visible
    segments. Centering removes that shortcut, as MAE's per-patch target normalization does for pixels.
    """
    return target.float() - masked_mean(target.float(), present)[:, None]


class MaskedDecoder(nn.Module):
    """MAE decoder: visible segment features plus one learned mask token per masked segment, each
    with a learned decoder position; predicts every segment's centered, standardized teacher feature.

    Used for training only and dropped afterwards. The target is the teacher minus the mean teacher of
    its recording's present segments, standardized per dimension with train-patient statistics of
    those centered targets, set once before training.
    """

    def __init__(self, segments: int, input_dim: int = 768, width: int = 256, depth: int = 2, heads: int = 8,
                 target_dim: int = 768, dropout: float = 0.):
        super().__init__()
        self.input = nn.Linear(input_dim, width)
        self.mask_token = nn.Parameter(torch.empty(width))
        self.positions = nn.Parameter(torch.empty(segments, width))
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.positions, std=.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, target_dim)
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("target_std", torch.ones(target_dim))

    def set_target_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.target_mean.dtype, device=self.target_mean.device)
        std = torch.as_tensor(std, dtype=self.target_std.dtype, device=self.target_std.device)
        if mean.shape != self.target_mean.shape or std.shape != self.target_std.shape:
            raise ValueError("Target normalization shape differs")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid target normalization")
        self.target_mean.copy_(mean)
        self.target_std.copy_(std)

    def standardize(self, target: Tensor, present: Tensor) -> Tensor:
        return (centered(target, present) - self.target_mean) / self.target_std

    def forward(self, visible: Tensor, index: Tensor, valid: Tensor, present: Tensor) -> tuple[Tensor, Tensor]:
        """visible[B,V,D] features at segments index[B,V]; present[B,S] marks segments that exist.

        Returns predictions[B,S,target_dim] and shown[B,S], the segments the encoder saw.
        """
        b, segments = present.shape
        projected = self.input(visible) * valid[..., None].to(visible.dtype)
        placed = torch.zeros((b, segments, projected.shape[-1]), dtype=projected.dtype, device=projected.device)
        placed = placed.scatter_add(1, index[..., None].expand(-1, -1, projected.shape[-1]), projected)
        shown = torch.zeros((b, segments), dtype=torch.long, device=present.device).scatter_add(1, index, valid.long()) > 0
        tokens = torch.where(shown[..., None], placed, self.mask_token.to(placed.dtype).expand(b, segments, -1))
        tokens = tokens + self.positions[:segments].to(tokens.dtype)
        hidden = self.blocks(tokens, src_key_padding_mask=~present)
        return self.output(self.norm(hidden)), shown


def masked_mse(prediction: Tensor, target: Tensor, masked: Tensor) -> Tensor:
    """Mean squared error over masked segments only."""
    if not masked.any():
        return prediction.sum() * 0
    return (prediction.float() - target.float()).pow(2).mean(-1)[masked].mean()


def reference_errors(target: Tensor, index: Tensor, valid: Tensor, masked: Tensor) -> dict[str, Tensor]:
    """Errors of two trivial predictors of masked targets, in the decoder's standardized space.

    copy: the nearest visible segment's target (earlier one on ties). mean: the mean of visible targets.
    The masked loss must be clearly below both.
    """
    b, segments, _ = target.shape
    visible_targets = target.gather(1, index[..., None].expand(-1, -1, target.shape[-1]))
    mean = masked_mean(visible_targets, valid)[:, None].expand_as(target)
    slots = torch.arange(segments, device=target.device)
    distance = (slots[None, :, None] - index[:, None, :]).abs().float()
    distance = distance.masked_fill(~valid[:, None, :], float("inf"))
    nearest = index.gather(1, distance.argmin(dim=2))
    copy = target.gather(1, nearest[..., None].expand(-1, -1, target.shape[-1]))
    return {"copy": masked_mse(copy, target, masked), "mean": masked_mse(mean, target, masked)}


class TemporalEncoder(nn.Module):
    """Inner tokens of each segment with within-segment and global positions; trained from scratch."""

    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.register_buffer("input_mean", torch.zeros(config.input_dim))
        self.register_buffer("input_std", torch.ones(config.input_dim))
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.within = nn.Parameter(torch.empty(config.tokens_per_segment, config.model_dim))
        self.global_positions = nn.Parameter(torch.empty(config.segments, config.model_dim))
        nn.init.normal_(self.within, std=.02)
        nn.init.normal_(self.global_positions, std=.02)
        layer = nn.TransformerEncoderLayer(config.model_dim, config.heads, 4 * config.model_dim,
                                            config.dropout, activation="gelu", batch_first=True,
                                            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.depth, enable_nested_tensor=False)
        # Each contextual token is projected before pooling and before both SSL losses.
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

    def encode(self, tokens: Tensor, token_mask: Tensor, segment_index: Tensor, segment_valid: Tensor) -> dict[str, Tensor]:
        """tokens[B,V,T,input_dim] of segments segment_index[B,V]; token_mask[B,V,T].

        Returns segment features local[B,V,output_dim], their mask, and the recording feature.
        """
        c = self.config
        b, count, inner, dim = tokens.shape
        if inner != c.tokens_per_segment or dim != c.input_dim:
            raise ValueError("Temporal input shape differs from the configured segment size")
        if token_mask.shape != tokens.shape[:3] or token_mask.dtype != torch.bool:
            raise ValueError("Expected a boolean token mask")
        present = token_mask & segment_valid[..., None]
        values = ((tokens.detach().float() - self.input_mean) / self.input_std).masked_fill(~present[..., None], 0)
        embedded = self.input_projection(values) + self.within[None, None] + self.global_positions[segment_index][:, :, None]
        embedded, flat = embedded.flatten(1, 2), present.flatten(1)
        active = flat.any(dim=1)
        contextual = torch.zeros_like(embedded)
        if active.any():
            rows = active.nonzero(as_tuple=False).flatten()
            encoded = self.transformer(embedded.index_select(0, rows), src_key_padding_mask=~flat.index_select(0, rows))
            contextual = contextual.index_copy(0, rows, encoded.masked_fill(~flat.index_select(0, rows)[..., None], 0))
        output = self.output_projection(contextual).masked_fill(~flat[..., None], 0).reshape(b, count, inner, -1)
        segment_mask = present.any(dim=2)
        local = (output.sum(dim=2) / present.sum(dim=2, keepdim=True).clamp_min(1).to(output.dtype)).masked_fill(~segment_mask[..., None], 0)
        return {"local": local, "mask": segment_mask, "embedding": masked_mean(local, segment_mask)}

    def forward(self, tokens: Tensor, token_mask: Tensor) -> dict[str, Tensor]:
        """Every segment, no masking: tokens[B,S,T,input_dim]."""
        b, segments = tokens.shape[:2]
        index = torch.arange(segments, device=tokens.device).expand(b, -1)
        return self.encode(tokens, token_mask, index, token_mask.any(dim=2))


class TemporalSSL(nn.Module):
    def __init__(self, config: TemporalConfig, decoder: bool = True):
        super().__init__()
        self.config = config
        self.encoder = TemporalEncoder(config)
        self.local_projector = nn.Sequential(nn.Linear(config.output_dim, config.projection_dim),
                                             nn.GELU(), nn.Linear(config.projection_dim, config.projection_dim))
        self.dynamic_projector = nn.Sequential(nn.Linear(config.output_dim, config.projection_dim),
                                               nn.GELU(), nn.Linear(config.projection_dim, config.output_dim))
        self.decoder = MaskedDecoder(config.segments, config.output_dim, config.decoder_width, config.decoder_depth,
                                     config.heads, config.input_dim) if decoder else None

    def forward(self, tokens, token_mask):
        return self.encoder(tokens, token_mask)

    def ssl_losses(self, local: Tensor, embedding: Tensor, teacher: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """L_local and L_dyn on present segments; teacher[B,V,D] is the frozen lower level on each segment alone."""
        projected = self.local_projector(local)
        rank = rank_local_loss(projected[mask], teacher[mask], self.config.temperature)
        dynamic = dynamics_loss(self.dynamic_projector(local), embedding, teacher, mask, self.config.high_affinity_fraction)
        return {"local": rank, "dynamic": dynamic}

    def losses(self, outputs: dict[str, Tensor], teacher: Tensor) -> dict[str, Tensor]:
        return self.ssl_losses(outputs["local"], outputs["embedding"], teacher, outputs["mask"])

    def masked_step(self, tokens: Tensor, token_mask: Tensor, teacher: Tensor,
                    generator: torch.Generator | None = None, references: bool = False) -> dict[str, Tensor]:
        """Level 2 training: 75% of present segments removed, MAE loss on them, SSL losses on the rest.

        teacher[B,S,D] is each segment's frozen lower-level feature; it is also the masked target.
        """
        if self.decoder is None:
            raise RuntimeError("This model has no decoder")
        present = token_mask.any(dim=2)
        index, valid = draw_visible(present, self.config.mask_ratio, generator)
        gathered = tokens.gather(1, index[..., None, None].expand(-1, -1, *tokens.shape[2:]))
        gathered_mask = token_mask.gather(1, index[..., None].expand(-1, -1, token_mask.shape[2])) & valid[..., None]
        output = self.encoder.encode(gathered, gathered_mask, index, valid)
        prediction, shown = self.decoder(output["local"], index, output["mask"], present)
        target = self.decoder.standardize(teacher, present)
        masked = present & ~shown
        visible_teacher = teacher.gather(1, index[..., None].expand(-1, -1, teacher.shape[-1]))
        ssl = self.ssl_losses(output["local"], output["embedding"], visible_teacher, output["mask"])
        result = {"masked": masked_mse(prediction, target, masked), **ssl}
        result["loss"] = result["masked"] + result["local"] + result["dynamic"]
        if references:
            result.update(reference_errors(target, index, output["mask"], masked))
            result["dynamic_unprojected"] = unprojected_dynamics(output["local"], output["embedding"], visible_teacher,
                                                                 output["mask"], self.config.high_affinity_fraction)
        return result
