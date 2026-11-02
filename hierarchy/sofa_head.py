"""Level 3 head: future SOFA-2 organ scores as ordinal outputs, as a baseline model plus a waveform correction."""
import torch
from torch import nn
import torch.nn.functional as F

THRESHOLDS = 4


class OrdinalHead(nn.Module):
    """P(score > k) for k = 0..3, per organ.

    A baseline-only path always runs. The waveform adds a correction to its logits, so its
    value is the gap to the baseline control. Training the waveform path alone gives wave_only.
    """

    def __init__(self, wave_dim=768, organs=6, hidden=256, dropout=.1, use_wave=True, use_baseline=True):
        super().__init__()
        self.use_wave, self.use_baseline, self.organs = use_wave, use_baseline, organs
        self.baseline = nn.Sequential(nn.Linear(2 * organs, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, organs * THRESHOLDS))
        self.wave = nn.Sequential(nn.Linear(wave_dim + 2 * organs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, organs * THRESHOLDS))
        nn.init.zeros_(self.wave[-1].weight)
        nn.init.zeros_(self.wave[-1].bias)

    def forward(self, embedding, base, base_mask):
        """logits[B,organs,4]. Baseline scores are scaled to 0..1; missing ones are zero with a false mask."""
        score = (base / 4).masked_fill(~base_mask, 0)
        context = torch.cat((score, base_mask.to(score.dtype)), dim=-1)
        if not self.use_baseline:
            context = torch.zeros_like(context)
        logits = self.baseline(context) if self.use_baseline else torch.zeros(len(base), self.organs * THRESHOLDS, device=base.device)
        if self.use_wave:
            logits = logits + self.wave(torch.cat((embedding, context), dim=-1))
        return logits.reshape(len(base), self.organs, THRESHOLDS)


def expected_score(logits):
    """Expected score, the sum of P(score > k). Always within 0..4."""
    return torch.sigmoid(logits).sum(-1)


def ordinal_targets(score):
    """score[...] in 0..4 to cumulative targets[...,4]."""
    return (score[..., None] > torch.arange(THRESHOLDS, device=score.device)).to(torch.float32)


def sofa_loss(logits, future, base, label, weights, total_weight=1.):
    """Organ-balanced ordinal loss, plus Huber on the summed change of organs labeled on each anchor.

    Carried values never enter, because a carried organ has a false label and so does not reach the sum.
    """
    weights = weights / weights.sum().clamp_min(1e-9)
    per = F.binary_cross_entropy_with_logits(logits, ordinal_targets(future), reduction="none").mean(-1)
    organ = (per * label * weights[:, None]).sum(0) / (label * weights[:, None]).sum(0).clamp_min(1e-9)
    organ = organ[label.any(0)].mean() if label.any() else logits.sum() * 0
    change = (expected_score(logits) - base) * label
    true_change = (future - base) * label
    total = (F.smooth_l1_loss(change.sum(1), true_change.sum(1), reduction="none") * label.any(1) * weights).sum()
    return {"organ": organ, "total": total, "task": organ + total_weight * total}
