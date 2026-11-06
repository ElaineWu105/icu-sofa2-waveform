"""Level 3 head: next-day SOFA-2 organ scores by CORN, and one composite output, as a baseline model plus a waveform correction.

CORN (Shi, Cao and Raschka 2023): for organ o and k = 0..3, logit f_ok gives P(future > k | future > k - 1).
Output k trains only on anchors whose future score is above k - 1, and the cumulative probabilities
P(future > k) are running products of the sigmoids, so they cannot cross. The composite output is the
probability of total change >= 2 or death (labels/README.md, sections 7.1 and 7.2).
"""
import torch
from torch import nn
import torch.nn.functional as F

THRESHOLDS = 4


class SofaHead(nn.Module):
    """6 organs x 4 CORN logits and 1 composite logit.

    The baseline path reads the EHR context (E0 or E2). The waveform path reads a waveform feature and the same
    context and adds to the same logits; its last layer starts at zero, so training starts from the baseline model.
    `context_dim=0` gives a waveform-only model; `wave_dim=0` gives a baseline-only control.
    """

    def __init__(self, context_dim, wave_dim=768, organs=6, hidden=256, dropout=.1):
        super().__init__()
        if not context_dim and not wave_dim:
            raise ValueError("The head needs a context or a waveform input")
        self.organs, self.outputs = organs, organs * THRESHOLDS + 1
        self.baseline = (nn.Sequential(nn.Linear(context_dim, 128), nn.GELU(), nn.LayerNorm(128), nn.Linear(128, self.outputs))
                         if context_dim else None)
        self.wave = (nn.Sequential(nn.Linear(wave_dim + context_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, self.outputs))
                     if wave_dim else None)
        if self.wave is not None:
            nn.init.zeros_(self.wave[-1].weight)
            nn.init.zeros_(self.wave[-1].bias)

    def forward(self, context, wave=None):
        rows = len(context) if context is not None else len(wave)
        device = context.device if context is not None else wave.device
        logits = torch.zeros(rows, self.outputs, device=device)
        if self.baseline is not None:
            logits = logits + self.baseline(context)
        if self.wave is not None:
            inputs = wave if self.baseline is None else torch.cat((wave, context), dim=-1)
            logits = logits + self.wave(inputs)
        return {"organ": logits[:, :-1].reshape(rows, self.organs, THRESHOLDS), "composite": logits[:, -1]}


def cumulative(logits):
    """P(future > k) for k = 0..3, the running product of CORN's conditional probabilities."""
    return torch.sigmoid(logits).cumprod(dim=-1)


def expected_score(logits):
    """Expected future score, the sum of P(future > k). Always within 0..4."""
    return cumulative(logits).sum(-1)


def corn_terms(logits, future, label):
    """Per organ, the sum and count of BCE terms over eligible anchor-threshold pairs.

    Pair (anchor, k) is eligible when the organ is labeled and future > k - 1; its target is future > k.
    """
    k = torch.arange(THRESHOLDS, device=logits.device)
    eligible = label[..., None] & (future[..., None] > k - 1)
    target = (future[..., None] > k).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * eligible).sum(dim=(0, 2)), eligible.sum(dim=(0, 2))


def composite_terms(logit, event, known):
    bce = F.binary_cross_entropy_with_logits(logit, event, reduction="none")
    return (bce * known).sum(), known.sum()


def sofa_loss_from_terms(organ_sum, organ_count, composite_sum, composite_count):
    """L_SOFA: mean over organs of each organ's mean CORN loss, plus the composite BCE. Organs weigh equally."""
    has = organ_count > 0
    organ = (organ_sum[has] / organ_count[has]).mean() if has.any() else organ_sum.sum() * 0
    composite = composite_sum / composite_count if composite_count > 0 else composite_sum * 0
    return {"organ": organ, "composite": composite, "task": organ + composite}


def sofa_loss(output, future, label, event, event_known):
    organ_sum, organ_count = corn_terms(output["organ"], future, label)
    composite_sum, composite_count = composite_terms(output["composite"], event, event_known)
    return sofa_loss_from_terms(organ_sum, organ_count, composite_sum, composite_count)
