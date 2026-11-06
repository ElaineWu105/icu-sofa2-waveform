"""The three-minute model: the original extension, a dropped bad slot that keeps its time position,
and masked window prediction in the place of the paper's task loss."""
import torch

from long_model import LongConfig, LongCSFM, dynamics_loss, rank_local_loss, select_high_affinity
from temporal_model import MASK_RATIO, MaskedDecoder, centered, masked_mse, reference_errors, visible_count

WINDOWS = 18


def make_config():
    """Tang et al. 2026 losses at weight 1, with the 10.24 model sizes."""
    return LongConfig(max_windows=WINDOWS, prompt_tokens=8, projection_dim=256, temperature=.1, lambda_local=1.,
                      lambda_dyn=1., high_affinity_fraction=.5, gradient_checkpointing=True)


class ClipEncoder(LongCSFM):
    def __init__(self, encoder, config=None):
        super().__init__(encoder, config)
        d = encoder.cfg.dim
        self.decoder = MaskedDecoder(self.config.max_windows, d, 256, 2, 8, d)

    def forward(self, signal, channel_ids, positions=None):
        if positions is None:
            return super().forward(signal, channel_ids)
        original = self.global_pos
        b, windows, channels, _ = self._validate(signal, channel_ids)
        wave = self._wave_embedding(signal, channel_ids)
        wave = (wave + original[positions][:, :, None, None, :]).flatten(1, 3)
        tokens = torch.cat((self.encoder.cls_token.expand(b, -1, -1), wave), dim=1)
        for depth, (attention, feed_forward) in enumerate(self.encoder.transformer.layers):
            tokens = torch.cat((self.prompts[depth][None].expand(b, -1, -1), tokens), dim=1)
            tokens = tokens + attention(tokens)
            tokens = tokens + feed_forward(tokens)
            tokens = tokens[:, self.config.prompt_tokens:]
        tokens = self.encoder.transformer.norm(tokens)
        local = tokens[:, 1:].reshape(b, windows, channels * self.encoder.cfg.num_patches, -1).mean(2)
        return {"embedding": local.mean(dim=1), "local": local}


def ssl_loss(core, output, teacher):
    """Local ranking and local-to-global dynamics, on the windows the student saw."""
    local = rank_local_loss(core.g_local(output["local"]).flatten(0, 1), teacher.detach().flatten(0, 1), core.config.temperature)
    dynamic = dynamics_loss(core.g_dyn(output["local"]), output["embedding"], teacher.detach(), core.config.high_affinity_fraction)
    return {"loss": core.config.lambda_local * local + core.config.lambda_dyn * dynamic, "local": local, "dynamic": dynamic}


def draw_windows(batch, windows=WINDOWS, ratio=MASK_RATIO, generator=None):
    """Visible windows of clean clips, index[B,V] sorted in time; 4 of 18 at 75%."""
    count = visible_count(windows, ratio)
    order = torch.rand(batch, windows, generator=generator).argsort(dim=1)
    return order[:, :count].sort(dim=1).values


def masked_forward(core, signal, channel_ids, teacher, visible):
    """Student on the visible windows only; the decoder predicts every window's standardized teacher.

    signal[B,W,3,L] holds all windows of clean clips, teacher[B,W,D] their frozen features.
    Returns the student output, the masked loss, and what the reference checks need.
    """
    rows = torch.arange(len(signal), device=signal.device)[:, None]
    output = core(signal[rows, visible], channel_ids[rows, visible], visible)
    valid = torch.ones(visible.shape, dtype=torch.bool, device=signal.device)
    present = torch.ones(teacher.shape[:2], dtype=torch.bool, device=signal.device)
    prediction, shown = core.decoder(output["local"], visible, valid, present)
    target = core.decoder.standardize(teacher, present)
    masked = ~shown
    return {"output": output, "masked": masked_mse(prediction, target, masked), "target": target, "valid": valid,
            "masked_slots": masked, "visible_teacher": teacher[rows, visible]}


def check_terms(core, step, visible):
    """Copy and mean predictors of masked windows, and L_dyn without its projector."""
    references = reference_errors(step["target"], visible, step["valid"], step["masked_slots"])
    output, teacher = step["output"], step["visible_teacher"].float()
    with torch.no_grad():
        chosen = select_high_affinity(teacher, core.config.high_affinity_fraction)
        local = output["local"].float().gather(1, chosen[..., None].expand(-1, -1, output["local"].shape[-1]))
        target = output["embedding"].float()[:, None].expand_as(local)
        unprojected = (1 - torch.nn.functional.cosine_similarity(local, target, dim=-1)).mean()
    return {**references, "dynamic_unprojected": unprojected}
