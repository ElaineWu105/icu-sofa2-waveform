"""The three-minute model: the original extension, plus a dropped bad slot that keeps its time position."""
import torch

from long_model import LongConfig, LongCSFM


def make_config():
    """Tang et al. 2026 losses at weight 1, with the 10.24 model sizes."""
    return LongConfig(max_windows=18, prompt_tokens=8, projection_dim=256, temperature=.1, lambda_local=1.,
                      lambda_dyn=1., high_affinity_fraction=.5, gradient_checkpointing=True)


class ClipEncoder(LongCSFM):
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
    """Local ranking and local-to-global dynamics. No task loss at this level."""
    from long_model import dynamics_loss, rank_local_loss
    local = rank_local_loss(core.g_local(output["local"]).flatten(0, 1), teacher.detach().flatten(0, 1), core.config.temperature)
    dynamic = dynamics_loss(core.g_dyn(output["local"]), output["embedding"], teacher.detach(), core.config.high_affinity_fraction)
    return {"loss": core.config.lambda_local * local + core.config.lambda_dyn * dynamic, "local": local, "dynamic": dynamic}
