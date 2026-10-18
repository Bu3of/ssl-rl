"""iQRL-style auxiliary head (adapted from Scannell et al., 2024, "iQRL:
Implicitly Quantized Representations for Sample-Efficient RL").

Same multi-step self-predictive structure as SPR (`models/spr_head.py`): a
transition model predicts a chain of future latents from the current latent
plus a chain of actions, trained to match the true future latents. The
difference is the anti-collapse mechanism: instead of a momentum target
network, iQRL argues that passing the latent through a *quantization*
bottleneck (FSQ -- Finite Scalar Quantization, Mentzer et al., 2023) is
itself sufficient to prevent collapse, since a collapsed (near-constant)
representation gets no special advantage under quantization the way it can
under a purely continuous embedding.

To isolate quantization as the variable under test -- this project's actual
research question -- this head deliberately does NOT use a momentum
encoder. The "target" branch is the online encoder itself, stop-gradient
only, exactly matching the `momentum_tau=0.0` SPR ablation already run; the
only difference between that ablation and this head is the FSQ bottleneck.
This also means it needs no separate target_encoder copy.

Adapted, not a literal reproduction: iQRL's original paper targets
DMControl/TD3 (continuous control) with its own FSQ level configuration;
this keeps the same environment/algorithm/encoder as every other head in
this project's comparison instead, per the project's controlled-comparison
design, and the FSQ dimensionality/levels here (8 dims x 5 levels) are a
reasonable choice in the same spirit as the paper's small per-dimension
level counts, not a copy of its exact published configuration.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spr_head import MLPHead, TransitionModel, random_shift


class FSQ(nn.Module):
    """Bounds each dimension to [-1, 1] (tanh) then rounds it to one of
    `levels` evenly spaced points, with a straight-through gradient
    estimator (forward uses the quantized value, backward passes the
    gradient through as if quantization were the identity)."""

    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(x)
        half = (self.levels - 1) / 2
        quantized = torch.round(bounded * half) / half
        return bounded + (quantized - bounded).detach()


class IQRLHead(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_actions: int,
        sequence_length: int = 5,
        fsq_dim: int = 8,
        fsq_levels: int = 5,
        aug_pad: int = 2,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.aug_pad = aug_pad

        embedding_dim = encoder.embedding_dim
        self.quantize_proj = nn.Linear(embedding_dim, fsq_dim)
        self.fsq = FSQ(levels=fsq_levels)

        self.transition_model = TransitionModel(fsq_dim, num_actions)
        self.online_projection = MLPHead(fsq_dim, proj_dim)
        self.prediction_head = MLPHead(proj_dim, proj_dim)

    def sample_and_compute_loss(self, online_encoder, agent):
        if agent.sequence_buffer is None or not agent.sequence_buffer.can_sample():
            return None
        obs_seq, actions_seq, mask = agent.sequence_buffer.sample(agent.aux_batch_size)
        obs_seq = torch.from_numpy(obs_seq).to(agent.device)
        actions_seq = torch.from_numpy(actions_seq).to(agent.device)
        mask = torch.from_numpy(mask).to(agent.device)
        return self.compute_loss(online_encoder, obs_seq, actions_seq, mask)

    def compute_loss(self, online_encoder, obs_seq, actions_seq, mask):
        b, k1, c, h, w = obs_seq.shape
        k = k1 - 1

        online_view = random_shift(obs_seq.reshape(b * k1, c, h, w), self.aug_pad).view(b, k1, c, h, w)
        target_view = random_shift(obs_seq.reshape(b * k1, c, h, w), self.aug_pad).view(b, k1, c, h, w)

        embeds = online_encoder(online_view.reshape(b * k1, c, h, w))
        q_all = self.fsq(self.quantize_proj(embeds)).view(b, k1, -1)
        q0 = q_all[:, 0]

        with torch.no_grad():
            target_embeds = online_encoder(target_view.reshape(b * k1, c, h, w))
            target_q = self.fsq(self.quantize_proj(target_embeds)).view(b, k1, -1)[:, 1:]  # steps 1..K

        preds = []
        q = q0
        for t in range(k):
            q = self.transition_model(q, actions_seq[:, t])
            preds.append(q)
        preds = torch.stack(preds, dim=1)  # (B, K, fsq_dim)

        online_proj = self.online_projection(preds.reshape(b * k, -1)).view(b, k, -1)
        online_pred = self.prediction_head(online_proj.reshape(b * k, -1)).view(b, k, -1)

        target_proj = self.online_projection(target_q.reshape(b * k, -1)).view(b, k, -1)

        online_pred = F.normalize(online_pred, dim=-1)
        target_proj = F.normalize(target_proj.detach(), dim=-1)

        cosine_sim = (online_pred * target_proj).sum(dim=-1)
        loss_per_step = 1.0 - cosine_sim
        return (loss_per_step * mask).sum() / mask.sum().clamp(min=1.0)

    def update_momentum_encoder(self, online_encoder):
        pass  # no momentum target -- quantization is the only anti-collapse mechanism under test
