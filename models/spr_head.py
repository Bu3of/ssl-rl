"""SPR (Self-Predictive Representations, Schwarzer et al., ICLR 2021) auxiliary head.

Trains the shared encoder to make its embeddings self-predictive: given the
embedding of the current frame plus a chain of actions, a learned transition
model predicts the embeddings of the next K frames, and those predictions are
trained (cosine similarity, SimSiam/BYOL-style stop-gradient) to match a
momentum ("target") encoder's real embeddings of those future frames. The
momentum encoder is an EMA copy of the online encoder, updated after every
gradient step -- distinct from Rainbow's own hard-synced target_net, which
exists purely for the Q-learning bootstrap target.

Two adaptations from the original (real-Atari) paper, made explicit here:
- Augmentation is a small random-shift crop sized for MinAtar's 10x10 grid
  (the original uses a larger shift + intensity jitter tuned for 84x84
  frames); see `random_shift`.
- Augmentation is sampled independently per frame rather than held constant
  across a whole trajectory, which is simpler to implement correctly and
  still serves the same purpose (preventing a trivial shortcut solution).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def random_shift(x: torch.Tensor, pad: int = 2) -> torch.Tensor:
    """Independent random translation per sample, via reflect-pad + crop."""
    n, c, h, w = x.shape
    if pad == 0:
        return x
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")

    shift_h = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    shift_w = torch.randint(0, 2 * pad + 1, (n,), device=x.device)

    rows = shift_h.view(n, 1) + torch.arange(h, device=x.device).view(1, h)
    cols = shift_w.view(n, 1) + torch.arange(w, device=x.device).view(1, w)

    rows_idx = rows.view(n, 1, h, 1).expand(n, c, h, w + 2 * pad)
    xp = torch.gather(xp, 2, rows_idx)

    cols_idx = cols.view(n, 1, 1, w).expand(n, c, h, w)
    return torch.gather(xp, 3, cols_idx)


class TransitionModel(nn.Module):
    def __init__(self, embedding_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(action, self.num_actions).float()
        return z + self.net(torch.cat([z, a_onehot], dim=-1))


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.out_features = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SPRHead(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_actions: int,
        sequence_length: int = 5,
        momentum_tau: float = 0.99,
        aug_pad: int = 2,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.momentum_tau = momentum_tau
        self.aug_pad = aug_pad

        embedding_dim = encoder.embedding_dim

        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.transition_model = TransitionModel(embedding_dim, num_actions)
        self.online_projection = MLPHead(embedding_dim, proj_dim)
        self.target_projection = copy.deepcopy(self.online_projection)
        for p in self.target_projection.parameters():
            p.requires_grad_(False)
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

        z0 = online_encoder(online_view[:, 0])

        with torch.no_grad():
            target_embeds = self.target_encoder(target_view.reshape(b * k1, c, h, w))
            target_proj = self.target_projection(target_embeds).view(b, k1, -1)[:, 1:]  # steps 1..K

        preds = []
        z = z0
        for t in range(k):
            z = self.transition_model(z, actions_seq[:, t])
            preds.append(z)
        preds = torch.stack(preds, dim=1)  # (B, K, embedding_dim)

        online_proj = self.online_projection(preds.reshape(b * k, -1)).view(b, k, -1)
        online_pred = self.prediction_head(online_proj.reshape(b * k, -1)).view(b, k, -1)

        online_pred = F.normalize(online_pred, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1)

        cosine_sim = (online_pred * target_proj).sum(dim=-1)  # (B, K)
        loss_per_step = 1.0 - cosine_sim
        return (loss_per_step * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def update_momentum_encoder(self, online_encoder):
        tau = self.momentum_tau
        for online_p, target_p in zip(online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)
        for online_p, target_p in zip(self.online_projection.parameters(), self.target_projection.parameters()):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)
