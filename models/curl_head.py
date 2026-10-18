"""CURL (Contrastive Unsupervised Representations for RL, Srinivas, Laskin &
Abbeel, ICML 2020) auxiliary head.

Instance discrimination: two independently-augmented views of the same
observation should be identifiable against a batch of other observations'
views. The "query" view goes through the online encoder, the "key" view
through a momentum (EMA) target encoder with stop-gradient -- same momentum
mechanism as SPR (`models/spr_head.py`, whose `random_shift` augmentation
this reuses), but the loss itself is contrastive (InfoNCE via a learned
bilinear similarity, as in the original paper) rather than SPR's multi-step
self-predictive regression. Unlike SPR, CURL needs only single frames, not
K-step sequences, so it samples directly from the agent's main replay buffer
instead of a separate sequence buffer.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spr_head import random_shift


class CURLHead(nn.Module):
    def __init__(self, encoder: nn.Module, momentum_tau: float = 0.99, aug_pad: int = 2):
        super().__init__()
        self.momentum_tau = momentum_tau
        self.aug_pad = aug_pad

        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        embedding_dim = encoder.embedding_dim
        # bilinear similarity matrix W (Srinivas et al.): logits = z_q @ W @ z_k^T
        self.W = nn.Parameter(torch.rand(embedding_dim, embedding_dim))

    def sample_and_compute_loss(self, online_encoder, agent):
        if len(agent.buffer) < agent.aux_batch_size:
            return None
        batch, _ = agent.buffer.sample(agent.aux_batch_size)
        obs = torch.from_numpy(batch["obs"]).to(agent.device)
        return self.compute_loss(online_encoder, obs)

    def compute_loss(self, online_encoder, obs):
        query_view = random_shift(obs, self.aug_pad)
        key_view = random_shift(obs, self.aug_pad)

        z_q = online_encoder(query_view)
        with torch.no_grad():
            z_k = self.target_encoder(key_view)

        logits = z_q @ self.W @ z_k.T  # (B, B); logits[i, j] = similarity between query i and key j
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # numerical stability only

        # the positive for query i is key i (same underlying observation, different augmentation);
        # every other key in the batch is a negative.
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def update_momentum_encoder(self, online_encoder):
        tau = self.momentum_tau
        for online_p, target_p in zip(online_encoder.parameters(), self.target_encoder.parameters()):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1 - tau)
