"""Representation-collapse diagnostic: effective rank of an encoder's embeddings.

Effective rank (Roy & Vetterli, 2007) treats the normalized singular value
spectrum of the (centered) embedding matrix as a probability distribution and
takes its exponentiated entropy. A collapsed representation (all embeddings
pointing the same direction, or many dead units) concentrates variance into
very few singular values and gets an effective rank near 1; a healthy,
information-rich representation spreads variance across many dimensions and
gets a higher effective rank, up to the embedding dimension.

This is the core measurement behind the project's research question -- which
self-supervised objective best resists collapse -- and is used identically
whether the encoder was trained with no auxiliary loss (baseline), SPR,
CURL, or iQRL.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def effective_rank(embeddings: torch.Tensor) -> float:
    x = embeddings - embeddings.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    p = s / s.sum().clamp(min=1e-12)
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def collect_embeddings(encoder, agent, env, n=512):
    """Rolls the agent's current policy out in `env` (a fresh instance -- do not
    reuse the training env, since this consumes steps) and encodes the visited
    observations, giving a diverse batch to measure representation quality on."""
    obs = env.reset()
    batch = []
    for _ in range(n):
        action = agent.act(obs)
        obs, _, done, _, _ = env.step(action)
        batch.append(obs)
        if done:
            obs = env.reset()
    import numpy as np

    obs_batch = torch.from_numpy(np.stack(batch)).float()
    return encoder(obs_batch)
