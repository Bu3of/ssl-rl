"""Categorical (C51) Bellman projection (Bellemare, Dabney & Munos, 2017).

Shifts and rescales the next-state atom support by the n-step reward and
discount, then splits each atom's probability mass linearly onto the two
nearest fixed support points, since the shifted atoms generally fall between
them.
"""
from __future__ import annotations

import torch


def project_distribution(next_probs, rewards, dones, gamma_n, support, v_min, v_max, num_atoms):
    batch_size = next_probs.shape[0]
    delta_z = (v_max - v_min) / (num_atoms - 1)

    Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * gamma_n * support.unsqueeze(0)
    Tz = Tz.clamp(v_min, v_max)
    b = (Tz - v_min) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()

    # When b lands exactly on a support point, lower == upper and no mass
    # would be distributed; nudge one side so the atom still receives it.
    lower[(upper > 0) & (lower == upper)] -= 1
    upper[(lower < num_atoms - 1) & (lower == upper)] += 1

    projected = torch.zeros(batch_size, num_atoms, device=next_probs.device)
    offset = (
        torch.arange(batch_size, device=next_probs.device) * num_atoms
    ).unsqueeze(1).expand(batch_size, num_atoms)

    projected.view(-1).index_add_(
        0, (lower + offset).reshape(-1), (next_probs * (upper.float() - b)).reshape(-1)
    )
    projected.view(-1).index_add_(
        0, (upper + offset).reshape(-1), (next_probs * (b - lower.float())).reshape(-1)
    )
    return projected
