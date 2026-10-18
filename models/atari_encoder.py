"""CNN encoder for Atari-100k observations: the standard "Nature DQN" architecture
(Mnih et al., 2015) that essentially all Atari-100k Rainbow/SPR-family work builds
on. A drop-in replacement for MinAtarEncoder (same `.embedding_dim` interface)
wherever the observation is (4, 84, 84) instead of MinAtar's (C, 10, 10).

Note: matching SPR's own published numbers exactly also requires matching its
other hyperparameters (replay ratio, learning rate schedule, etc.) beyond just
this architecture -- worth double-checking against the paper before treating a
result here as a literal reproduction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AtariEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9x9 -> 7x7
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return torch.relu(self.fc(x))
