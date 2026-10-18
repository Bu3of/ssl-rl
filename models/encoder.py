"""Shared CNN encoder for MinAtar observations.

Architecture follows the MinAtar DQN baseline (Young & Tian, 2019): a single
3x3 stride-1 conv layer over the 10x10 binary grid, flattened into one fully
connected layer. This embedding is the shared input to the Q-head and to
whichever SSL auxiliary head (SPR/CURL/iQRL) is attached, so its output stays
a plain embedding vector with no task-specific head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_GRID_SIZE = 10
_CONV_OUT_SIZE = _GRID_SIZE - 3 + 1  # 3x3 kernel, stride 1, no padding -> 8


class MinAtarEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        self.fc = nn.Linear(16 * _CONV_OUT_SIZE * _CONV_OUT_SIZE, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = x.flatten(start_dim=1)
        return torch.relu(self.fc(x))
