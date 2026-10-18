"""Rainbow Q-network: dueling value/advantage streams over a categorical (C51)
value distribution, with NoisyLinear layers standing in for epsilon-greedy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.noisy_linear import NoisyLinear


class RainbowNet(nn.Module):
    def __init__(self, encoder, num_actions, num_atoms=51, v_min=-10.0, v_max=10.0, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        embedding_dim = encoder.embedding_dim
        self.value_hidden = NoisyLinear(embedding_dim, hidden_dim)
        self.value_out = NoisyLinear(hidden_dim, num_atoms)
        self.advantage_hidden = NoisyLinear(embedding_dim, hidden_dim)
        self.advantage_out = NoisyLinear(hidden_dim, num_actions * num_atoms)

    def reset_noise(self):
        for module in (self.value_hidden, self.value_out, self.advantage_hidden, self.advantage_out):
            module.reset_noise()

    def dist(self, x):
        z = self.encoder(x)

        value = self.value_out(F.relu(self.value_hidden(z)))
        value = value.view(-1, 1, self.num_atoms)

        advantage = self.advantage_out(F.relu(self.advantage_hidden(z)))
        advantage = advantage.view(-1, self.num_actions, self.num_atoms)

        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=-1)

    def forward(self, x):
        probs = self.dist(x)
        return (probs * self.support).sum(dim=-1)
