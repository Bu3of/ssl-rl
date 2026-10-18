"""Rainbow agent (Hessel et al., 2017): double Q-learning + dueling architecture
+ categorical (C51) value distribution + prioritized replay + n-step returns
+ noisy-net exploration, wired around a shared encoder.

Optionally takes an `aux_head` (SPRHead, CURLHead, ...) to add a self-supervised
auxiliary loss on top of the RL loss -- each head implements its own
`sample_and_compute_loss(online_encoder, agent)`, pulling whatever data shape
it needs from `agent.buffer` (single frames) or `agent.sequence_buffer`
(raw K-step windows, only needed by SPR-style multi-step heads). With
aux_head left as None this is exactly the Phase 0 baseline agent, unchanged.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from agents.categorical_projection import project_distribution
from agents.rainbow_net import RainbowNet
from agents.replay_buffer import PrioritizedReplayBuffer


class RainbowAgent:
    def __init__(
        self,
        encoder: nn.Module,
        num_actions: int,
        obs_shape: tuple,
        device: torch.device,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        gamma: float = 0.99,
        n_step: int = 3,
        buffer_capacity: int = 100_000,
        batch_size: int = 32,
        lr: float = 1e-4,
        target_update_interval: int = 1000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        aux_head=None,
        sequence_buffer=None,
        aux_batch_size: int = 32,
        aux_loss_weight: float = 1.0,
    ):
        self.device = device
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        self.online_net = RainbowNet(encoder, num_actions, num_atoms, v_min, v_max).to(device)
        self.target_net = copy.deepcopy(self.online_net).to(device)
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.aux_head = aux_head.to(device) if aux_head is not None else None
        self.sequence_buffer = sequence_buffer
        self.aux_batch_size = aux_batch_size
        self.aux_loss_weight = aux_loss_weight

        self.trainable_params = list(self.online_net.parameters())
        if self.aux_head is not None:
            self.trainable_params += [p for p in self.aux_head.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable_params, lr=lr, eps=1.5e-4)

        self.buffer = PrioritizedReplayBuffer(
            buffer_capacity, obs_shape, alpha=per_alpha, beta_start=per_beta_start, beta_frames=per_beta_frames
        )
        self.n_step_buffer: deque = deque(maxlen=n_step)
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        self.online_net.eval()
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        q_values = self.online_net(obs_t)
        self.online_net.train()
        return int(q_values.argmax(dim=1).item())

    def _n_step_transition(self):
        reward = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r
        obs, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_obs, done = self.n_step_buffer[-1]
        return obs, action, reward, next_obs, done

    def store_transition(self, obs, action, reward, next_obs, done):
        if self.sequence_buffer is not None:
            self.sequence_buffer.add(obs, action, done)

        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(self.n_step_buffer) == self.n_step:
            self.buffer.add(*self._n_step_transition())
            self.n_step_buffer.popleft()
        if done:
            while len(self.n_step_buffer) > 0:
                self.buffer.add(*self._n_step_transition())
                self.n_step_buffer.popleft()

    def _rl_loss(self):
        batch, tree_indices = self.buffer.sample(self.batch_size)
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones = torch.from_numpy(batch["dones"]).to(self.device)
        weights = torch.from_numpy(batch["weights"]).to(self.device)
        batch_indices = torch.arange(obs.shape[0], device=self.device)

        with torch.no_grad():
            next_actions = self.online_net(next_obs).argmax(dim=1)
            next_probs = self.target_net.dist(next_obs)[batch_indices, next_actions]
            gamma_n = self.gamma ** self.n_step
            target_probs = project_distribution(
                next_probs, rewards, dones, gamma_n,
                self.online_net.support, self.v_min, self.v_max, self.num_atoms,
            )

        probs = self.online_net.dist(obs)
        probs_a = probs[batch_indices, actions]
        log_probs_a = torch.log(probs_a.clamp(min=1e-8))

        td_errors = -(target_probs * log_probs_a).sum(dim=1)
        loss = (td_errors * weights).mean()
        return loss, tree_indices, td_errors

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        loss, tree_indices, td_errors = self._rl_loss()

        aux_loss_value = None
        if self.aux_head is not None:
            aux_loss = self.aux_head.sample_and_compute_loss(self.online_net.encoder, self)
            if aux_loss is not None:
                loss = loss + self.aux_loss_weight * aux_loss
                aux_loss_value = aux_loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.trainable_params, 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(tree_indices, td_errors.detach().cpu().numpy())

        self.online_net.reset_noise()
        self.target_net.reset_noise()

        if self.aux_head is not None:
            self.aux_head.update_momentum_encoder(self.online_net.encoder)

        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        if aux_loss_value is not None:
            return loss.item(), aux_loss_value
        return loss.item()

    def state_dict(self):
        state = {
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learn_steps": self.learn_steps,
        }
        if self.aux_head is not None:
            state["aux_head"] = self.aux_head.state_dict()
        return state

    def load_state_dict(self, state):
        self.online_net.load_state_dict(state["online_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.learn_steps = state["learn_steps"]
        if self.aux_head is not None and "aux_head" in state:
            self.aux_head.load_state_dict(state["aux_head"])
