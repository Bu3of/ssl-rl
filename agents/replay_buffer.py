"""Proportional prioritized experience replay (Schaul et al., 2016).

Backed by an array-based sum-tree so both sampling and priority updates are
O(log capacity) instead of the O(capacity) an argsort/np.random.choice
approach would cost at the buffer sizes Rainbow needs.

Stores n-step-folded transitions (obs at the start of the n-step window,
next_obs n steps later) for the Q-learning loss. SPR's multi-step latent
prediction needs raw, un-folded per-step sequences instead — see
`sequence_buffer.py` for that.
"""
from __future__ import annotations

import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)

    def add(self, priority, data_index):
        tree_index = data_index + self.capacity - 1
        self.update(tree_index, priority)

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_index = leaf - (self.capacity - 1)
        return leaf, self.tree[leaf], data_index

    @property
    def total_priority(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, alpha=0.6, beta_start=0.4, beta_frames=100_000, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        self.frame = 1
        self.max_priority = 1.0  # tracked incrementally; scanning the tree would be O(capacity)

        self.tree = SumTree(capacity)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def __len__(self):
        return self.size

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.tree.add(self.max_priority, self.ptr)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _beta(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def sample(self, batch_size):
        beta = self._beta()
        self.frame += 1

        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float32)

        segment = self.tree.total_priority / batch_size
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            value = np.random.uniform(lo, hi)
            tree_idx, priority, data_idx = self.tree.get_leaf(value)
            tree_indices[i] = tree_idx
            priorities[i] = priority
            indices[i] = data_idx

        probs = priorities / self.tree.total_priority
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        batch = dict(
            obs=self.obs[indices], actions=self.actions[indices], rewards=self.rewards[indices],
            next_obs=self.next_obs[indices], dones=self.dones[indices], weights=weights.astype(np.float32),
        )
        return batch, tree_indices

    def update_priorities(self, tree_indices, td_errors):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), float(priority))
        self.max_priority = max(self.max_priority, float(priorities.max()))
